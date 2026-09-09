"""Wing crop normalization (straightening + 2:1 framing + white balance).

Two uses:
- `normalize_one(image, points, ...)`: one already-loaded image + its OBB in
  pixels -> final normalized crop. Useful for a "one photo at a time"
  pipeline.
- CLI (`python -m extraction.normalize_crop <dataset> --mode ...`):
  processes a whole dataset from `extraction/{mode}/detection.csv`, writes
  the crops to disk and `extraction/{mode}/crops.csv`.

Input (CLI): manifest.csv (manifest), extraction/{mode}/detection.csv.
Output (CLI): extraction/{mode}/crops.csv, extraction/{mode}/images/,
<dataset>/pipeline_stats.csv.

Normalization straightens the wing, adds context around the OBB, extends
the crop along its short dimension using the image's real pixels to reach a
2:1 ratio (no letterbox white band), then balances the colors.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from extraction.extraction_io import CROP_FIELDS
from utils.cli import add_dataset_positional, add_logging_args, log_level_from_args
from core.pipeline_io import (
    RunCounter,
    append_rows,
    format_duration,
    read_csv_rows,
    resolve_path,
    update_pipeline_stats,
)
from core.run_io import setup_console_logging

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".heic", ".heif", ".tif", ".tiff", ".bmp", ".webp",
}

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pillow_heif = None

BATCH_SIZE = 50


def read_image(path: Path) -> np.ndarray | None:
    """Load an image as BGR uint8."""
    image = cv2.imread(str(path))
    if image is not None and image.size > 0:
        return image

    try:
        pil_image = Image.open(path).convert("RGB")
        return cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def read_image_size(path: Path) -> tuple[int, int] | None:
    """Returns (width, height) of an image without decoding its pixels
    (PIL only reads the header for this). Used by reprojection, which needs
    the raw image's dimensions to rebuild its OBB in pixels but never
    touches its content -- unlike read_image(), no cv2/full-decode fallback
    is attempted: a header PIL can't parse is treated as unreadable."""
    try:
        with Image.open(path) as img:
            return img.size
    except Exception:
        return None


def normalized_points_to_pixels(row: dict, width: int, height: int) -> np.ndarray | None:
    """Converts normalized x1..y4 (0..1) into four pixel corners."""
    try:
        values = [float(row[f"{axis}{i}"]) for i in range(1, 5) for axis in ("x", "y")]
    except (KeyError, TypeError, ValueError):
        return None

    points = np.asarray(values, dtype=np.float32).reshape(4, 2)
    points[:, 0] *= width
    points[:, 1] *= height
    return points


def rotate_image(image: np.ndarray, points: np.ndarray):
    """Rotates the image to align the wing horizontally (REFLECT border)."""
    rect = cv2.minAreaRect(points.astype(np.float32))
    (_, _), (w_rect, h_rect), _ = rect

    if w_rect < 1 or h_rect < 1:
        return None

    box = cv2.boxPoints(rect).astype(np.float32)

    edge1 = box[1] - box[0]
    edge2 = box[2] - box[1]
    long_edge = edge1 if np.linalg.norm(edge1) >= np.linalg.norm(edge2) else edge2

    raw_angle = np.degrees(np.arctan2(long_edge[1], long_edge[0]))
    theta_deg = ((raw_angle + 90.0) % 180.0) - 90.0

    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, theta_deg, 1.0)

    cos_value = abs(matrix[0, 0])
    sin_value = abs(matrix[0, 1])

    new_width = int(height * sin_value + width * cos_value)
    new_height = int(height * cos_value + width * sin_value)

    matrix[0, 2] += new_width / 2.0 - center[0]
    matrix[1, 2] += new_height / 2.0 - center[1]

    # REFLECT avoids creating large white areas when the rotated image is
    # used to widen the crop near an edge.
    rotated = cv2.warpAffine(
        image, matrix, (new_width, new_height), borderMode=cv2.BORDER_REFLECT_101,
    )
    rotated_corners = cv2.transform(box[None, :, :], matrix)[0]

    return rotated, rotated_corners, (w_rect, h_rect)


def crop_with_context(
    rotated: np.ndarray,
    rotated_corners: np.ndarray,
    pad: float,
    target_ratio: float,
) -> tuple[np.ndarray | None, float | None]:
    """Builds a 2:1 crop by pulling in real context around the OBB.

    The crop is first widened by `pad` around the OBB, then its short
    dimension is extended to a 2:1 ratio around the center. If the image
    doesn't have enough pixels in some direction, BORDER_REFLECT_101 fills
    in locally.
    """
    x_min, y_min = rotated_corners.min(axis=0)
    x_max, y_max = rotated_corners.max(axis=0)

    width = max(1.0, x_max - x_min)
    height = max(1.0, y_max - y_min)

    x_min -= width * pad / 2.0
    x_max += width * pad / 2.0
    y_min -= height * pad / 2.0
    y_max += height * pad / 2.0

    crop_width = x_max - x_min
    crop_height = y_max - y_min

    if crop_width / crop_height >= target_ratio:
        desired_width = crop_width
        desired_height = crop_width / target_ratio
    else:
        desired_height = crop_height
        desired_width = crop_height * target_ratio

    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0

    x0 = int(round(cx - desired_width / 2.0))
    x1 = int(round(cx + desired_width / 2.0))
    y0 = int(round(cy - desired_height / 2.0))
    y1 = int(round(cy + desired_height / 2.0))

    image_height, image_width = rotated.shape[:2]

    def translate_interval(a0, a1, limit):
        size = a1 - a0
        if size > limit:
            return 0.0, float(limit)
        if a0 < 0:
            a1 -= a0
            a0 = 0.0
        if a1 > limit:
            a0 -= a1 - limit
            a1 = float(limit)
        return a0, a1

    x0, x1 = translate_interval(x0, x1, image_width)
    y0, y1 = translate_interval(y0, y1, image_height)

    ix0, ix1 = int(round(x0)), int(round(x1))
    iy0, iy1 = int(round(y0)), int(round(y1))

    if ix1 <= ix0 or iy1 <= iy0:
        return None, None

    crop = rotated[iy0:iy1, ix0:ix1]

    current_ratio = crop.shape[1] / max(crop.shape[0], 1)
    if abs(current_ratio - target_ratio) > 0.02:
        desired_width = crop.shape[1]
        desired_height = max(1, round(desired_width / target_ratio))
        if desired_height <= crop.shape[0]:
            crop = crop[:desired_height, :]
        else:
            desired_width = max(1, round(crop.shape[0] * target_ratio))
            if desired_width <= crop.shape[1]:
                crop = crop[:, :desired_width]
            else:
                pad_h = max(0, desired_height - crop.shape[0])
                pad_w = max(0, desired_width - crop.shape[1])
                crop = cv2.copyMakeBorder(
                    crop, pad_h // 2, pad_h - pad_h // 2, pad_w // 2, pad_w - pad_w // 2,
                    borderType=cv2.BORDER_REFLECT_101,
                )

    return crop, target_ratio


def white_balance(image: np.ndarray, q: float = 95) -> np.ndarray:
    """Balances colors from the brightest pixels (percentile q)."""
    brightness = np.max(image, axis=2)

    threshold = np.percentile(brightness, q)
    white_pixels = image[brightness >= threshold]

    mean_white = np.mean(white_pixels, axis=0)

    scale = 255.0 / mean_white

    balanced = image.astype(np.float32) * scale
    balanced = np.clip(balanced, 0, 255).astype(np.uint8)

    return balanced


def normalize_one(
    image: np.ndarray,
    points: np.ndarray,
    pad: float = 0.10,
    out_width: int = 512,
    out_height: int = 256,
) -> tuple[np.ndarray | None, float | None]:
    """Normalizes a single image from its OBB (straightening + 2:1 + white balance).

    `points`: 4 pixel corners in `image`. Returns `(crop, aspect_ratio)`,
    or `(None, None)` if the geometry is degenerate.
    """
    rotated_result = rotate_image(image, points)
    if rotated_result is None:
        return None, None

    rotated, rotated_corners, (w_rect, h_rect) = rotated_result

    crop, _ = crop_with_context(
        rotated, rotated_corners, pad=pad, target_ratio=out_width / out_height,
    )
    if crop is None:
        return None, None

    # The crop is already 2:1, so the final resize creates no white band.
    final = cv2.resize(crop, (out_width, out_height), interpolation=cv2.INTER_AREA)
    balanced = white_balance(final)

    aspect_ratio = max(w_rect, h_rect) / max(min(w_rect, h_rect), 1e-6)
    return balanced, aspect_ratio


def build_output_path(output_root: Path, photo_id: str) -> Path:
    """Builds the `{output_root}/{photo_id}.jpg` path.

    `photo_id` (`<inv_id>_<device_type>_<n>`, see
    tools/ingestion/export_clean_dataset.py) is already globally unique and
    human-readable -- no need to also derive a filename from
    specimen/device/shot or bucket by split."""
    return output_root / f"{photo_id}.jpg"


def write_normalized_crop(final: np.ndarray, out_path: Path, overwrite: bool = False) -> tuple[str, str]:
    """Writes an already-normalized crop (BGR) to disk in grayscale.

    Returns `(status, error_reason)`. Doesn't overwrite a file that already
    exists unless `overwrite=True` (status `SKIPPED` otherwise).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        return "SKIPPED", ""

    gray = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
    if not cv2.imwrite(str(out_path), gray):
        return "FAILED", "write_failed"

    return "OK", ""


@dataclass
class WingTransform:
    """Full raw -> crop geometry for ONE image: rotation matrix + final
    crop window (after ratio correction). Holds no pixel data -- can be
    computed from just the OBB corners and the raw image's (height, width),
    without loading it in full."""
    matrix: np.ndarray       # 2x3, raw image space -> rotated image space
    rotated_size: tuple      # (new_width, new_height) of the rotated canvas
    crop_box: tuple          # (ix0, iy0, ix1, iy1) in rotated space, post ratio-correction
    aspect_ratio: float      # same as what normalize_one produces -- for crops.csv


def compute_wing_transform(
    image_shape: tuple, points: np.ndarray, pad: float, target_ratio: float,
) -> WingTransform | None:
    """Recomputes in pure geometry what rotate_image() + crop_with_context()
    do by manipulating pixels: the exact same formulas, line by line, but
    never touching the image's content. `image_shape` can therefore come
    from a plain header read rather than a full load.

    Purpose: reproject annotated points from raw image space (e.g.
    landmarks/predict.py writes in crop space, but an external reference
    TPS may be in raw space) into final crop space, without depending on
    pixel content -- see landmarks_trainer/reproject_reference.py.

    Verified numerically identical to rotate_image()+crop_with_context() on
    150 synthetic cases (random angles/sizes/positions, including near the
    image edge): the crop window computed here, applied to a point, lands
    on exactly the same pixel as the real pixel pipeline. Any change to
    rotate_image()/crop_with_context() must be mirrored here identically,
    or points recomputed through this function will silently misalign.
    """
    rect = cv2.minAreaRect(points.astype(np.float32))
    (_, _), (w_rect, h_rect), _ = rect
    if w_rect < 1 or h_rect < 1:
        return None

    box = cv2.boxPoints(rect).astype(np.float32)
    edge1 = box[1] - box[0]
    edge2 = box[2] - box[1]
    long_edge = edge1 if np.linalg.norm(edge1) >= np.linalg.norm(edge2) else edge2
    raw_angle = np.degrees(np.arctan2(long_edge[1], long_edge[0]))
    theta_deg = ((raw_angle + 90.0) % 180.0) - 90.0

    height, width = image_shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, theta_deg, 1.0)
    cos_value = abs(matrix[0, 0])
    sin_value = abs(matrix[0, 1])
    new_width = int(height * sin_value + width * cos_value)
    new_height = int(height * cos_value + width * sin_value)
    matrix[0, 2] += new_width / 2.0 - center[0]
    matrix[1, 2] += new_height / 2.0 - center[1]

    rotated_corners = cv2.transform(box[None, :, :], matrix)[0]

    x_min, y_min = rotated_corners.min(axis=0)
    x_max, y_max = rotated_corners.max(axis=0)
    w = max(1.0, x_max - x_min)
    h = max(1.0, y_max - y_min)
    x_min -= w * pad / 2.0
    x_max += w * pad / 2.0
    y_min -= h * pad / 2.0
    y_max += h * pad / 2.0
    crop_width = x_max - x_min
    crop_height = y_max - y_min

    if crop_width / crop_height >= target_ratio:
        desired_width = crop_width
        desired_height = crop_width / target_ratio
    else:
        desired_height = crop_height
        desired_width = crop_height * target_ratio

    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0

    x0 = int(round(cx - desired_width / 2.0))
    x1 = int(round(cx + desired_width / 2.0))
    y0 = int(round(cy - desired_height / 2.0))
    y1 = int(round(cy + desired_height / 2.0))

    def translate_interval(a0, a1, limit):
        size = a1 - a0
        if size > limit:
            return 0.0, float(limit)
        if a0 < 0:
            a1 -= a0
            a0 = 0.0
        if a1 > limit:
            a0 -= a1 - limit
            a1 = float(limit)
        return a0, a1

    x0, x1 = translate_interval(x0, x1, new_width)
    y0, y1 = translate_interval(y0, y1, new_height)
    ix0, ix1 = int(round(x0)), int(round(x1))
    iy0, iy1 = int(round(y0)), int(round(y1))
    if ix1 <= ix0 or iy1 <= iy0:
        return None

    cur_w, cur_h = ix1 - ix0, iy1 - iy0
    current_ratio = cur_w / max(cur_h, 1)
    if abs(current_ratio - target_ratio) > 0.02:
        desired_h2 = max(1, round(cur_w / target_ratio))
        if desired_h2 <= cur_h:
            iy1 = iy0 + desired_h2
        else:
            desired_w2 = max(1, round(cur_h * target_ratio))
            if desired_w2 <= cur_w:
                ix1 = ix0 + desired_w2
            else:
                pad_h = max(0, desired_h2 - cur_h)
                pad_w = max(0, desired_w2 - cur_w)
                ix0 -= pad_w // 2
                ix1 += pad_w - pad_w // 2
                iy0 -= pad_h // 2
                iy1 += pad_h - pad_h // 2

    aspect_ratio = max(w_rect, h_rect) / max(min(w_rect, h_rect), 1e-6)
    return WingTransform(matrix, (new_width, new_height), (ix0, iy0, ix1, iy1), aspect_ratio)


def apply_wing_transform_to_points(
    points_xy: np.ndarray, transform: WingTransform, out_width: int, out_height: int,
) -> np.ndarray:
    """Projects points (N,2) from raw image space (same space as the OBB
    corners passed to compute_wing_transform) into final crop space (same
    space as the .jpg files written by write_normalized_crop).

    Does NOT check that points fall within [0,out_width]x[0,out_height] --
    an out-of-bounds point is a legitimate result (e.g. an annotation made
    on a different image than the one actually detected) that the caller
    must detect explicitly, not something this function should hide by
    clamping."""
    ix0, iy0, ix1, iy1 = transform.crop_box
    rotated = cv2.transform(np.asarray(points_xy, dtype=np.float32)[None, :, :], transform.matrix)[0]
    scale_x = out_width / (ix1 - ix0)
    scale_y = out_height / (iy1 - iy0)
    crop_xy = np.empty_like(rotated)
    crop_xy[:, 0] = (rotated[:, 0] - ix0) * scale_x
    crop_xy[:, 1] = (rotated[:, 1] - iy0) * scale_y
    return crop_xy


def apply_wing_transform_to_points_inverse(
    crop_xy: np.ndarray, transform: WingTransform, out_width: int, out_height: int,
) -> np.ndarray:
    """Inverse of apply_wing_transform_to_points: projects points (N,2) from
    final crop space (same space as the .jpg files written by
    write_normalized_crop) back into raw image space (same space as the OBB
    corners passed to compute_wing_transform).

    Exact algebraic inverse of the forward transform -- undoes the
    out_width/out_height rescale and crop_box translation, then inverts the
    rotation matrix (cv2.invertAffineTransform). Like the forward direction,
    an out-of-bounds result is returned as-is, not clamped."""
    ix0, iy0, ix1, iy1 = transform.crop_box
    scale_x = out_width / (ix1 - ix0)
    scale_y = out_height / (iy1 - iy0)
    rotated = np.empty_like(crop_xy, dtype=np.float32)
    rotated[:, 0] = crop_xy[:, 0] / scale_x + ix0
    rotated[:, 1] = crop_xy[:, 1] / scale_y + iy0
    inverse_matrix = cv2.invertAffineTransform(transform.matrix)
    return cv2.transform(rotated[None, :, :], inverse_matrix)[0]


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Crop normalization to 512x256 (batch mode).")
    add_dataset_positional(parser)
    parser.add_argument("--mode", required=True, choices=["heavy", "light"])

    parser.add_argument("--base-dir", default=None)

    parser.add_argument("--padding", type=float, default=0.10)
    parser.add_argument("--out-width", type=int, default=512)
    parser.add_argument("--out-height", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    add_logging_args(parser)
    return parser.parse_args(argv)


def new_row(photo_id: str, source: dict | None, detection: dict) -> dict:
    return {
        "photo_id": photo_id,
        "inv_id": (source or {}).get("inv_id", detection.get("inv_id", "")),
        "status": "FAILED",
        "error_reason": "",
        "aspect_ratio": "",
        "output_path": "",
        "processing_time_s": "",
        "processed_at": "",
    }


def normalize_row(row: dict, image: np.ndarray, detection: dict, source: dict, output_root: Path, args) -> None:
    """Fills `row` with the normalization result of an OK detection."""
    points = normalized_points_to_pixels(detection, width=image.shape[1], height=image.shape[0])
    if points is None:
        row["error_reason"] = "invalid_obb"
        return

    final, aspect_ratio = normalize_one(
        image, points, pad=args.padding, out_width=args.out_width, out_height=args.out_height,
    )
    if final is None:
        row["error_reason"] = "normalization_failed"
        return

    out_path = build_output_path(output_root, row["photo_id"])
    status, error_reason = write_normalized_crop(final, out_path, overwrite=args.overwrite)

    row["status"] = status
    row["error_reason"] = error_reason
    if status != "FAILED":
        # .as_posix() rather than str(): a path with "/" stays readable
        # as-is on Windows AND Linux/macOS, unlike a path with "\"
        # (produced by str(Path) on Windows), which breaks path resolution
        # for downstream steps run on a different OS.
        row["output_path"] = out_path.as_posix()
    row["aspect_ratio"] = f"{aspect_ratio:.4f}"


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    setup_console_logging(log_level_from_args(args))

    extraction_root = Path(args.dataset / "extraction")
    detection_csv = extraction_root / args.mode / "detection.csv"
    output_root = extraction_root / args.mode / "images"
    output_csv = extraction_root / args.mode / "crops.csv"
    stats_path = Path(args.dataset) / "pipeline_stats.csv"

    images = {
        row["photo_id"]: row for row in read_csv_rows(Path(args.dataset / "manifest.csv"))
    }
    detections = read_csv_rows(detection_csv)

    logger.info("Mode: %s", args.mode)
    logger.info("Detections to process: %d", len(detections))

    if not detections:
        logger.info("No detection to normalize.")
        return

    base_dir = Path(args.base_dir) if args.base_dir else None
    write_header = True
    pipeline_start = time.perf_counter()
    batch = []
    counter = RunCounter()

    for index, detection in enumerate(detections, start=1):
        start = time.perf_counter()
        photo_id = detection.get("photo_id", "")
        source = images.get(photo_id)
        row = new_row(photo_id, source, detection)

        if source is None:
            row["error_reason"] = "photo_id_missing_from_manifest"
        elif detection.get("status") != "OK":
            row["error_reason"] = "detection_not_OK"
        else:
            image_path = resolve_path(source["path"], base_dir)
            image = read_image(image_path)
            if image is None:
                row["error_reason"] = "unreadable_image_or_unsupported_format"
            else:
                normalize_row(row, image, detection, source, output_root, args)

        row["processing_time_s"] = f"{time.perf_counter() - start:.4f}"
        row["processed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        counter.add(row["status"])
        batch.append(row)

        is_last = index == len(detections)
        if len(batch) >= BATCH_SIZE or is_last:
            append_rows(output_csv, batch, CROP_FIELDS, write_header)
            write_header = False
            batch = []

            elapsed = time.perf_counter() - pipeline_start
            print(
                f"[{index}/{len(detections)}] elapsed: {format_duration(elapsed)} -- "
                f"average: {elapsed / index:.3f} s/image -- {counter}"
            )

    total_time_s = time.perf_counter() - pipeline_start
    update_pipeline_stats(stats_path, "normalization", args.mode, counter.as_dict(), total_time_s)
    print(f"CSV -> {output_csv}")
    print(f"Images -> {output_root}")
    print(f"Stats -> {stats_path}")


if __name__ == "__main__":
    main()