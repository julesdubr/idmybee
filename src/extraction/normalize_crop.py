"""Normalisation commune des crops issus de YOLOE ou YOLO-OBB.

Entrée
------
- images.csv
- CSV d'extraction contenant x1..y4 normalisés dans l'image source

Sortie
------
- images grayscale 512x256
- CSV de normalisation

La normalisation redresse l'aile, ajoute un contexte autour de l'OBB, puis
étend le crop dans la dimension courte avec les pixels réels de l'image afin
d'obtenir un rapport 2:1. Il n'y a pas de bande blanche de letterbox.
"""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".heic", ".heif", ".tif", ".tiff", ".bmp", ".webp",
}


try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pillow_heif = None


OUTPUT_FIELDS = [
    "image_id",
    "specimen_id",
    "dataset",
    "detection_status",
    "normalization_status",
    "error_reason",
    "confidence",
    "aspect_ratio",
    "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4",
    "output_path",
    "processing_time_s",
    "processed_at",
]


def read_image(path: Path) -> np.ndarray | None:
    """Charge une image en BGR uint8."""
    image = cv2.imread(str(path))
    if image is not None and image.size > 0:
        return image

    try:
        pil_image = Image.open(path).convert("RGB")
        return cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def normalized_points_to_pixels(row: dict, width: int, height: int) -> np.ndarray | None:
    """Convertit x1..y4 normalisés en quatre sommets pixel."""
    try:
        values = [float(row[f"{axis}{i}"]) for i in range(1, 5) for axis in ("x", "y")]
    except (KeyError, TypeError, ValueError):
        return None

    points = np.asarray(values, dtype=np.float32).reshape(4, 2)
    points[:, 0] *= width
    points[:, 1] *= height
    return points


def rotate_image(image: np.ndarray, points: np.ndarray):
    """Tourne l'image comme le pipeline lourd, avec une bordure non blanche."""
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

    # REFLECT évite de créer de grandes zones blanches lorsque l'image tournée
    # est utilisée pour élargir le crop près du bord.
    rotated = cv2.warpAffine(
        image,
        matrix,
        (new_width, new_height),
        borderMode=cv2.BORDER_REFLECT_101,
    )

    rotated_corners = cv2.transform(box[None, :, :], matrix)[0]

    return rotated, rotated_corners, (w_rect, h_rect)


def crop_with_context(
    rotated: np.ndarray,
    rotated_corners: np.ndarray,
    pad: float,
    target_ratio: float,
) -> tuple[np.ndarray | None, float | None]:
    """Construit un crop 2:1 en récupérant le contexte réel autour de l'OBB.

    Le crop est d'abord élargi de `pad` autour de l'OBB, puis la dimension courte
    est augmentée jusqu'au ratio 2:1 autour du centre. Si l'image ne contient pas
    assez de pixels dans une direction, BORDER_REFLECT_101 complète localement.
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

    # L'orientation précédente garantit normalement crop_width > crop_height.
    # Dans tous les cas, on force un ratio exact sans déformer l'aile.
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

    # On utilise d'abord le maximum de pixels réels disponible.
    # Si le rectangle 2:1 dépasse le bord, on le translate vers l'intérieur
    # avant tout éventuel recours à une bordure synthétique.
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

    # En principe aucune bordure blanche n'est nécessaire.
    # Si l'image est trop petite pour contenir le rectangle 2:1 demandé,
    # réflexion de bord comme dernier recours pour conserver 512x256.
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
                    crop,
                    pad_h // 2, pad_h - pad_h // 2,
                    pad_w // 2, pad_w - pad_w // 2,
                    borderType=cv2.BORDER_REFLECT_101,
                )

    return crop, target_ratio


def white_balance(image, q=99):
    brightness = np.max(image, axis=2)
    
    threshold = np.percentile(brightness, q)
    white_pixels = image[brightness >= threshold]
    
    mean_white = np.mean(white_pixels, axis=0)
    
    scale = 255.0 / mean_white
    
    balanced = image.astype(np.float32) * scale
    balanced = np.clip(balanced, 0, 255).astype(np.uint8)

    return balanced


def normalize_crop(
    image: np.ndarray,
    points: np.ndarray,
    pad: float = 0.10,
    out_width: int = 512,
    out_height: int = 256,
):
    """Normalise une image à partir de son OBB."""
    rotated_result = rotate_image(image, points)
    if rotated_result is None:
        return None, None

    rotated, rotated_corners, (w_rect, h_rect) = rotated_result

    crop, _ = crop_with_context(
        rotated,
        rotated_corners,
        pad=pad,
        target_ratio=out_width / out_height,
    )

    if crop is None:
        return None, None

    # Le crop est déjà 2:1. Le resize final ne crée donc aucune bande blanche.
    final = cv2.resize(
        crop,
        (out_width, out_height),
        interpolation=cv2.INTER_AREA,
    )
    balanced = white_balance(final, 95)
    
    aspect_ratio = max(w_rect, h_rect) / max(min(w_rect, h_rect), 1e-6)

    return balanced, aspect_ratio


def build_output_path(output_root: Path, source: dict, image_id: str) -> Path:
    """Construit le chemin de sortie standard `{dataset}/{dataset}_{specimen}_{device}{shot}_{image_id}.jpg`."""
    dataset = source.get("dataset") or "dataset"
    specimen = source.get("specimen_id") or "unknown"
    device = source.get("device_type") or source.get("collector") or "x"
    shot = source.get("shot_index") or "0"

    return (
        output_root
        / dataset
        / f"{dataset}_{specimen}_{device}{shot}_{image_id}.jpg"
    )


def write_normalized_crop(
    final: np.ndarray,
    out_path: Path,
    overwrite: bool = False,
) -> tuple[str, str]:
    """Écrit un crop déjà normalisé (BGR) en niveaux de gris sur disque.

    Retourne `(normalization_status, error_reason)`. Ne réécrit pas un fichier
    déjà présent sauf si `overwrite=True` (statut `SKIPPED` dans ce cas).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        return "SKIPPED", ""

    gray = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
    if not cv2.imwrite(str(out_path), gray):
        return "FAILED", "ecriture_impossible"

    return "OK", ""


def parse_args():
    parser = argparse.ArgumentParser(description="Normalisation des crops en 512x256.")
    parser.add_argument("--images-csv", required=True)
    parser.add_argument("--detections-csv", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--padding", type=float, default=0.10)
    parser.add_argument("--out-width", type=int, default=512)
    parser.add_argument("--out-height", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    images_csv = Path(args.images_csv)
    detections_csv = Path(args.detections_csv)
    output_root = Path(args.output_root)
    output_csv = Path(args.csv) if args.csv else output_root / "normalized_crops.csv"

    with images_csv.open("r", newline="", encoding="utf-8-sig") as handle:
        images = {row["image_id"]: row for row in csv.DictReader(handle)}

    with detections_csv.open("r", newline="", encoding="utf-8-sig") as handle:
        detections = list(csv.DictReader(handle))

    rows_out = []

    for index, detection in enumerate(detections, start=1):
        start = time.perf_counter()
        image_id = detection.get("image_id", "")
        source = images.get(image_id)

        base_row = {
            "image_id": image_id,
            "specimen_id": source.get("specimen_id", "") if source else detection.get("specimen_id", ""),
            "dataset": source.get("dataset", "") if source else detection.get("dataset", ""),
            "detection_status": detection.get("status", ""),
            "normalization_status": "FAILED",
            "error_reason": "",
            "confidence": detection.get("confidence", ""),
            "aspect_ratio": "",
            "x1": detection.get("x1", ""), "y1": detection.get("y1", ""),
            "x2": detection.get("x2", ""), "y2": detection.get("y2", ""),
            "x3": detection.get("x3", ""), "y3": detection.get("y3", ""),
            "x4": detection.get("x4", ""), "y4": detection.get("y4", ""),
            "output_path": "",
            "processing_time_s": "",
            "processed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

        if source is None:
            base_row["error_reason"] = "image_id_absent_de_images_csv"
            base_row["processing_time_s"] = f"{time.perf_counter() - start:.4f}"
            rows_out.append(base_row)
            continue

        if detection.get("status") != "OK":
            base_row["error_reason"] = "detection_non_OK"
            base_row["processing_time_s"] = f"{time.perf_counter() - start:.4f}"
            rows_out.append(base_row)
            continue

        raw_path = Path(source["raw_path"])
        image = read_image(raw_path)
        if image is None:
            base_row["error_reason"] = "image_illisible_ou_format_non_supporte"
            base_row["processing_time_s"] = f"{time.perf_counter() - start:.4f}"
            rows_out.append(base_row)
            continue

        points = normalized_points_to_pixels(
            detection,
            width=image.shape[1],
            height=image.shape[0],
        )

        if points is None:
            base_row["error_reason"] = "obb_invalide"
            base_row["processing_time_s"] = f"{time.perf_counter() - start:.4f}"
            rows_out.append(base_row)
            continue

        final, aspect_ratio = normalize_crop(
            image,
            points,
            pad=args.padding,
            out_width=args.out_width,
            out_height=args.out_height,
        )

        if final is None:
            base_row["error_reason"] = "normalisation_impossible"
            base_row["processing_time_s"] = f"{time.perf_counter() - start:.4f}"
            rows_out.append(base_row)
            continue

        out_path = build_output_path(output_root, source, image_id)
        status, error_reason = write_normalized_crop(final, out_path, overwrite=args.overwrite)

        base_row["normalization_status"] = status
        base_row["error_reason"] = error_reason
        if status != "FAILED":
            base_row["output_path"] = str(out_path)

        base_row["aspect_ratio"] = f"{aspect_ratio:.4f}"
        base_row["processing_time_s"] = f"{time.perf_counter() - start:.4f}"
        rows_out.append(base_row)

        if index % 100 == 0 or index == len(detections):
            print(f"[{index}/{len(detections)}]")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"CSV : {output_csv}")
    print(f"Images : {output_root}")


if __name__ == "__main__":
    main()