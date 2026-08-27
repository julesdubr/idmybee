"""Normalisation d'un crop d'aile (redressement + cadrage 2:1 + balance des blancs).

Deux usages :
- `normalize_one(image, points, ...)` : une image déjà chargée + son OBB en
  pixels -> crop final normalisé. Utile pour un pipeline "une photo à la fois".
- CLI (`python normalize_crop.py --images-csv ... --detection-csv ...`) :
  traite tout un dataset à partir de `extraction/{mode}/detection.csv`, écrit
  les crops sur disque et `extraction/{mode}/crops.csv`.

Entrée (CLI) : manifest.csv (manifest), extraction/{mode}/detection.csv.
Sortie (CLI) : extraction/{mode}/crops.csv, extraction/{mode}/images/, <dataset>/pipeline_stats.csv.

La normalisation redresse l'aile, ajoute un contexte autour de l'OBB, étend le
crop dans la dimension courte avec les pixels réels de l'image pour obtenir un
rapport 2:1 (pas de bande blanche de letterbox), puis équilibre les couleurs.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parent))

from extraction_io import CROP_FIELDS
from utils.pipeline_io import (
    RunCounter,
    append_rows,
    format_duration,
    read_csv_rows,
    resolve_path,
    update_pipeline_stats,
)

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
    """Convertit x1..y4 normalisés (0..1) en quatre sommets pixel."""
    try:
        values = [float(row[f"{axis}{i}"]) for i in range(1, 5) for axis in ("x", "y")]
    except (KeyError, TypeError, ValueError):
        return None

    points = np.asarray(values, dtype=np.float32).reshape(4, 2)
    points[:, 0] *= width
    points[:, 1] *= height
    return points


def rotate_image(image: np.ndarray, points: np.ndarray):
    """Tourne l'image pour aligner l'aile à l'horizontale (bordure REFLECT)."""
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
    """Construit un crop 2:1 en récupérant le contexte réel autour de l'OBB.

    Le crop est d'abord élargi de `pad` autour de l'OBB, puis la dimension
    courte est augmentée jusqu'au ratio 2:1 autour du centre. Si l'image ne
    contient pas assez de pixels dans une direction, BORDER_REFLECT_101
    complète localement.
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
    """Équilibre les couleurs à partir des pixels les plus brillants (percentile q)."""
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
    """Normalise une image unique à partir de son OBB (redressement + 2:1 + balance des blancs).

    `points` : 4 sommets en pixels dans `image`. Retourne `(crop, aspect_ratio)`,
    ou `(None, None)` si la géométrie est dégénérée.
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

    # Le crop est déjà 2:1, le resize final ne crée donc aucune bande blanche.
    final = cv2.resize(crop, (out_width, out_height), interpolation=cv2.INTER_AREA)
    balanced = white_balance(final)

    aspect_ratio = max(w_rect, h_rect) / max(min(w_rect, h_rect), 1e-6)
    return balanced, aspect_ratio


def build_output_path(output_root: Path, source: dict, image_id: str) -> Path:
    """Construit le chemin `{output_root}/{split}/{specimen}_{device}{shot}.jpg`."""
    split = source.get("split") or "split"
    specimen = source.get("specimen_id") or "unknown"
    device = source.get("device_type") or source.get("collector") or "x"
    shot = source.get("shot_index") or "0"

    return output_root / split / f"{specimen}-{device}{shot}.jpg"


def write_normalized_crop(final: np.ndarray, out_path: Path, overwrite: bool = False) -> tuple[str, str]:
    """Écrit un crop déjà normalisé (BGR) en niveaux de gris sur disque.

    Retourne `(status, error_reason)`. Ne réécrit pas un fichier déjà présent
    sauf si `overwrite=True` (statut `SKIPPED` sinon).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        return "SKIPPED", ""

    gray = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
    if not cv2.imwrite(str(out_path), gray):
        return "FAILED", "ecriture_impossible"

    return "OK", ""


def parse_args():
    parser = argparse.ArgumentParser(description="Normalisation des crops en 512x256 (mode dataset).")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--mode", required=True, choices=["heavy", "light"])

    parser.add_argument("--base-dir", default=None)

    parser.add_argument("--padding", type=float, default=0.10)
    parser.add_argument("--out-width", type=int, default=512)
    parser.add_argument("--out-height", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def new_row(image_id: str, source: dict | None, detection: dict) -> dict:
    return {
        "image_id": image_id,
        "specimen_id": (source or {}).get("specimen_id", detection.get("specimen_id", "")),
        "split": (source or {}).get("split", detection.get("split", "")),
        "status": "FAILED",
        "error_reason": "",
        "aspect_ratio": "",
        "output_path": "",
        "processing_time_s": "",
        "processed_at": "",
    }


def normalize_row(row: dict, image: np.ndarray, detection: dict, source: dict, output_root: Path, args) -> None:
    """Complète `row` avec le résultat de la normalisation d'une détection OK."""
    points = normalized_points_to_pixels(detection, width=image.shape[1], height=image.shape[0])
    if points is None:
        row["error_reason"] = "obb_invalide"
        return

    final, aspect_ratio = normalize_one(
        image, points, pad=args.padding, out_width=args.out_width, out_height=args.out_height,
    )
    if final is None:
        row["error_reason"] = "normalisation_impossible"
        return

    out_path = build_output_path(output_root, source, row["image_id"])
    status, error_reason = write_normalized_crop(final, out_path, overwrite=args.overwrite)

    row["status"] = status
    row["error_reason"] = error_reason
    if status != "FAILED":
        # .as_posix() plutôt que str() : un chemin avec des "/" reste lisible
        # tel quel sous Windows ET Linux/macOS, contrairement à un chemin
        # avec des "\" (produit par str(Path) sous Windows), qui casse la
        # résolution de chemin des étapes suivantes lancées sur un autre OS.
        row["output_path"] = out_path.as_posix()
    row["aspect_ratio"] = f"{aspect_ratio:.4f}"


def main():
    args = parse_args()

    extraction_root = Path(args.dataset / "extraction")
    detection_csv = extraction_root / args.mode / "detection.csv"
    output_root = extraction_root / args.mode / "images"
    output_csv = extraction_root / args.mode / "crops.csv"
    stats_path = Path(args.dataset) / "pipeline_stats.csv"

    images = {
        row["image_id"]: row for row in read_csv_rows(Path(args.dataset / "manifest.csv"))
    }
    detections = read_csv_rows(detection_csv)

    print(f"Mode : {args.mode}")
    print(f"Détections à traiter : {len(detections)}")

    if not detections:
        print("Aucune détection à normaliser.")
        return

    base_dir = Path(args.base_dir) if args.base_dir else None
    write_header = True
    pipeline_start = time.perf_counter()
    batch = []
    counter = RunCounter()

    for index, detection in enumerate(detections, start=1):
        start = time.perf_counter()
        image_id = detection.get("image_id", "")
        source = images.get(image_id)
        row = new_row(image_id, source, detection)

        if source is None:
            row["error_reason"] = "image_id_absent_de_images_csv"
        elif detection.get("status") != "OK":
            row["error_reason"] = "detection_non_OK"
        else:
            raw_path = resolve_path(source["raw_path"], base_dir)
            image = read_image(raw_path)
            if image is None:
                row["error_reason"] = "image_illisible_ou_format_non_supporte"
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
                f"[{index}/{len(detections)}] "
                f"temps écoulé : {format_duration(elapsed)} — "
                f"moyenne : {elapsed / index:.3f} s/image — "
                f"{counter}"
            )

    total_time_s = time.perf_counter() - pipeline_start
    update_pipeline_stats(stats_path, "normalization", args.mode, counter.as_dict(), total_time_s)
    print(f"CSV : {output_csv}")
    print(f"Images : {output_root}")
    print(f"Stats : {stats_path}")


if __name__ == "__main__":
    main()