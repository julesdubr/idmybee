"""Détection de l'aile sur une image (mode heavy YOLOE ou light YOLO-OBB).

Deux usages :
- `detect_one_image(mode, image, args)` : une image déjà chargée -> box + status.
  Utile pour un pipeline "une photo à la fois" (ex. capture depuis un appareil).
- CLI (`python detect_wing.py --dataset ... --mode ...`) : traite tout un
  dataset à partir de son manifest, écrit `extraction/{mode}/detection.csv`.

Entrée (CLI) : manifest.csv (manifest).
Sortie (CLI) : extraction/{mode}/detection.csv, <dataset>/pipeline_stats.csv.

Pour normaliser les crops à partir de detection.csv, voir `normalize_crop.py`.
Pour enchaîner détection + normalisation sur un dataset, voir `extract_wings.py`.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parent))

from extraction_io import DETECTION_FIELDS, read_images_csv, select_images
from normalize_crop import read_image
from utils.pipeline_io import RunCounter, append_rows, format_duration, resolve_path, update_pipeline_stats

BATCH_SIZE = 50


def get_backend(mode: str):
    """Importe le backend (heavy ou light) correspondant à `mode`."""
    if mode == "heavy":
        from heavy import detection as backend
    else:
        from light import detection as backend
    return backend


def detect_one_image(mode: str, ctx: dict, image: np.ndarray, args) -> dict:
    """Détecte l'aile sur une image déjà chargée. Ne fait aucune I/O.

    `ctx` vient de `backend.load_model(args)`. Retourne le dict de
    `backend.detect_one` : status, error_reason, confidence, n_detections, box.
    """
    return get_backend(mode).detect_one(ctx, image, args)


def box_to_row_fields(box: np.ndarray | None) -> dict:
    """Éclate une box 4x2 (ou None) en champs x1..y4 pour le CSV."""
    if box is None:
        return {f"{axis}{i}": "" for i in range(1, 5) for axis in ("x", "y")}

    values = box.reshape(-1)
    return {
        f"{axis}{i}": f"{values[2 * (i - 1) + (0 if axis == 'x' else 1)]:.8f}"
        for i in range(1, 5)
        for axis in ("x", "y")
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Détection des ailes sur un dataset, mode heavy (YOLOE) ou light (YOLO-OBB)."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--mode", default="light", choices=["heavy", "light"])
    parser.add_argument("--split", default=None)

    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--device", default=None)

    parser.add_argument("--base-dir", default=None)

    known_args, _ = parser.parse_known_args()
    backend = get_backend(known_args.mode)
    backend.add_arguments(parser)

    args = parser.parse_args()
    args.backend = backend
    return args


def new_row(source: dict) -> dict:
    return {
        "image_id": source.get("image_id", ""),
        "specimen_id": source.get("specimen_id", ""),
        "split": source.get("split", ""),
        "status": "FAILED",
        "error_reason": "",
        "confidence": "",
        "n_detections": "0",
        **box_to_row_fields(None),
    }


def main():
    args = parse_args()
    images = read_images_csv(Path(args.dataset / "manifest.csv"))
    targets = select_images(images, split=args.split)

    print(f"Mode : {args.mode}")
    print(f"Images à traiter : {len(targets)}")

    if not targets:
        print("Aucune image sélectionnée, rien à faire.")
        return

    ctx = args.backend.load_model(args)

    extraction_root = Path(args.dataset / "extraction")
    output_path = extraction_root / args.mode / "detection.csv"
    stats_path = Path(args.dataset) / "pipeline_stats.csv"
    write_header = True

    base_dir = Path(args.base_dir) if args.base_dir else None
    pipeline_start = time.perf_counter()
    batch = []
    counter = RunCounter()

    for index, source in enumerate(targets, start=1):
        item_start = time.perf_counter()
        row = new_row(source)

        raw_path = resolve_path(source["raw_path"], base_dir)
        image = read_image(raw_path)

        if image is None:
            row["error_reason"] = "image_illisible_ou_format_non_supporte"
        else:
            detection = detect_one_image(args.mode, ctx, image, args)
            row["status"] = detection["status"]
            row["error_reason"] = detection["error_reason"]
            row["n_detections"] = str(detection["n_detections"])

            if detection["confidence"] is not None:
                row["confidence"] = f"{detection['confidence']:.4f}"

            row.update(box_to_row_fields(detection["box"]))

        row["processing_time_s"] = f"{time.perf_counter() - item_start:.4f}"
        row["processed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        counter.add(row["status"])
        batch.append(row)

        is_last = index == len(targets)
        if len(batch) >= BATCH_SIZE or is_last:
            append_rows(output_path, batch, DETECTION_FIELDS, write_header)
            write_header = False
            batch = []

            elapsed = time.perf_counter() - pipeline_start
            print(
                f"[{index}/{len(targets)}] "
                f"temps écoulé : {format_duration(elapsed)} — "
                f"moyenne : {elapsed / index:.3f} s/image — "
                f"{counter}"
            )

    total_time_s = time.perf_counter() - pipeline_start
    update_pipeline_stats(stats_path, "detection", args.mode, counter.as_dict(), total_time_s)
    print(f"CSV : {output_path}")
    print(f"Stats : {stats_path}")


if __name__ == "__main__":
    main()
