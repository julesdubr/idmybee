"""Point d'entrée unique pour la détection + normalisation des ailes (heavy ou light).

Ce script porte la seule boucle detection -> normalisation, en une seule
lecture d'image par specimen. Chaque backend (heavy/detection.py,
light/detection.py) ne fournit que `load_model` et `process_one` ; la lecture
de images.csv, le timing, la normalisation et l'écriture incrémentale du CSV
de sortie sont mutualisés ici.

Normalisation
-------------
- Mode light : `process_one` ne renvoie jamais de crop, la box détectée est
  normalisée ici même, une seule fois.
- Mode heavy : le scoring CLIP normalise déjà chaque candidat top-k pour le
  comparer aux références visuelles. Le crop du candidat gagnant est donc
  déjà calculé côté backend et réutilisé tel quel ici, sans normaliser une
  seconde fois la même image.

Pour normaliser un `detections.csv` déjà produit sans redétecter, voir
`normalize_crop.py`, qui reste utilisable de façon autonome.

Exemples
--------
Heavy (YOLOE + SAM + CLIP) :

    python -m extraction.extract_wings --mode heavy `
        --images-csv .\\data\\manifest\\images.csv `
        --output-csv .\\data\\extraction\\heavy\\detections.csv `
        --output-root .\\data\\crops\\heavy `
        --ref .\\data\\references\\ref-obb.json `
        --ref-crops .\\data\\references\\ref-crops `
        --model yoloe-11s-seg.pt `
        --imgsz 1024 `
        --conf 0.05

Light (YOLO-OBB) :

    python -m extraction.extract_wings --mode light `
        --images-csv .\\data\\manifest\\images.csv `
        --output-csv .\\data\\extraction\\light\\detections.csv `
        --output-root .\\data\\crops\\light `
        --model .\\runs\\obb\\weights\\best.pt `
        --imgsz 1024 `
        --conf 0.10
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

from detection_io import (
    EXTRACT_FIELDS,
    append_rows,
    read_images_csv,
    resolve_raw_path,
    select_images,
)
from normalize_crop import (
    build_output_path,
    normalize_crop,
    normalized_points_to_pixels,
    read_image,
    write_normalized_crop,
)

BATCH_SIZE = 50


def parse_args():
    parser = argparse.ArgumentParser(
        description="Détection + normalisation des ailes, mode heavy (YOLOE+SAM) ou light (YOLO-OBB)."
    )
    parser.add_argument("--mode", required=True, choices=["heavy", "light"])
    parser.add_argument("--images-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-root", required=True, help="Racine des crops normalisés.")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--image-id", action="append", default=None)
    parser.add_argument("--base-dir", default=None)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--device", default=None)
    parser.add_argument("--keep-duplicates", action="store_true")

    # Args de normalisation, communs aux deux modes.
    parser.add_argument("--padding", type=float, default=0.10)
    parser.add_argument("--out-width", type=int, default=512)
    parser.add_argument("--out-height", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")

    # Args communs déclarés, puis args spécifiques ajoutés par le backend choisi.
    known_args, _ = parser.parse_known_args()

    if known_args.mode == "heavy":
        from heavy import detection as backend
    else:
        from light import detection as backend

    backend.add_arguments(parser)
    args = parser.parse_args()
    args.backend = backend
    return args


def new_detection_row(source: dict) -> dict:
    """Row au format attendu par `backend.process_one` (vocabulaire détection)."""
    return {
        "image_id": source.get("image_id", ""),
        "specimen_id": source.get("specimen_id", ""),
        "dataset": source.get("dataset", ""),
        "status": "FAILED",
        "error_reason": "",
        "confidence": "",
        "selection_score": "",
        "n_detections": "0",
        "x1": "", "y1": "", "x2": "", "y2": "",
        "x3": "", "y3": "", "x4": "", "y4": "",
    }


def to_extract_row(detection_row: dict) -> dict:
    """Convertit une row de détection en row combinée détection+normalisation.

    Renomme `status` -> `detection_status` et ajoute les champs de
    normalisation (initialisés à vide/FAILED), sans toucher au reste.
    """
    row = dict(detection_row)
    row["detection_status"] = row.pop("status", "FAILED")
    row["normalization_status"] = "FAILED"
    row["aspect_ratio"] = ""
    row["output_path"] = ""
    row["processing_time_s"] = ""
    row["processed_at"] = ""
    return row


def format_duration(seconds: float) -> str:
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{int(hours)}h{int(minutes):02d}m{secs:04.1f}s"
    if minutes:
        return f"{int(minutes)}m{secs:04.1f}s"
    return f"{secs:.1f}s"


def normalize_row(row: dict, image, source: dict, crop, args) -> None:
    """Complète `row` avec le résultat de la normalisation.

    `crop` est le crop déjà normalisé fourni par le backend (mode heavy,
    candidat gagnant), ou `None` si le backend n'en a pas produit (mode
    light) : dans ce cas la box détectée (`row["x1"]..["y4"]`) est
    normalisée ici, une seule fois.
    """
    if row["detection_status"] != "OK":
        row["error_reason"] = row["error_reason"] or "detection_non_OK"
        return

    final = crop
    aspect_ratio = None

    if final is None:
        points = normalized_points_to_pixels(row, width=image.shape[1], height=image.shape[0])
        if points is None:
            row["error_reason"] = "obb_invalide"
            return

        final, aspect_ratio = normalize_crop(
            image,
            points,
            pad=args.padding,
            out_width=args.out_width,
            out_height=args.out_height,
        )

    if final is None:
        row["error_reason"] = "normalisation_impossible"
        return

    out_path = build_output_path(Path(args.output_root), source, row["image_id"])
    status, error_reason = write_normalized_crop(final, out_path, overwrite=args.overwrite)

    row["normalization_status"] = status
    if error_reason:
        row["error_reason"] = error_reason
    if status != "FAILED":
        row["output_path"] = str(out_path)
    if aspect_ratio is not None:
        row["aspect_ratio"] = f"{aspect_ratio:.4f}"


def main():
    args = parse_args()
    images = read_images_csv(Path(args.images_csv))
    targets = select_images(
        images,
        dataset=args.dataset,
        image_ids=set(args.image_id) if args.image_id else None,
        skip_duplicates=not args.keep_duplicates,
    )

    print(f"Mode : {args.mode}")
    print(f"Images à traiter : {len(targets)}")

    if not targets:
        print("Aucune image sélectionnée, rien à faire.")
        return

    ctx = args.backend.load_model(args)

    output_path = Path(args.output_csv)
    write_header = True  # Le premier flush (re)crée le fichier.

    base_dir = Path(args.base_dir) if args.base_dir else None
    pipeline_start = time.perf_counter()
    batch = []

    for index, source in enumerate(targets, start=1):
        item_start = time.perf_counter()

        raw_path = resolve_raw_path(source["raw_path"], base_dir)
        image = read_image(raw_path)

        if image is None:
            row = to_extract_row(new_detection_row(source))
            row["error_reason"] = "image_illisible_ou_format_non_supporte"
        else:
            detection_row, crop = args.backend.process_one(
                ctx, image, new_detection_row(source), args
            )
            row = to_extract_row(detection_row)
            normalize_row(row, image, source, crop, args)

        row["processing_time_s"] = f"{time.perf_counter() - item_start:.4f}"
        row["processed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        batch.append(row)

        is_last = index == len(targets)
        if len(batch) >= BATCH_SIZE or is_last:
            append_rows(output_path, batch, EXTRACT_FIELDS, write_header)
            write_header = False
            batch = []

            elapsed = time.perf_counter() - pipeline_start
            avg_per_image = elapsed / index
            print(
                f"[{index}/{len(targets)}] "
                f"temps écoulé : {format_duration(elapsed)} — "
                f"moyenne : {avg_per_image:.3f} s/image"
            )

    print(f"CSV : {output_path}")
    print(f"Images : {args.output_root}")


if __name__ == "__main__":
    main()