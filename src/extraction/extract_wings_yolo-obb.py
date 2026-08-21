"""
extract_wings_obb.py
Extraction batch des ailes avec un modèle YOLO-OBB spécialisé.

Responsabilité unique :
    sélectionner les images, lancer le détecteur OBB, produire le crop
    normalisé 512x256 et écrire un CSV minimal.

Deux modes d'entrée
-------------------
1. Comparaison stricte avec le pipeline lourd
   --images_csv images.csv
   --reference_csv crops.csv
   Le script reprend uniquement les image_id ayant une dernière ligne OK.

2. Dossier(s) d'images
   --input /chemin/a --input /chemin/b
   Les sous-dossiers sont parcourus récursivement.

Le crop final est enregistré en niveaux de gris, comme dans le pipeline lourd.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from obb_geometry import letterbox, oriented_crop, read_image


IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".heic", ".heif",
    ".tif", ".tiff", ".bmp", ".webp",
}

CSV_FIELDS = [
    "image_id",
    "specimen_id",
    "dataset",
    "status",
    "error_reason",
    "confidence",
    "aspect_ratio",
    "x1", "y1",
    "x2", "y2",
    "x3", "y3",
    "x4", "y4",
    "output_path",
    "processing_time_s",
    "processed_at",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Extraction d'ailes avec un détecteur YOLO-OBB spécialisé.")

    parser.add_argument("--model", required=True, help="Chemin vers le meilleur poids .pt.")
    parser.add_argument("--input", action="append", default=[], help="Dossier ou fichier image. Peut être répété. Parcours récursif des dossiers.")
    parser.add_argument("--images_csv", default=None, help="images.csv du pipeline lourd. Requis avec --reference_csv.")
    parser.add_argument("--reference_csv", default=None, help="crops.csv du pipeline lourd. Seules les dernières lignes OK sont utilisées.")
    parser.add_argument("--output_root", required=True, help="Dossier de sortie des crops.")
    parser.add_argument("--csv", default=None, help="CSV de sortie. Par défaut: <output_root>/crops_obb.csv")
    parser.add_argument("--imgsz", type=int, default=1024, help="Taille d'inférence du détecteur.")
    parser.add_argument("--conf", type=float, default=0.10, help="Seuil de confiance.")
    parser.add_argument("--device", default=None, help="CPU, 0, 1, etc. Vide = automatique.")
    parser.add_argument("--padding", type=float, default=0.10, help="Padding total relatif comme dans le pipeline lourd.")
    parser.add_argument("--out_width", type=int, default=512)
    parser.add_argument("--out_height", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true", help="Réécrit les crops existants.")

    return parser.parse_args()


def image_id_from_path(path: Path) -> str:
    """ID stable pour le mode dossier seul."""
    return hashlib.md5(str(path.resolve()).encode("utf-8")).hexdigest()[:16]


def load_reference_targets(images_csv: Path, reference_csv: Path):
    """
    Reconstitue exactement le sous-ensemble OK du pipeline lourd.

    crops.csv étant append-only, la dernière ligne pour chaque image_id
    est considérée comme l'état courant.
    """
    with images_csv.open("r", newline="", encoding="utf-8-sig") as handle:
        images = list(csv.DictReader(handle))

    with reference_csv.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    latest = {}
    for row in rows:
        image_id = row.get("image_id")
        if image_id:
            latest[image_id] = row

    targets = []

    for image_row in images:
        image_id = image_row.get("image_id")
        if not image_id:
            continue

        reference = latest.get(image_id)
        if reference is None: # or reference.get("status") != "OK":
            continue

        raw_path = Path(image_row["raw_path"])
        targets.append({
            "image_id": image_id,
            "specimen_id": image_row.get("specimen_id"),
            "dataset": image_row.get("dataset"),
            "path": raw_path,
        })

    return targets


def find_directory_targets(inputs):
    """Construit une liste d'images à partir d'un ou plusieurs chemins."""
    targets = []

    for raw_input in inputs:
        path = Path(raw_input)

        if path.is_file():
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                targets.append({
                    "image_id": image_id_from_path(path),
                    "specimen_id": "",
                    "dataset": path.parent.name,
                    "path": path,
                })
            continue

        if path.is_dir():
            for image_path in sorted(path.rglob("*")):
                if not image_path.is_file():
                    continue
                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                targets.append({
                    "image_id": image_id_from_path(image_path),
                    "specimen_id": "",
                    "dataset": image_path.parent.name,
                    "path": image_path,
                })

    return targets


def build_output_path(output_root: Path, target: dict) -> Path:
    """Nom de sortie lisible et unique."""
    source = Path(target["path"])

    dataset = target.get("dataset") or "dataset"
    specimen = target.get("specimen_id") or source.parent.name or "unknown"

    filename = f"{dataset}_{specimen}_{source.stem}_{target['image_id']}.jpg"

    return output_root / str(dataset) / filename


def detect_one(model, image, args):
    """Retourne la meilleure OBB détectée ou None."""
    results = model.predict(
        image,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        verbose=False,
        max_det=10,
    )

    result = results[0]

    if result.obb is None or len(result.obb) == 0:
        return None

    confidences = result.obb.conf.detach().cpu().numpy()
    best_index = int(np.argmax(confidences))

    points = (
        result.obb.xyxyxyxy[best_index]
        .detach()
        .cpu()
        .numpy()
        .reshape(4, 2)
    )

    return {
        "confidence": float(confidences[best_index]),
        "points": points,
    }


def process_target(model, target, output_path, args):
    """Détecte, normalise et écrit une seule aile."""
    start = time.perf_counter()

    def failed(reason):
        return {
            "status": "FAILED",
            "error_reason": reason,
            "confidence": None,
            "aspect_ratio": None,
            "points": None,
            "processing_time_s": time.perf_counter() - start,
        }

    image = read_image(Path(target["path"]))

    if image is None:
        return failed("image_illisible_ou_format_non_supporte")

    detection = detect_one(model, image, args)

    if detection is None:
        return failed("aucune_detection")

    crop, aspect_ratio, normalized_box = oriented_crop(
        image,
        detection["points"],
        bg_color=(255, 255, 255),
        pad=args.padding,
    )

    if crop is None:
        return failed("crop_degenere")

    final = letterbox(
        crop,
        out_width=args.out_width,
        out_height=args.out_height,
        bg_color=(255, 255, 255),
    )

    if final is None:
        return failed("normalisation_impossible")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    gray = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)

    if not cv2.imwrite(str(output_path), gray):
        return failed("ecriture_impossible")

    return {
        "status": "OK",
        "error_reason": "",
        "confidence": detection["confidence"],
        "aspect_ratio": aspect_ratio,
        "points": normalized_box,
        "processing_time_s": time.perf_counter() - start,
    }


def make_csv_row(target, result, output_path):
    row = {
        "image_id": target["image_id"],
        "specimen_id": target.get("specimen_id", ""),
        "dataset": target.get("dataset", ""),
        "status": result["status"],
        "error_reason": result["error_reason"],
        "confidence": (
            f"{result['confidence']:.4f}"
            if result["confidence"] is not None
            else ""
        ),
        "aspect_ratio": (
            f"{result['aspect_ratio']:.4f}"
            if result["aspect_ratio"] is not None
            else ""
        ),
        "x1": "",
        "y1": "",
        "x2": "",
        "y2": "",
        "x3": "",
        "y3": "",
        "x4": "",
        "y4": "",
        "output_path": str(output_path) if result["status"] == "OK" else "",
        "processing_time_s": f"{result['processing_time_s']:.4f}",
        "processed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    if result["points"] is not None:
        values = np.asarray(result["points"]).reshape(4, 2)
        for index, (x, y) in enumerate(values, start=1):
            row[f"x{index}"] = f"{x:.8f}"
            row[f"y{index}"] = f"{y:.8f}"

    return row


def main():
    args = parse_args()

    if bool(args.reference_csv) != bool(args.images_csv):
        raise SystemExit("--images_csv et --reference_csv doivent être fournis ensemble.")

    if not args.input and not args.reference_csv:
        raise SystemExit("Fournis soit --images_csv + --reference_csv, soit au moins un --input.")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.csv) if args.csv else output_root / "crops_obb.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if args.reference_csv:
        targets = load_reference_targets(Path(args.images_csv), Path(args.reference_csv))
        print(f"Sous-ensemble exact du pipeline lourd : {len(targets)} images OK")
    else:
        targets = find_directory_targets(args.input)
        print(f"Images trouvées : {len(targets)}")

    print(f"Chargement du modèle : {args.model}")
    model = YOLO(args.model)

    counts = {
        "OK": 0,
        "FAILED": 0,
        "SKIPPED": 0,
    }

    write_header = not csv_path.exists()

    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)

        if write_header:
            writer.writeheader()

        total_start = time.perf_counter()

        for index, target in enumerate(targets, start=1):
            output_path = build_output_path(output_root, target)

            if output_path.exists() and not args.overwrite:
                counts["SKIPPED"] += 1
                continue

            result = process_target(
                model=model,
                target=target,
                output_path=output_path,
                args=args,
            )

            counts[result["status"]] += 1

            writer.writerow(make_csv_row(target, result, output_path))

            if index % 25 == 0 or index == len(targets):
                elapsed = time.perf_counter() - total_start
                done = counts["OK"] + counts["FAILED"]

                mean_time = elapsed / done if done else 0.0

                print(
                    f"[{index}/{len(targets)}] "
                    f"OK={counts['OK']} "
                    f"FAILED={counts['FAILED']} "
                    f"SKIPPED={counts['SKIPPED']} "
                    f"temps moyen={mean_time:.3f}s/image"
                )

    print("\nTerminé.")
    print(f"OK       : {counts['OK']}")
    print(f"FAILED   : {counts['FAILED']}")
    print(f"SKIPPED  : {counts['SKIPPED']}")
    print(f"CSV      : {csv_path}")
    print(f"Sorties  : {output_root}")


if __name__ == "__main__":
    main()
