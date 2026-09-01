"""
Rend des overlays (image + landmarks numérotés) à partir d'un fichier .tps
quelconque -- réutilisable partout où le pipeline produit un TPS avec
IMAGE= pointant vers une image lisible : sorties de landmarks/predict.py
(landmarks.tps), de landmarks_trainer/reproject_reference.py, ou tout autre
TPS respectant le format lu par utils.tps_io.parse_tps.

    python -m utils.tps_overlay \\
        --tps data/Bombus/landmarks/landmarks.tps \\
        --output-dir data/Bombus/landmarks/overlays \\
        --csv data/Bombus/landmarks/landmarks.csv

Tri en sous-dossiers : si le TPS a des COMMENT= (image_id) ET qu'un --csv
est fourni, chaque image part dans <output-dir>/<status>/<image_id>.png,
`status` étant lu dans le CSV (colonne --status-col, "status" par défaut)
en joignant sur --image-id-col ("image_id" par défaut). Un image_id présent
dans le TPS mais absent du CSV part dans <output-dir>/_unmatched/ plutôt
que d'être perdu silencieusement ou mélangé au reste.

Sans --csv (ou TPS sans COMMENT=) : toutes les images vont directement
dans <output-dir>/, à plat, nommées par image_id si connu sinon par ID=
(tps_id) sinon par index de spécimen.

Ne modifie ni ne dépend de rien d'autre que utils.tps_io -- ce fichier n'a
pas d'opinion sur QUI produit le TPS, juste sur comment le dessiner.
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from tqdm import tqdm

from utils.pipeline_io import read_csv_rows, resolve_path
from utils.tps_io import parse_tps

UNMATCHED_DIR = "_unmatched"


def draw_landmarks(image: np.ndarray, points_xy: np.ndarray, radius: int = 4,
                    point_color: tuple = (0, 0, 255), text_color: tuple = (0, 255, 0),
                    font_scale: float = 0.35) -> np.ndarray:
    """Dessine des points numérotés (0, 1, 2, ...) sur une copie de `image`.
    Ne modifie pas `image` en place."""
    annotated = image.copy()
    for i, (x, y) in enumerate(points_xy):
        xi, yi = int(round(x)), int(round(y))
        cv2.circle(annotated, (xi, yi), radius, point_color, -1)
        cv2.putText(annotated, str(i), (xi + 5, yi - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, text_color, 1, cv2.LINE_AA)
    return annotated


def load_status_by_image_id(csv_path: str, image_id_col: str, status_col: str) -> dict:
    rows = read_csv_rows(Path(csv_path))
    missing = [c for c in (image_id_col, status_col) if rows and c not in rows[0]]
    if missing:
        raise ValueError(f"{csv_path} n'a pas de colonne {missing} (colonnes présentes : {list(rows[0].keys())})")
    return {row[image_id_col]: row[status_col] for row in rows}


def output_name(specimen, index: int) -> str:
    if specimen.image_id:
        return f"{specimen.image_id}.png"
    if specimen.tps_id is not None:
        return f"tps{specimen.tps_id}.png"
    return f"specimen_{index:04d}.png"


def render_tps_overlays(
    tps_path, output_dir, base_dir=None, csv_path=None,
    image_id_col: str = "image_id", status_col: str = "status",
) -> dict:
    """Fonction réutilisable directement en Python (pas seulement en CLI) --
    voir reproject_reference.py pour un exemple d'appel après avoir écrit un
    TPS + son CSV compagnon dans le même run."""
    specimens, tps_errors = parse_tps(Path(tps_path), strict=False)

    status_by_id = load_status_by_image_id(csv_path, image_id_col, status_col) if csv_path else {}
    sortable = bool(csv_path)  # condition posée par Jules : COMMENT= ET csv fourni

    output_dir = Path(output_dir)
    written, unmatched, skipped = 0, 0, []

    for idx, sp in tqdm(enumerate(specimens), total=len(specimens)):
        image_path = resolve_path(sp.image_path, base_dir)
        image = cv2.imread(str(image_path))
        if image is None:
            skipped.append({"tps_id": sp.tps_id, "image_id": sp.image_id or "", "reason": f"image introuvable : {image_path}"})
            continue

        annotated = draw_landmarks(image, sp.landmarks)
        name = output_name(sp, idx)

        subdir = output_dir
        if sortable and sp.image_id:
            status = status_by_id.get(sp.image_id)
            if status is None:
                subdir = output_dir / UNMATCHED_DIR
                unmatched += 1
            else:
                subdir = output_dir / status
        elif sortable and not sp.image_id:
            subdir = output_dir / UNMATCHED_DIR
            unmatched += 1

        subdir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(subdir / name), annotated)
        written += 1

    if skipped:
        skipped_path = output_dir / "skipped.csv"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(skipped_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["tps_id", "image_id", "reason"])
            writer.writeheader()
            writer.writerows(skipped)

    return {
        "n_specimens": len(specimens),
        "n_tps_errors": len(tps_errors),
        "written": written,
        "unmatched": unmatched,
        "skipped": len(skipped),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tps", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-dir", default=None, help="Racine pour résoudre IMAGE= si relatif, même sens que resolve_path")
    parser.add_argument("--csv", default=None, help="CSV compagnon (image_id + status) pour trier en sous-dossiers -- optionnel")
    parser.add_argument("--image-id-col", default="image_id")
    parser.add_argument("--status-col", default="status")
    args = parser.parse_args()

    base_dir = Path(args.base_dir) if args.base_dir else None
    summary = render_tps_overlays(
        args.tps, args.output_dir, base_dir=base_dir, csv_path=args.csv,
        image_id_col=args.image_id_col, status_col=args.status_col,
    )
    print(summary)


if __name__ == "__main__":
    main()
