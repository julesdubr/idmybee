"""
Reprojette la référence de Tancrede (19 landmarks, espace image brute) vers
l'espace crop final (512x256 par défaut), en réutilisant EXACTEMENT la même
géométrie que extraction/normalize_crop.py (voir compute_wing_transform,
vérifiée numériquement identique à rotate_image()+crop_with_context() sur
150 cas synthétiques -- toute image dont le crop a été produit avec les
mêmes --padding/--out-width/--out-height obtient donc des points alignés au
pixel près sur ce crop).

    python reproject_reference.py \\
        --dataset data/Bombus --mode light \\
        --tancrede-tps data/reference/reference_landmarks.tps \\
        --output-tps data/Bombus/landmarks/tancrede_reference_crop_space.tps

Produit :
  - <output-tps>            : nouvelle référence, espace crop, avec
                               COMMENT=image_id=...;specimen_id=... déjà
                               renseigné (utilisable directement par
                               export_dataset.py, plus de "KNOWN GAP").
  - <output-tps>.review.csv : une ligne par spécimen de la référence Tancrede
                               (matché ou non), avec une colonne `keep` que
                               tu remplis à la main après avoir regardé les
                               overlays.
  - <overlays-dir>/{OK,SUSPECT}/*.png : le crop final + les 19 points
                               numérotés dessus, triés par statut (voir
                               utils/tps_overlay.py, généralisé pour marcher
                               aussi sur les .tps produits par
                               landmarks/predict.py). Pour repérer les
                               specimens mal alignés (Tancrede a travaillé
                               sur des photos parfois déjà un peu croppées à
                               la main -- l'OBB automatique peut donc
                               correspondre à un cadrage différent de celui
                               qu'il a annoté).

Jointure specimen -> image_id : le TPS de Tancrede n'a pas de COMMENT=
(digitisé hors pipeline), donc pas d'image_id direct. On matche IMAGE= à
manifest.csv par chemin (les 3 derniers segments -- dossier espèce/sexe/
fichier -- lèvent les ambiguïtés de nom de fichier dupliqué entre espèces ;
à défaut, simple nom de fichier). Non testé sur l'intégralité du jeu de
Tancrede, seulement sur les fichiers fournis (610/625 matchés, 599 avec
detection+crop OK) -- les non-matchés sont listés dans review.csv, pas
perdus silencieusement.

IMPORTANT : --padding/--out-width/--out-height doivent être EXACTEMENT ceux
utilisés pour produire extraction/{mode}/crops.csv, sinon les points calculés
ne correspondent pas aux crops réels -- valeurs par défaut = celles par
défaut de normalize_crop.py, à changer ensemble si l'un des deux a changé.

Convention Y : si les points de Tancrede tombent bien horizontalement mais
sont inversés verticalement sur les overlays, passe --flip-y (probable :
son TPS vient d'un outil tiers type tpsDig, qui utilise Y-vers-le-haut au
lieu de la convention pixel standard Y-vers-le-bas). La correction se fait
côté image brute avant la rotation -- inverser après coup dans l'espace
crop ne marche PAS dès que l'aile est tournée (réflexion et rotation ne
commutent pas), d'où l'option ici plutôt qu'un post-traitement séparé.

Flux de révision (2 passes) :
    1) Premier run (sans --exclude-csv) : produit tout, review.csv a la
       colonne `keep` vide/TRUE partout, `auto_suspect` marque les points
       hors du crop (indice fort de mauvais alignement).
    2) Tu regardes les overlays, tu mets `keep=FALSE` sur les mauvais dans
       review.csv.
    3) Deuxième run avec --exclude-csv pointant sur CE review.csv : les
       décisions `keep` sont reportées (image_id par image_id) dans le
       nouveau review.csv, et les specimens `keep=FALSE` sont exclus de
       <output-tps>.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from extraction.normalize_crop import (
    compute_wing_transform, apply_wing_transform_to_points,
    normalized_points_to_pixels, read_image,
)
from utils.pipeline_io import RunCounter, read_csv_rows, resolve_path
from utils.tps_io import ImageLandmarks, parse_tps, write_tps
from utils.tps_overlay import render_tps_overlays

FALSY = {"false", "0", "non", "no", "n"}


def norm_path(p: str) -> str:
    return p.replace("\\", "/")


def path_tail(p: str, n: int = 3) -> str:
    parts = Path(norm_path(p)).parts
    return "/".join(parts[-n:]).lower()


def path_basename(p: str) -> str:
    return Path(norm_path(p)).name.lower()


def build_manifest_index(manifest_rows: list) -> tuple[dict, dict]:
    by_tail, by_basename = {}, {}
    for row in manifest_rows:
        raw = row["raw_path"]
        by_tail.setdefault(path_tail(raw), []).append(row)
        by_basename.setdefault(path_basename(raw), []).append(row)
    return by_tail, by_basename


def match_manifest_row(image_path: str, by_tail: dict, by_basename: dict):
    """Retourne (row, methode) ou (None, raison_echec)."""
    tail_candidates = by_tail.get(path_tail(image_path))
    if tail_candidates and len(tail_candidates) == 1:
        return tail_candidates[0], "tail3"

    base_candidates = by_basename.get(path_basename(image_path))
    if base_candidates and len(base_candidates) == 1:
        return base_candidates[0], "basename"
    if base_candidates and len(base_candidates) > 1:
        return None, f"nom de fichier ambigu ({len(base_candidates)} candidats dans manifest.csv)"

    return None, "aucune correspondance dans manifest.csv"


def load_previous_keep_decisions(path: str) -> dict:
    decisions = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            keep = (row.get("keep") or "").strip()
            if keep:
                decisions[row["image_id"]] = keep
    return decisions


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--mode", default="light", choices=["heavy", "light"])
    parser.add_argument("--tancrede-tps", required=True)
    parser.add_argument("--base-dir", default=None, help="Racine pour résoudre raw_path (manifest.csv), même sens que normalize_crop.py --base-dir")
    parser.add_argument("--crops-base-dir", default=None, help="Racine pour résoudre output_path (crops.csv), même sens que predict.py --base-dir -- PAS forcément le même que --base-dir")
    parser.add_argument("--padding", type=float, default=0.10, help="DOIT correspondre à ce qui a produit crops.csv")
    parser.add_argument("--out-width", type=int, default=512, help="idem")
    parser.add_argument("--out-height", type=int, default=256, help="idem")
    parser.add_argument("--flip-y", action="store_true",
                         help="Inverse Y (y -> hauteur_image_brute - y) sur les points de Tancrede avant "
                              "la reprojection -- son TPS semble utiliser la convention Y-vers-le-haut de "
                              "tpsDig plutôt que la convention pixel standard. Se corrige AVANT la rotation, "
                              "pas après (voir commentaire dans le code) -- ce flag est le bon endroit, pas "
                              "un post-traitement sur <output-tps>.")
    parser.add_argument("--output-tps", required=True)
    parser.add_argument("--review-output", default=None, help="Défaut : <output-tps>.review.csv")
    parser.add_argument("--overlays-dir", default=None, help="Défaut : <output-tps>.overlays/")
    parser.add_argument("--exclude-csv", default=None,
                         help="review.csv d'un run précédent : reporte les décisions `keep` déjà prises")
    args = parser.parse_args()

    target_ratio = args.out_width / args.out_height
    base_dir = Path(args.base_dir) if args.base_dir else None
    crops_base_dir = Path(args.crops_base_dir) if args.crops_base_dir else None

    output_tps = Path(args.output_tps)
    review_path = Path(args.review_output) if args.review_output else Path(f"{args.output_tps}.review.csv")
    overlays_dir = Path(args.overlays_dir) if args.overlays_dir else Path(f"{args.output_tps}.overlays")

    manifest_rows = read_csv_rows(args.dataset / "manifest.csv")
    by_tail, by_basename = build_manifest_index(manifest_rows)

    detection_csv = args.dataset / "extraction" / args.mode / "detection.csv"
    crops_csv = args.dataset / "extraction" / args.mode / "crops.csv"
    detections = {row["image_id"]: row for row in read_csv_rows(detection_csv)}
    crops = {row["image_id"]: row for row in read_csv_rows(crops_csv)}

    specimens, errors = parse_tps(Path(args.tancrede_tps), strict=False)
    print(f"{len(specimens)} spécimen(s) dans {args.tancrede_tps} ({len(errors)} bloc(s) illisible(s))")

    previous_keep = load_previous_keep_decisions(args.exclude_csv) if args.exclude_csv else {}
    if args.exclude_csv:
        print(f"{len(previous_keep)} décision(s) `keep` reportée(s) depuis {args.exclude_csv}")

    counter = RunCounter()
    review_rows = []
    output_specimens = []
    image_cache_shape = {}  # image_id -> (height, width), évite de relire une image déjà vue

    for sp in specimens:
        review_row = {
            "image_id": "", "specimen_id": "", "tancrede_image_path": sp.image_path,
            "matched_via": "", "status": "SKIPPED", "reason": "", "n_out_of_bounds": "",
            "aspect_ratio_obb": "", "aspect_ratio_crops_csv": "", "keep": "",
        }

        manifest_row, method_or_reason = match_manifest_row(sp.image_path, by_tail, by_basename)
        if manifest_row is None:
            review_row["reason"] = method_or_reason
            counter.add("SKIPPED")
            review_rows.append(review_row)
            continue

        image_id = manifest_row["image_id"]
        review_row["image_id"] = image_id
        review_row["specimen_id"] = manifest_row.get("specimen_id", "")
        review_row["matched_via"] = method_or_reason

        detection = detections.get(image_id)
        if detection is None or detection.get("status") != "OK":
            review_row["reason"] = "detection.csv absent ou status != OK pour ce image_id"
            counter.add("SKIPPED")
            review_rows.append(review_row)
            continue

        crop_row = crops.get(image_id)
        if crop_row is None or crop_row.get("status") != "OK":
            review_row["reason"] = "crops.csv absent ou status != OK pour ce image_id"
            counter.add("SKIPPED")
            review_rows.append(review_row)
            continue

        if image_id not in image_cache_shape:
            raw_path = resolve_path(manifest_row["raw_path"], base_dir)
            image = read_image(raw_path)
            if image is None:
                review_row["reason"] = f"image brute illisible : {raw_path}"
                counter.add("SKIPPED")
                review_rows.append(review_row)
                continue
            image_cache_shape[image_id] = image.shape
        image_shape = image_cache_shape[image_id]

        obb_points = normalized_points_to_pixels(detection, width=image_shape[1], height=image_shape[0])
        if obb_points is None:
            review_row["reason"] = "OBB invalide dans detection.csv"
            counter.add("SKIPPED")
            review_rows.append(review_row)
            continue

        transform = compute_wing_transform(image_shape, obb_points, pad=args.padding, target_ratio=target_ratio)
        if transform is None:
            review_row["reason"] = "géométrie OBB dégénérée (compute_wing_transform)"
            counter.add("SKIPPED")
            review_rows.append(review_row)
            continue

        points_raw = sp.landmarks.astype(np.float64).copy()
        if args.flip_y:
            # Le TPS de Tancrede a l'air d'utiliser la convention Y-vers-le-haut
            # de tpsDig (Y=0 en bas de l'image) plutôt que la convention pixel
            # standard (Y=0 en haut) qu'utilisent detection.csv/l'OBB. Il FAUT
            # corriger ça ICI, dans l'espace image brute, avant la rotation --
            # inverser Y après coup, dans l'espace crop final, NE MARCHE PAS
            # dès que l'aile est tournée (réflexion et rotation ne commutent
            # pas) : vérifié, un point inversé après coup ne retombe pas au
            # bon endroit dès que l'angle de rotation de l'OBB n'est pas 0.
            points_raw[:, 1] = image_shape[0] - points_raw[:, 1]

        crop_xy = apply_wing_transform_to_points(points_raw, transform, args.out_width, args.out_height)

        n_oob = int(np.sum(
            (crop_xy[:, 0] < 0) | (crop_xy[:, 0] >= args.out_width) |
            (crop_xy[:, 1] < 0) | (crop_xy[:, 1] >= args.out_height)
        ))
        review_row["n_out_of_bounds"] = n_oob
        review_row["aspect_ratio_obb"] = f"{transform.aspect_ratio:.4f}"
        review_row["aspect_ratio_crops_csv"] = crop_row.get("aspect_ratio", "")

        status = "SUSPECT" if n_oob > 0 else "OK"
        review_row["status"] = status
        if n_oob > 0:
            review_row["reason"] = f"{n_oob} landmark(s) hors du crop {args.out_width}x{args.out_height}"

        keep = previous_keep.get(image_id, "")
        review_row["keep"] = keep
        counter.add(status)
        review_rows.append(review_row)

        if keep.strip().lower() in FALSY:
            continue  # exclu par une décision de révision précédente

        output_specimens.append(ImageLandmarks.from_image(
            n_points=len(crop_xy), landmarks=crop_xy, image_path=crop_row["output_path"],
            image_id=image_id, specimen_id=manifest_row.get("specimen_id"),
        ))

    output_tps.parent.mkdir(parents=True, exist_ok=True)
    write_tps(output_tps, output_specimens)

    review_path.parent.mkdir(parents=True, exist_ok=True)
    with open(review_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(review_rows[0].keys()) if review_rows else [])
        writer.writeheader()
        writer.writerows(review_rows)

    # Overlays -- délégué à utils.tps_overlay (partagé avec landmarks/predict.py) :
    # relit output_tps qu'on vient d'écrire, trie OK/SUSPECT dans overlays_dir/
    # en joignant sur image_id via review_path (mêmes colonnes image_id/status
    # que n'importe quel autre CSV du pipeline).
    overlay_summary = render_tps_overlays(
        output_tps, overlays_dir, base_dir=crops_base_dir, csv_path=review_path,
    ) if output_specimens else {"written": 0, "unmatched": 0, "skipped": 0}

    print(f"{counter}")
    print(f"{len(output_specimens)} spécimen(s) écrit(s) -> {output_tps}")
    print(f"Revue -> {review_path}")
    print(f"Overlays ({overlay_summary['written']} écrit(s)) -> {overlays_dir}")
    if not args.exclude_csv:
        print(
            "\nPremier run : regarde les overlays, mets `keep=FALSE` sur les mauvais dans "
            f"{review_path}, puis relance avec --exclude-csv {review_path}"
        )


if __name__ == "__main__":
    main()