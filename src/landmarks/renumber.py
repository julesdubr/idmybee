"""renumber.py
Phase 3 : renumérote les landmarks non-ordonnés d'un détecteur (ex: UNet de
Gabriel, sortie de landmarks/predict.py) vers la numérotation canonique
d'une forme de référence FIGÉE (voir landmarks/build_reference.py), via une
méthode de numérotation respectant le contrat landmarks.methods.base --
puis flague les registrations suspectes par comparaison à la population de
la même espèce.

Deux usages, même logique de numérotation :
- `numerate_one(landmarks, zones)` : un nuage de points déjà chargé -> un
  NumberingResult. C'est exactement ce que fait le CLI pour chaque
  spécimen, indépendamment des autres -- rien n'empêche de l'appeler sur
  une seule image en mode terrain, à condition d'utiliser la MÊME forme de
  référence figée qu'en mode batch (raison d'être de build_reference.py :
  sans artefact figé, recalculer le consensus à chaque run risquerait de
  désynchroniser légèrement numérotation batch et numérotation terrain).
- CLI (`python -m landmarks.renumber --dataset ...`) : traite tout un TPS
  d'un dataset, écrit les TPS renumérotés + un log détaillé par spécimen.

Entrée (CLI) :
  - <dataset>/landmarks/<tps>        (sortie de landmarks/predict.py)
  - <reference>                      (artefact figé, voir build_reference.py)
  - <dataset>/specimens.csv          (optionnel : active le diagnostic
    outlier par espèce -- sans lui, tous les spécimens numérotés restent OK)

Sortie (CLI) :
  - <dataset>/landmarks/<tps stem>_numbered.tps : tous les spécimens
    numérotés avec succès (OK + SUSPECT).
  - <dataset>/landmarks/landmarks_numbered.csv : un statut par spécimen
    (OK/SUSPECT/FAILED) + motif, pour TOUS les spécimens d'entrée -- y
    compris les FAILED absents du TPS : rien n'est perdu sans trace.
  - <dataset>/pipeline_stats.csv (step="renumbering")

Statuts :
  - FAILED  : nombre de landmarks incompatible avec la référence (jamais
    numéroté). C'est le sort de TOUT spécimen SUSPECT en sortie de
    predict.py (Phase 2) : moins de landmarks que la référence n'en attend,
    donc échec automatique ici, quelle que soit la qualité de la
    registration -- ce n'est pas un bug, numerate_one() ne gère pas les
    nombres de points différents (voir landmarks.methods.hungarian_umeyama).
  - SUSPECT : numéroté avec succès, mais outlier post-GPA par rapport à sa
    propre espèce (voir utils.outliers.flag_by_species) -- nécessite
    --specimens ; sans lui, ou pour un spécimen non labellisé, reste OK.
  - OK      : numéroté, aucun signal d'anomalie.

  NB -- bug corrigé par rapport à l'ancien numbering/reconstruct_tps.py :
  ce diagnostic SUSPECT importait déjà utils.outliers.flag_by_species mais
  ne l'appelait jamais nulle part dans main() (ni --specimens, ni la
  fonction de chargement des labels, n'étaient réellement câblés) -- toute
  registration réussie ressortait donc OK, quelle que soit sa position par
  rapport à son espèce.

Usage :
    python -m landmarks.build_reference --ref data/references/ref-landmarks.tps \\
        --drop 3 --out data/references/reference_shape.npz          # une fois

    python -m landmarks.renumber --dataset data/Bombus --tps landmarks.tps \\
        --reference data/references/reference_shape.npz \\
        --specimens data/Bombus/specimens.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parent))
from landmarks.build_reference import load_reference
from landmarks.methods.base import NumberingResult
from landmarks.methods.hungarian_umeyama import numerate as numerate_hungarian_umeyama
from utils.outliers import HEAVY_LANDMARK_FRAC, MAD_FACTOR, MIN_GROUP_SIZE, flag_by_species
from utils.pipeline_io import update_pipeline_stats
from utils.tps_io import ImageLandmarks, parse_tps, write_tps

# Une seule méthode implémentée pour l'instant (graph_matching.py testée
# puis retirée -- moins bonne sur ce jeu de données). Ce registre évite de
# retoucher le CLI le jour où une deuxième méthode conforme au contrat de
# landmarks.methods.base est ajoutée : `--method <nom>`.
METHODS = {"hungarian_umeyama": numerate_hungarian_umeyama}


def numerate_one(
    landmarks: np.ndarray, zones: np.ndarray, method: str = "hungarian_umeyama",
) -> NumberingResult:
    """Numérote UN spécimen (batch ou terrain) contre une forme de référence
    déjà chargée (voir build_reference.load_reference). Le seul FAILED
    possible ici est intrinsèque à la méthode (nombre de points
    incompatible) -- le diagnostic SUSPECT par espèce (population) n'a de
    sens qu'en présence d'un groupe, voir flag_by_species / le CLI ci-dessous."""
    if landmarks.shape[0] != zones.shape[0]:
        return NumberingResult(
            numbered=landmarks, status="FAILED", score=float("inf"),
            reason=f"{landmarks.shape[0]} landmarks, {zones.shape[0]} attendus",
        )
    return METHODS[method](landmarks, zones)


def load_specimen_labels(specimens_csv: Path) -> tuple[dict[str, str], set[str]]:
    """Lit specimens.csv -> (species par specimen_id labellisé, ensemble des
    specimen_id labellisés). Un spécimen absent du dict/set est traité comme
    non labellisé (pool de prédiction), y compris s'il est absent de
    specimens_csv."""
    df = pd.read_csv(specimens_csv)
    required = {"specimen_id", "species", "is_labeled"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Colonnes manquantes dans {specimens_csv} : {missing}")
    labeled = df[df["is_labeled"].astype(bool)]
    species_by_id = dict(zip(labeled["specimen_id"], labeled["species"]))
    return species_by_id, set(species_by_id)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--tps", type=Path, default=Path("landmarks.tps"),
                     help="TPS à renuméroter, landmarks non-ordonnés (dans <dataset>/landmarks/).")
    ap.add_argument("--reference", type=Path, required=True,
                     help="Forme de référence figée (voir landmarks.build_reference).")
    ap.add_argument("--specimens", type=Path, default=None,
                     help="specimens.csv -- active le diagnostic SUSPECT par espèce. "
                          "Sans lui, tous les spécimens numérotés restent OK.")
    ap.add_argument("--method", default="hungarian_umeyama", choices=sorted(METHODS),
                     help="Méthode de numérotation.")
    ap.add_argument("--log", type=Path, default=None,
                     help="Chemin du CSV de statut (défaut : <dataset>/landmarks/landmarks_numbered.csv)")
    ap.add_argument("--min-group-size", type=int, default=MIN_GROUP_SIZE,
                     help=f"Taille minimale d'un groupe espèce pour évaluer le seuil d'outlier "
                          f"post-GPA (défaut : {MIN_GROUP_SIZE})")
    ap.add_argument("--outlier-mad-factor", type=float, default=MAD_FACTOR,
                     help=f"Seuil = médiane + facteur*MAD de la distance à la position médiane du "
                          f"landmark, par espèce (défaut : {MAD_FACTOR})")
    ap.add_argument("--outlier-landmark-frac", type=float, default=HEAVY_LANDMARK_FRAC,
                     help=f"Fraction de landmarks outliers au-delà de laquelle un spécimen entier "
                          f"est marqué SUSPECT (défaut : {HEAVY_LANDMARK_FRAC})")
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    input_path = args.dataset / "landmarks" / args.tps

    zones, ref_meta = load_reference(args.reference)
    print(
        f"Référence : {len(zones)} zones ({args.reference}, "
        f"{ref_meta.get('n_specimens_used', '?')} spécimen(s) utilisés, "
        f"figée le {ref_meta.get('created_at', '?')})"
    )
    drop = ref_meta.get("drop")
    if drop is not None and drop >= 0:
        zone_orig_idx = ref_meta.get("zone_orig_idx")
        idx_repr = [int(i) for i in zone_orig_idx] if zone_orig_idx is not None else "?"
        print(f"  ordre = référence source {idx_repr} (LM{drop} exclu)")

    inputs, parse_errors = parse_tps(input_path, strict=False)
    if not inputs:
        raise SystemExit(
            f"Aucun spécimen valide dans {input_path} ({len(parse_errors)} erreur(s) de parsing)."
        )
    if parse_errors:
        print(f"{len(parse_errors)} erreur(s) de parsing TPS ignorée(s) dans {input_path}")

    # --- Renumérotation, spécimen par spécimen ---------------------------------
    pipeline_start = time.perf_counter()
    results: list[NumberingResult] = []
    processing_times: list[float] = []
    specimen_refs = []  # (image_id, specimen_id, tps_id, image_path), parallèle à `results`
    for sp in inputs:
        item_start = time.perf_counter()
        results.append(numerate_one(sp.landmarks, zones, method=args.method))
        processing_times.append(time.perf_counter() - item_start)
        specimen_refs.append((sp.image_id, sp.specimen_id, sp.tps_id, sp.image_path))

    numbered_specimens = [
        replace(sp, landmarks=r.numbered) for sp, r in zip(inputs, results) if r.status != "FAILED"
    ]
    # index dans `results`/`specimen_refs` de chaque entrée de numbered_specimens, dans le même ordre
    numbered_indices = [i for i, r in enumerate(results) if r.status != "FAILED"]

    # --- Diagnostic outlier post-GPA par espèce (SUSPECT) -----------------------
    # Seule source de SUSPECT désormais : contrairement à l'ancien
    # `ambiguity_ratio` de hungarian_umeyama.numerate() (comparaison
    # par-spécimen du meilleur et du deuxième-meilleur départ, retiré --
    # marquait SUSPECT quasi 100% des spécimens sur ce jeu de données), ce
    # diagnostic compare chaque spécimen à la POPULATION de son espèce, ce
    # qui le rend beaucoup plus spécifique. Nécessite --specimens.
    n_outlier_by_index: dict[int, int] = {}
    if args.specimens is not None:
        species_by_id, labeled_ids = load_specimen_labels(args.specimens)
        labeled_positions = [
            j for j, sp in enumerate(numbered_specimens) if sp.specimen_id in labeled_ids
        ]
        if labeled_positions:
            labeled_specimens = [numbered_specimens[j] for j in labeled_positions]
            species = np.array([species_by_id[s.specimen_id] for s in labeled_specimens])
            n_outlier, heavy = flag_by_species(
                labeled_specimens, species,
                heavy_frac=args.outlier_landmark_frac,
                min_group_size=args.min_group_size,
                mad_factor=args.outlier_mad_factor,
            )
            for local_j, is_heavy, n_out in zip(labeled_positions, heavy, n_outlier):
                result_idx = numbered_indices[local_j]
                n_outlier_by_index[result_idx] = int(n_out)
                if is_heavy:
                    results[result_idx].status = "SUSPECT"
    else:
        print("Pas de --specimens fourni : diagnostic outlier par espèce désactivé (tous OK).")

    # --- Écriture du TPS renuméroté ---------------------------------------------
    landmarks_dir = args.dataset / "landmarks"
    landmarks_dir.mkdir(parents=True, exist_ok=True)
    stem = args.tps.stem
    out_path = landmarks_dir / f"{stem}_numbered.tps"
    write_tps(out_path, numbered_specimens)

    n_failed = len(inputs) - len(numbered_specimens)
    print(f"\nÉcrit {len(numbered_specimens)}/{len(inputs)} spécimen(s) -> {out_path}")
    if n_failed:
        print(f"  ({n_failed} FAILED exclus)")

    n_by_status = {s: sum(r.status == s for r in results) for s in ("OK", "SUSPECT", "FAILED")}
    print(f"Statuts : {n_by_status}")

    # --- Log détaillé + stats ----------------------------------------------------
    log_path = args.log or (landmarks_dir / "landmarks_numbered.csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "image_id", "specimen_id", "tps_id", "image_path", "status",
            "registration_cost", "n_outlier_landmarks", "error_reason", "processing_time_s",
        ])
        for i, ((image_id, specimen_id, tps_id, image_path), r, pt) in enumerate(
            zip(specimen_refs, results, processing_times)
        ):
            w.writerow([
                image_id, specimen_id, tps_id, image_path, r.status, r.score,
                n_outlier_by_index.get(i, ""), r.reason, f"{pt:.4f}",
            ])
    print(f"Statuts détaillés -> {log_path}")

    total_time_s = time.perf_counter() - pipeline_start
    counter_dict = {
        "total": len(inputs), "ok": n_by_status["OK"], "suspect": n_by_status["SUSPECT"],
        "skipped": 0, "failed": n_by_status["FAILED"],
    }
    stats_path = args.dataset / "pipeline_stats.csv"
    update_pipeline_stats(stats_path, "renumbering", args.method, counter_dict, total_time_s)
    print(f"Stats -> {stats_path}")


if __name__ == "__main__":
    main()
