"""reconstruct_tps.py
Renumérote les landmarks non-ordonnés d'un détecteur (ex: UNet de Gabriel, 18
points) pour matcher la numérotation de référence (ex: Tancrède, 19 points),
via une méthode de numérotation respectant le contrat numbering.base --
puis flague les registrations suspectes et sépare le résultat en TPS
labellisé/non-labellisé/tous.

Écrit trois TPS (mêmes landmarks, sous-ensembles différents des spécimens
OK/SUSPECT -- un FAILED, nombre de points incompatible avec le template,
n'est jamais écrit dans aucun des trois) :
- <output_path> : tous les spécimens numérotés (ex: landmarks_numbered.tps)
- <output_path stem>_labeled<suffix>   : seulement ceux avec une espèce
  connue dans specimens_csv (is_labeled=True) -- jeu d'entraînement pour
  lda.py et jeu des analyses de variance (analysis/report_variance.py).
- <output_path stem>_unlabeled<suffix> : le reste -- pool de prédiction pour
  predict.py, spécimens sans détermination biologique.

Écrit aussi <manifest-dir>/landmarks_numbered.csv : un statut par spécimen
(OK/SUSPECT/FAILED) + le motif explicite, pour TOUS les spécimens d'entrée
-- y compris les FAILED absents des TPS : rien n'est perdu sans trace, la
trace est dans ce CSV plutôt que dans les TPS eux-mêmes. Directement
utilisable comme --exclude-ids de lda.py pour écarter les SUSPECT/FAILED de
l'entraînement (optionnel -- SUSPECT signale un outlier post-GPA, pas
forcément une erreur).

Le statut SUSPECT vient d'une SEULE source désormais : un outlier post-GPA
détecté PAR ESPÈCE (voir utils.outliers) sur le sous-ensemble labellisé --
la position renumérotée du spécimen s'écarte trop de la position médiane de
son espèce sur plusieurs landmarks. Contrairement à l'ancien
`ambiguity_ratio` de numbering.hungarian_umeyama (comparaison par-spécimen
du meilleur et du deuxième-meilleur départ de registration, retiré : sur ce
jeu de données il marquait SUSPECT quasi 100% des spécimens, y compris des
registrations correctes -- un critère trop bruité pour être utile), ce
diagnostic compare chaque spécimen à la POPULATION de son espèce, ce qui le
rend beaucoup plus spécifique. Les spécimens non labellisés ne peuvent pas
en bénéficier (pas d'espèce connue pour choisir le bon groupe de
comparaison) : ils restent OK dès lors que la renumérotation a réussi.

Usage:
    python -m numbering.reconstruct_tps data/annotations/tancrede.tps data/annotations/landmarks_unet.tps \\
        data/manifest/specimens.csv data/annotations/landmarks_numbered.tps --drop 3
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parent))
from numbering.base import NumberingResult
from numbering.hungarian_umeyama import numerate
from utils.gpa import gpagen
from utils.outliers import HEAVY_LANDMARK_FRAC, MAD_FACTOR, MIN_GROUP_SIZE, flag_by_species
from utils.tps_io import ImageLandmarks, parse_tps, write_tps


def build_reference_shape(ref_specimens: list, expected_lm: int) -> np.ndarray:
    """Consensus GPA du template de référence. Prend les spécimens déjà
    parsés (pas un chemin) pour éviter de reparser deux fois le même TPS."""
    matching = [s for s in ref_specimens if s.n_points == expected_lm]
    if not matching:
        raise SystemExit(
            f"Aucun spécimen à {expected_lm} landmarks dans le TPS de référence "
            f"(essayer --ref-landmarks pour forcer un autre nombre)."
        )
    return gpagen([s.landmarks for s in matching]).mean_shape


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


def split_output_paths(output_path: Path) -> tuple[Path, Path, Path]:
    """(tous, labellisés, non-labellisés) dérivés d'un seul chemin de base."""
    labeled = output_path.with_name(f"{output_path.stem}_labeled{output_path.suffix}")
    unlabeled = output_path.with_name(f"{output_path.stem}_unlabeled{output_path.suffix}")
    return output_path, labeled, unlabeled


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("reference_path", type=Path, help="TPS de référence, numérotation cible (ex: tancrede.tps)")
    ap.add_argument("input_path", type=Path, help="TPS à renuméroter, landmarks non-ordonnés (ex: landmarks_unet.tps)")
    ap.add_argument("specimens_csv", type=Path,
                     help="data/manifest/specimens.csv (specimen_id, species, is_labeled) -- pour le split "
                          "labellisé/non-labellisé et le flag d'outliers post-GPA par espèce")
    ap.add_argument("output_path", type=Path,
                     help="TPS de sortie 'tous' (ex: landmarks_numbered.tps) -- <stem>_labeled et "
                          "<stem>_unlabeled sont dérivés et écrits en plus")
    ap.add_argument("--drop", type=int, default=None,
                     help="Landmark(s) de reference_path sans équivalent dans input_path (ex: 3)")
    ap.add_argument("--ref-landmarks", type=int, default=None,
                     help="Nombre de landmarks attendu dans reference_path (défaut: le plus fréquent trouvé)")
    ap.add_argument("--manifest-dir", type=Path, default=Path("data/manifest"),
                     help="Dossier manifest où écrire landmarks_numbered.csv (défaut: data/manifest)")
    ap.add_argument("--log", type=Path, default=None,
                     help="Chemin du CSV de statut (défaut: <manifest-dir>/landmarks_numbered.csv)")
    ap.add_argument("--min-group-size", type=int, default=MIN_GROUP_SIZE,
                     help=f"Taille minimale d'un groupe espèce pour évaluer le seuil d'outlier post-GPA "
                          f"(défaut: {MIN_GROUP_SIZE})")
    ap.add_argument("--outlier-mad-factor", type=float, default=MAD_FACTOR,
                     help=f"Seuil = médiane + facteur*MAD de la distance à la position médiane du landmark, "
                          f"par espèce (défaut: {MAD_FACTOR})")
    ap.add_argument("--outlier-landmark-frac", type=float, default=HEAVY_LANDMARK_FRAC,
                     help=f"Fraction de landmarks outliers au-delà de laquelle un spécimen entier est marqué "
                          f"SUSPECT (défaut: {HEAVY_LANDMARK_FRAC})")
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)

    for label, path in (("reference_path", args.reference_path), ("input_path", args.input_path)):
        if path.suffix.lower() != ".tps":
            print(f"ATTENTION : {label}={path} n'a pas l'extension .tps -- vérifier que c'est le bon fichier.")

    ref_specimens, ref_errors = parse_tps(args.reference_path, strict=False)
    if not ref_specimens:
        raise SystemExit(
            f"Aucun spécimen valide dans {args.reference_path} ({len(ref_errors)} erreur(s) de "
            f"parsing). Vérifier que ce fichier est bien un .tps, pas un CSV de métadonnées."
        )

    if args.ref_landmarks is not None:
        expected_ref_lm = args.ref_landmarks
    else:
        counts = np.bincount([s.n_points for s in ref_specimens])
        expected_ref_lm = int(np.argmax(counts))
    consensus = build_reference_shape(ref_specimens, expected_ref_lm)

    if args.drop is not None:
        zones = np.delete(consensus, args.drop, axis=0)
        zone_orig_idx = [i for i in range(len(consensus)) if i != args.drop]
    else:
        zones = consensus
        zone_orig_idx = list(range(len(consensus)))

    inputs, parse_errors = parse_tps(args.input_path, strict=False)
    if parse_errors:
        print(f"{len(parse_errors)} erreur(s) de parsing TPS ignorée(s) dans {args.input_path}")

    # --- Renumérotation ------------------------------------------------------
    out_specimens: list[ImageLandmarks] = []
    out_result_indices: list[int] = []  # index dans `results`/`specimen_refs`, aligné sur out_specimens
    results: list[NumberingResult] = []
    specimen_refs = []  # (image_id, specimen_id, tps_id, image_path), parallèle à results, pour le log
    for sp in inputs:
        if sp.n_points != len(zones):
            result = NumberingResult(
                numbered=sp.landmarks, status="FAILED", score=float("inf"),
                reason=f"{sp.n_points} landmarks, {len(zones)} attendus",
            )
        else:
            result = numerate(sp.landmarks, zones)
        results.append(result)
        specimen_refs.append((sp.image_id, sp.specimen_id, sp.tps_id, sp.image_path))
        if result.status != "FAILED":
            out_result_indices.append(len(results) - 1)
            out_specimens.append(replace(sp, landmarks=result.numbered))

    # --- Flag SUSPECT post-GPA, par espèce, sur le sous-ensemble labellisé --
    species_by_id, labeled_ids = load_specimen_labels(args.specimens_csv)
    n_outlier_by_result_idx: dict[int, int] = {}
    labeled_positions = [i for i, sp in enumerate(out_specimens) if sp.specimen_id in species_by_id]
    if labeled_positions:
        labeled_specimens = [out_specimens[i] for i in labeled_positions]
        labeled_species = np.array([species_by_id[out_specimens[i].specimen_id] for i in labeled_positions])
        print(f"\nDétection d'outliers post-GPA par espèce ({len(labeled_positions)} spécimen(s) labellisé(s)) :")
        n_outlier, heavy = flag_by_species(
            labeled_specimens, labeled_species,
            heavy_frac=args.outlier_landmark_frac,
            min_group_size=args.min_group_size,
            mad_factor=args.outlier_mad_factor,
        )
        for local_i, global_i in enumerate(labeled_positions):
            ridx = out_result_indices[global_i]
            n_outlier_by_result_idx[ridx] = int(n_outlier[local_i])
            if heavy[local_i]:
                results[ridx].status = "SUSPECT"
                results[ridx].reason = (
                    f"outlier post-GPA au sein de l'espèce {labeled_species[local_i]} "
                    f"({n_outlier[local_i]} landmark(s) hors seuil)"
                )
        print(f"{sum(heavy)}/{len(labeled_positions)} spécimen(s) labellisé(s) marqué(s) SUSPECT")
    else:
        print("\nAucun spécimen labellisé (is_labeled=True) trouvé parmi les résultats -- "
              "détection d'outliers post-GPA sautée.")

    # --- Split tous / labellisés / non-labellisés ----------------------------
    labeled_out = [sp for sp in out_specimens if sp.specimen_id in labeled_ids]
    unlabeled_out = [sp for sp in out_specimens if sp.specimen_id not in labeled_ids]

    all_path, labeled_path, unlabeled_path = split_output_paths(args.output_path)
    for path in (all_path, labeled_path, unlabeled_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    write_tps(all_path, out_specimens)
    write_tps(labeled_path, labeled_out)
    write_tps(unlabeled_path, unlabeled_out)

    n_failed = len(inputs) - len(out_specimens)
    print(f"\nÉcrit {len(out_specimens)}/{len(inputs)} spécimen(s) -> {all_path}")
    print(f"  dont {len(labeled_out)} labellisé(s)     -> {labeled_path}")
    print(f"  dont {len(unlabeled_out)} non labellisé(s) -> {unlabeled_path}")
    if n_failed:
        print(f"  ({n_failed} FAILED exclus des 3 TPS, détail dans le log)")
    print(f"Ordre landmarks = référence {zone_orig_idx}" + (f" (LM{args.drop} exclu)" if args.drop is not None else ""))

    n_by_status = {s: sum(r.status == s for r in results) for s in ("OK", "SUSPECT", "FAILED")}
    print(f"Statuts : {n_by_status}")

    log_path = args.log or (args.manifest_dir / "landmarks_numbered.csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "image_id", "specimen_id", "tps_id", "image_path", "is_labeled",
            "status", "registration_cost", "n_outlier_landmarks", "error_reason",
        ])
        for i, ((image_id, specimen_id, tps_id, image_path), r) in enumerate(zip(specimen_refs, results)):
            w.writerow([
                image_id, specimen_id, tps_id, image_path, specimen_id in labeled_ids,
                r.status, r.score, n_outlier_by_result_idx.get(i, ""), r.reason,
            ])
    print(f"Statuts détaillés -> {log_path}")


if __name__ == "__main__":
    main()