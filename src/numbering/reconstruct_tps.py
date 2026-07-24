"""reconstruct_tps.py
Renumérote les landmarks non-ordonnés d'un détecteur (ex: UNet de Gabriel, 18
points) pour matcher la numérotation de référence (ex: Tancrède, 19 points),
via une méthode de numérotation respectant le contrat numbering.base.

Écrit systématiquement :
- le TPS renuméroté (TOUS les spécimens, aucun exclu silencieusement -- même
  principe que le reste du pipeline : ne jamais perdre une image sans trace).
- <output>_status.csv : un statut d'alignement par spécimen (OK/SUSPECT/
  FAILED) + le motif explicite. Le filtrage (ex: --exclude-ids de lda.py)
  reste une décision de l'étape suivante, pas de celle-ci.

Usage:
    python -m numbering.reconstruct_tps data/annotations/tancrede.tps data/annotations/gabriel.tps \
        data/annotations/gabriel_reordered.tps --drop 3
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parent))
from numbering.base import NumberingResult
from numbering.hungarian_umeyama import numerate
from utils.gpa import gpagen
from utils.tps_io import parse_tps, write_tps

OUTLIER_MAD_FACTOR = 5


def build_reference(ref_path: Path, expected_lm: int) -> np.ndarray:
    specimens, _ = parse_tps(ref_path, strict=False)
    specimens = [s for s in specimens if s.n_points == expected_lm]
    if not specimens:
        raise ValueError(f"Aucun spécimen à {expected_lm} landmarks dans {ref_path}")
    return gpagen([s.landmarks for s in specimens]).mean_shape


def flag_population_outliers(results: list[NumberingResult]) -> None:
    """Fait passer OK -> SUSPECT (en place) pour les coûts anormalement
    élevés (médiane + 5*MAD), parmi les spécimens déjà OK uniquement
    (un FAILED reste FAILED, ce n'est pas une question de seuil)."""
    ok_costs = np.array([r.score for r in results if r.status == "OK"])
    if len(ok_costs) < 2:
        return
    med = float(np.median(ok_costs))
    mad = float(np.median(np.abs(ok_costs - med)))
    thresh = med + OUTLIER_MAD_FACTOR * mad
    for r in results:
        if r.status == "OK" and r.score > thresh:
            r.status = "SUSPECT"
            r.reason = f"coût registration {r.score:.5f} > seuil population {thresh:.5f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("reference_path", type=Path, help="TPS de référence, numérotation cible (ex: tancrede.tps)")
    ap.add_argument("input_path", type=Path, help="TPS à renuméroter, landmarks non-ordonnés (ex: gabriel.tps)")
    ap.add_argument("output_path", type=Path, help="TPS de sortie, dans l'ordre de reference_path")
    ap.add_argument("--drop", type=int, default=None,
                     help="Landmark(s) de reference_path sans équivalent dans input_path (ex: 3)")
    ap.add_argument("--ref-landmarks", type=int, default=None,
                     help="Nombre de landmarks attendu dans reference_path (défaut: le plus fréquent trouvé)")
    ap.add_argument("--log", type=Path, default=None,
                     help="CSV de statut par spécimen (défaut: <output_path stem>_status.csv)")
    args = ap.parse_args()

    ref_specimens, _ = parse_tps(args.reference_path, strict=False)
    if args.ref_landmarks is not None:
        expected_ref_lm = args.ref_landmarks
    else:
        counts = np.bincount([s.n_points for s in ref_specimens])
        expected_ref_lm = int(np.argmax(counts))
    consensus = build_reference(args.reference_path, expected_ref_lm)

    if args.drop is not None:
        zones = np.delete(consensus, args.drop, axis=0)
        zone_orig_idx = [i for i in range(len(consensus)) if i != args.drop]
    else:
        zones = consensus
        zone_orig_idx = list(range(len(consensus)))

    inputs, parse_errors = parse_tps(args.input_path, strict=False)
    if parse_errors:
        print(f"{len(parse_errors)} erreur(s) de parsing TPS ignorée(s) dans {args.input_path}")

    out_specimens = []
    results: list[NumberingResult] = []
    specimen_refs = []  # (sid, image_path) parallèle à results, pour le log
    for sp in inputs:
        if sp.n_points != len(zones):
            result = NumberingResult(
                numbered=sp.landmarks, status="FAILED", score=float("inf"),
                reason=f"{sp.n_points} landmarks, {len(zones)} attendus",
            )
        else:
            result = numerate(sp.landmarks, zones)
        results.append(result)
        specimen_refs.append((sp.sid, sp.image_path))
        out_specimens.append(replace(sp, landmarks=result.numbered))

    flag_population_outliers(results)

    write_tps(args.output_path, out_specimens)
    print(f"Écrit {len(out_specimens)} spécimen(s) -> {args.output_path}")
    print(f"Ordre landmarks = référence {zone_orig_idx}" + (f" (LM{args.drop} exclu)" if args.drop is not None else ""))

    n_by_status = {s: sum(r.status == s for r in results) for s in ("OK", "SUSPECT", "FAILED")}
    print(f"Statuts : {n_by_status}")

    log_path = args.log or args.output_path.with_name(args.output_path.stem + "_status.csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "image", "status", "score", "reason"])
        for (sid, image), r in zip(specimen_refs, results):
            w.writerow([sid, image, r.status, r.score, r.reason])
    print(f"Statuts détaillés -> {log_path}")


if __name__ == "__main__":
    main()