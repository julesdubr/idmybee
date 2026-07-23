"""Renumérote les landmarks non-ordonnés de Gabriel (UNet) pour matcher la
numérotation de Tancrède, via alignement rigide + assignation hongroise
(utils.register). Le TPS produit est directement utilisable par lda.py.

Gabriel ne produit que 18 landmarks (Tancrède en a 19) ; --drop indique
le landmark Tancrède sans équivalent (confirmé : LM3).

Usage:
    python reconstruct_tps.py data/annotations/tancrede.tps data/annotations/gabriel.tps \
        data/annotations/gabriel_reordered.tps --drop 3 --cost-log out/gabriel_costs.csv
"""
import argparse
import csv
from dataclasses import replace
from pathlib import Path

import numpy as np

from utils.gpa import gpagen
from numbering.hungarian_umeyama import register_unlabeled
from utils.tps_io import parse_tps, write_tps


def build_reference(tancrede_path, expected_lm=19):
    specimens, _ = parse_tps(tancrede_path, strict=False)
    specimens = [s for s in specimens if s.n_points == expected_lm]
    return gpagen([s.landmarks for s in specimens]).mean_shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tancrede_path")
    ap.add_argument("gabriel_path")
    ap.add_argument("output_path")
    ap.add_argument("--drop", type=int, default=3, help="Landmark Tancrède sans équivalent Gabriel")
    ap.add_argument("--cost-log", type=Path, default=None, help="CSV optionnel: coût de registration par spécimen")
    args = ap.parse_args()

    consensus = build_reference(args.tancrede_path)
    zones = np.delete(consensus, args.drop, axis=0)
    zone_orig_idx = [i for i in range(len(consensus)) if i != args.drop]

    gab, errors = parse_tps(args.gabriel_path, strict=False)
    gab = [s for s in gab if s.n_points == len(zones)]
    print(f"{len(gab)} spécimens valides ({len(errors)} erreur(s) de parsing ignorée(s))")

    out_specimens, cost_rows = [], []
    for s in gab:
        assign, cost, _ = register_unlabeled(s.landmarks, zones)
        inv = np.empty(len(zones), dtype=int)
        inv[assign] = np.arange(len(assign))  # slot j (zone j) <- point assigné à j
        out_specimens.append(replace(s, landmarks=s.landmarks[inv]))
        cost_rows.append((s.sid, s.image_path, cost))

    write_tps(args.output_path, out_specimens)
    print(f"Écrit {len(out_specimens)} spécimens -> {args.output_path}")
    print(f"Ordre landmarks = Tancrède {zone_orig_idx} (LM{args.drop} exclu)")

    costs = np.array([c for _, _, c in cost_rows])
    med, mad = np.median(costs), np.median(np.abs(costs - np.median(costs)))
    thresh = med + 5 * mad
    n_flagged = int((costs > thresh).sum())
    print(f"Coût registration: médiane={med:.5f}, {n_flagged} spécimen(s) > {thresh:.5f} (à vérifier)")

    if args.cost_log:
        args.cost_log.parent.mkdir(parents=True, exist_ok=True)
        with open(args.cost_log, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "image", "cost", "flagged"])
            for sid, image, cost in cost_rows:
                w.writerow([sid, image, cost, cost > thresh])
        print(f"Log -> {args.cost_log}")


if __name__ == "__main__":
    main()
