"""Outil de diagnostic ponctuel : détermine quel landmark Tancrède (19 LM)
n'a pas d'équivalent chez Gabriel (18 LM), en testant les 19 hypothèses et
en gardant celle qui minimise le coût moyen de registration. Déjà résolu
une fois (LM3, voir reconstruct_tps.py --drop) -- à relancer seulement si
le schéma de landmarks change.

Usage: python tools/find_missing_landmark.py data/annotations/tancrede.tps data/annotations/gabriel.tps
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/, quel que soit le cwd
from utils.gpa import gpagen
from utils.register import register_unlabeled
from utils.tps_io import parse_tps


def main():
    tancrede_path, gabriel_path = sys.argv[1], sys.argv[2]

    tan, _ = parse_tps(tancrede_path, strict=False)
    tan = [s for s in tan if s.n_points == 19]
    consensus = gpagen([s.landmarks for s in tan]).mean_shape

    gab, _ = parse_tps(gabriel_path, strict=False)
    gab_points = [s.landmarks for s in gab if s.n_points == 18]

    best_k, best_cost = None, np.inf
    for k in range(19):
        zones = np.delete(consensus, k, axis=0)
        mean_cost = np.mean([register_unlabeled(pts, zones)[1] for pts in gab_points])
        print(f"  drop LM{k:2d}: mean_cost={mean_cost:.6f}")
        if mean_cost < best_cost:
            best_k, best_cost = k, mean_cost

    print(f"\nMeilleure hypothèse: LM{best_k} (coût={best_cost:.6f})")


if __name__ == "__main__":
    main()
