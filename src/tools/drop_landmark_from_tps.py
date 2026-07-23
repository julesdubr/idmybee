"""Retire un landmark de chaque spécimen d'un TPS -- utile pour comparer
Tancrède (19 LM) à un fichier reconstruit auquel il manque ce landmark.

Usage: python drop_landmark_from_tps.py tancrede.tps tancrede_minus_lm3.tps --drop 3
"""
import argparse
from dataclasses import replace

import numpy as np

from utils.tps_io import parse_tps, write_tps

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input_path")
    ap.add_argument("output_path")
    ap.add_argument("--drop", type=int, default=3)
    args = ap.parse_args()

    specimens, _ = parse_tps(args.input_path, strict=False)
    out = [replace(s, landmarks=np.delete(s.landmarks, args.drop, axis=0), n_points=s.n_points - 1)
           for s in specimens]
    write_tps(args.output_path, out)
    print(f"Écrit {len(out)} spécimens ({out[0].n_points} landmarks) -> {args.output_path}")
