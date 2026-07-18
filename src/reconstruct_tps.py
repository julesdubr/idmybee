"""
Reconstruct a TPS file with Gabriel's landmarks reordered to match
Tancrede's numbering, so it can be fed directly into the existing
GPA -> PCA -> LDA validation loop.

Since Gabriel's scheme is missing one landmark relative to Tancrede's 19
(confirmed: LM3), the output has 18 landmarks per specimen, in the order
Tancrede's 0,1,2,4,5,...,18 (i.e. Tancrede's numbering with index
`dropped` removed). Use drop_landmark_from_tps.py to produce a matching
18-landmark version of Tancrede's own file for a fair, apples-to-apples
comparison in the LDA loop.

Also writes a per-specimen registration-cost CSV so you can flag/exclude
likely-bad detections before running the classification loop, in the same
diagnostic-first spirit as validate_landmarks_lda_v2.py.

Usage:
    python reconstruct_tps.py <tancrede.tps> <gabriel.tps> <output.tps> [--drop 3] [--cost-log cost.csv]
"""
import argparse
import csv
import numpy as np

from utils.tps_io import parse_tps, as_array, filter_valid
from gpa import gpa
from register import register_unlabeled, apply_transform, umeyama


def write_tps(path, specimens_coords, images, ids):
    with open(path, 'w', newline='\r\n') as f:
        for coords, image, specid in zip(specimens_coords, images, ids):
            f.write(f"LM={len(coords)}\n")
            for x, y in coords:
                f.write(f"{x:.4f} {y:.4f}\n")
            if image is not None:
                f.write(f"IMAGE={image}\n")
            if specid is not None:
                f.write(f"ID={specid}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tancrede_path")
    ap.add_argument("gabriel_path")
    ap.add_argument("output_path")
    ap.add_argument("--drop", type=int, default=3,
                     help="Index (0-based) of the Tancrede landmark with no Gabriel counterpart")
    ap.add_argument("--cost-log", default=None,
                     help="Optional CSV path to log per-specimen registration cost")
    args = ap.parse_args()

    print("Building reference template from Tancrede's hand-digitized data...")
    tan = parse_tps(args.tancrede_path)
    tan, _ = filter_valid(tan, 19, label="tancrede")
    X = as_array(tan)
    _, consensus, _ = gpa(X)

    zones = np.delete(consensus, args.drop, axis=0)
    zone_orig_idx = [i for i in range(consensus.shape[0]) if i != args.drop]

    print("Loading Gabriel's unordered predictions...")
    gab = parse_tps(args.gabriel_path)
    gab, skipped = filter_valid(gab, len(zones), label="gabriel")

    out_coords, out_images, out_ids, cost_rows = [], [], [], []
    costs = []
    for s in gab:
        assign, cost, _ = register_unlabeled(s['coords'], zones)
        # invert assignment: output slot j (Tancrede landmark zone_orig_idx[j])
        # gets the raw Gabriel point that was assigned to it
        inv = np.empty(len(zones), dtype=int)
        inv[assign] = np.arange(len(assign))
        reordered = s['coords'][inv]
        out_coords.append(reordered)
        out_images.append(s['image'])
        out_ids.append(s['id'])
        costs.append(cost)
        cost_rows.append((s['id'], s['image'], cost))

    write_tps(args.output_path, out_coords, out_images, out_ids)
    print(f"\nWrote {len(out_coords)} specimens to {args.output_path}")
    print(f"Landmark order in output = Tancrede's {zone_orig_idx} "
          f"(i.e. Tancrede's numbering with LM{args.drop} removed)")
    print(f"{len(skipped)} specimen(s) skipped (see warnings above)")

    costs = np.array(costs)
    med, mad = np.median(costs), np.median(np.abs(costs - np.median(costs)))
    flag_thresh = med + 5 * mad
    n_flagged = (costs > flag_thresh).sum()
    print(f"\nRegistration cost: median={med:.5f}, {n_flagged} specimen(s) "
          f"above {flag_thresh:.5f} (median + 5*MAD) -- worth a manual look before LDA")

    if args.cost_log:
        with open(args.cost_log, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['id', 'image', 'registration_cost', 'flagged'])
            for specid, image, cost in cost_rows:
                w.writerow([specid, image, cost, cost > flag_thresh])
        print(f"Per-specimen cost log written to {args.cost_log}")


if __name__ == "__main__":
    main()