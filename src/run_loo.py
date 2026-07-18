"""
Resolve landmark correspondence for Gabriel's unordered 18-point UNet
predictions against Tancrede's ordered 19-point hand-digitized reference.

Two questions this answers:
  1. Which of Tancrede's 19 landmarks has no counterpart in Gabriel's
     18-point scheme? (leave-one-out registration cost sweep)
  2. Given correspondence, does the simple top/bottom - left/right
     tie-break heuristic (for the two ambiguous point clusters) agree
     with the full geometric registration?

Usage:
    python run_leave_one_out.py <tancrede.tps> <gabriel.tps>
"""
import sys
import numpy as np

from utils.tps_io import parse_tps, as_array, filter_valid
from gpa import gpa
from register import register_unlabeled, apply_transform, umeyama


def build_reference(tancrede_path, expected_lm=19):
    tan = parse_tps(tancrede_path)
    tan, _ = filter_valid(tan, expected_lm, label="tancrede")
    X = as_array(tan)
    aligned, consensus, n_iter = gpa(X)
    return aligned, consensus


def leave_one_out_sweep(gabriel_points, consensus):
    """For each of the 19 template landmarks, drop it and register all
    Gabriel specimens against the remaining 18. Returns dict k -> (per_spec, mean_cost)."""
    n_lm = consensus.shape[0]
    results = {}
    for k in range(n_lm):
        print(f"{k}/{n_lm}")
        zones = np.delete(consensus, k, axis=0)
        per_spec = [register_unlabeled(pts, zones) for pts in gabriel_points]
        mean_cost = np.mean([c for _, c, _ in per_spec])
        results[k] = (per_spec, mean_cost)
    return results


def validate_axis_heuristic(gabriel_points, per_spec, consensus, dropped_k,
                             left_cluster=(0, 1, 14), right_pair=(9, 10)):
    """Check how often a simple top>mid>bottom / right>left rule (applied
    in the registered frame) agrees with the full geometric assignment."""
    zone_orig_idx = [i for i in range(consensus.shape[0]) if i != dropped_k]
    zones = np.delete(consensus, dropped_k, axis=0)
    expected_order = [left_cluster[i] for i in
                       np.argsort(-consensus[list(left_cluster), 1])]

    n_left_match, n_right_match, n = 0, 0, len(gabriel_points)
    for pts, (assign, cost, _) in zip(gabriel_points, per_spec):
        pts0 = pts - pts.mean(axis=0)
        R, scale, t = umeyama(pts0, zones[assign])
        transformed = apply_transform(pts0, R, scale, t)

        loc = {lm: transformed[np.where(assign == zone_orig_idx.index(lm))[0][0]]
               for lm in left_cluster if lm != dropped_k}
        if len(loc) == len(left_cluster):
            order = sorted(loc, key=lambda lm: -loc[lm][1])
            n_left_match += int(order == expected_order)

        if all(lm != dropped_k for lm in right_pair):
            p9 = transformed[np.where(assign == zone_orig_idx.index(right_pair[0]))[0][0]]
            p10 = transformed[np.where(assign == zone_orig_idx.index(right_pair[1]))[0][0]]
            n_right_match += int(p9[0] > p10[0])

    print(f"Left-cluster axis heuristic agreement: {n_left_match}/{n}")
    print(f"Right-pair axis heuristic agreement:   {n_right_match}/{n}")


if __name__ == "__main__":
    tancrede_path, gabriel_path = sys.argv[1], sys.argv[2]

    print("Building reference template from Tancrede's hand-digitized data...")
    _, consensus = build_reference(tancrede_path)

    print("Loading Gabriel's unordered predictions...")
    gab = parse_tps(gabriel_path)
    gab, _ = filter_valid(gab, 18, label="gabriel")
    gabriel_points = [s['coords'] for s in gab]

    print("Running leave-one-out sweep (which of 19 landmarks has no Gabriel counterpart)...")
    results = leave_one_out_sweep(gabriel_points, consensus)
    for k, (_, mean_cost) in sorted(results.items()):
        print(f"  drop LM{k:2d}: mean_cost={mean_cost:.6f}")

    k_best = min(results, key=lambda k: results[k][1])
    print(f"\nBest hypothesis: LM{k_best} is the unmatched landmark "
          f"(lowest mean registration cost = {results[k_best][1]:.6f})")

    print("\nValidating the top/bottom - left/right tie-break heuristic:")
    per_spec, _ = results[k_best]
    validate_axis_heuristic(gabriel_points, per_spec, consensus, k_best)