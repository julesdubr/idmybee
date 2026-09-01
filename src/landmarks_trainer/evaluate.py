"""
Evaluate a trained UNet's landmark localization quality, using the exact
same inference code path as production (landmarks/predict.py) rather than
a separate decode implementation -- so these numbers reflect what actually
happens when the model is deployed, plateau-grouping bugfixes included.

    python evaluate.py \
        --weights data/models/unet_landmarks/2026-08-26_finetune1/weights.pth \
        --manifest data/models/unet_landmarks/2026-08-26_finetune1/test_manifest.csv \
        --output data/models/unet_landmarks/2026-08-26_finetune1/eval.json

Scope: this measures raw point localization (predicted points vs ground
truth points, matched by nearest position via the Hungarian algorithm,
since predict.py's output isn't numbered/identified yet -- that's
renumber.py's job downstream). Don't conflate this with numbering/identity
accuracy: a model can localize points well here and still make numbering
mistakes on zones 0/1/13 because of genuine anatomical ambiguity, not a
localization failure.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from landmarks.predict import predict_landmarks_from_path

from landmarks_trainer.constants import N_LANDMARKS
from landmarks_trainer.dataset import load_manifest
from landmarks_trainer.model import load_weights
from landmarks_trainer.train import resolve_device


def match_points(pred_xy: np.ndarray, gt_xy: np.ndarray):
    """Hungarian-match predicted points to ground truth by pixel distance.
    Returns per-match distances. len(pred) and len(gt) need not be equal --
    unmatched points (missed GT, or spurious predictions) are reported
    separately rather than silently ignored."""
    if len(pred_xy) == 0 or len(gt_xy) == 0:
        return np.array([]), len(gt_xy), len(pred_xy)

    cost = cdist(pred_xy, gt_xy)
    pred_idx, gt_idx = linear_sum_assignment(cost)
    distances = cost[pred_idx, gt_idx]

    n_missed = len(gt_xy) - len(gt_idx)
    n_spurious = len(pred_xy) - len(pred_idx)
    return distances, n_missed, n_spurious


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--n-landmarks", type=int, default=N_LANDMARKS)
    parser.add_argument("--output", default=None, help="Write metrics as JSON here")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    device = resolve_device(args.device)
    model = load_weights(args.weights, device=device)
    model.eval()

    df = load_manifest(args.manifest, n_landmarks=args.n_landmarks)

    all_distances = []
    total_missed = 0
    total_spurious = 0
    status_counts = {"OK": 0, "SUSPECT": 0, "FAILED": 0}

    for _, row in df.iterrows():
        status, _, pred_xy, _ = predict_landmarks_from_path(
            Path(row["crop_path"]), model, device, args.n_landmarks,
        )
        status_counts[status] += 1

        gt_xy = np.array([[row[f"x{i}"], row[f"y{i}"]] for i in range(args.n_landmarks)])
        pred_xy = pred_xy if pred_xy is not None else np.empty((0, 2))

        distances, missed, spurious = match_points(pred_xy, gt_xy)
        all_distances.extend(distances.tolist())
        total_missed += missed
        total_spurious += spurious

    all_distances = np.array(all_distances)
    results = {
        "n_specimens": len(df),
        "status_counts": status_counts,
        "n_matched_points": len(all_distances),
        "n_missed_points": total_missed,
        "n_spurious_points": total_spurious,
        "mean_error_px": float(all_distances.mean()) if len(all_distances) else None,
        "median_error_px": float(np.median(all_distances)) if len(all_distances) else None,
        "pct_within_5px": float((all_distances <= 5).mean()) if len(all_distances) else None,
        "pct_within_10px": float((all_distances <= 10).mean()) if len(all_distances) else None,
    }

    print(json.dumps(results, indent=2))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
