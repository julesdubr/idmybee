"""
Evaluate a trained UNet's landmark localization quality.

    python evaluate.py \
        --weights data/models/unet_landmarks/2026-08-26_finetune1/weights.pth \
        --manifest data/models/unet_landmarks/2026-08-26_finetune1/test_manifest.csv \
        --output data/models/unet_landmarks/2026-08-26_finetune1/eval.json

Scope: this measures raw point localization (predicted heatmap peaks vs
ground truth points, matched by nearest position via the Hungarian
algorithm since the heatmap doesn't carry landmark identity). It does NOT
measure numbering/identity accuracy -- that's the renumber.py step
downstream, with its own LOOCV LDA accuracy figures. Don't conflate the
two: a model can localize points well here and still have a hard time on
zones 0/1/13 identity assignment because of genuine anatomical ambiguity.
"""

import argparse
import json

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader

from constants import IMG_HEIGHT, IMG_WIDTH, N_LANDMARKS
from dataset import load_manifest, LandmarkHeatmapDataset
from heatmap import decode_heatmap
from model import load_weights
from train import resolve_device


def match_points(pred_rc: np.ndarray, gt_rc: np.ndarray):
    """Hungarian-match predicted points to ground truth by pixel distance.
    Returns per-match distances. len(pred) and len(gt) need not be equal --
    unmatched points (missed GT, or spurious predictions) are reported
    separately rather than silently ignored."""
    if len(pred_rc) == 0 or len(gt_rc) == 0:
        return np.array([]), len(gt_rc), len(pred_rc)

    cost = cdist(pred_rc, gt_rc)
    pred_idx, gt_idx = linear_sum_assignment(cost)
    distances = cost[pred_idx, gt_idx]

    n_missed = len(gt_rc) - len(gt_idx)  # ground truth points with no predicted match at all
    n_spurious = len(pred_rc) - len(pred_idx)
    return distances, n_missed, n_spurious


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default=None, help="Write metrics as JSON here")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--min-distance", type=int, default=8, help="decode_heatmap min_distance")
    parser.add_argument("--threshold-abs", type=float, default=0.05, help="decode_heatmap threshold_abs")
    args = parser.parse_args()

    device = resolve_device(args.device)
    model = load_weights(args.weights, device=device)
    model.eval()

    df = load_manifest(args.manifest)
    ds = LandmarkHeatmapDataset(df, img_shape=(IMG_HEIGHT, IMG_WIDTH), train_augment=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    all_distances = []
    total_missed = 0
    total_spurious = 0

    with torch.no_grad():
        for images, heatmaps in loader:
            images = images.to(device)
            pred_heatmap = model(images)[0, 0].cpu().numpy()
            gt_heatmap = heatmaps[0, 0].numpy()

            pred_points = decode_heatmap(pred_heatmap, n_points=N_LANDMARKS,
                                          min_distance=args.min_distance,
                                          threshold_abs=args.threshold_abs)
            gt_points = decode_heatmap(gt_heatmap, n_points=N_LANDMARKS,
                                        min_distance=args.min_distance, threshold_abs=0.5)

            distances, missed, spurious = match_points(pred_points, gt_points)
            all_distances.extend(distances.tolist())
            total_missed += missed
            total_spurious += spurious

    all_distances = np.array(all_distances)
    results = {
        "n_specimens": len(df),
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
