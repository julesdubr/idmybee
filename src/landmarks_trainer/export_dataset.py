"""
Build the UNet fine-tuning manifest: pairs each crop image with its
ground-truth landmark points, ready for dataset.py / train.py.

    python export_dataset.py \
        --tps path/to/tancrede_reference.tps \
        --crops-csv path/to/extraction/<backend>/crops.csv \
        --output data/models/unet_landmarks/train_manifest.csv

ASSUMPTIONS -- please check these before trusting the output (see the chat
message this shipped with for the exact questions):

1. `utils.tps_io.parse_tps(path)` exists and returns a list of records each
   exposing `.image_id` (str) and `.points` (N,2 array, (x, y) pixel coords,
   19 points for Tancrede's blueprint). If the real signature differs, this
   import will need a one-line fix -- it's deliberately NOT reimplemented
   here to avoid a second, possibly-inconsistent TPS parser living in the
   codebase.
2. Tancrede's TPS points are already in the SAME pixel space as the crop
   images in crops.csv (i.e. he digitized directly on the crops, not on the
   raw uncropped photos). If that's wrong, every exported point is silently
   misaligned -- which is exactly why this script always renders debug
   overlays (crop + plotted points) for a sample of exported specimens.
   Look at those before launching a training run.
3. crops.csv has an image_id column and a column with the crop file path
   (name configurable via --crop-path-col, since exact naming wasn't
   confirmed). If crops.csv accumulates multiple rows per image_id, the
   LAST row wins (same convention as elsewhere in the pipeline) -- a
   warning is printed if duplicates are found.

Landmark drop: Tancrede's blueprint has 19 landmarks; LM3 (1-indexed, index
2 zero-indexed) has no counterpart in the UNet's 18-point scheme and is
dropped by default, matching reconstruct_tps.py --drop 3.
"""

import argparse
import csv
import random
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

try:
    from utils.tps_io import parse_tps
except ImportError as e:
    raise ImportError(
        "export_dataset.py expects utils.tps_io.parse_tps(path) -> records "
        "with .image_id and .points (N,2 array of x,y). Adjust the import "
        "at the top of this file if the real function name/signature "
        "differs, then re-run."
    ) from e


def load_crop_lookup(crops_csv: str, image_id_col: str, crop_path_col: str) -> dict:
    df = pd.read_csv(crops_csv)
    for col in (image_id_col, crop_path_col):
        if col not in df.columns:
            raise ValueError(f"{crops_csv} has no column '{col}' (columns: {list(df.columns)})")

    dupes = df[image_id_col].duplicated().sum()
    if dupes:
        warnings.warn(f"{crops_csv}: {dupes} duplicate image_id rows, keeping the last one")
    df = df.drop_duplicates(subset=image_id_col, keep="last")

    manifest_dir = Path(crops_csv).resolve().parent
    lookup = {}
    for _, row in df.iterrows():
        p = row[crop_path_col]
        p = p if Path(p).is_absolute() else str((manifest_dir / p).resolve())
        lookup[row[image_id_col]] = p
    return lookup


def save_debug_overlay(crop_path: str, points_xy: np.ndarray, out_path: Path):
    image = cv2.imread(crop_path)
    if image is None:
        return
    for x, y in points_xy:
        cv2.circle(image, (int(round(x)), int(round(y))), 4, (0, 0, 255), -1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), image)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tps", required=True, help="Path to Tancrede's reference TPS file")
    parser.add_argument("--crops-csv", required=True, help="crops.csv from extraction")
    parser.add_argument("--image-id-col", default="image_id")
    parser.add_argument("--crop-path-col", default="crop_path")
    parser.add_argument("--drop", type=int, default=3, help="1-indexed landmark to drop (default: LM3)")
    parser.add_argument("--output", required=True, help="Output manifest CSV path")
    parser.add_argument("--skipped-output", default=None,
                         help="Where to log specimens skipped for lack of a matching crop "
                              "(default: <output>.skipped.csv)")
    parser.add_argument("--debug-overlays-dir", default=None,
                         help="If set, render N random exported crops with points overlaid, "
                              "for visual sanity-checking before training")
    parser.add_argument("--n-debug-overlays", type=int, default=20)
    parser.add_argument("--seed", type=int, default=58)
    args = parser.parse_args()

    drop_idx = args.drop - 1

    records = parse_tps(args.tps)
    crop_lookup = load_crop_lookup(args.crops_csv, args.image_id_col, args.crop_path_col)

    exported_rows = []
    skipped_rows = []

    for rec in records:
        crop_path = crop_lookup.get(rec.image_id)
        if crop_path is None:
            skipped_rows.append({"image_id": rec.image_id, "reason": "no matching crop_path"})
            continue

        points = np.asarray(rec.points, dtype=np.float64)
        points = np.delete(points, drop_idx, axis=0)
        if len(points) != 18:
            skipped_rows.append({
                "image_id": rec.image_id,
                "reason": f"expected 18 points after drop, got {len(points)}",
            })
            continue

        row = {"image_id": rec.image_id, "crop_path": crop_path}
        specimen_id = getattr(rec, "specimen_id", None)
        if specimen_id is not None:
            row["specimen_id"] = specimen_id
        for i, (x, y) in enumerate(points):
            row[f"x{i}"] = x
            row[f"y{i}"] = y
        exported_rows.append(row)

    if not exported_rows:
        print("Nothing exported -- check the ASSUMPTIONS in this file's docstring.", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(exported_rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(exported_rows)

    skipped_path = Path(args.skipped_output or f"{args.output}.skipped.csv")
    with open(skipped_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "reason"])
        writer.writeheader()
        writer.writerows(skipped_rows)

    print(f"Exported {len(exported_rows)} specimens -> {out_path}")
    print(f"Skipped {len(skipped_rows)} specimens -> {skipped_path}")

    if args.debug_overlays_dir:
        rng = random.Random(args.seed)
        sample = rng.sample(exported_rows, min(args.n_debug_overlays, len(exported_rows)))
        overlays_dir = Path(args.debug_overlays_dir)
        for row in sample:
            points_xy = np.array([[row[f"x{i}"], row[f"y{i}"]] for i in range(18)])
            save_debug_overlay(row["crop_path"], points_xy,
                                overlays_dir / f"{row['image_id']}.png")
        print(f"Wrote {len(sample)} debug overlays -> {overlays_dir}  "
              f"(LOOK AT THESE before training -- see assumption #2 in the docstring)")


if __name__ == "__main__":
    main()
