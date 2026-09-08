"""export_dataset.py
Builds the UNet fine-tuning manifest: pairs each crop image with its
ground-truth landmark points, ready for dataset.py / train.py.

    python export_dataset.py data/Bombus --mode light \\
        --tps data/Bombus/landmarks/tancrede_reference.tps \\
        --output data/models/unet_landmarks/train_manifest.csv

Goal of this fine-tuning round: train the UNet to predict Tancrede's full
19-point blueprint (LM3 included), instead of Gabriel's original 18-point
scheme (LM3 dropped -- see reconstruct_tps.py --drop 3). By default nothing
is dropped; pass --drop <n> (1-indexed) to reproduce the old 18-point
export if ever needed for comparison.

Reuses the pipeline's own building blocks rather than re-parsing crops.csv
or TPS files independently:
  - landmarks.predict.load_target_crops -- same crops.csv dedup/status/split
    filtering predict.py itself uses (OK+SKIPPED, latest row per photo_id).
  - core.pipeline_io.resolve_path -- same relative-path resolution as
    predict.py's crop_path handling.
  - core.tps_io.parse_tps -- same TPS parser predict.py uses to reload its
    own output.

ASSUMPTION -- please check before trusting the output: `ImageLandmarks`
records returned by parse_tps() expose `.landmarks` as an (N, 2) array of
(x, y) pixel coordinates in the SAME pixel space as the crop images (i.e.
Tancrede digitized directly on the crops, not on raw uncropped photos). If
that's wrong, every exported point is silently misaligned -- which is
exactly why this script always renders debug overlays (crop + plotted
points) for a sample of exported specimens. Look at those before launching
a training run.

KNOWN GAP: joining a TPS specimen to its crop requires `.photo_id`, which
is only set if the TPS has a `COMMENT=photo_id=...` line (this pipeline's
own convention). If your reference TPS doesn't have that (e.g. Tancrede's
raw-space reference, digitized with a third-party tool), see
reproject_reference.py first -- it resolves photo_id via manifest.csv
matching and writes a new TPS with COMMENT= already set, ready for this
script.
"""
from __future__ import annotations

import argparse
import csv
import logging
import random
from pathlib import Path

import cv2
import numpy as np

from landmarks.predict import load_target_crops
from utils.cli import add_dataset_positional, add_logging_args, log_level_from_args
from core.pipeline_io import resolve_path
from core.run_io import setup_console_logging
from core.tps_io import parse_tps

logger = logging.getLogger(__name__)


def save_debug_overlay(crop_path: str, points_xy: np.ndarray, out_path: Path):
    image = cv2.imread(crop_path)
    if image is None:
        return
    for x, y in points_xy:
        cv2.circle(image, (int(round(x)), int(round(y))), 4, (0, 0, 255), -1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), image)


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dataset_positional(parser, help="Dataset root, e.g. data/Bombus")
    parser.add_argument("--mode", default="light", choices=["heavy", "light"],
                         help="Backend whose crops.csv to join against (extraction/{mode}/crops.csv).")
    parser.add_argument("--tps", required=True, help="Path to Tancrede's reference TPS file")
    parser.add_argument("--base-dir", default=None, help="Root for resolving crops.csv's relative output_path, "
                                                           "same meaning as predict.py --base-dir")
    parser.add_argument("--drop", type=int, default=0,
                         help="1-indexed landmark to drop (0 = keep all 19, the default for this round)")
    parser.add_argument("--output", required=True, help="Output manifest CSV path")
    parser.add_argument("--skipped-output", default=None,
                         help="Where to log specimens skipped for lack of a matching crop "
                              "(default: <output>.skipped.csv)")
    parser.add_argument("--debug-overlays-dir", default=None,
                         help="If set, render N random exported crops with points overlaid, "
                              "for visual sanity-checking before training")
    parser.add_argument("--n-debug-overlays", type=int, default=20)
    parser.add_argument("--seed", type=int, default=58)
    add_logging_args(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    setup_console_logging(log_level_from_args(args))

    crops_path = args.dataset / "extraction" / args.mode / "crops.csv"
    base_dir = Path(args.base_dir) if args.base_dir else None
    drop_idx = args.drop - 1 if args.drop else None

    crop_rows = load_target_crops(crops_path, split_filter=None)
    crop_lookup = {row["photo_id"]: row for row in crop_rows}
    logger.info("%d crop(s) available in %s", len(crop_lookup), crops_path)

    specimens, errors = parse_tps(Path(args.tps), strict=False)
    if errors:
        logger.warning("%d unreadable block(s) in %s (skipped):", len(errors), args.tps)
        for e in errors[:10]:
            logger.warning("  specimen #%s, line %s: %s", e.specimen_index, e.line_no, e.message)
    logger.info("%d specimen(s) in %s", len(specimens), args.tps)

    n_with_photo_id = sum(1 for sp in specimens if sp.photo_id is not None)
    logger.info("%d/%d specimen(s) have photo_id set (COMMENT= present)", n_with_photo_id, len(specimens))
    if n_with_photo_id == 0:
        raise SystemExit(
            "No specimen has a COMMENT= photo_id= in this TPS -- likely Tancrede's reference, "
            "digitized with a third-party tool (tpsDig or equivalent), not this pipeline. tps_io.py "
            "explicitly states the caller must then join via core.dataset -- this script does NOT "
            "do that yet (unknown signature). Nothing will be exported until that's wired in, rather "
            "than guessing a filename-based match that could silently pair wrong points with the "
            "wrong image."
        )

    exported_rows = []
    skipped_rows = []
    reported_shape = None

    for sp in specimens:
        if sp.photo_id is None:
            skipped_rows.append({"photo_id": f"(tps_id={sp.tps_id})", "reason": "no photo_id (no COMMENT= in this TPS block)"})
            continue

        crop_row = crop_lookup.get(sp.photo_id)
        if crop_row is None:
            skipped_rows.append({"photo_id": sp.photo_id, "reason": "no matching crop in crops.csv"})
            continue

        crop_path = resolve_path(crop_row["output_path"], base_dir).resolve()  # absolute:
        # crops.csv's output_path is relative to the dataset root (or --base-dir), not to
        # wherever the manifest CSV ends up living. dataset.py's load_manifest() re-resolves
        # any relative crop_path against the *manifest's own* directory (see its docstring) --
        # leaving crop_path relative here silently double-joins it with the run dir
        # (data/models/unet_landmarks/...) instead of the dataset root. Absolute sidesteps
        # the mismatch entirely.
        if not crop_path.exists():
            skipped_rows.append({"photo_id": sp.photo_id, "reason": f"crop file missing: {crop_path}"})
            continue

        points = np.asarray(sp.landmarks, dtype=np.float64)
        if drop_idx is not None:
            points = np.delete(points, drop_idx, axis=0)

        if reported_shape is None:
            img = cv2.imread(str(crop_path))
            if img is not None:
                reported_shape = img.shape[:2]
                logger.info("First crop image shape (height, width): %s "
                            "-- confirm this matches constants.IMG_HEIGHT/IMG_WIDTH", reported_shape)

        row = {
            "photo_id": sp.photo_id,
            "inv_id": getattr(sp, "inv_id", crop_row.get("inv_id", "")),
            "crop_path": str(crop_path),
        }
        for i, (x, y) in enumerate(points):
            row[f"x{i}"] = x
            row[f"y{i}"] = y
        exported_rows.append(row)

    if not exported_rows:
        raise SystemExit("Nothing exported -- check the ASSUMPTION in this file's docstring.")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(exported_rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(exported_rows)

    skipped_path = Path(args.skipped_output or f"{args.output}.skipped.csv")
    with open(skipped_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["photo_id", "reason"])
        writer.writeheader()
        writer.writerows(skipped_rows)

    n_landmarks = len(points)
    print(f"Exported {len(exported_rows)} specimens ({n_landmarks} landmarks each) -> {out_path}")
    print(f"Skipped {len(skipped_rows)} specimens -> {skipped_path}")

    if args.debug_overlays_dir:
        rng = random.Random(args.seed)
        sample = rng.sample(exported_rows, min(args.n_debug_overlays, len(exported_rows)))
        overlays_dir = Path(args.debug_overlays_dir)
        for row in sample:
            points_xy = np.array([[row[f"x{i}"], row[f"y{i}"]] for i in range(n_landmarks)])
            save_debug_overlay(row["crop_path"], points_xy,
                                overlays_dir / f"{row['photo_id']}.png")
        print(f"Wrote {len(sample)} debug overlays -> {overlays_dir}  "
              f"(LOOK AT THESE before training -- see the ASSUMPTION in the docstring)")


if __name__ == "__main__":
    main()
