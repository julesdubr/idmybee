"""reproject_reference.py
Reprojects Tancrede's reference (19 landmarks, raw image space) into the
final crop space (512x256 by default), reusing EXACTLY the same geometry as
extraction/normalize_crop.py (see compute_wing_transform, verified
numerically identical to rotate_image()+crop_with_context() on 150
synthetic cases -- any image whose crop was produced with the same
--padding/--out-width/--out-height therefore gets points aligned to the
pixel on that crop).

    python reproject_reference.py data/Bombus --mode light \\
        --tancrede-tps data/reference/reference_landmarks.tps \\
        --output-tps data/Bombus/landmarks/tancrede_reference_crop_space.tps

Produces:
  - <output-tps>            : new reference, crop space, with
                               COMMENT=image_id=...;specimen_id=... already
                               filled in (usable directly by
                               export_dataset.py, no more "KNOWN GAP").
  - <output-tps>.review.csv : one row per specimen of Tancrede's reference
                               (matched or not), with a `keep` column you
                               fill in by hand after looking at the
                               overlays.
  - <overlays-dir>/{OK,SUSPECT}/*.png : the final crop + the 19 numbered
                               points drawn on it, sorted by status (see
                               utils/tps_overlay.py, generalized to also
                               work on .tps files produced by
                               landmarks/predict.py). For spotting
                               misaligned specimens (Tancrede sometimes
                               worked from photos already lightly cropped
                               by hand -- so the automatic OBB can
                               correspond to a different framing than the
                               one he annotated).

specimen -> image_id join: Tancrede's TPS has no COMMENT= (digitized
outside this pipeline), so no direct image_id. IMAGE= is matched against
manifest.csv by path (the last 3 segments -- species/sex/file folder --
lift duplicate-filename ambiguity between species; falling back to plain
filename otherwise). Not tested on the whole of Tancrede's set, only on the
files provided (610/625 matched, 599 with detection+crop OK) -- unmatched
ones are listed in review.csv, not silently dropped.

IMPORTANT: --padding/--out-width/--out-height must be EXACTLY the ones used
to produce extraction/{mode}/crops.csv, otherwise the computed points don't
correspond to the actual crops -- default values = normalize_crop.py's
defaults, change both together if either one changes.

Y convention: if Tancrede's points land correctly horizontally but come out
vertically flipped on the overlays, pass --flip-y (likely: his TPS comes
from a third-party tool like tpsDig, which uses Y-up instead of the
standard pixel Y-down convention). The correction happens on the raw-image
side, before rotation -- flipping afterwards in crop space does NOT work
once the wing is rotated (reflection and rotation don't commute), hence the
flag here rather than a separate post-processing step.

Review flow (2 passes):
    1) First run (without --exclude-csv): produces everything, review.csv
       has the `keep` column empty/TRUE throughout, `auto_suspect` flags
       points outside the crop (a strong sign of misalignment).
    2) You look at the overlays, set `keep=FALSE` on the bad ones in
       review.csv.
    3) Second run with --exclude-csv pointing at THIS review.csv: the
       `keep` decisions are carried over (image_id by image_id) into the
       new review.csv, and specimens with `keep=FALSE` are excluded from
       <output-tps>.
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np

from extraction.normalize_crop import (
    compute_wing_transform, apply_wing_transform_to_points,
    normalized_points_to_pixels, read_image,
)
from utils.cli import add_dataset_positional, add_logging_args, log_level_from_args
from utils.pipeline_io import RunCounter, read_csv_rows, resolve_path
from utils.run_io import setup_console_logging
from core.tps_io import ImageLandmarks, parse_tps, write_tps
from utils.tps_overlay import render_tps_overlays

logger = logging.getLogger(__name__)

FALSY = {"false", "0", "non", "no", "n"}


def norm_path(p: str) -> str:
    return p.replace("\\", "/")


def path_tail(p: str, n: int = 3) -> str:
    parts = Path(norm_path(p)).parts
    return "/".join(parts[-n:]).lower()


def path_basename(p: str) -> str:
    return Path(norm_path(p)).name.lower()


def build_manifest_index(manifest_rows: list) -> tuple[dict, dict]:
    by_tail, by_basename = {}, {}
    for row in manifest_rows:
        raw = row["raw_path"]
        by_tail.setdefault(path_tail(raw), []).append(row)
        by_basename.setdefault(path_basename(raw), []).append(row)
    return by_tail, by_basename


def match_manifest_row(image_path: str, by_tail: dict, by_basename: dict):
    """Returns (row, method) or (None, failure_reason)."""
    tail_candidates = by_tail.get(path_tail(image_path))
    if tail_candidates and len(tail_candidates) == 1:
        return tail_candidates[0], "tail3"

    base_candidates = by_basename.get(path_basename(image_path))
    if base_candidates and len(base_candidates) == 1:
        return base_candidates[0], "basename"
    if base_candidates and len(base_candidates) > 1:
        return None, f"ambiguous filename ({len(base_candidates)} candidates in manifest.csv)"

    return None, "no match in manifest.csv"


def load_previous_keep_decisions(path: str) -> dict:
    decisions = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            keep = (row.get("keep") or "").strip()
            if keep:
                decisions[row["image_id"]] = keep
    return decisions


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dataset_positional(parser)
    parser.add_argument("--mode", default="light", choices=["heavy", "light"])
    parser.add_argument("--tancrede-tps", required=True)
    parser.add_argument("--base-dir", default=None, help="Root for resolving raw_path (manifest.csv), same meaning as normalize_crop.py --base-dir")
    parser.add_argument("--crops-base-dir", default=None, help="Root for resolving output_path (crops.csv), same meaning as predict.py --base-dir -- NOT necessarily the same as --base-dir")
    parser.add_argument("--padding", type=float, default=0.10, help="MUST match what produced crops.csv")
    parser.add_argument("--out-width", type=int, default=512, help="idem")
    parser.add_argument("--out-height", type=int, default=256, help="idem")
    parser.add_argument("--flip-y", action="store_true",
                         help="Flips Y (y -> raw_image_height - y) on Tancrede's points before "
                              "reprojection -- his TPS appears to use tpsDig's Y-up convention rather "
                              "than the standard pixel convention. Corrects BEFORE rotation, not after "
                              "(see comment in the code) -- this flag is the right place, not a "
                              "post-processing step on <output-tps>.")
    parser.add_argument("--output-tps", required=True)
    parser.add_argument("--review-output", default=None, help="Default: <output-tps>.review.csv")
    parser.add_argument("--overlays-dir", default=None, help="Default: <output-tps>.overlays/")
    parser.add_argument("--exclude-csv", default=None,
                         help="review.csv from a previous run: carries over `keep` decisions already made")
    add_logging_args(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    setup_console_logging(log_level_from_args(args))

    target_ratio = args.out_width / args.out_height
    base_dir = Path(args.base_dir) if args.base_dir else None
    crops_base_dir = Path(args.crops_base_dir) if args.crops_base_dir else None

    output_tps = Path(args.output_tps)
    review_path = Path(args.review_output) if args.review_output else Path(f"{args.output_tps}.review.csv")
    overlays_dir = Path(args.overlays_dir) if args.overlays_dir else Path(f"{args.output_tps}.overlays")

    manifest_rows = read_csv_rows(args.dataset / "manifest.csv")
    by_tail, by_basename = build_manifest_index(manifest_rows)

    detection_csv = args.dataset / "extraction" / args.mode / "detection.csv"
    crops_csv = args.dataset / "extraction" / args.mode / "crops.csv"
    detections = {row["image_id"]: row for row in read_csv_rows(detection_csv)}
    crops = {row["image_id"]: row for row in read_csv_rows(crops_csv)}

    specimens, errors = parse_tps(Path(args.tancrede_tps), strict=False)
    logger.info("%d specimen(s) in %s (%d unreadable block(s))", len(specimens), args.tancrede_tps, len(errors))

    previous_keep = load_previous_keep_decisions(args.exclude_csv) if args.exclude_csv else {}
    if args.exclude_csv:
        logger.info("%d `keep` decision(s) carried over from %s", len(previous_keep), args.exclude_csv)

    counter = RunCounter()
    review_rows = []
    output_specimens = []
    image_cache_shape = {}  # image_id -> (height, width), avoids re-reading an image already seen

    for sp in specimens:
        review_row = {
            "image_id": "", "specimen_id": "", "tancrede_image_path": sp.image_path,
            "matched_via": "", "status": "SKIPPED", "reason": "", "n_out_of_bounds": "",
            "aspect_ratio_obb": "", "aspect_ratio_crops_csv": "", "keep": "",
        }

        manifest_row, method_or_reason = match_manifest_row(sp.image_path, by_tail, by_basename)
        if manifest_row is None:
            review_row["reason"] = method_or_reason
            counter.add("SKIPPED")
            review_rows.append(review_row)
            continue

        image_id = manifest_row["image_id"]
        review_row["image_id"] = image_id
        review_row["specimen_id"] = manifest_row.get("specimen_id", "")
        review_row["matched_via"] = method_or_reason

        detection = detections.get(image_id)
        if detection is None or detection.get("status") != "OK":
            review_row["reason"] = "detection.csv missing or status != OK for this image_id"
            counter.add("SKIPPED")
            review_rows.append(review_row)
            continue

        crop_row = crops.get(image_id)
        if crop_row is None or crop_row.get("status") != "OK":
            review_row["reason"] = "crops.csv missing or status != OK for this image_id"
            counter.add("SKIPPED")
            review_rows.append(review_row)
            continue

        if image_id not in image_cache_shape:
            raw_path = resolve_path(manifest_row["raw_path"], base_dir)
            image = read_image(raw_path)
            if image is None:
                review_row["reason"] = f"unreadable raw image: {raw_path}"
                counter.add("SKIPPED")
                review_rows.append(review_row)
                continue
            image_cache_shape[image_id] = image.shape
        image_shape = image_cache_shape[image_id]

        obb_points = normalized_points_to_pixels(detection, width=image_shape[1], height=image_shape[0])
        if obb_points is None:
            review_row["reason"] = "invalid OBB in detection.csv"
            counter.add("SKIPPED")
            review_rows.append(review_row)
            continue

        transform = compute_wing_transform(image_shape, obb_points, pad=args.padding, target_ratio=target_ratio)
        if transform is None:
            review_row["reason"] = "degenerate OBB geometry (compute_wing_transform)"
            counter.add("SKIPPED")
            review_rows.append(review_row)
            continue

        points_raw = sp.landmarks.astype(np.float64).copy()
        if args.flip_y:
            # Tancrede's TPS appears to use tpsDig's Y-up convention (Y=0 at
            # the bottom of the image) rather than the standard pixel
            # convention (Y=0 at the top) used by detection.csv/the OBB.
            # This MUST be corrected HERE, in raw-image space, before
            # rotation -- flipping Y afterwards, in final crop space, does
            # NOT work once the wing is rotated (reflection and rotation
            # don't commute): verified, a point flipped after the fact
            # doesn't land in the right place as soon as the OBB's rotation
            # angle isn't 0.
            points_raw[:, 1] = image_shape[0] - points_raw[:, 1]

        crop_xy = apply_wing_transform_to_points(points_raw, transform, args.out_width, args.out_height)

        n_oob = int(np.sum(
            (crop_xy[:, 0] < 0) | (crop_xy[:, 0] >= args.out_width) |
            (crop_xy[:, 1] < 0) | (crop_xy[:, 1] >= args.out_height)
        ))
        review_row["n_out_of_bounds"] = n_oob
        review_row["aspect_ratio_obb"] = f"{transform.aspect_ratio:.4f}"
        review_row["aspect_ratio_crops_csv"] = crop_row.get("aspect_ratio", "")

        status = "SUSPECT" if n_oob > 0 else "OK"
        review_row["status"] = status
        if n_oob > 0:
            review_row["reason"] = f"{n_oob} landmark(s) outside the {args.out_width}x{args.out_height} crop"

        keep = previous_keep.get(image_id, "")
        review_row["keep"] = keep
        counter.add(status)
        review_rows.append(review_row)

        if keep.strip().lower() in FALSY:
            continue  # excluded by a previous review decision

        output_specimens.append(ImageLandmarks.from_image(
            n_points=len(crop_xy), landmarks=crop_xy, image_path=crop_row["output_path"],
            image_id=image_id, specimen_id=manifest_row.get("specimen_id"),
        ))

    output_tps.parent.mkdir(parents=True, exist_ok=True)
    write_tps(output_tps, output_specimens)

    review_path.parent.mkdir(parents=True, exist_ok=True)
    with open(review_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(review_rows[0].keys()) if review_rows else [])
        writer.writeheader()
        writer.writerows(review_rows)

    # Overlays -- delegated to utils.tps_overlay (shared with
    # landmarks/predict.py): re-reads output_tps we just wrote, sorts
    # OK/SUSPECT into overlays_dir/ by joining on image_id via review_path
    # (same image_id/status columns as any other CSV in the pipeline).
    overlay_summary = render_tps_overlays(
        output_tps, overlays_dir, base_dir=crops_base_dir, csv_path=review_path,
    ) if output_specimens else {"written": 0, "unmatched": 0, "skipped": 0}

    print(f"{counter}")
    print(f"{len(output_specimens)} specimen(s) written -> {output_tps}")
    print(f"Review -> {review_path}")
    print(f"Overlays ({overlay_summary['written']} written) -> {overlays_dir}")
    if not args.exclude_csv:
        print(
            "\nFirst run: look at the overlays, set `keep=FALSE` on the bad ones in "
            f"{review_path}, then rerun with --exclude-csv {review_path}"
        )


if __name__ == "__main__":
    main()
