"""export_final_landmarks.py
Exports a clean, self-contained landmarks package for external use (e.g.
Adrien's R/geomorph pipeline) from a dataset already processed by the
landmarking pipeline (extraction -> landmarks/predict.py -> renumber.py).

Reuses core.dataset.load_dataset() for the TPS/manifest/biological_data
join and OK/SUSPECT/FAILED exclusion -- see utils.cli.add_dataset_args()
for the exact filtering semantics (--tps chooses the 19- or 18-landmark
file, --devices/--species/--castes, --include-outliers). Exports the WHOLE
dataset root given -- one call per dataset root (e.g. once for
data/Bombus/collection, once for data/Bombus/terrain), there is no
train/test split within a single call anymore.

Canonical output directory: `<dataset>/export/` (override with --output-dir).
This is the R-facing package for the dataset, kept separate from the
working files (`extraction/`, `landmarks/`) and from the root
`biological_data.csv` (specimen-level, from tools/ingestion/export_clean_dataset.py).
The `biological_data.csv` written HERE is photo-level, row-aligned to the
TPS.

Output (in --output-dir, default `<dataset>/export/`), for the requested specimens:
    landmarks_<n>lm_crop.tps       crop-space coordinates
    landmarks_<n>lm_original.tps   raw-image-space coordinates
                                    (omitted with --no-original-space)
    biological_data.csv            same row order/IDs as the TPS (photo-level)
    failed.csv                     excluded photos, by stage

In both TPS files: sequential integer IDs (ID=1, 2, ...), no COMMENT=,
identical order to biological_data.csv's first column -- this is the
final, R-facing export, deliberately not using
core.tps_io.assign_sequential_ids()'s photo_id ordering.

Original-image-space reprojection uses extraction.normalize_crop's wing
transform (inverse of raw -> crop) together with extraction/<mode>/
detection.csv's OBB and the clean dataset's own image dimensions (the same
`path` detect_wing.py itself reads to run detection -- there is no
separate "raw" file anymore, see tools/ingestion/export_clean_dataset.py). A photo
that fails reprojection is excluded from ALL outputs (not just the
original-space TPS), to keep the three files in lockstep; it's listed in
failed.csv under stage "reprojection". --padding/--out-width/--out-height
must match whatever normalize_crop.py actually used to produce the crops.

Biological columns come straight from biological_data.csv (via
core.dataset.load_dataset()'s meta_df, joined by inv_id) -- no separate
identification CSV merge here anymore (tools/ingestion/export_clean_dataset.py
already produced a clean, single-schema biological_data.csv per source).
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np

from core.tps_io import ImageLandmarks, write_tps
from extraction.normalize_crop import (
    apply_wing_transform_to_points_inverse,
    compute_wing_transform,
    normalized_points_to_pixels,
    read_image_size,
)
from utils.cli import (
    add_dataset_args,
    add_dataset_positional,
    add_logging_args,
    dataset_kwargs,
    log_level_from_args,
)
from core.dataset import load_dataset
from core.pipeline_io import dataset_export_dir, read_csv_rows, resolve_path
from core.run_io import setup_console_logging

logger = logging.getLogger(__name__)

FAILED_FIELDS = ["photo_id", "inv_id", "stage", "reason"]
BASE_BIO_COLUMNS = ["id", "inv_id", "species", "caste", "device", "device_tag"]

# Consecutive "image unreadable" failures before aborting reprojection
# entirely -- a handful is a few corrupt/missing files (kept in
# failed.csv, the run continues); this many in a row almost always
# means the clean dataset's image folder isn't where --base-dir expects it,
# so fail fast with a clear message instead of grinding through the rest of
# the dataset one slow failure at a time.
UNREADABLE_IMAGE_ABORT_THRESHOLD = 5


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a clean TPS (crop and/or original image space) + "
                    "biological data CSV + failure report from a processed dataset."
    )
    add_dataset_positional(parser)
    add_dataset_args(parser)
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory to write the exported files to "
             "(default: <dataset>/export/ -- see core.pipeline_io.dataset_export_dir).",
    )
    parser.add_argument(
        "--no-original-space", action="store_true",
        help="Skip original-image-space reprojection: crop-space TPS + biological_data only, "
             "no need for raw image access.",
    )
    parser.add_argument(
        "--mode", default="light", choices=["heavy", "light"],
        help="extraction/<mode>/detection.csv to use for reprojection (default: light).",
    )
    parser.add_argument(
        "--base-dir", type=Path, default=None,
        help="Root to resolve manifest.csv's path against, if it isn't already absolute.",
    )
    parser.add_argument(
        "--padding", type=float, default=0.10,
        help="Must match the --padding normalize_crop.py used to produce these crops (default: 0.10).",
    )
    parser.add_argument(
        "--out-width", type=int, default=512,
        help="Must match the --out-width normalize_crop.py used to produce these crops (default: 512).",
    )
    parser.add_argument(
        "--out-height", type=int, default=256,
        help="Must match the --out-height normalize_crop.py used to produce these crops (default: 256).",
    )
    add_logging_args(parser)
    return parser.parse_args(argv)


def build_failure_report(
    dataset: Path, kept_tps_ids: set[int], extra_filters_active: bool,
) -> list[dict]:
    """Explains, for every photo in the dataset absent from the final
    export, at which stage it was excluded and why -- see module docstring
    for the exact scope (extra_filters_active caveat)."""
    manifest = {row["photo_id"]: row for row in read_csv_rows(dataset / "manifest.csv")}
    landmarks_status = read_csv_rows(dataset / "landmarks" / "landmarks_numbered.csv")
    biological_data = {row["inv_id"]: row for row in read_csv_rows(dataset / "biological_data.csv")}

    rows: list[dict] = []
    for lm_row in landmarks_status:
        photo_id = lm_row.get("photo_id", "")
        manifest_row = manifest.get(photo_id)
        if manifest_row is None:
            continue  # not part of this dataset's manifest at all
        inv_id = lm_row.get("inv_id") or manifest_row.get("inv_id", "")

        status = lm_row.get("status", "")

        if status == "FAILED":
            rows.append({
                "photo_id": photo_id, "inv_id": inv_id,
                "stage": "landmark_placement", "reason": lm_row.get("error_reason", ""),
            })
        elif status == "SUSPECT":
            rows.append({
                "photo_id": photo_id, "inv_id": inv_id, "stage": "outlier_registration",
                "reason": lm_row.get("error_reason") or "flagged by per-species MAD outlier detection",
            })
        elif status == "OK":
            try:
                tps_id = int(lm_row["tps_id"])
            except (KeyError, ValueError):
                tps_id = None
            if tps_id in kept_tps_ids or extra_filters_active:
                continue
            bio = biological_data.get(inv_id)
            if bio is None:
                reason, stage = "inv_id not found in biological_data.csv", "biological_metadata"
            elif not bio.get("species"):
                reason, stage = "species missing in biological_data.csv", "biological_metadata"
            else:
                reason, stage = "excluded while loading (e.g. inconsistent landmark count) -- see logs", "other"
            rows.append({"photo_id": photo_id, "inv_id": inv_id, "stage": stage, "reason": reason})
        else:
            rows.append({
                "photo_id": photo_id, "inv_id": inv_id,
                "stage": "landmark_placement", "reason": f"unknown status {status!r}",
            })
    return rows


def _clean(value) -> str:
    """NaN/None -> "" (never the literal text "nan"/"None" in an output CSV)."""
    if value is None:
        return ""
    try:
        if isinstance(value, float) and np.isnan(value):
            return ""
    except TypeError:
        pass
    return str(value)


def reproject_to_raw_space(
    sp: ImageLandmarks, manifest: dict, detection_by_photo: dict, args: argparse.Namespace,
) -> tuple[np.ndarray | None, str, str | None]:
    """Projects one specimen's crop-space landmarks back to raw image space.
    Returns (coords, "", image_path) on success, or (None, reason, None) --
    reason is exactly "unreadable_image" on a read failure, used by main()
    to detect a likely misconfigured --base-dir early. image_path is the
    resolved path to the *clean dataset's* image (manifest.csv's `path`) --
    callers use it as the original-space TPS's IMAGE= entry, since the crop
    path in sp.image_path doesn't apply to these coordinates."""
    photo_id = sp.photo_id
    manifest_row = manifest.get(photo_id) if photo_id else None
    if manifest_row is None:
        return None, "photo_id not found in manifest.csv", None

    detection_row = detection_by_photo.get(photo_id)
    if detection_row is None or detection_row.get("status") != "OK":
        return None, "no OK detection row in detection.csv (OBB unavailable)", None

    image_path = resolve_path(manifest_row["path"], args.base_dir)
    size = read_image_size(image_path)
    if size is None:
        return None, "unreadable_image", None
    width, height = size

    points = normalized_points_to_pixels(detection_row, width=width, height=height)
    if points is None:
        return None, "invalid OBB corners in detection.csv", None

    transform = compute_wing_transform(
        (height, width), points, pad=args.padding, target_ratio=args.out_width / args.out_height,
    )
    if transform is None:
        return None, "degenerate OBB geometry", None

    original_xy = apply_wing_transform_to_points_inverse(
        sp.landmarks.astype(np.float32), transform, args.out_width, args.out_height,
    )
    return original_xy, "", str(image_path)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    setup_console_logging(log_level_from_args(args))

    dataset = args.dataset
    if args.output_dir is None:
        args.output_dir = dataset_export_dir(dataset)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.landmarks_tps is not None and args.landmarks_status_csv is None:
        # load_dataset() only defaults the status CSV for the *default* TPS path -- with an
        # explicit --tps (e.g. the 18-landmark variant) it otherwise skips outlier exclusion
        # entirely with a warning, even though landmarks_numbered.csv applies to both variants.
        args.landmarks_status_csv = dataset / "landmarks" / "landmarks_numbered.csv"
        logger.info("No --landmarks-status-csv given with --tps: defaulting to %s", args.landmarks_status_csv)

    specimens, meta_df = load_dataset(dataset, labeled_only=True, **dataset_kwargs(args))
    n_points = specimens[0].n_points

    extra_filters_active = bool(args.devices or args.species or args.castes)
    kept_tps_ids = {sp.tps_id for sp in specimens}
    failed_rows = build_failure_report(dataset, kept_tps_ids, extra_filters_active)

    reproject = not args.no_original_space
    manifest_by_photo = {row["photo_id"]: row for row in read_csv_rows(dataset / "manifest.csv")}
    detection_by_photo: dict = {}
    if reproject:
        detection_csv = dataset / "extraction" / args.mode / "detection.csv"
        if not detection_csv.exists():
            raise FileNotFoundError(
                f"{detection_csv} not found -- pass --mode or --no-original-space."
            )
        detection_by_photo = {row["photo_id"]: row for row in read_csv_rows(detection_csv)}

    crop_out, original_out, bio_rows = [], [], []
    consecutive_unreadable = 0
    new_id = 1

    for sp, meta in zip(specimens, meta_df.itertuples()):
        original_xy = None
        image_path = None
        if reproject:
            original_xy, reason, image_path = reproject_to_raw_space(sp, manifest_by_photo, detection_by_photo, args)
            if reason == "unreadable_image":
                consecutive_unreadable += 1
                if consecutive_unreadable >= UNREADABLE_IMAGE_ABORT_THRESHOLD:
                    raise FileNotFoundError(
                        f"{consecutive_unreadable} consecutive images could not be read -- check "
                        f"--base-dir, or pass --no-original-space to export crop-space landmarks only."
                    )
            else:
                consecutive_unreadable = 0
            if original_xy is None:
                failed_rows.append({
                    "photo_id": sp.photo_id or "", "inv_id": sp.inv_id or "",
                    "stage": "reprojection", "reason": reason,
                })
                continue  # excluded from ALL outputs, so ids stay dense and the three files match

        crop_out.append(ImageLandmarks(sp.n_points, sp.landmarks, sp.image_path, new_id))
        if reproject:
            original_out.append(ImageLandmarks(sp.n_points, original_xy, image_path, new_id))

        row = {
            "id": new_id, "inv_id": _clean(meta.inv_id), "species": _clean(meta.species), "caste": _clean(meta.caste),
            "device": _clean(meta.device), "device_tag": _clean(meta.device_tag),
        }
        bio_rows.append(row)

        new_id += 1

    n_lm = f"{n_points}lm"
    tps_crop_path = args.output_dir / f"landmarks_{n_lm}_crop.tps"
    write_tps(tps_crop_path, crop_out)

    tps_original_path = None
    if reproject:
        tps_original_path = args.output_dir / f"landmarks_{n_lm}_original.tps"
        write_tps(tps_original_path, original_out)

    bio_path = args.output_dir / "biological_data.csv"
    with bio_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=BASE_BIO_COLUMNS)
        writer.writeheader()
        writer.writerows(bio_rows)

    failed_path = args.output_dir / "failed.csv"
    with failed_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FAILED_FIELDS)
        writer.writeheader()
        writer.writerows(failed_rows)

    print()
    print("=" * 60)
    print("Export complete")
    print("=" * 60)
    print(f"Dataset                  : {dataset}")
    print(f"Specimens exported       : {len(bio_rows)}")
    print(f"Failed/excluded (report) : {len(failed_rows)}")
    print()
    print(f"TPS (crop)     : {tps_crop_path}")
    if tps_original_path:
        print(f"TPS (original) : {tps_original_path}")
    print(f"Biological data : {bio_path}")
    print(f"Failed report   : {failed_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
