"""Build a clean, canonically-named copy of one identification source
(collection, terrain, or any future source) into `--output-dir`.

One run processes ONE source: reads that source's raw ingest manifest and
raw identification CSV, resolves specimen identity conflicts and duplicate
rows, assigns stable `inv_id`/`photo_id` values, and writes a renamed copy
of the images plus clean CSVs. Source files are never modified -- read-only
export, safe to point at production data. See `manifest/identification.py`
for the identity-resolution logic itself.

This tool is OPTIONAL and only produces a clean dataset -- it does not
build the pipeline's own `manifest.csv`/`biological_data.csv` (see
`tools/ingestion/build_manifest.py` for that, which works on this tool's output just
as well as on any other compliant dataset). Run it only when the raw data
actually needs identity resolution.

Outputs: `biological_data_all.csv` (every specimen, photographed or not --
useful even without a photo, no equivalent downstream) and `dataset.csv`
(one row per successfully copied photo, biological columns merged in
directly -- the compliant, per-photo input `tools/ingestion/build_manifest.py`
expects; `--device-column`/`--device-name-column` add a per-photo `device`
column). `--mapping-file` is read-modify-written: existing `inv_id`
assignments are frozen, only newly-seen specimens get a number appended --
safe to re-run.

`--origin-codes` picks the identity derivation mode -- see
`manifest/identification.py::assign_inv_name`/`extend_mapping`. Given
(collection-style): `inv_name` looked up from `--origin-column`,
`inv_num` assigned sequentially. Omitted (terrain-style): `--key-column`
must already be a canonical `<name>_<digits>` id, both parsed from it
directly -- there, the raw key value and `inv_id` are identical, so it is
not kept as a separate `original_id` column in the output (unlike the
collection-style case, where it is genuinely distinct information).

Usage (collection):
    python -m tools.ingestion.export_clean_dataset data/bombus_collection_raw/manifest.csv \\
        --identification-csv /path/to/IDMB_Bombus_collect.csv \\
        --source-type collection --key-column inv_id --device-column device_type \\
        --compare-columns genus,species,caste,collection_origin,identification_year,dd,mm,yyyy \\
        --origin-codes data/clean/collection/collection_origin_codes.csv \\
        --image-group-by genus,species,caste \\
        --mapping-file data/Bombus/collection/inv_id_mapping.csv \\
        --output-dir data/clean/collection

Usage (terrain):
    python -m tools.ingestion.export_clean_dataset data/bombus_terrain_raw/manifest.csv \\
        --identification-csv /path/to/IDMB_Bombus_terrain.csv \\
        --source-type terrain --key-column inv_id \\
        --compare-columns genus,species,caste,dd,mm,yyyy \\
        --image-group-by photographer \\
        --mapping-file data/Bombus/terrain/inv_id_mapping.csv \\
        --output-dir data/clean/terrain

If several sources make up one dataset (e.g. collection + terrain), run
this once per source, pointing `--mapping-file` at the same file so
`inv_id` stays unique across sources, run `tools/ingestion/build_manifest.py` on
each source's `dataset.csv`, then combine the results with
`tools/ingestion/combine_manifests.py`.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
import time
from pathlib import Path

import pandas as pd

from manifest import identification as ident
from utils.cli import add_logging_args, log_level_from_args
from core.pipeline_io import RunCounter, format_duration
from core.run_io import setup_console_logging

logger = logging.getLogger(__name__)

BATCH_SIZE = 50  # console progress checkpoint, see the image-copy loop in main()


def _compute_hash(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()[:16]


def _copy_and_verify(raw_path: str, dest: Path) -> tuple[str | None, str | None, int | None, str | None]:
    """Copy one image to its clean destination and hash/size the COPY
    itself, never the source -- `manifest.csv` must describe the clean
    dataset independently of wherever the raw file lives. Returns
    (path, content_hash, file_size_bytes, error); the first three are None
    on failure, with error set. Never raises -- a handful of unreachable
    files (e.g. an unmounted external drive) should not abort the whole
    export."""
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(raw_path, dest)
        return str(dest), _compute_hash(dest), dest.stat().st_size, None
    except OSError as exc:
        return None, None, None, str(exc)


def _sanitize_path_component(value) -> str:
    """Path components come from free-text data (species names, photographer
    names...) -- neutralize path separators so a stray "/" can't create an
    unintended nested directory."""
    text = str(value).replace("/", "_").replace("\\", "_").strip()
    return text or "unknown"


def _relative_image_path(row: pd.Series, group_by_columns: list[str], photo_id: str, ext: str) -> Path:
    """Build `<group_by_1>/<group_by_2>/.../<photo_id><ext>` from `row`.
    A missing/empty value for a group-by column becomes `unknown_<column>`
    rather than silently collapsing rows into the wrong bucket."""
    parts = []
    for col in group_by_columns:
        value = row.get(col)
        if pd.isna(value) or str(value) == "":
            value = f"unknown_{col}"
        parts.append(_sanitize_path_component(value))
    parts.append(f"{photo_id}{ext}")
    return Path(*parts)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", help="Path to this source's ingest manifest CSV (one source per run, see module docstring).")
    parser.add_argument("--identification-csv", required=True, help="Raw identification CSV for this source.")
    parser.add_argument("--source-type", required=True, help="Label for this source (e.g. 'collection', 'terrain'). Kept apart in the mapping file and in reports.")
    parser.add_argument("--key-column", required=True, help="Identification CSV column matching the raw manifest's original_id.")
    parser.add_argument("--device-column", default=None, help="Identification CSV column distinguishing device per row. Omit if this source has no per-device rows.")
    parser.add_argument("--device-name-column", default="device", help="Identification CSV column holding the human-readable device/camera name for --device-column (default: %(default)s). Ignored if --device-column is omitted.")
    parser.add_argument("--compare-columns", default=",".join(ident.DEFAULT_COMPARE_COLUMNS), help="Comma-separated identification CSV columns compared to detect identity conflicts (default: %(default)s).")
    parser.add_argument("--origin-codes", default=None, help="Origin -> inv_name lookup CSV. If given, inv_name is looked up and inv_num assigned sequentially, and the raw --key-column value is kept as original_id. If omitted, inv_name/inv_num are parsed directly from --key-column (must already be a canonical <name>_<digits> id) and original_id is not added, since it would equal inv_id.")
    parser.add_argument("--origin-column", default="collection_origin", help="Identification CSV column looked up in --origin-codes (default: %(default)s). Ignored if --origin-codes is omitted.")
    parser.add_argument("--default-device-type", default="S", help="Fallback manifest device_type for rows where it's missing (default: %(default)s).")
    parser.add_argument("--image-group-by", default=None, help="Comma-separated columns used to group copied images into subfolders, e.g. 'genus,species,caste' or 'photographer'. Default: no subfolders (flat images/).")
    parser.add_argument("--mapping-file", required=True, help="Frozen original_id -> inv_id mapping CSV (read-modify-written; share the same file across a dataset's sources to keep inv_id unique across them).")
    parser.add_argument("--output-dir", required=True, help="Directory to write this source's clean export into.")
    parser.add_argument("--no-copy-images", action="store_true", help="Write CSVs/reports only, skip copying image files.")
    parser.add_argument("--dry-run", action="store_true", help="Do not write anything, just print the summary.")
    add_logging_args(parser)
    args = parser.parse_args(argv)
    setup_console_logging(log_level_from_args(args))

    manifest_df = pd.read_csv(args.manifest)
    if "source_type" in manifest_df.columns:
        manifest_df = manifest_df[manifest_df["source_type"] == args.source_type].copy()
    if "status" in manifest_df.columns:
        failed = manifest_df["status"] == "FAILED"
        if failed.any():
            logger.info("%d manifest row(s) dropped: status=FAILED in the raw scan (see tools/ingestion/ingest_raw.py)", int(failed.sum()))
        # the raw scan's own status is spent once used to drop FAILED rows here --
        # this run's export produces its own status (copy outcome), see below.
        manifest_df = manifest_df[~failed].drop(columns=["status", "status_reason"], errors="ignore").copy()

    identification_df = pd.read_csv(args.identification_csv).copy()
    for col in ("dd", "mm", "yyyy"):
        if col not in identification_df.columns:
            identification_df[col] = pd.NA

    compare_columns = [c.strip() for c in args.compare_columns.split(",") if c.strip()]
    image_group_by = [c.strip() for c in args.image_group_by.split(",") if c.strip()] if args.image_group_by else []

    required_columns = {args.key_column, *compare_columns}
    if args.device_column:
        required_columns.add(args.device_column)
    if args.origin_codes:
        required_columns.add(args.origin_column)
    missing_required = sorted(required_columns - set(identification_df.columns))
    if missing_required:
        parser.error(f"column(s) not found in {args.identification_csv}: {missing_required}")

    device_types = None
    if args.device_column and args.device_name_column in identification_df.columns:
        device_types = ident.build_device_type_table(identification_df, args.device_column, args.device_name_column)

    manifest_df["device_type"] = manifest_df["device_type"].astype("object")
    device_missing = manifest_df["device_type"].isna() | (manifest_df["device_type"] == "")
    if device_missing.any():
        logger.info("%d manifest row(s) had no device_type, defaulted to %r", int(device_missing.sum()), args.default_device_type)
        manifest_df.loc[device_missing, "device_type"] = args.default_device_type

    origin_codes = ident.load_origin_codes(args.origin_codes) if args.origin_codes else None
    identification_df["inv_name"] = ident.assign_inv_name(
        identification_df, args.key_column, origin_codes, args.origin_column,
    )

    mapping = ident.load_frozen_mapping(args.mapping_file)

    keep_original_id = origin_codes is not None

    resolved_all, conflicts = ident.resolve_identification(
        identification_df, args.key_column, args.device_column, compare_columns,
    )
    conflicted_keys = set(conflicts[args.key_column])
    n_conflicted_specimens = len(conflicted_keys)

    specimens = resolved_all.drop_duplicates(args.key_column)[[args.key_column, "inv_name", "dd", "mm", "yyyy"]].copy()
    specimens = specimens.rename(columns={args.key_column: "original_id"})
    specimens["source_type"] = args.source_type
    specimens["resolved_conflict"] = specimens["original_id"].isin(conflicted_keys)

    frozen_at = pd.Timestamp.now().isoformat(timespec="seconds")
    mapping = ident.extend_mapping(mapping, specimens, frozen_at, embedded_numbering=origin_codes is None)

    biological_data_all = ident.build_specimen_table(mapping, resolved_all, args.key_column, args.source_type, args.device_column, keep_original_id)

    kept_resolved, excluded_no_image = ident.restrict_to_present_images(resolved_all, manifest_df, args.key_column)
    biological_data = ident.build_specimen_table(mapping, kept_resolved, args.key_column, args.source_type, args.device_column, keep_original_id)

    # Both reports below are per-device rows of resolved_all/identification_df
    # at this point (device is meaningless for a conflict or an exclusion --
    # see manifest/identification.py module docstring) -- attach the real
    # canonical inv_id (mapping is now final; attach_canonical_inv_id already
    # renames args.key_column to original_id) and collapse the excluded-rows
    # report to one row per specimen. `conflicts` already has one row per
    # distinct identity variant (not per device), it only needs the correct
    # inv_id column.
    conflicts = ident.attach_canonical_inv_id(conflicts, mapping, args.key_column, args.source_type, keep_original_id)
    conflicts["source_type"] = args.source_type

    excluded_no_image = ident.attach_canonical_inv_id(excluded_no_image, mapping, args.key_column, args.source_type, keep_original_id)
    excluded_no_image = excluded_no_image.drop_duplicates("inv_id").reset_index(drop=True)

    known_original_ids = set(identification_df[args.key_column])
    orphans_report = ident.find_orphan_images(manifest_df, known_original_ids)

    specimen_devices = None
    if args.device_column and args.device_name_column in identification_df.columns:
        specimen_devices = ident.build_specimen_device_table(
            mapping, resolved_all, args.key_column, args.source_type, args.device_column, args.device_name_column,
        )

    source_mapping = mapping[mapping["source_type"] == args.source_type]
    photo_manifest = manifest_df[manifest_df["original_id"].isin(kept_resolved[args.key_column])].copy()
    photos = ident.assign_photo_ids(photo_manifest, source_mapping).reset_index(drop=True)
    photos["source_type"] = args.source_type
    if specimen_devices is not None:
        photos = photos.merge(specimen_devices, on=["inv_id", "device_type"], how="left")
    if "photographer" in biological_data.columns:
        # The raw manifest's own photographer column (from ingest_raw.py's
        # photographer_subfolder convention) only exists for some sources
        # (e.g. terrain) and is always empty for others (e.g. collection,
        # organized by specimen, not by photographer) -- the identification
        # CSV has it for every source, so it's the single source of truth.
        photos = photos.drop(columns=["photographer"], errors="ignore")
        photos = photos.merge(biological_data[["inv_id", "photographer"]], on="inv_id", how="left")

    # `group_by_source` is used only to compute each photo's destination
    # subfolder below -- biological columns (e.g. genus/species/caste)
    # joined in here are NOT part of the final manifest.csv (join on
    # inv_id against biological_data.csv instead, see module docstring).
    group_by_source = photos
    bio_join_cols = [c for c in image_group_by if c in biological_data.columns and c not in photos.columns]
    if bio_join_cols:
        group_by_source = photos.merge(biological_data[["inv_id"] + bio_join_cols], on="inv_id", how="left")
    missing_group_cols = [c for c in image_group_by if c not in group_by_source.columns]
    if missing_group_cols:
        parser.error(f"--image-group-by references column(s) not found on the manifest or biological data: {missing_group_cols}")

    print(f"Specimens (all, source={args.source_type}): {len(biological_data_all)}")
    print(f"Specimens (with image): {len(biological_data)}")
    print(f"Photos: {len(photos)}")
    print(f"New inv_id assigned this run: see {args.mapping_file}")
    print(f"Identity conflicts (kept first occurrence, needs manual review): {n_conflicted_specimens} specimen(s), {len(conflicts)} row(s)")
    print(f"Identification rows excluded (no matching image): {len(excluded_no_image)}")
    print(f"Images with no identification row at all (excluded, need biological data): {orphans_report['original_id'].nunique() if len(orphans_report) else 0} specimen(s), {len(orphans_report)} photo(s)")
    if device_types is not None:
        print(f"Device types: see {Path(args.output_dir) / 'device_types.csv'}")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    output_dir = Path(args.output_dir)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    mapping.to_csv(args.mapping_file, index=False)
    if device_types is not None:
        device_types.to_csv(output_dir / "device_types.csv", index=False)
    biological_data_all.to_csv(output_dir / "biological_data_all.csv", index=False)
    conflicts.to_csv(reports_dir / "identification_conflicts.csv", index=False)
    excluded_no_image.to_csv(reports_dir / "excluded_identification_rows.csv", index=False)
    orphans_report.to_csv(reports_dir / "missing_biological_data.csv", index=False)

    print(f"\n=== Copying images ({args.source_type}) ===")
    paths, hashes, sizes, statuses, reasons = [], [], [], [], []
    copy_failures = []
    counter = RunCounter()
    pipeline_start = time.perf_counter()
    n_photos = len(photos)
    for i in range(n_photos):
        photo_row = photos.iloc[i]
        if args.no_copy_images:
            paths.append(None)
            hashes.append(photo_row.get("content_hash"))
            sizes.append(photo_row.get("file_size_bytes"))
            statuses.append("SKIPPED")
            reasons.append("--no-copy-images: no clean copy written (hash/size below are from the raw scan, not verified)")
            counter.add("SKIPPED")
        else:
            dest = output_dir / "images" / _relative_image_path(group_by_source.iloc[i], image_group_by, photo_row["photo_id"], photo_row["ext"])
            path, content_hash, size, error = _copy_and_verify(photo_row["raw_path"], dest)
            paths.append(path)
            hashes.append(content_hash)
            sizes.append(size)
            statuses.append("OK" if error is None else "FAILED")
            reasons.append("" if error is None else error)
            counter.add(statuses[-1])
            if error:
                copy_failures.append({"photo_id": photo_row["photo_id"], "raw_path": photo_row["raw_path"], "error": error})

        index = i + 1
        if index % BATCH_SIZE == 0 or index == n_photos:
            elapsed = time.perf_counter() - pipeline_start
            print(
                f"[{index}/{n_photos}] elapsed: {format_duration(elapsed)} -- "
                f"average: {elapsed / index:.3f} s/image -- {counter}"
            )

    photos = photos.assign(path=paths, content_hash=hashes, file_size_bytes=sizes, status=statuses, status_reason=reasons)
    if copy_failures:
        pd.DataFrame(copy_failures).to_csv(reports_dir / "copy_failures.csv", index=False)
        print(f"\n{len(copy_failures)} image(s) could not be copied -- see reports/copy_failures.csv")

    # `dataset.csv` is this script's actual deliverable: one row per
    # successfully copied photo, biological columns merged in directly
    # (denormalized) -- the compliant, per-photo input tools/ingestion/build_manifest.py
    # turns into manifest.csv/biological_data.csv. Rows whose copy failed or
    # was skipped (--no-copy-images) have no valid path to hand off, so
    # they're excluded (already traced via reports/copy_failures.csv or
    # status=SKIPPED above).
    admin_columns = {"inv_id", "inv_name", "resolved_conflict"}
    bio_merge_columns = ["inv_id"] + [
        c for c in biological_data.columns
        if c not in admin_columns and not c.startswith("n_photos") and c not in photos.columns
    ]
    dataset_df = photos[photos["status"] == "OK"].merge(biological_data[bio_merge_columns], on="inv_id", how="left")

    dataset_columns = ["photo_id", "inv_id", "device_type"]
    if "device" in dataset_df.columns:
        dataset_columns.append("device")
    dataset_columns.append("photo_index")
    if "photographer" in dataset_df.columns:
        dataset_columns.append("photographer")
    dataset_columns.append("path")
    dataset_columns += [c for c in bio_merge_columns if c != "inv_id"]
    dataset_df[dataset_columns].to_csv(output_dir / "dataset.csv", index=False)
    print(
        f"\nWritten to {output_dir} ({len(dataset_df)} photo(s) in dataset.csv -- "
        "run tools/ingestion/build_manifest.py on it next to get manifest.csv/biological_data.csv)"
    )


if __name__ == "__main__":
    main()