"""build_manifest.py
Builds `manifest.csv`/`biological_data.csv` -- the contract every
downstream pipeline stage reads via `utils.dataset.load_dataset()` -- from
ANY per-photo dataset CSV, whether or not it came from
`tools/export_clean_dataset.py`. See `manifest/build.py` for the input
contract (mandatory/optional columns) and the pure validation logic itself.

This tool never copies or renames files -- image paths (absolute, or
relative to `--base-dir`) are used exactly as given, wherever they live (a
repo folder, an external volume, anywhere else on disk). It does
structural validation only (files present/readable, no duplicate
`photo_id`, biological columns consistent per `inv_id`), never identity
arbitration -- run `tools/export_clean_dataset.py` first if the input
actually needs that (the same raw label reused across two different
specimens, messy multi-convention filenames, ...).

Output: `manifest.csv` + `biological_data.csv` if every row validates OK.
Otherwise `manifest_raw.csv` only (same schema, `status`/`status_reason`
explain what's wrong per row) -- fix it by hand, or run
`tools/export_clean_dataset.py` on the underlying raw data if it turns out
to need real identity resolution, then re-run this tool on its output.

Usage:
    python -m tools.build_manifest data/clean/collection/dataset.csv \\
        --output-dir data/clean/collection
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from manifest import build as mbuild
from utils.cli import add_logging_args, log_level_from_args
from utils.pipeline_io import RunCounter
from utils.run_io import setup_console_logging

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_csv", help="Per-photo CSV describing the dataset (see module docstring).")
    parser.add_argument("--output-dir", required=True, help="Directory to write manifest.csv/biological_data.csv (or manifest_raw.csv) into.")
    parser.add_argument("--path-column", default="path", help="Column giving each photo's image file (default: %(default)s).")
    parser.add_argument("--base-dir", default=None, help="Resolve relative paths against this directory (default: dataset_csv's own parent directory).")
    parser.add_argument("--default-device-type", default="S", help="Fallback device_type for rows missing it (default: %(default)s).")
    add_logging_args(parser)
    args = parser.parse_args(argv)
    setup_console_logging(log_level_from_args(args))

    dataset_csv = Path(args.dataset_csv)
    df = pd.read_csv(dataset_csv)

    missing = mbuild.validate_required_columns(df, args.path_column)
    if missing:
        parser.error(f"column(s) not found in {dataset_csv}: {missing}")

    # Computed from the CSV as read, before resolve_and_verify_images/
    # assign_sequential_photo_ids add their own technical columns (ext,
    # content_hash, ...) -- those must never be mistaken for biological data.
    bio_columns = mbuild.biological_columns(df, args.path_column)

    base_dir = Path(args.base_dir) if args.base_dir else dataset_csv.parent

    df = mbuild.resolve_and_verify_images(df, args.path_column, base_dir)
    df = mbuild.assign_sequential_photo_ids(df, args.path_column, args.default_device_type)
    df = mbuild.flag_duplicate_photo_ids(df)
    df, conflicts = mbuild.check_biological_consistency(df, bio_columns)

    counter = RunCounter()
    for status in df["status"]:
        counter.add(status)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = mbuild.build_manifest_table(df)

    if (df["status"] == "OK").all():
        biological_data = mbuild.build_biological_data(df, bio_columns)
        manifest.to_csv(output_dir / "manifest.csv", index=False)
        biological_data.to_csv(output_dir / "biological_data.csv", index=False)
        print(f"\n--- Summary ---\nmanifest.csv       : {len(manifest)} row(s) -- {counter}")
        print(f"biological_data.csv: {len(biological_data)} specimen(s)")
        print(f"\nWritten to: {output_dir.resolve()}")
    else:
        manifest.to_csv(output_dir / "manifest_raw.csv", index=False)
        if len(conflicts):
            reports_dir = output_dir / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            conflicts.to_csv(reports_dir / "biological_inconsistencies.csv", index=False)
            print(f"Biological inconsistencies: see {reports_dir / 'biological_inconsistencies.csv'}")
        print(f"\n--- Summary ---\nmanifest_raw.csv : {len(manifest)} row(s) -- {counter}")
        print(
            f"\n{dataset_csv} does not meet the compliant-dataset standard -- see manifest_raw.csv's "
            "status/status_reason column. Fix it by hand, or run tools/export_clean_dataset.py on the "
            "underlying raw data if identity resolution is needed, then re-run this tool."
        )


if __name__ == "__main__":
    main()
