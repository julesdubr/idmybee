"""combine_manifests.py
Concatenates `manifest.csv`/`biological_data.csv` from several dataset
roots -- each already built by `tools/ingestion/build_manifest.py` -- into one
combined pair, for downstream stages that need a single dataset root
spanning more than one source (e.g. collection + terrain).

No rescan, no hashing, no filename parsing: each input root's files are
already validated, this only concatenates them. Raises if `photo_id`/
`inv_id` collide across roots -- a sign their `inv_id` spaces overlap (see
`CONVENTIONS.md` "Identification des spécimens").

Usage:
    python -m tools.ingestion.combine_manifests data/clean/collection data/clean/terrain \\
        --output-dir data/clean/combined
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from utils.cli import add_logging_args, log_level_from_args
from core.run_io import setup_console_logging

logger = logging.getLogger(__name__)


def combine(roots: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.concat([pd.read_csv(root / "manifest.csv") for root in roots], ignore_index=True)
    duplicated_photo_ids = sorted(manifest.loc[manifest["photo_id"].duplicated(), "photo_id"].unique())
    if duplicated_photo_ids:
        raise ValueError(f"photo_id collision across combined roots: {duplicated_photo_ids}")

    biological_data = pd.concat([pd.read_csv(root / "biological_data.csv") for root in roots], ignore_index=True)
    duplicated_inv_ids = sorted(biological_data.loc[biological_data["inv_id"].duplicated(), "inv_id"].unique())
    if duplicated_inv_ids:
        raise ValueError(f"inv_id collision across combined roots: {duplicated_inv_ids}")

    return manifest, biological_data


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roots", nargs="+", type=Path, help="Dataset roots to combine (each must already have manifest.csv/biological_data.csv, e.g. from tools/ingestion/build_manifest.py).")
    parser.add_argument("--output-dir", required=True, help="Directory to write the combined manifest.csv/biological_data.csv into.")
    add_logging_args(parser)
    args = parser.parse_args(argv)
    setup_console_logging(log_level_from_args(args))

    manifest, biological_data = combine(args.roots)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_dir / "manifest.csv", index=False)
    biological_data.to_csv(output_dir / "biological_data.csv", index=False)

    print(f"\n--- Summary ---\nmanifest.csv       : {len(manifest)} row(s) from {len(args.roots)} root(s)")
    print(f"biological_data.csv: {len(biological_data)} specimen(s)")
    print(f"\nWritten to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
