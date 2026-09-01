"""cli.py
Shared command-line building blocks.

add_logging_args()/log_level_from_args() : --verbose/--quiet, used by every
script (see utils.run_io.setup_console_logging).

add_dataset_args() : --split, --devices, --species, --castes,
--include-outliers, --non-strict, --tps, --landmarks-status-csv,
--run-label -- shared by classifiers/train.py, classifiers/predict.py
(batch) and analysis/variance_report.py.

resolve_split() : explicit --split > "all" if a custom --tps is given
without --split > default_split.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path


def add_logging_args(parser: argparse.ArgumentParser) -> None:
    """--verbose/--quiet, mutually exclusive. See log_level_from_args()."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--verbose", action="store_true", help="Debug-level logging.")
    group.add_argument("--quiet", action="store_true", help="Warnings and errors only.")


def log_level_from_args(args: argparse.Namespace) -> int:
    if getattr(args, "verbose", False):
        return logging.DEBUG
    if getattr(args, "quiet", False):
        return logging.WARNING
    return logging.INFO


def add_dataset_positional(parser: argparse.ArgumentParser, help: str = "Dataset root (contains manifest.csv).") -> None:
    parser.add_argument("dataset", type=Path, help=help)


def add_dataset_args(parser: argparse.ArgumentParser, default_split: str = "train") -> None:
    parser.add_argument(
        "--split", type=str, default=None,
        help=f"Value of manifest.csv's 'split' column to keep, or 'all'. "
             f"Default: {default_split!r} -- unless --tps is given, in which case 'all' "
             f"(see utils.cli.resolve_split).",
    )
    parser.add_argument("--devices", type=str, nargs="+", default=None,
                         help="Keep only these photos (e.g. --devices P1 S1). See utils.dataset._device_tag.")
    parser.add_argument("--species", type=str, nargs="+", default=None, help="Keep only these species.")
    parser.add_argument("--castes", type=str, nargs="+", default=None, help="Keep only these castes.")
    parser.add_argument(
        "--include-outliers", dest="exclude_outliers", action="store_false", default=True,
        help="Include SUSPECT/FAILED photos (excluded by default, see --landmarks-status-csv).",
    )
    parser.add_argument("--non-strict", action="store_true", help="Tolerate malformed TPS blocks.")
    parser.add_argument(
        "--tps", type=Path, default=None, dest="landmarks_tps",
        help="Use this .tps file instead of root/landmarks/landmarks_numbered.tps "
             "(e.g. reference manual annotations). specimens.csv/manifest.csv are still "
             "used as-is for the join -- see utils.dataset.load_dataset.",
    )
    parser.add_argument(
        "--landmarks-status-csv", type=Path, default=None,
        help="Status CSV (tps_id+status) paired with --tps, for --exclude-outliers. "
             "Optional: without --tps, defaults to landmarks_numbered.csv; with --tps, "
             "no default status (exclusion skipped with a warning).",
    )
    parser.add_argument(
        "--run-label", type=str, default=None,
        help="Human-readable label appended to the run id (e.g. 'tancrede19lm'). "
             "Defaults to the --tps filename if given, otherwise omitted.",
    )


def resolve_split(args: argparse.Namespace, default_split: str = "train") -> str:
    """Resolve the effective --split value: explicit > 'all' if a custom --tps > default_split."""
    if args.split is not None:
        return args.split
    if getattr(args, "landmarks_tps", None) is not None:
        return "all"
    return default_split


def dataset_kwargs(args: argparse.Namespace, default_split: str = "train") -> dict:
    """Build the load_dataset() kwargs shared across scripts from a Namespace
    populated by add_dataset_args()."""
    return dict(
        split=resolve_split(args, default_split=default_split),
        devices=args.devices,
        species=args.species,
        castes=args.castes,
        exclude_outliers=args.exclude_outliers,
        strict=not args.non_strict,
        landmarks_tps=args.landmarks_tps,
        landmarks_status_csv=args.landmarks_status_csv,
    )
