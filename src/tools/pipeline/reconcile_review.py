"""reconcile_review.py
Re-applies a (possibly hand-edited) review CSV written by
tools.pipeline.export_review -- or by app/build_dataset.py's validation
steps -- as an override consumed downstream with no other code change:

    <dataset>/review/crops_review.csv     -> <dataset>/extraction/<mode>/crops_reviewed.csv
                                              (landmarks.predict --crops-csv)
    <dataset>/review/landmarks_review.csv -> <dataset>/landmarks/landmarks_reviewed.csv
                                              (--landmarks-status-csv, see utils.cli.add_dataset_args)

Reconciles whichever of the two review CSVs exists under <dataset>/review/
-- pass --crops-only/--landmarks-only to restrict to one. See utils.review
for the shared logic (same functions the UI calls after an in-app edit).

Usage:
    python -m tools.pipeline.reconcile_review data/Bombus/collection
"""
from __future__ import annotations

import argparse
import logging

from utils.cli import add_dataset_positional, add_logging_args, log_level_from_args
from utils.review import read_review_csv, review_audit_path, write_crop_review, write_landmarks_review
from core.run_io import setup_console_logging

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dataset_positional(parser)
    parser.add_argument("--mode", default="light", choices=["heavy", "light"],
                         help="extraction/<mode>/crops.csv this reconciles against (default: light).")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--crops-only", action="store_true", help="Reconcile the crop review CSV only.")
    group.add_argument("--landmarks-only", action="store_true", help="Reconcile the landmark review CSV only.")
    add_logging_args(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    setup_console_logging(log_level_from_args(args))

    did_something = False

    crops_review_path = review_audit_path(args.dataset, "crops")
    if not args.landmarks_only and crops_review_path.exists():
        df = read_review_csv(crops_review_path)
        out_path, _audit = write_crop_review(args.dataset, args.mode, df)
        print(f"Crops reconciled     -> {out_path}")
        did_something = True

    landmarks_review_path = review_audit_path(args.dataset, "landmarks")
    if not args.crops_only and landmarks_review_path.exists():
        df = read_review_csv(landmarks_review_path)
        out_path, _audit = write_landmarks_review(args.dataset, df)
        print(f"Landmarks reconciled -> {out_path}")
        did_something = True

    if not did_something:
        raise SystemExit(
            f"No review CSV found under {args.dataset}/review/ -- run "
            "tools.pipeline.export_review first, edit its 'reviewed_status' column, then re-run this."
        )


if __name__ == "__main__":
    main()
