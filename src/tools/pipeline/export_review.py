"""export_review.py
Writes the two validation-review CSVs (crops and landmark placement) for a
dataset already through extraction.normalize_crop / landmarks.renumber --
the CLI counterpart of app/build_dataset.py's two validation steps, sharing
the exact same core logic (utils.review).

Each review CSV (<dataset>/review/{crops,landmarks}_review.csv) has a
`reviewed_status` column, initialized to the auto status, ready for manual
editing in a spreadsheet: change a row's `reviewed_status` to OK/SUSPECT/
FAILED, save, then run tools.pipeline.reconcile_review on the edited file to
apply the overrides.

--overlays additionally renders the landmark-placement overlays (numbered
points on the crop) sorted by status into <dataset>/review/overlays/<status>/
(see utils.tps_overlay.render_tps_overlays) -- there is no equivalent need
for the crop step: crops.csv's own output_path already points at a viewable
image, no overlay to draw yet at that stage.

Usage:
    python -m tools.pipeline.export_review data/Bombus/collection --overlays
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from utils.cli import add_dataset_positional, add_logging_args, log_level_from_args
from utils.review import build_crop_review_df, build_landmark_review_df, review_audit_path
from utils.tps_overlay import render_tps_overlays
from core.run_io import setup_console_logging

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dataset_positional(parser)
    parser.add_argument("--mode", default="light", choices=["heavy", "light"],
                         help="extraction/<mode>/crops.csv to read (default: light).")
    parser.add_argument("--overlays", action="store_true",
                         help="Also render numbered-landmark overlays, sorted by status, "
                              "into <dataset>/review/overlays/.")
    add_logging_args(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    setup_console_logging(log_level_from_args(args))

    crop_df = build_crop_review_df(args.dataset, args.mode)
    crop_path = review_audit_path(args.dataset, "crops")
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    crop_df.to_csv(crop_path, index=False)
    print(f"Crop review    -> {crop_path} ({len(crop_df)} photo(s))")

    landmark_df = build_landmark_review_df(args.dataset)
    landmark_path = review_audit_path(args.dataset, "landmarks")
    landmark_df.to_csv(landmark_path, index=False)
    print(f"Landmark review -> {landmark_path} ({len(landmark_df)} photo(s))")

    if args.overlays:
        overlays_dir = Path(args.dataset) / "review" / "overlays"
        summary = render_tps_overlays(
            Path(args.dataset) / "landmarks" / "landmarks_numbered.tps", overlays_dir,
            csv_path=Path(args.dataset) / "landmarks" / "landmarks_numbered.csv",
        )
        print(f"Overlays        -> {overlays_dir} ({summary['written']} written, sorted by status)")

    print(
        "\nEdit the 'reviewed_status' column (OK/SUSPECT/FAILED) by hand, then run:\n"
        f"  python -m tools.pipeline.reconcile_review {args.dataset} --mode {args.mode}"
    )


if __name__ == "__main__":
    main()
