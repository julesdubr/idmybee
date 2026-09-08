"""predict_dataset.py
Process ANY clean dataset (detection -> crop -> landmarks -> renumbering),
write its R-facing landmarks package to <dataset>/export/, then classify
every specimen with an already-trained model (see tools/train_dataset.py).

Dataset-agnostic: pass the dataset root as the positional argument -- there
is no default, and nothing here is specific to collection vs terrain vs a
future source. --unet-model/--n-landmarks/--reference must match the ones
used to train --model (a mismatched landmark scheme gives wrong predictions
with no error -- classifiers.predict only checks the point count, not the
order).

Stages (each stage's own main(argv), in-process -- see
utils.landmarking_pipeline and PIPELINE.md):
    extraction.detect_wing            -> <dataset>/extraction/<mode>/detection.csv
    extraction.normalize_crop         -> <dataset>/extraction/<mode>/images/, crops.csv
    landmarks.predict                 -> <dataset>/landmarks/landmarks.{tps,csv}
    landmarks.renumber                -> <dataset>/landmarks/landmarks_numbered.{tps,csv}
    tools.export_final_landmarks      -> <dataset>/export/
    classifiers.predict batch         -> data/models/lda/<run_id>/predict/<eval_tag>/predictions.csv

If biological_data.csv has little or no known species/caste, the printed
top-1/top-3 "accuracy" only reflects the labeled rows -- exploratory, not a
held-out test set.

Usage:
    python -m tools.predict_dataset data/Bombus/terrain \\
        --model data/models/lda/species_collection/train/model.joblib \\
        --unet-model data/models/unet_landmarks/2026-08-29_131929/weights.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

from classifiers.predict import run_batch
from utils.cli import add_dataset_args, add_dataset_positional, add_logging_args, log_level_from_args
from utils.landmarking_pipeline import add_landmarking_args, run_export, run_landmarking
from utils.run_io import setup_console_logging


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process a dataset, export its landmarks package, and classify it with a trained model."
    )
    add_dataset_positional(parser, help="Clean dataset root (contains manifest.csv + biological_data.csv).")
    parser.add_argument("--model", dest="model_path", type=Path, required=True,
                         help="Trained model (see tools/train_dataset.py).")
    add_landmarking_args(parser)
    add_dataset_args(parser)
    parser.add_argument("--low-confidence-threshold", type=float, default=0.6,
                         help="classifiers.predict batch --low-confidence-threshold.")
    add_logging_args(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    setup_console_logging(log_level_from_args(args))

    run_landmarking(args)
    export_dir = run_export(args)

    print(f"\n=== Classification (model={args.model_path}) ===")
    run_batch(args)

    print(f"\nLandmarks package -> {export_dir}")


if __name__ == "__main__":
    main()
