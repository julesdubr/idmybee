"""train_dataset.py
Process ANY clean dataset (detection -> crop -> landmarks -> renumbering),
write its R-facing landmarks package to <dataset>/export/, then fit a
GPA-PCA-LDA model on it.

Dataset-agnostic: pass the dataset root as the positional argument -- there
is no default, and nothing here is specific to collection vs terrain vs a
future source. See tools/pipeline/predict_dataset.py to classify another dataset
with the model this produces.

Stages (each stage's own main(argv), in-process -- see
utils.landmarking_pipeline and PIPELINE.md):
    extraction.detect_wing            -> <dataset>/extraction/<mode>/detection.csv
    extraction.normalize_crop         -> <dataset>/extraction/<mode>/images/, crops.csv
    landmarks.predict                 -> <dataset>/landmarks/landmarks.{tps,csv}
    landmarks.renumber                -> <dataset>/landmarks/landmarks_numbered.{tps,csv}
    tools.pipeline.export_final_landmarks      -> <dataset>/export/
    classifiers.train                 -> models/lda/<run_id>/train/model.joblib

Usage:
    python -m tools.pipeline.train_dataset data/Bombus/collection \\
        --unet-model models/unet_landmarks/2026-08-29_131929/weights.pt

    python -m tools.pipeline.train_dataset data/Bombus/terrain \\
        --unet-model models/unet_landmarks/legacy_baseline/weights.pt \\
        --n-landmarks 18 --overwrite
"""
from __future__ import annotations

import argparse

from classifiers.train import main as train_main
from utils.cli import add_dataset_args, add_dataset_positional, add_logging_args, log_level_from_args
from utils.cli import verbosity_argv
from utils.landmarking_pipeline import add_landmarking_args, dataset_filter_argv, run_export, run_landmarking
from core.run_io import setup_console_logging



def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process a dataset, export its landmarks package, and fit an LDA model."
    )
    add_dataset_positional(parser, help="Clean dataset root (contains manifest.csv + biological_data.csv).")
    add_landmarking_args(parser)
    add_dataset_args(parser)
    parser.add_argument("--level", default="species", choices=["species", "caste"], help="classifiers.train --level.")
    parser.add_argument("--lda-components", type=int, default=2)
    parser.add_argument("--model-name", type=str, default=None, help="classifiers.train --model-name.")
    add_logging_args(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    setup_console_logging(log_level_from_args(args))

    run_landmarking(args)
    export_dir = run_export(args)

    print(f"\n=== LDA training (level={args.level}) ===")
    train_argv = [
        str(args.dataset), "--level", args.level,
        "--lda-components", str(args.lda_components), *verbosity_argv(args), *dataset_filter_argv(args),
    ]
    if args.model_name:
        train_argv += ["--model-name", args.model_name]
    train_main(train_argv)

    print(f"\nDone.")
    print(f"  Landmarks package -> {export_dir}")
    print(f"  LDA model         -> models/lda/<run_id>/train/model.joblib")


if __name__ == "__main__":
    main()
