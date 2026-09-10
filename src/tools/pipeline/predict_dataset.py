"""predict_dataset.py
Classifies a dataset already prepared through detection -> crop -> landmark
placement -> renumbering -> export (see app/setup_dataset.py / PIPELINE.md
"Orchestrator scripts") with an already-trained model (see
tools/pipeline/train_dataset.py) -- this tool never runs any of that
landmarking itself, it just points classifiers.predict at a dataset root
that's already ready, same as app/predict_dataset.py.

Dataset-agnostic: pass the dataset root as the positional argument -- there
is no default, and nothing here is specific to collection vs terrain vs a
future source. --model-name must name a model trained on a compatible
landmark scheme (a mismatch gives wrong predictions with no error --
classifiers.predict only checks the point count, not the order).

--model-name is resolved to models/lda/<name>/model.joblib by
core.run_io.find_model_path -- pass the same --model-name given to
tools/pipeline/train_dataset.py (or the run_id folder itself, e.g. for a
model with no --model-name at train time).

If biological_data.csv has little or no known species/caste, the printed
top-1/top-3 "accuracy" only reflects the labeled rows -- exploratory, not a
held-out test set.

Usage:
    python -m tools.pipeline.predict_dataset data/Bombus/terrain \\
        --model-name "Identification bourdons (collection)"
"""
from __future__ import annotations

import argparse

from classifiers.predict import run_batch
from utils.cli import add_dataset_args, add_dataset_positional, add_logging_args, log_level_from_args
from core.pipeline_io import dataset_export_dir
from core.run_io import FAMILY_LDA, find_model_path, setup_console_logging


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify an already-prepared, already-exported dataset with a trained model."
    )
    add_dataset_positional(parser, help="Clean dataset root, already prepared by app/setup_dataset.py "
                                         "(contains manifest.csv + biological_data.csv + landmarks/).")
    parser.add_argument("--model-name", required=True,
                         help="Trained model's --model-name (see tools/pipeline/train_dataset.py), "
                              "or its run_id folder name under models/lda/.")
    add_dataset_args(parser)
    parser.add_argument("--low-confidence-threshold", type=float, default=0.6,
                         help="classifiers.predict batch --low-confidence-threshold.")
    add_logging_args(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    setup_console_logging(log_level_from_args(args))

    if not (args.dataset / "manifest.csv").exists() or not (args.dataset / "biological_data.csv").exists():
        raise SystemExit(
            f"{args.dataset}: manifest.csv/biological_data.csv not found -- "
            "prepare this dataset with app/setup_dataset.py first."
        )
    export_dir = dataset_export_dir(args.dataset)
    if not export_dir.exists():
        print(f"Warning: {export_dir} not found -- run app/setup_dataset.py's export step on this dataset first.")

    args.model_path = find_model_path(args.model_name, family=FAMILY_LDA)
    print(f"\n=== Classification (model={args.model_name!r} -> {args.model_path}) ===")
    run_batch(args)


if __name__ == "__main__":
    main()
