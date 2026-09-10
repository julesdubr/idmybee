"""train_dataset.py
Fits a GPA -> PCA -> LDA model (classifiers.train) on a dataset already
prepared through detection -> crop -> landmark placement -> renumbering ->
export (see app/setup_dataset.py / PIPELINE.md "Orchestrator scripts") --
this tool never runs any of that landmarking itself, it just points
classifiers.train at a dataset root that's already ready, same as
app/train_model.py.

Dataset-agnostic: pass the dataset root as the positional argument -- there
is no default, and nothing here is specific to collection vs terrain vs a
future source. See tools/pipeline/predict_dataset.py to classify another
dataset with the model this produces.

--model-name is optional: omit it and classifiers.train derives a
deterministic name from level/dataset/devices/landmarks-source instead --
either way the model is saved automatically to
models/lda/<name>/model.joblib (see core.run_io.resolve_model_slug), with
its LOOCV performance record under runs/lda/<name>/train/.

Usage:
    python -m tools.pipeline.train_dataset data/Bombus/collection \\
        --model-name "Identification bourdons (collection)"
"""
from __future__ import annotations

import argparse

from classifiers.train import main as train_main
from utils.cli import add_dataset_args, add_dataset_positional, add_logging_args, log_level_from_args
from utils.cli import verbosity_argv
from utils.landmarking_pipeline import dataset_filter_argv
from core.pipeline_io import dataset_export_dir
from core.run_io import setup_console_logging


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a GPA-PCA-LDA model on an already-prepared, already-exported dataset."
    )
    add_dataset_positional(parser, help="Clean dataset root, already prepared by app/setup_dataset.py "
                                         "(contains manifest.csv + biological_data.csv + landmarks/).")
    add_dataset_args(parser)
    parser.add_argument("--level", default="species", choices=["species", "caste"], help="classifiers.train --level.")
    parser.add_argument("--lda-components", type=int, default=2)
    parser.add_argument("--model-name", type=str, default=None, help="classifiers.train --model-name.")
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

    train_argv = [
        str(args.dataset), "--level", args.level,
        "--lda-components", str(args.lda_components), *verbosity_argv(args), *dataset_filter_argv(args),
    ]
    if args.model_name:
        train_argv += ["--model-name", args.model_name]
    result = train_main(train_argv)

    print(f"\nDone.")
    print(f"  LDA model -> {result.model_path}")
    print(f"  Performance record -> {result.runs_dir}")


if __name__ == "__main__":
    main()
