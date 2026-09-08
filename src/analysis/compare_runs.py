"""compare_runs.py
Compares the accuracy of several runs (train.py or predict.py batch) side
by side, from their metrics.json (e.g. which landmark source classifies
best).

Each entry is a path relative to data/models/<family>/ down to the folder
containing metrics.json: "<run_id>/train" for a LOOCV run, or
"<run_id>/predict/<eval_tag>" for an evaluation -- both can be mixed in the
same comparison.

Usage:
    python -m classifiers.train data/Bombus --level species --tps .../tancrede_19lm.tps --run-label tancrede19lm
    python -m classifiers.train data/Bombus --level species --tps .../tancrede_18lm.tps --run-label tancrede18lm
    python -m classifiers.train data/Bombus --level species --tps .../auto_19lm.tps --run-label auto19lm
    python -m classifiers.train data/Bombus --level species --tps .../auto_18lm.tps --run-label auto18lm

    python -m analysis.compare_runs \\
        species_all_tancrede19lm/train species_all_tancrede18lm/train \\
        species_all_auto19lm/train species_all_auto18lm/train \\
        --label landmarks_source_comparison
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from utils.cli import add_logging_args, log_level_from_args
from core.run_io import ANALYSIS_ROOT, FAMILY_LDA, MODELS_ROOT, read_metrics, setup_console_logging, slugify

logger = logging.getLogger(__name__)


def load_comparison_table(steps: list[str], family: str) -> pd.DataFrame:
    rows = []
    for step in steps:
        step_path = MODELS_ROOT / family / step
        metrics_path = step_path / "metrics.json"
        if not metrics_path.exists():
            raise SystemExit(f"{metrics_path} not found.")
        metrics = read_metrics(step_path)
        metrics["run"] = step
        rows.append(metrics)
    df = pd.DataFrame(rows).set_index("run")
    preferred = ["landmarks_source", "n", "n_points", "accuracy_top1", "accuracy_top3"]
    ordered = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return df[ordered]


def plot_accuracy_comparison(df: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(max(6.0, 1.4 * len(df)), 5.5))
    x = range(len(df))
    width = 0.35
    if "accuracy_top1" in df.columns:
        ax.bar([i - width / 2 for i in x], df["accuracy_top1"], width, label="top-1", color="steelblue")
    if "accuracy_top3" in df.columns:
        ax.bar([i + width / 2 for i in x], df["accuracy_top3"], width, label="top-3", color="lightsteelblue")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df.index, rotation=30, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Comparison plot -> {out_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare the accuracy of several train.py/predict.py runs side by side")
    parser.add_argument("steps", type=str, nargs="+",
                         help="Paths relative to data/models/<family>/, e.g. species_train_P1-S1/train "
                              "or species_train_P1-S1/predict/test")
    parser.add_argument("--label", type=str, default=None, help="Output folder name (default: derived from the paths)")
    parser.add_argument("--family", type=str, default=FAMILY_LDA)
    add_logging_args(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    setup_console_logging(log_level_from_args(args))

    table = load_comparison_table(args.steps, args.family)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print(table)

    label = args.label or slugify("-vs-".join(args.steps))[:120]
    out_dir = ANALYSIS_ROOT / "compare" / label
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "comparison.csv"
    table.to_csv(csv_path)
    print(f"\nTable -> {csv_path}")

    plot_accuracy_comparison(table, out_dir / "comparison.png", title=f"Comparison -- {label}")
    print(f"Run -> {out_dir}")


if __name__ == "__main__":
    main()
