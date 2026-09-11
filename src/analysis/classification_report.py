"""classification_report.py
Figures and tables for a classification run (train or predict): confusion
matrix, per-species table, per-device table, GPA alignment, LDA projection.
Reads outputs already written by train.py/predict.py (predictions.csv,
params.json) and reloads the model to project specimens into its space --
refits nothing.

Shape dispersion (shape_variance) stays exclusively in
analysis/variance_report.py.

Usage:
    python -m analysis.classification_report species_train_P1-S1 --step train
    python -m analysis.classification_report species_train_P1-S1 --step predict --eval-tag test
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report as sk_classification_report
from sklearn.metrics import confusion_matrix

from utils.cli import add_logging_args, log_level_from_args
from core.dataset import load_dataset, target_groupe
from core.gpa import align_to_reference, two_d_array
from core.model_io import load_model
from core.run_io import FAMILY_LDA, RUNS_ROOT, read_params, result_path, run_path, setup_console_logging, write_params

logger = logging.getLogger(__name__)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda v: f"{v:.4f}")


def confusion_matrix_df(df: pd.DataFrame, level: str) -> pd.DataFrame:
    truth, predicted = df[f"true_{level}"], df[f"predicted_{level}"]
    labels = sorted(truth.unique())
    cm = confusion_matrix(truth, predicted, labels=labels)
    return pd.DataFrame(cm, index=labels, columns=labels)


def species_report_df(df: pd.DataFrame, level: str) -> pd.DataFrame:
    """Precision/recall/f1 per class (species or species_caste)."""
    truth, predicted = df[f"true_{level}"], df[f"predicted_{level}"]
    labels = sorted(truth.unique())
    report = sk_classification_report(truth, predicted, labels=labels, output_dict=True, zero_division=0)
    table = pd.DataFrame(report).T.loc[labels, ["precision", "recall", "f1-score"]]
    table.columns = ["precision", "recall", "f1"]
    table.insert(0, "n", truth.value_counts().reindex(labels).astype(int))
    table.index.name = level
    return table


def device_report_df(merged: pd.DataFrame, level: str) -> pd.DataFrame | None:
    """Accuracy per device. None if the 'device' column is absent or has
    only a single value (nothing to compare)."""
    if "device" not in merged.columns or merged["device"].dropna().nunique() <= 1:
        return None
    known = merged["device"].notna()
    sub = merged[known]
    correct = (sub[f"predicted_{level}"] == sub[f"true_{level}"]).values
    rows = []
    for d in sorted(sub["device"].dropna().unique()):
        mask = (sub["device"] == d).values
        rows.append({"device": d, "n": int(mask.sum()), "accuracy": float(correct[mask].mean())})
    return pd.DataFrame(rows).set_index("device")


def plot_confusion_matrix(cm_df: pd.DataFrame, out_path: Path, title: str) -> None:
    n = len(cm_df)
    fig, ax = plt.subplots(figsize=(max(6.0, 0.6 * n), max(5.0, 0.55 * n)))
    im = ax.imshow(cm_df.values, cmap="Blues")

    ax.set_xticks(range(len(cm_df.columns)))
    ax.set_xticklabels(cm_df.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(cm_df.index)))
    ax.set_yticklabels(cm_df.index)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title(title)

    values = cm_df.values
    thresh = values.max() / 2 if values.max() > 0 else 0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if v:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=8,
                        color="white" if v > thresh else "black")

    fig.colorbar(im, ax=ax, shrink=0.8, label="n specimens")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Confusion matrix -> {out_path}")


def plot_gpa_alignment(aligned: np.ndarray, groupe: pd.Series, out_path: Path, title: str) -> None:
    """Displays shapes (already aligned to the model's reference) in 2D, colored by group."""
    if aligned.ndim != 3 or aligned.shape[2] != 2:
        raise ValueError("aligned must be an array of shape (n_specimens, n_points, 2).")

    unique_groups = sorted(groupe.unique())
    cmap = plt.get_cmap("tab20")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    for i, group in enumerate(unique_groups):
        mask = (groupe == group).values
        specimens_coords = aligned[mask]
        if specimens_coords.size == 0:
            continue
        color = cmap(i % 20)

        group_coords = np.vstack(specimens_coords)
        ax1.scatter(group_coords[:, 0], group_coords[:, 1], color=color, alpha=0.05, label=group, s=10)
        centroids = specimens_coords.mean(axis=0)
        ax2.scatter(centroids[:, 0], centroids[:, 1], color=color, s=20, label=group, marker="x")

    for ax, subtitle in ((ax1, "all"), (ax2, "centroids")):
        ax.set_xlabel("X coordinate")
        ax.set_ylabel("Y coordinate")
        ax.set_title(f"{title} ({subtitle})")
        ax.legend(title="Group", loc="best", fontsize="small")
        ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"GPA alignment -> {out_path}")


def plot_lda(lda_scores: np.ndarray, groupe: pd.Series, out_path: Path, title: str) -> None:
    if lda_scores.shape[1] == 1:
        x, y = lda_scores[:, 0], np.zeros(lda_scores.shape[0])
    else:
        x, y = lda_scores[:, 0], lda_scores[:, 1]

    plt.figure(figsize=(10, 8))
    unique_groups = sorted(groupe.unique())
    if len(unique_groups) > 20:
        cmap_combined = np.vstack([plt.cm.tab20c(np.linspace(0, 1, 20)), plt.cm.tab20b(np.linspace(0, 1, 20))])
        colors = cmap_combined[: len(unique_groups)]
    else:
        colors = [plt.cm.tab20(i) for i in range(len(unique_groups))]

    for i, group in enumerate(unique_groups):
        mask = (groupe == group).values
        plt.scatter(x[mask], y[mask], label=group, color=colors[i], alpha=0.25, s=20)

    plt.xlabel("LDA 1")
    plt.ylabel("LDA 2" if lda_scores.shape[1] > 1 else "Constant")
    plt.title(title)
    plt.legend(title="Group", loc="best", fontsize="small")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"LDA projection -> {out_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Figures and tables for a classification run (see module docstring)"
    )
    parser.add_argument("run_id", type=str, help="Run identifier (see core.run_io.build_run_id)")
    parser.add_argument("--step", type=str, choices=["train", "predict"], default="train",
                         help="Which predictions to analyze: LOOCV's (train) or a predict.py batch's (predict).")
    parser.add_argument("--eval-tag", type=str, default=None,
                         help="Required if --step predict -- see runs/<family>/<run_id>/predict/ for the list.")
    parser.add_argument("--family", type=str, default=FAMILY_LDA)
    add_logging_args(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    setup_console_logging(log_level_from_args(args))

    if args.step == "predict" and not args.eval_tag:
        available = result_path(args.family, args.run_id, "predict", root=RUNS_ROOT)
        options = sorted(p.name for p in available.iterdir()) if available.exists() else []
        raise SystemExit(f"--eval-tag required with --step predict. Available for {args.run_id!r}: {options}")

    model_path = result_path(args.family, args.run_id) / "model.joblib"
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path} (was train.py run with --no-save-model?)")
    model = load_model(model_path)

    # model.dataset_label is the exact dataset_name train.py nested its
    # performance record under (runs/<family>/<run_id>/train/<dataset_name>/).
    dataset_tag = args.eval_tag if args.step == "predict" else model.dataset_label
    step_path = result_path(args.family, args.run_id, args.step, dataset_tag, root=RUNS_ROOT)
    if not step_path.exists():
        raise SystemExit(
            f"{step_path} not found -- run classifiers.train or classifiers.predict batch first "
            f"for run_id={args.run_id!r}."
        )
    params = read_params(step_path)

    pred_csv = step_path / ("loocv_predictions.csv" if args.step == "train" else "predictions.csv")
    df = pd.read_csv(pred_csv)
    true_cols = [c for c in df.columns if c.startswith("true_")]
    if not true_cols:
        raise SystemExit(
            f"{pred_csv} has no 'true_<level>' column -- no known truth, no report possible "
            "(e.g. predict.py single outputs, which have no ground truth)."
        )
    level = true_cols[0].removeprefix("true_")

    dataset_root = Path(params["dataset"])
    specimens, meta_df = load_dataset(
        dataset_root,
        devices=params.get("devices"),
        species=params.get("species"),
        castes=params.get("castes"),
        exclude_outliers=bool(params.get("exclude_outliers", False)),
        strict=not bool(params.get("non_strict", False)),
        landmarks_tps=params.get("landmarks_tps"),
        landmarks_status_csv=params.get("landmarks_status_csv"),
        labeled_only=True,
    )
    meta_df = meta_df.copy()
    meta_df["tps_id"] = [sp.tps_id for sp in specimens]

    run_label = f"{args.run_id}/{args.step}/{dataset_tag}"
    out_dir = run_path(args.family, args.run_id, args.step, dataset_tag, "report", root=RUNS_ROOT)

    cm_df = confusion_matrix_df(df, level)
    cm_df.to_csv(out_dir / "confusion_matrix.csv")
    plot_confusion_matrix(cm_df, out_dir / "confusion_matrix.png", title=f"Confusion matrix -- {run_label}")
    print("\nConfusion matrix:\n" + str(cm_df))

    species_df = species_report_df(df, level)
    species_df.to_csv(out_dir / "species_report.csv")
    print(f"\nTable by {level} -> {out_dir / 'species_report.csv'}\n" + str(species_df))

    merged = df.merge(meta_df[["tps_id", "device", "device_tag"]], on="tps_id", how="left")
    device_df = device_report_df(merged, level)
    if device_df is not None:
        device_df.to_csv(out_dir / "device_report.csv")
        print(f"\nTable by device -> {out_dir / 'device_report.csv'}\n" + str(device_df))

    pred_tps_ids = set(df["tps_id"])
    keep_mask = [sp.tps_id in pred_tps_ids for sp in specimens]
    plot_specimens = [sp for sp, k in zip(specimens, keep_mask) if k]
    plot_meta = meta_df[keep_mask].reset_index(drop=True)
    if plot_specimens:
        groupe = target_groupe(plot_meta, level)
        aligned = np.stack([align_to_reference(sp.landmarks, model.mean_shape) for sp in plot_specimens])
        plot_gpa_alignment(aligned, groupe, out_dir / "gpa_alignment.png", title=f"GPA alignment -- {run_label}")

        scores = model.pca.transform(two_d_array(aligned))
        lda_scores = model.lda.transform(scores)
        plot_lda(lda_scores, groupe, out_dir / "lda_projection.png", title=f"LDA projection -- {run_label}")

    write_params(out_dir, args, extra={"run_id": args.run_id, "eval_tag": args.eval_tag, "family": args.family, "step": args.step})
    print(f"\nRun -> {out_dir}")


if __name__ == "__main__":
    main()
