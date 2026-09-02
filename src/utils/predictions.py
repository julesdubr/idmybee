"""predictions.py
Prediction schema shared by classifiers/train.py (LOOCV) and
classifiers/predict.py (batch): same columns, so
analysis/classification_report.py can read either without distinction.

Columns produced by build_predictions_df():
    tps_id, image_id, specimen_id, image_path,
    predicted_<level>, confidence, second_choice, second_confidence,
    third_choice, third_confidence,
    [procrustes_distance]                 -- only if provided (predict.py)
    [true_<level>, correct_top1, correct_top3]  -- only if truth is provided
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from utils.tps_io import ImageLandmarks

CHOICE_RANKS = ("second", "third")  # beyond top-1 (predicted/confidence)


def build_predictions_df(
    specimens: list[ImageLandmarks],
    level: str,
    predicted: np.ndarray,
    proba: np.ndarray,
    classes: np.ndarray,
    truth_by_tps_id: dict[int, str] | None = None,
    procrustes_distances: list[float] | None = None,
) -> pd.DataFrame:
    """Builds the predictions DataFrame from an LDA's outputs
    (classes/predict/predict_proba). `specimens` must be in the same
    order as the rows of `predicted`/`proba`. Reports the top 3 choices
    (top-1/2/3)."""
    confidence = proba.max(axis=1)
    order = np.argsort(-proba, axis=1)

    columns = {
        "tps_id": [sp.tps_id for sp in specimens],
        "image_id": [sp.image_id for sp in specimens],
        "specimen_id": [sp.specimen_id for sp in specimens],
        "image_path": [sp.image_path for sp in specimens],
        f"predicted_{level}": predicted,
        "confidence": confidence,
    }
    for rank, name in enumerate(CHOICE_RANKS, start=1):  # rank 1 = 2nd choice, rank 2 = 3rd choice
        if len(classes) > rank:
            idx = order[:, rank]
            columns[f"{name}_choice"] = classes[idx]
            columns[f"{name}_confidence"] = proba[np.arange(len(specimens)), idx]
        else:
            columns[f"{name}_choice"] = [None] * len(specimens)
            columns[f"{name}_confidence"] = [None] * len(specimens)
    if procrustes_distances is not None:
        columns["procrustes_distance"] = procrustes_distances

    df = pd.DataFrame(columns)

    if truth_by_tps_id is not None:
        true_col = f"true_{level}"
        df[true_col] = df["tps_id"].map(truth_by_tps_id)
        choice_cols = [f"predicted_{level}"] + [f"{name}_choice" for name in CHOICE_RANKS]
        df["correct_top1"] = df[f"predicted_{level}"] == df[true_col]
        df["correct_top3"] = df.apply(
            lambda row: row[true_col] in {row[c] for c in choice_cols if pd.notna(row[c])}, axis=1
        )

    return df


def accuracy_summary(df: pd.DataFrame, level: str) -> dict:
    """Top-1/top-3 accuracy from a DataFrame produced by
    build_predictions_df() with truth_by_tps_id provided. Raises if the
    correct_top1/correct_top3 columns are missing (no known truth)."""
    if "correct_top1" not in df.columns:
        raise ValueError("No known truth in this DataFrame (truth_by_tps_id not provided at construction).")
    return {
        "n": int(len(df)),
        "accuracy_top1": float(df["correct_top1"].mean()),
        "accuracy_top3": float(df["correct_top3"].mean()),
    }


def print_predictions_report(df: pd.DataFrame, level: str, low_confidence_threshold: float = 0.6) -> None:
    """Terminal-readable summary: prediction breakdown, confidence, and
    predictions below the confidence threshold for manual review."""
    print(f"\nPrediction breakdown ({level}):")
    print(df[f"predicted_{level}"].value_counts())
    print(
        f"\nMean confidence: {df['confidence'].mean():.3f} "
        f"(min={df['confidence'].min():.3f}, max={df['confidence'].max():.3f})"
    )
    low_conf = df[df["confidence"] < low_confidence_threshold]
    if len(low_conf):
        cols = [c for c in ["tps_id", "specimen_id", "image_path", f"predicted_{level}", "confidence", "second_choice"]
                if c in low_conf.columns]
        print(
            f"\n{len(low_conf)} prediction(s) below the confidence threshold "
            f"({low_confidence_threshold}) -- for manual review:"
        )
        print(low_conf[cols].to_string(index=False))
