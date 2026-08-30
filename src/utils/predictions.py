"""predictions.py
Schéma de prédiction commun à classifiers/train.py (LOOCV) et
classifiers/predict.py (batch) : mêmes colonnes, pour que
analysis/classification_report.py sache lire l'un ou l'autre sans
distinction.

Colonnes produites par build_predictions_df() :
    tps_id, image_id, specimen_id, image_path,
    predicted_<level>, confidence, second_choice, second_confidence,
    third_choice, third_confidence,
    [procrustes_distance]                 -- seulement si fourni (predict.py)
    [true_<level>, correct_top1, correct_top3]  -- seulement si truth fournie
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from utils.tps_io import ImageLandmarks

CHOICE_RANKS = ("second", "third")  # au-delà du top-1 (predicted/confidence)


def build_predictions_df(
    specimens: list[ImageLandmarks],
    level: str,
    predicted: np.ndarray,
    proba: np.ndarray,
    classes: np.ndarray,
    truth_by_tps_id: dict[int, str] | None = None,
    procrustes_distances: list[float] | None = None,
) -> pd.DataFrame:
    """Construit le DataFrame de prédictions à partir des sorties d'un LDA
    (classes/predict/predict_proba). `specimens` doit être dans le même
    ordre que les lignes de `predicted`/`proba`. Rapporte les 3 meilleurs
    choix (top-1/2/3)."""
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
    for rank, name in enumerate(CHOICE_RANKS, start=1):  # rank 1 = 2e choix, rank 2 = 3e choix
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
    """Top-1/top-3 accuracy à partir d'un DataFrame produit par
    build_predictions_df() avec truth_by_tps_id fourni. Lève si les colonnes
    correct_top1/correct_top3 sont absentes (pas de vérité connue)."""
    if "correct_top1" not in df.columns:
        raise ValueError("Aucune vérité connue dans ce DataFrame (truth_by_tps_id non fourni à la construction).")
    return {
        "n": int(len(df)),
        "accuracy_top1": float(df["correct_top1"].mean()),
        "accuracy_top3": float(df["correct_top3"].mean()),
    }


def print_predictions_report(df: pd.DataFrame, level: str, low_confidence_threshold: float = 0.6) -> None:
    """Résumé lisible en terminal : répartition des prédictions, confiance,
    et prédictions sous le seuil de confiance à vérifier manuellement."""
    print(f"\nRépartition des prédictions ({level}) :")
    print(df[f"predicted_{level}"].value_counts())
    print(
        f"\nConfiance moyenne : {df['confidence'].mean():.3f} "
        f"(min={df['confidence'].min():.3f}, max={df['confidence'].max():.3f})"
    )
    low_conf = df[df["confidence"] < low_confidence_threshold]
    if len(low_conf):
        cols = [c for c in ["tps_id", "specimen_id", "image_path", f"predicted_{level}", "confidence", "second_choice"]
                if c in low_conf.columns]
        print(
            f"\n{len(low_conf)} prédiction(s) sous le seuil de confiance "
            f"({low_confidence_threshold}) -- à vérifier manuellement :"
        )
        print(low_conf[cols].to_string(index=False))
