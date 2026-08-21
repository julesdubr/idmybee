"""Sauvegarde et chargement d'un modèle GPA -> PCA -> LDA entraîné.

Un `TrainedModel` regroupe tout ce qu'il faut pour classer de nouveaux
spécimens sans réentraîner : la forme de référence GPA (`mean_shape`), le
PCA et le LDA ajustés sur le jeu d'entraînement complet (pas la version
LOOCV, qui ne sert qu'à estimer l'accuracy). Voir lda.py (produit le
modèle via --save-model) et predict.py (le consomme).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


@dataclass
class TrainedModel:
    mean_shape: np.ndarray               # (n_points, 2) -- référence GPA
    n_points: int
    pca: PCA
    lda: LinearDiscriminantAnalysis
    level: str                           # "espece" ou "caste" (colonne classée)
    classes: list[str] = field(default_factory=list)
    device: str | None = None            # filtre --device utilisé à l'entraînement, si any
    dataset: str | None = None           # filtre --dataset utilisé à l'entraînement, si any
    source_tps: str = ""
    n_train: int = 0


def save_model(model: TrainedModel, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Modèle -> {path}")


def load_model(path: str | Path) -> TrainedModel:
    model = joblib.load(Path(path))
    if not isinstance(model, TrainedModel):
        raise TypeError(f"{path} ne contient pas un TrainedModel valide")
    return model