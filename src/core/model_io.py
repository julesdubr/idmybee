"""Save and load a trained GPA -> PCA -> LDA model.

A `TrainedModel` bundles everything needed to classify new specimens
without retraining: the GPA reference shape (`mean_shape`), the PCA and
LDA fitted on the full training set (not the LOOCV version, which is only
used to estimate accuracy). See train.py (produces the model via
--save-model) and predict.py (consumes it).
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
    mean_shape: np.ndarray               # (n_points, 2) -- GPA reference
    n_points: int
    pca: PCA
    lda: LinearDiscriminantAnalysis
    level: str                           # "species" or "caste" (classified column)
    classes: list[str] = field(default_factory=list)
    split: str | None = None             # --split used at training time (e.g. "train"), if any
    devices: list[str] | None = None     # --devices used at training time (e.g. ["P1","S1"]), if any
    source_tps: str = ""
    n_train: int = 0


def save_model(model: TrainedModel, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Model -> {path}")


def load_model(path: str | Path) -> TrainedModel:
    model = joblib.load(Path(path))
    if not isinstance(model, TrainedModel):
        raise TypeError(f"{path} does not contain a valid TrainedModel")
    return model
