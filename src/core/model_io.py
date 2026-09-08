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
    dataset_label: str = ""              # dataset root's name used at training time (e.g. "collection")
    devices: list[str] | None = None     # --devices used at training time (e.g. ["P1","S1"]), if any
    source_tps: str = ""
    n_train: int = 0
    model_name: str = ""                 # human-facing name (--model-name), e.g. "Red-rumped bumblebee identifier".
                                          # Purely descriptive: never used to build run_id/the output path (see
                                          # core.run_io.build_run_id), which stays derived from
                                          # level/dataset_label/devices for reproducibility. Falls back to run_id
                                          # for display wherever unset (see display_name()).

    def display_name(self, run_id: str = "") -> str:
        """model_name if one was given at training time, otherwise run_id
        (or "" if neither is known) -- what a model picker (CLI or UI)
        should show instead of a bare, abstract run_id."""
        return self.model_name or run_id


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
