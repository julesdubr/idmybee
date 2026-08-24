"""variance.py
Primitives de variance sur des coordonnées GPA-alignées (n_specimens,
n_landmarks, 2) : à quel point les spécimens d'un même groupe se
ressemblent (within_group_variance), et à quel point les groupes eux-mêmes
diffèrent (between_group_variance). Toute la "distance" utilisée ici est la
distance de Procrustes au carré (somme des carrés des écarts de coordonnées
après superposition), cohérente avec ce que fait déjà la GPA en amont.

Ces deux fonctions sont le socle commun de :
- utils.reporting (tables par groupe/appareil de lda.py, within seulement),
- analysis.variance (décomposition ANOVA espèce x appareil complète,
  within + between + emboîtement + tests de permutation).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _flatten(aligned: np.ndarray) -> np.ndarray:
    return aligned.reshape(len(aligned), -1)


def within_group_variance(aligned: np.ndarray, groupe: pd.Series) -> pd.Series:
    """Variance de Procrustes par groupe : distance quadratique moyenne de
    chaque spécimen au centroïde de forme de son propre groupe."""
    X = _flatten(aligned)
    out = {}
    for g in groupe.unique():
        mask = (groupe == g).values
        # n = int(mask.sum())
        centroid = X[mask].mean(axis=0)
        out[g] = float(((X[mask] - centroid) ** 2).sum(axis=1).mean()) # (n - 1)
        # différence pour tous les centroids de tous les groupes => nb_dist
        # puis / par nb_dist - nb_g
    return pd.Series(out, name="shape_variance")


def between_group_variance(aligned: np.ndarray, groupe: pd.Series) -> float:
    """Variance de Procrustes ENTRE les groupes : distance quadratique
    moyenne (pondérée par la taille de chaque groupe) du centroïde de
    chaque groupe au centroïde global."""
    X = _flatten(aligned)
    grand_mean = X.mean(axis=0)
    total = 0.0

    n_groups = len(groupe.unique())

    for g in groupe.unique():
        mask = (groupe == g).values
        n = int(mask.sum())
        centroid = X[mask].mean(axis=0)
        total += n * float(((centroid - grand_mean) ** 2).sum())

    return total / (n_groups - 1)