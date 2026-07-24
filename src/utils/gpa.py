"""gpa.py
Generalized Procrustes Analysis (2D, sans réflexion).

Reproduit geomorph::gpagen() : similarité uniquement (translation +
échelle isotrope + rotation). La réflexion est explicitement interdite
via det(R)=+1 (voir utils.alignment.kabsch_umeyama) -- sans ça, la moitié
des spécimens s'alignent silencieusement en miroir.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from utils.alignment import kabsch_umeyama


def centroid_size(coords: np.ndarray) -> float:
    centered = coords - coords.mean(axis=0)
    return float(np.sqrt(np.sum(centered**2)))


def _center_and_scale(coords: np.ndarray) -> np.ndarray:
    centered = coords - coords.mean(axis=0)
    cs = centroid_size(coords)
    if cs == 0:
        raise ValueError("Spécimen dégénéré : landmarks confondus (centroid size = 0)")
    return centered / cs


@dataclass
class GPAResult:
    aligned: np.ndarray        # (n_specimens, n_points, 2)
    mean_shape: np.ndarray     # (n_points, 2) -- consensus
    centroid_sizes: np.ndarray  # (n_specimens,) -- tailles brutes pré-standardisation
    n_iterations: int


def gpagen(landmarks: list[np.ndarray], max_iter: int = 100, tol: float = 1e-8) -> GPAResult:
    """landmarks: un (n_points, 2) par spécimen, même nombre/ordre de points partout."""
    n_specimens = len(landmarks)
    if n_specimens == 0:
        raise ValueError("Aucun spécimen fourni")
    n_points = landmarks[0].shape[0]
    for idx, lm in enumerate(landmarks):
        if lm.shape != (n_points, 2):
            raise ValueError(f"Spécimen {idx}: forme {lm.shape}, ({n_points}, 2) attendue")

    raw_cs = np.array([centroid_size(lm) for lm in landmarks])
    standardized = np.stack([_center_and_scale(lm) for lm in landmarks])

    reference = standardized[0].copy()
    aligned = standardized.copy()

    n_iterations = 0
    for n_iterations in range(1, max_iter + 1):
        for i in range(n_specimens):
            # scale=1 : les formes sont déjà normalisées à centroid size 1
            # par _center_and_scale, seule la rotation reste à optimiser.
            R, _ = kabsch_umeyama(standardized[i], reference, estimate_scale=False)
            aligned[i] = standardized[i] @ R.T

        new_mean = aligned.mean(axis=0)
        new_mean = new_mean / np.sqrt(np.sum(new_mean**2))

        shift = float(np.sqrt(np.sum((new_mean - reference) ** 2)))
        reference = new_mean
        if shift < tol:
            break

    return GPAResult(aligned, reference, raw_cs, n_iterations)


def two_d_array(aligned: np.ndarray) -> np.ndarray:
    """Équivalent geomorph::two.d.array() : (n, p, 2) -> (n, p*2), colonnes x1,y1,x2,y2,..."""
    return aligned.reshape(aligned.shape[0], -1)


def align_to_reference(coords: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Aligne UN spécimen sur une forme de référence déjà standardisée (ex: le
    `mean_shape` d'un GPAResult précédent), en un seul passage (pas d'itération
    de consensus). Utilisé pour projeter de nouveaux spécimens dans l'espace
    de forme d'un modèle déjà entraîné (voir predict.py), là où `gpagen()`
    calculerait un nouveau consensus commun à un groupe de spécimens.

    `reference` doit avoir la même forme (n_points, 2) que `coords`.
    """
    if coords.shape != reference.shape:
        raise ValueError(
            f"Forme incompatible avec la référence : {coords.shape} vs {reference.shape} "
            "(nombre de landmarks différent -- vérifier que le schéma de landmarks du "
            "nouveau TPS correspond bien à celui utilisé pour entraîner le modèle)."
        )
    standardized = _center_and_scale(coords)
    R, _ = kabsch_umeyama(standardized, reference, estimate_scale=False)
    return standardized @ R.T


def procrustes_distance(aligned_coords: np.ndarray, reference: np.ndarray) -> float:
    """Distance de Procrustes (racine de la somme des carrés des écarts) entre
    un spécimen déjà aligné (via `align_to_reference`) et la référence.
    Sert de score de typicité : une valeur nettement supérieure à ce qui est
    observé sur le jeu d'entraînement signale une forme atypique ou un
    problème de landmarks (mauvais ordre, détection ratée, etc.)."""
    return float(np.sqrt(np.sum((aligned_coords - reference) ** 2)))