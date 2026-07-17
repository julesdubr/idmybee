"""Generalized Procrustes Analysis (GPA) pour landmarks 2D.

Reproduit le comportement de geomorph::gpagen() : transformation de
similarité uniquement (translation + mise à l'échelle isotrope +
rotation), SANS réflexion. Chaque spécimen est centré, mis à l'échelle
sur une taille de centroïde unitaire, puis itérativement tourné sur la
forme moyenne (consensus) jusqu'à convergence.

La réflexion est explicitement interdite en imposant det(R) = +1 dans la
rotation de Kabsch par spécimen (cf. l'algorithme de Kabsch/Umeyama) :
c'est ce point précis qui, laissé de côté, produit des alignements
silencieusement en miroir sur la moitié des spécimens.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def centroid_size(coords: np.ndarray) -> float:
    """Racine de la somme des carrés des distances des landmarks à leur centroïde."""
    centered = coords - coords.mean(axis=0)
    return float(np.sqrt(np.sum(centered**2)))


def _center_and_scale(coords: np.ndarray) -> np.ndarray:
    centered = coords - coords.mean(axis=0)
    cs = centroid_size(coords)
    if cs == 0:
        raise ValueError("Spécimen dégénéré : tous les landmarks sont confondus (centroid size = 0)")
    return centered / cs


def _kabsch_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rotation 2x2 optimale (sans réflexion) alignant `source` sur `target`.

    `source` et `target` sont (n_points, 2), supposés déjà centrés en 0.
    Retourne R tel que `source @ R.T` est la configuration tournée.
    """
    H = source.T @ target  # (2, 2)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    if d == 0:
        d = 1.0
    R = Vt.T @ np.diag([1.0, d]) @ U.T
    return R


@dataclass
class GPAResult:
    aligned: np.ndarray  # (n_specimens, n_points, 2) -- coords procrustes-alignées
    mean_shape: np.ndarray  # (n_points, 2) -- forme consensus
    centroid_sizes: np.ndarray  # (n_specimens,) -- tailles brutes pré-standardisation
    n_iterations: int


def gpagen(landmarks: list[np.ndarray], max_iter: int = 100, tol: float = 1e-8) -> GPAResult:
    """Generalized Procrustes Analysis.

    Parameters
    ----------
    landmarks:
        Liste de tableaux (n_points, 2), un par spécimen. Tous doivent avoir
        le même nombre de points, dans le même ordre.
    max_iter, tol:
        Contrôle de la convergence sur la mise à jour de la forme moyenne.

    Returns
    -------
    GPAResult : coordonnées alignées (taille de centroïde unitaire, rotation
    superposée), forme moyenne convergée, et taille de centroïde brute de
    chaque spécimen en entrée (utile comme covariable de taille ou métrique
    de QC, équivalent de gpagen()$Csize en R).
    """
    n_specimens = len(landmarks)
    if n_specimens == 0:
        raise ValueError("Aucun spécimen fourni")
    n_points = landmarks[0].shape[0]
    for idx, lm in enumerate(landmarks):
        if lm.shape != (n_points, 2):
            raise ValueError(
                f"Le spécimen {idx} a une forme {lm.shape}, ({n_points}, 2) attendue. "
                "Tous les spécimens doivent avoir le même nombre/ordre de landmarks pour la GPA."
            )

    raw_cs = np.array([centroid_size(lm) for lm in landmarks])
    standardized = np.stack([_center_and_scale(lm) for lm in landmarks])  # (n, p, 2)

    reference = standardized[0].copy()
    aligned = standardized.copy()

    n_iterations = 0
    for n_iterations in range(1, max_iter + 1):
        for i in range(n_specimens):
            R = _kabsch_rotation(standardized[i], reference)
            aligned[i] = standardized[i] @ R.T

        new_mean = aligned.mean(axis=0)
        new_mean = new_mean / np.sqrt(np.sum(new_mean**2))  # forme consensus remise à taille unitaire

        shift = float(np.sqrt(np.sum((new_mean - reference) ** 2)))
        reference = new_mean
        if shift < tol:
            break

    return GPAResult(
        aligned=aligned,
        mean_shape=reference,
        centroid_sizes=raw_cs,
        n_iterations=n_iterations,
    )


def two_d_array(aligned: np.ndarray) -> np.ndarray:
    """Équivalent de geomorph::two.d.array().

    (n_specimens, n_points, 2) -> (n_specimens, n_points * 2), colonnes
    ordonnées x1, y1, x2, y2, ..., xp, yp.
    """
    n_specimens = aligned.shape[0]
    return aligned.reshape(n_specimens, -1)