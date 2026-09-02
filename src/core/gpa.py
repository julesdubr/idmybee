"""gpa.py
Generalized Procrustes Analysis (2D, no reflection).

Mirrors geomorph::gpagen(): similarity only (translation + isotropic
scale + rotation). Reflection is explicitly forbidden via det(R)=+1 (see
core.alignment.kabsch_umeyama) -- without that, half the specimens would
silently align as mirror images.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.alignment import kabsch_umeyama


def centroid_size(coords: np.ndarray) -> float:
    centered = coords - coords.mean(axis=0)
    return float(np.sqrt(np.sum(centered**2)))


def _center_and_scale(coords: np.ndarray) -> np.ndarray:
    centered = coords - coords.mean(axis=0)
    cs = centroid_size(coords)
    if cs == 0:
        raise ValueError("Degenerate specimen: landmarks coincide (centroid size = 0)")
    return centered / cs


@dataclass
class GPAResult:
    aligned: np.ndarray        # (n_specimens, n_points, 2)
    mean_shape: np.ndarray     # (n_points, 2) -- consensus
    centroid_sizes: np.ndarray  # (n_specimens,) -- raw sizes, pre-standardization
    n_iterations: int


def gpagen(landmarks: list[np.ndarray], max_iter: int = 100, tol: float = 1e-8) -> GPAResult:
    """landmarks: one (n_points, 2) array per specimen, same point count/order everywhere."""
    n_specimens = len(landmarks)
    if n_specimens == 0:
        raise ValueError("No specimens provided")
    n_points = landmarks[0].shape[0]
    for idx, lm in enumerate(landmarks):
        if lm.shape != (n_points, 2):
            raise ValueError(f"Specimen {idx}: shape {lm.shape}, expected ({n_points}, 2)")

    raw_cs = np.array([centroid_size(lm) for lm in landmarks])
    standardized = np.stack([_center_and_scale(lm) for lm in landmarks])

    reference = standardized[0].copy()
    aligned = standardized.copy()

    n_iterations = 0
    for n_iterations in range(1, max_iter + 1):
        for i in range(n_specimens):
            # scale=1: shapes are already normalized to centroid size 1 by
            # _center_and_scale, so only the rotation needs optimizing.
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
    """Equivalent of geomorph::two.d.array(): (n, p, 2) -> (n, p*2), columns x1,y1,x2,y2,..."""
    return aligned.reshape(aligned.shape[0], -1)


def align_to_reference(coords: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Align ONE specimen onto an already-standardized reference shape (e.g.
    a previous GPAResult's `mean_shape`), in a single pass (no consensus
    iteration). Used to project new specimens into the shape space of an
    already-trained model (see predict.py), where `gpagen()` would compute
    a new consensus shared by a group of specimens.

    `reference` must have the same shape (n_points, 2) as `coords`.
    """
    if coords.shape != reference.shape:
        raise ValueError(
            f"Shape mismatch with the reference: {coords.shape} vs {reference.shape} "
            "(different landmark count -- check that the new TPS's landmark scheme "
            "matches the one used to train the model)."
        )
    standardized = _center_and_scale(coords)
    R, _ = kabsch_umeyama(standardized, reference, estimate_scale=False)
    return standardized @ R.T


def procrustes_distance(aligned_coords: np.ndarray, reference: np.ndarray) -> float:
    """Procrustes distance (root sum of squared deviations) between an
    already-aligned specimen (via `align_to_reference`) and the reference.
    Used as a typicality score: a value well above what's observed on the
    training set flags an atypical shape or a landmark issue (wrong order,
    failed detection, etc.)."""
    return float(np.sqrt(np.sum((aligned_coords - reference) ** 2)))
