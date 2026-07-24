"""hungarian_umeyama.py
Numérotation par assignation hongroise + Umeyama (similarité sans réflexion),
alternées jusqu'à convergence, avec multi-départs (rotation grossière x
miroir) pour éviter les optima locaux.

Anciennement register.py. Renommé pour cohabiter avec d'autres méthodes
(ex: graph_matching.py à venir) sous un nom qui décrit l'approche plutôt
que le rôle générique ; implémente le contrat numbering.base.

La rotation+réflexion est déléguée à utils.alignment.kabsch_umeyama (le même
cœur SVD que la GPA), pour ne plus réimplémenter cette algèbre séparément.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from numbering.base import NumberingResult
from utils.alignment import kabsch_umeyama


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Similarité complète (R, scale, t), sans réflexion : dst_hat = scale*(src@R.T)+t."""
    mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
    R, scale = kabsch_umeyama(src - mu_src, dst - mu_dst, estimate_scale=True)
    t = mu_dst - scale * (R @ mu_src)
    return R, scale, t


def apply_transform(pts: np.ndarray, R: np.ndarray, scale: float, t: np.ndarray) -> np.ndarray:
    return scale * (pts @ R.T) + t


def _register_from_start(pts0: np.ndarray, cur_init: np.ndarray, zones: np.ndarray, n_iter: int):
    """Assignation+Umeyama alternés jusqu'à convergence.

    `pts0` : nuage centré original -- le transform est TOUJOURS recalculé
    depuis celui-ci, jamais depuis `cur` de l'itération précédente (sinon les
    rotations/miroirs initiaux se composeraient au lieu de ne servir qu'à
    orienter la toute première assignation).
    `cur_init` : pts0 après rotation/miroir initial, sert uniquement à ce
    premier calcul de coût, pour éviter un optimum local de l'assignation.
    """
    cur = cur_init
    assign_prev = None
    transform = None
    assign = None
    for _ in range(n_iter):
        cost = ((cur[:, None, :] - zones[None, :, :]) ** 2).sum(axis=2)
        row_ind, col_ind = linear_sum_assignment(cost)
        assign = col_ind[np.argsort(row_ind)]
        if assign_prev is not None and np.array_equal(assign, assign_prev):
            break
        assign_prev = assign
        transform = umeyama(pts0, zones[assign])
        cur = apply_transform(pts0, *transform)
    final_cost = float(((cur - zones[assign]) ** 2).sum(axis=1).mean())
    return assign, final_cost, transform


def numerate(
    landmarks: np.ndarray,
    reference: np.ndarray,
    n_iter: int = 15,
    angle_inits: tuple[float, ...] = (0, 90, 180, 270),
    mirror_options: tuple[bool, ...] = (False, True),
) -> NumberingResult:
    """Implémente le contrat numbering.base : renumérote `landmarks` (k, 2)
    selon l'ordre de `reference` (n_zones, 2).

    Ne gère pour l'instant que le cas k == n_zones (nombre de landmarks
    détectés = nombre attendu) ; le cas k != n_zones (points en trop/en
    moins) reste un FAILED explicite plutôt qu'une assignation rectangulaire
    non testée -- à traiter séparément si le détecteur en a besoin un jour.
    """
    if landmarks.shape[0] != reference.shape[0]:
        return NumberingResult(
            numbered=landmarks,
            status="FAILED",
            score=float("inf"),
            reason=(
                f"{landmarks.shape[0]} landmarks détectés, "
                f"{reference.shape[0]} attendus dans le template"
            ),
        )

    pts0 = landmarks - landmarks.mean(axis=0)
    best = None
    for angle in angle_inits:
        th = np.deg2rad(angle)
        rot0 = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        for mirror in mirror_options:
            mir = np.array([[-1, 0], [0, 1]]) if mirror else np.eye(2)
            cur_init = pts0 @ (rot0 @ mir).T
            assign, cost, transform = _register_from_start(pts0, cur_init, zones=reference, n_iter=n_iter)
            if best is None or cost < best[1]:
                best = (assign, cost, transform)

    assign, cost, _ = best
    inv = np.empty(len(reference), dtype=int)
    inv[assign] = np.arange(len(assign))  # slot j (zone j) <- point assigné à j
    numbered = landmarks[inv]
    return NumberingResult(numbered=numbered, status="OK", score=cost)