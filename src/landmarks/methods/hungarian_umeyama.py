"""hungarian_umeyama.py
Numbering via Hungarian assignment + Umeyama (similarity, no reflection),
alternated until convergence, with multi-start (coarse rotation x mirror)
to avoid local optima.

Formerly numbering/hungarian_umeyama.py (register.py before that). Moved
under landmarks/methods/ to sit alongside other methods (e.g.
graph_matching.py, tried then dropped -- worse on this dataset) under a
name that describes the approach rather than the generic role; implements
the landmarks.methods.base contract.

Rotation+scale is delegated to core.alignment.kabsch_umeyama (the same
SVD core as GPA), so this algebra isn't reimplemented separately.

Gist in one sentence: we don't know in advance which detected landmark
corresponds to which reference zone (the UNet produces an unordered point
cloud) -- so we alternate "given the current orientation, what's the best
point<->zone assignment (Hungarian)" and "given this assignment, what's the
best rotation/scale (Umeyama)" until the assignment stops changing. Since
this alternation can get stuck on a local optimum (e.g. the wing numbered
upside down, a bad but stable minimum), it's restarted from several initial
orientations and the best one is kept.
"""
# NB: the old per-specimen orientation-ambiguity detection (comparison of
# the best vs. second-best start, `ambiguity_ratio` parameter) has been
# removed -- see the `numerate()` docstring.
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from landmarks.methods.base import NumberingResult
from core.alignment import kabsch_umeyama


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Full similarity (R, scale, t), no reflection: dst_hat = scale*(src@R.T)+t."""
    mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
    R, scale = kabsch_umeyama(src - mu_src, dst - mu_dst, estimate_scale=True)
    t = mu_dst - scale * (R @ mu_src)
    return R, scale, t


def apply_transform(pts: np.ndarray, R: np.ndarray, scale: float, t: np.ndarray) -> np.ndarray:
    return scale * (pts @ R.T) + t


def _register_from_start(pts0: np.ndarray, cur_init: np.ndarray, zones: np.ndarray, n_iter: int):
    """Assignment+Umeyama alternated until convergence, from ONE start.

    `pts0`: original centered cloud -- the transform is ALWAYS recomputed
    from this one, never from the previous iteration's `cur` (otherwise the
    initial rotation/mirror would keep compounding instead of only serving
    to orient the very first assignment).
    `cur_init`: pts0 after the initial rotation/mirror, used only for this
    first cost computation, to avoid a local optimum of the assignment.

    Loop: (1) cost = squared distance between each current point and each
    zone -> Hungarian gives the assignment minimizing the summed cost;
    (2) Umeyama realigns the whole pts0 onto the zones in that order;
    (3) repeat with the new position -- until the assignment itself stops
    changing (convergence), or n_iter is reached.
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
    """Implements the landmarks.methods.base contract: renumbers
    `landmarks` (k, 2) to match `reference`'s (n_zones, 2) order.

    Only handles the k == n_zones case (detected landmark count = expected
    count) for now; the k != n_zones case (too many/too few points) stays
    an explicit FAILED rather than an untested rectangular assignment -- to
    handle separately if the detector ever needs it (see also
    landmarks/renumber.py, which already short-circuits this case before
    calling numerate()).

    Always returns the best of the 8 starts (4 rotations x 2 mirrors: we
    don't know a priori in which orientation or chirality the wing was
    photographed), with its registration cost in `score`. This module no
    longer judges that cost's reliability itself (the old
    `ambiguity_ratio`, comparing the best and second-best starts): on real
    data, that per-specimen criterion turned out far too aggressive (it
    flagged nearly every specimen as SUSPECT, including visibly correct
    registrations), because two starts can legitimately converge to close
    costs without the assignment being wrong.

    See landmarks.methods.base: detecting an abnormal cost is a
    population-level decision, not a single specimen's -- landmarks/renumber.py
    now handles it via a more robust method (post-GPA comparison to the
    landmark's median position, PER SPECIES, see core.outliers), which
    better distinguishes a genuine registration error from a plain shape
    variation.
    """
    if landmarks.shape[0] != reference.shape[0]:
        return NumberingResult(
            numbered=landmarks,
            status="FAILED",
            score=float("inf"),
            reason=(
                f"{landmarks.shape[0]} landmarks detected, "
                f"{reference.shape[0]} expected in the template"
            ),
        )

    pts0 = landmarks - landmarks.mean(axis=0)
    best_assign, best_cost = None, float("inf")
    for angle in angle_inits:
        th = np.deg2rad(angle)
        rot0 = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        for mirror in mirror_options:
            mir = np.array([[-1, 0], [0, 1]]) if mirror else np.eye(2)
            cur_init = pts0 @ (rot0 @ mir).T
            assign, cost, _ = _register_from_start(pts0, cur_init, zones=reference, n_iter=n_iter)
            if cost < best_cost:
                best_assign, best_cost = assign, cost

    # `best_assign[k]` = index of the zone assigned to the k-th detected
    # landmark. We want the inverse: for each zone j (in reference order),
    # which detected landmark maps to it -- `inv` is therefore the inverse
    # permutation of `best_assign`.
    inv = np.empty(len(reference), dtype=int)
    inv[best_assign] = np.arange(len(best_assign))  # slot j (zone j) <- point assigned to j
    numbered = landmarks[inv]

    return NumberingResult(numbered=numbered, status="OK", score=best_cost)
