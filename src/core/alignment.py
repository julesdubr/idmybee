"""alignment.py
Shared SVD core for any rigid 2D alignment without reflection (Kabsch /
Umeyama): optimal rotation, with optional scale.

Used by:
- core/gpa.py (rotation only -- shapes are already centered/scaled
  upstream by the GPA standardization, so scale=1 always).
- landmarks/methods/hungarian_umeyama.py (full similarity -- rotation and
  scale, since a detector's raw landmarks are neither centered nor at the
  reference template's scale).

Before this extraction, gpa.py and register.py each reimplemented their own
version of this computation (same algebra, formulated differently --
cross-covariance transposed depending on the file), risking a bug or
convention change in one going unreported in the other.
"""
from __future__ import annotations

import numpy as np


def _kabsch_svd(source_c: np.ndarray, target_c: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """SVD of the cross-covariance + anti-reflection correction.

    `source_c`/`target_c`: (n_points, 2), already centered (zero mean).
    Returns (R, D, d):
    - R: optimal 2x2 rotation (det(R) = +1), aligning source onto target
      via `source_c @ R.T`.
    - D: singular values of the cross-covariance (used for the optimal
      scale, see kabsch_umeyama).
    - d: anti-reflection correction sign applied to the lowest-variance
      axis (+1 or -1), also needed for the scale.
    """
    H = source_c.T @ target_c
    U, D, Vt = np.linalg.svd(H)
    d = float(np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0)
    R = Vt.T @ np.diag([1.0, d]) @ U.T
    return R, D, d


def kabsch_umeyama(
    source_c: np.ndarray, target_c: np.ndarray, estimate_scale: bool = False
) -> tuple[np.ndarray, float]:
    """Rotation (and optional isotropic scale) minimizing
    ||target_c - scale * source_c @ R.T||^2, without reflection (det(R)=+1).

    `source_c`/`target_c` must already be centered (zero mean) -- translation,
    if needed, is left to the caller (see
    landmarks/methods/hungarian_umeyama.umeyama for the version with
    translation).

    `estimate_scale=False` (GPA case: shapes are already normalized to
    centroid size 1, so the optimal scale is always 1) returns scale=1.0
    without computing it. `estimate_scale=True` (registration case: a
    detector's raw landmarks are not at the template's scale) computes the
    least-squares optimal scale.

    FIXED BUG (see the Phase 3 graph_matching session): `_kabsch_svd`
    computes `D` from `H = source_c.T @ target_c`, NOT normalized by n. For
    the Umeyama scale formula (`scale = trace(D)/var_source`) to be
    correct, `D` must come from the cross-covariance normalized by n --
    consistent with `var_source` below, which IS already divided by
    `len(source_c)`. Without this division, the computed scale is
    overestimated by a factor exactly equal to n (verified empirically:
    n=18 -> scale 18x too large, on the trivial case "Tancrede's ground
    truth against its own leave-one-out GPA consensus", where the identity
    assignment should give a near-zero cost). Rotation R is NOT affected
    (direction of U/Vt, independent of H's scale) -- so GPA
    (estimate_scale=False, which never uses D) was never impacted.
    """
    R, D, d = _kabsch_svd(source_c, target_c)
    if not estimate_scale:
        return R, 1.0
    n = len(source_c)
    var_source = float((source_c**2).sum() / n)
    if var_source == 0:
        raise ValueError("Degenerate specimen: landmarks coincide (zero variance)")
    scale = (D[0] + d * D[1]) / n / var_source
    return R, scale
