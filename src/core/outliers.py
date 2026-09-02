"""outliers.py
Post-GPA diagnostic: distinguishes pointwise detection noise (1-2 shifted
landmarks) from full registration failures (nearly all of a specimen's
landmarks shifted -- the specimen as a whole is misaligned, often an
out-of-template wing shape, or a bad renumbering).

Formerly tools/flag_outlier_specimens.py (standalone CLI script). Extracted
here as a pure module (no more __main__/argparse) to be called directly by
landmarks/renumber.py right after renumbering: the specimen population and
their numbering were just computed at the same place, no need to write an
intermediate TPS and reparse it in a second script for this diagnostic.

The threshold (median + `mad_factor`*MAD of the distance to the median
position, per landmark) is computed PER SPECIES, not over the whole
dataset: species naturally have different wing shapes (that's the whole
basis of classification), so a global threshold would confuse "different
wing because it's a different species" with "badly aligned wing" --
artificially inflating the outlier rate of the least-represented species or
those with the most atypical wings (e.g. B. rupestris). A group with fewer
than `min_group_size` specimens doesn't have a reliable median/MAD: it's
left alone (considered OK) rather than inventing a threshold.
"""
from __future__ import annotations

import logging

import numpy as np

from core.gpa import gpagen
from core.tps_io import ImageLandmarks

logger = logging.getLogger(__name__)

MIN_GROUP_SIZE = 10   # below this, a per-landmark median/MAD isn't reliable
MAD_FACTOR = 6.0       # median + MAD_FACTOR*MAD -> "landmark outlier" threshold
HEAVY_LANDMARK_FRAC = 0.55  # beyond this fraction of outlier landmarks -> whole specimen flagged


def outlier_matrix(aligned: np.ndarray, mad_factor: float = MAD_FACTOR) -> np.ndarray:
    """(n_specimens, n_landmarks) bool: True if the point is far from its
    landmark's median position (median + mad_factor*MAD)."""
    med = np.median(aligned, axis=0)
    dist = np.linalg.norm(aligned - med[None, :, :], axis=2)
    mad = np.median(np.abs(dist - np.median(dist, axis=0)), axis=0)
    thresh = np.median(dist, axis=0) + mad_factor * mad
    return dist > thresh[None, :]


def flag_by_species(
    specimens: list[ImageLandmarks],
    species: np.ndarray,
    heavy_frac: float = HEAVY_LANDMARK_FRAC,
    min_group_size: int = MIN_GROUP_SIZE,
    mad_factor: float = MAD_FACTOR,
) -> tuple[np.ndarray, np.ndarray]:
    """GPA + MAD threshold computed separately for each species (see the
    module docstring). Returns (n_outlier, heavy), aligned on `specimens`:
    n_outlier = number of outlier landmarks for the specimen, heavy = True
    if >= heavy_frac of the specimen's landmarks are outliers (whole
    specimen suspect, not just a noisy point)."""
    if not specimens:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=bool)

    n_landmarks = specimens[0].n_points
    n_outlier = np.zeros(len(specimens), dtype=int)
    heavy = np.zeros(len(specimens), dtype=bool)

    for sp_name in sorted(set(species)):
        idx = np.where(species == sp_name)[0]
        if len(idx) < min_group_size:
            logger.info("%s: %d specimen(s), < %d -- not evaluated (considered OK)", sp_name, len(idx), min_group_size)
            continue
        group = [specimens[i] for i in idx]
        result = gpagen([s.landmarks for s in group])
        group_outlier = outlier_matrix(result.aligned, mad_factor=mad_factor)
        group_n_outlier = group_outlier.sum(axis=1)
        n_outlier[idx] = group_n_outlier
        heavy[idx] = group_n_outlier >= heavy_frac * n_landmarks

    return n_outlier, heavy
