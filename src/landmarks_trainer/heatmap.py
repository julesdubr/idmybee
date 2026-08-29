"""
Ground-truth heatmap generation for landmark localization training.

Convention note (important, read this before touching coordinates):
    encode_heatmap takes points as (row, col) i.e. (y, x), matching numpy
    array indexing (arr[row, col]) and opencv image arrays
    (img.shape == (height, width, channels)).
    This is DIFFERENT from the (x, y) convention used in TPS files and the
    rest of the pipeline (geomorph, manifests, landmarks/predict.py's
    public functions). The conversion happens once, at the dataset
    boundary (see dataset.py) -- keep it that way, don't let (x, y) leak in
    here, it's exactly the kind of mixup that broke Gabriel's original code
    (see `raw_annotation[:,::-1]` sprinkled around UNet_class_and_functions.py).

encode_heatmap: ground-truth heatmap used as the training target.
    For each point, intensity falls off linearly with distance within
    `radius` pixels, raised to `power` (a high power keeps the bump tight
    around the point rather than a broad gaussian-like blob). Multiple
    points are combined with a per-pixel max (not sum), so overlapping
    landmarks don't create an artificially bright blob between them.
    Equivalent to creation_relief_ulti_v2 in the legacy code (verified
    numerically identical), but vectorized instead of a double for-loop
    over every pixel.

Peak DECODING (predicted heatmap -> points) deliberately does NOT live
here: landmarks/predict.py already owns a well-tested implementation
(extract_top_landmarks, connected-component grouping over local_maxima
plateaus, two documented bugfixes over the original notebook). evaluate.py
imports that function directly rather than duplicating decode logic with
a second, possibly-diverging implementation.
"""

import numpy as np

from constants import DEFAULT_HEATMAP_RADIUS, DEFAULT_HEATMAP_POWER, IMG_HEIGHT, IMG_WIDTH


def encode_heatmap(points: np.ndarray, shape: tuple = (IMG_HEIGHT, IMG_WIDTH),
                    radius: float = DEFAULT_HEATMAP_RADIUS,
                    power: float = DEFAULT_HEATMAP_POWER) -> np.ndarray:
    """Build a single-channel ground-truth heatmap from landmark points.

    points: (N, 2) array of (row, col), any N (19 for training, but not
            hardcoded -- callers decide what they pass in)
    shape:  (height, width) of the output heatmap
    Returns: (height, width) float32 array, values in [0, 1]
    """
    if len(points) == 0:
        return np.zeros(shape, dtype=np.float32)

    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    heat = np.zeros(shape, dtype=np.float32)

    for row, col in points:
        dist = np.sqrt((yy - row) ** 2 + (xx - col) ** 2)
        falloff = np.clip((radius - dist) / radius, 0.0, 1.0) ** power
        heat = np.maximum(heat, falloff)

    return heat.astype(np.float32)
