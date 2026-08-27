"""
Data augmentation for landmark heatmap training.

Two transforms, both operate on (image, points) together so landmark
coordinates always stay in sync with the image content:

- random_zoom_and_shift: re-implements Gabriel's RandomZoomAndShift, but as
  a pure function returning the new points instead of mutating instance
  attributes that the caller had to read back afterwards (that stateful
  design in the original is what made the annotation-shifting math in
  UNet_training.ipynb cell 2 so hard to follow). Verified numerically
  equivalent in behaviour.
- random_downscale_quality: re-implements RandomDownQuality (down/upsample
  to simulate lower-quality source photos). Doesn't touch points.

Points convention: (row, col), same as heatmap.py -- see that file's
docstring for why.
"""

import cv2
import numpy as np


def random_zoom_and_shift(image: np.ndarray, points: np.ndarray,
                           zoom_range: tuple = (0.8, 1.2),
                           rng: np.random.Generator = None):
    """Randomly zoom in/out and shift the crop, keeping every point inside
    the zoomed frame when possible (shift range is bounded by the sampled
    zoom so this holds by construction, same guarantee as the original).

    Returns (new_image, new_points). Points can end up outside the final
    frame at extreme zoom-out + shift combinations -- callers that need
    strict in-frame guarantees should check bounds after calling this.
    """
    rng = rng or np.random.default_rng()
    h, w = image.shape[:2]
    zoom = rng.uniform(*zoom_range)

    max_shift_h = (zoom_range[1] - zoom) / 2 * h
    max_shift_w = (zoom_range[1] - zoom) / 2 * w
    shift_h = rng.uniform(-max_shift_h, max_shift_h)
    shift_w = rng.uniform(-max_shift_w, max_shift_w)

    new_h, new_w = int(h * zoom), int(w * zoom)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad = 200
    padded = np.pad(resized, ((pad, pad), (pad, pad), (0, 0)), mode="constant")

    top = int((new_h - h) / 2 + shift_h + pad)
    left = int((new_w - w) / 2 + shift_w + pad)
    cropped = padded[top:top + h, left:left + w]

    new_points = points.astype(np.float64) * zoom
    new_points[:, 0] -= (top - pad)
    new_points[:, 1] -= (left - pad)

    return cropped, new_points


def random_downscale_quality(image: np.ndarray, quality_delta: float = 0.5,
                              rng: np.random.Generator = None) -> np.ndarray:
    """Simulate a lower-quality source photo by downsampling then
    upsampling back to the original size. Never upscales beyond the
    original (factor is capped at 1.0). Doesn't move points."""
    rng = rng or np.random.default_rng()
    factor = min(rng.uniform(1 - quality_delta, 1 + quality_delta), 1.0)

    h, w = image.shape[:2]
    small = cv2.resize(image, (int(w * factor), int(h * factor)),
                        interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def augment(image: np.ndarray, points: np.ndarray, rng: np.random.Generator = None):
    """Apply the full training augmentation pipeline: zoom+shift, then
    quality degradation. Use this from dataset.py; use the two functions
    above directly if you need finer control (e.g. for a debug script)."""
    rng = rng or np.random.default_rng()
    image, points = random_zoom_and_shift(image, points, rng=rng)
    image = random_downscale_quality(image, rng=rng)
    return image, points
