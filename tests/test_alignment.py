"""Tests for core.alignment.kabsch_umeyama.

The scale test pins down a bug that was fixed once already (H not
normalized by n -> scale overestimated by a factor of n, see the
function's docstring): identity assignment must always yield scale=1.0,
regardless of n_points. A regression here means the bug is back.
"""
import numpy as np
import pytest

from core.alignment import kabsch_umeyama

SHAPE = np.array([
    [0.0, 0.0], [1.0, 0.2], [2.0, -0.3], [1.5, 1.0],
    [0.5, 1.3], [-0.5, 0.8], [-0.2, -0.5], [1.0, -1.0],
])
CENTERED = SHAPE - SHAPE.mean(axis=0)


@pytest.mark.parametrize("n_points", [3, 8, 19])
def test_identity_assignment_scale_is_one(n_points):
    """scale must be 1.0 for identical source/target, whatever n_points is --
    the historical bug scaled this by a factor of n_points."""
    source = CENTERED[:n_points]
    _, scale = kabsch_umeyama(source, source, estimate_scale=True)
    assert scale == pytest.approx(1.0, abs=1e-8)


def test_no_scale_estimation_returns_one_without_computing():
    R, scale = kabsch_umeyama(CENTERED, CENTERED * 2, estimate_scale=False)
    assert scale == 1.0


def test_recovers_known_rotation():
    theta = np.deg2rad(37)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    target = CENTERED @ rot.T

    R, scale = kabsch_umeyama(CENTERED, target, estimate_scale=False)
    aligned = CENTERED @ R.T

    assert aligned == pytest.approx(target, abs=1e-8)
    assert scale == 1.0


def test_recovers_known_scale():
    factor = 2.7
    target = CENTERED * factor

    _, scale = kabsch_umeyama(CENTERED, target, estimate_scale=True)

    assert scale == pytest.approx(factor, abs=1e-6)


def test_rotation_matrix_never_reflects():
    """det(R) must be +1: a mirror-flipped specimen should still register via
    a real rotation, not a reflection (see gpa.py module docstring)."""
    mirrored = CENTERED.copy()
    mirrored[:, 0] *= -1

    R, _ = kabsch_umeyama(CENTERED, mirrored, estimate_scale=False)

    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-8)
