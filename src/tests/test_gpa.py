"""Tests for utils.gpa."""
import numpy as np
import pytest

from utils.gpa import align_to_reference, centroid_size, gpagen, procrustes_distance

BASE_SHAPE = np.array([
    [0.0, 0.0], [1.0, 0.2], [2.0, -0.3], [1.5, 1.0],
    [0.5, 1.3], [-0.5, 0.8], [-0.2, -0.5], [1.0, -1.0],
])


def _rotate(coords: np.ndarray, degrees: float) -> np.ndarray:
    theta = np.deg2rad(degrees)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return coords @ rot.T


def test_centroid_size_unit_square():
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    # centered: (+-0.5, +-0.5) -> sqrt(4 * 0.5) = sqrt(2)
    assert centroid_size(square) == pytest.approx(np.sqrt(2))


def test_centroid_size_degenerate_raises_in_gpagen():
    degenerate = np.zeros((4, 2))
    with pytest.raises(ValueError):
        gpagen([degenerate, BASE_SHAPE])


def test_gpagen_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        gpagen([BASE_SHAPE, BASE_SHAPE[:-1]])


def test_gpagen_aligns_rotated_and_scaled_copies():
    """Same shape, rotated/scaled/translated N times: GPA must converge with
    near-zero residual Procrustes distance to the mean shape for every copy."""
    specimens = [
        BASE_SHAPE + 5.0,
        _rotate(BASE_SHAPE, 90) * 2.0,
        _rotate(BASE_SHAPE, 213) * 0.3 - 10.0,
        _rotate(BASE_SHAPE, -48) * 1.7,
    ]

    result = gpagen(specimens)

    assert result.aligned.shape == (4, 8, 2)
    assert result.centroid_sizes.shape == (4,)
    for i in range(4):
        dist = procrustes_distance(result.aligned[i], result.mean_shape)
        assert dist == pytest.approx(0.0, abs=1e-6)
    # mean shape itself must be unit centroid size (GPA standardization)
    assert centroid_size(result.mean_shape) == pytest.approx(1.0, abs=1e-6)


def test_align_to_reference_matches_gpagen_member():
    specimens = [BASE_SHAPE, _rotate(BASE_SHAPE, 120) * 1.5]
    result = gpagen(specimens)

    aligned_again = align_to_reference(specimens[1], result.mean_shape)

    assert aligned_again == pytest.approx(result.aligned[1], abs=1e-6)


def test_align_to_reference_rejects_shape_mismatch():
    reference = BASE_SHAPE[:-1] / centroid_size(BASE_SHAPE[:-1])
    with pytest.raises(ValueError):
        align_to_reference(BASE_SHAPE, reference)
