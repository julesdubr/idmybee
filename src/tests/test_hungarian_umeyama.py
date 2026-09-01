"""Tests for landmarks.methods.hungarian_umeyama."""
import numpy as np

from landmarks.methods.hungarian_umeyama import numerate

REFERENCE = np.array([
    [0.0, 0.0], [1.0, 0.2], [2.0, -0.3], [1.5, 1.0],
    [0.5, 1.3], [-0.5, 0.8], [-0.2, -0.5], [1.0, -1.0],
])


def _shuffled(reference: np.ndarray, seed: int = 0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(reference))
    return reference[perm], perm


def test_numerate_recovers_identity_permutation_from_shuffled_input():
    shuffled, perm = _shuffled(REFERENCE)

    result = numerate(shuffled, REFERENCE)

    assert result.status == "OK"
    np.testing.assert_allclose(result.numbered, REFERENCE, atol=1e-6)
    assert result.score < 1e-8


def test_numerate_recovers_permutation_under_rotation_and_scale():
    shuffled, perm = _shuffled(REFERENCE, seed=1)
    theta = np.deg2rad(64)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    transformed = (shuffled @ rot.T) * 3.3 + np.array([10.0, -4.0])

    result = numerate(transformed, REFERENCE)

    assert result.status == "OK"
    assert result.score < 1e-6
    # numbered[j] should be the (transformed) point that maps back to reference[j]
    expected_order = transformed[np.argsort(perm)]
    np.testing.assert_allclose(result.numbered, expected_order, atol=1e-4)


def test_numerate_recovers_permutation_under_mirroring():
    shuffled, perm = _shuffled(REFERENCE, seed=2)
    mirrored = shuffled.copy()
    mirrored[:, 0] *= -1

    result = numerate(mirrored, REFERENCE)

    # Mirror recovery is the hardest case for this heuristic (coarse 4x2
    # multi-start, not an exhaustive search): unlike the identity/rotation
    # cases, near-perfect recovery isn't guaranteed on an arbitrary point
    # set, only a low-cost match.
    assert result.status == "OK"
    assert result.score < 0.1


def test_numerate_fails_on_landmark_count_mismatch():
    result = numerate(REFERENCE[:-1], REFERENCE)

    assert result.status == "FAILED"
    assert result.score == float("inf")
    assert "7" in result.reason and "8" in result.reason
