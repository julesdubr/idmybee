"""Tests for core.outliers."""
import numpy as np

from core.outliers import flag_by_species, outlier_matrix
from core.tps_io import ImageLandmarks

BASE_SHAPE = np.array([
    [0.0, 0.0], [1.0, 0.2], [2.0, -0.3], [1.5, 1.0],
    [0.5, 1.3], [-0.5, 0.8], [-0.2, -0.5], [1.0, -1.0],
])


def _make_specimens(n: int, shift_landmark: int | None = None, shift_amount: float = 5.0):
    specimens = []
    rng = np.random.default_rng(0)
    for i in range(n):
        pts = BASE_SHAPE + rng.normal(scale=0.01, size=BASE_SHAPE.shape)
        if shift_landmark is not None and i == 0:
            pts = pts.copy()
            pts[shift_landmark] += shift_amount
        specimens.append(
            ImageLandmarks(n_points=pts.shape[0], landmarks=pts, image_path=f"img{i}.jpg", tps_id=i)
        )
    return specimens


def test_outlier_matrix_flags_shifted_landmark():
    n = 15
    specimens = _make_specimens(n, shift_landmark=3, shift_amount=5.0)
    aligned = np.stack([s.landmarks for s in specimens])

    flags = outlier_matrix(aligned)

    assert flags[0, 3]
    # the other landmarks of specimen 0 should stay clean
    assert flags[0].sum() == 1


def test_flag_by_species_below_min_group_size_is_ok():
    specimens = _make_specimens(5, shift_landmark=0, shift_amount=10.0)
    species = np.array(["rare_sp"] * 5)

    n_outlier, heavy = flag_by_species(specimens, species, min_group_size=10)

    assert (n_outlier == 0).all()
    assert not heavy.any()


def test_flag_by_species_heavy_flag_on_globally_shifted_specimen():
    n = 20
    specimens = _make_specimens(n)
    # make specimen 0 an overall bad registration: every landmark off
    rng = np.random.default_rng(1)
    specimens[0].landmarks = specimens[0].landmarks + rng.normal(scale=3.0, size=BASE_SHAPE.shape)
    species = np.array(["sp_a"] * n)

    n_outlier, heavy = flag_by_species(specimens, species, min_group_size=10)

    assert heavy[0]
    assert not heavy[1:].any()
