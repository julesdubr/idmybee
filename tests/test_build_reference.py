"""Tests for landmarks.build_reference."""
from __future__ import annotations

import numpy as np
import pytest

from core.tps_io import ImageLandmarks, parse_tps, write_tps
from landmarks.build_reference import build_reference_shape, load_reference, main

BASE_SHAPE = np.array([
    [0.0, 0.0], [1.0, 0.2], [2.0, -0.3], [1.5, 1.0],
    [0.5, 1.3], [-0.5, 0.8], [-0.2, -0.5], [1.0, -1.0],
])


def _write_ground_truth_tps(path, shapes) -> None:
    specimens = [
        ImageLandmarks(n_points=len(shape), landmarks=shape, image_path="", tps_id=i)
        for i, shape in enumerate(shapes, start=1)
    ]
    write_tps(path, specimens)


def test_build_reference_shape_matches_expected_landmark_count_only():
    incomplete = ImageLandmarks(n_points=4, landmarks=BASE_SHAPE[:4], image_path="", tps_id=1)
    full = ImageLandmarks(n_points=8, landmarks=BASE_SHAPE, image_path="", tps_id=2)
    consensus = build_reference_shape([incomplete, full], expected_lm=8)
    assert consensus.shape == (8, 2)


def test_build_reference_shape_raises_if_no_specimen_matches():
    incomplete = ImageLandmarks(n_points=4, landmarks=BASE_SHAPE[:4], image_path="", tps_id=1)
    with pytest.raises(SystemExit):
        build_reference_shape([incomplete], expected_lm=8)


def test_write_then_load_reference_round_trips_zones(tmp_path):
    consensus = build_reference_shape(
        [ImageLandmarks(n_points=8, landmarks=BASE_SHAPE, image_path="", tps_id=1)], expected_lm=8,
    )
    out_path = tmp_path / "reference.tps"
    write_tps(out_path, [ImageLandmarks(n_points=len(consensus), landmarks=consensus, image_path="", tps_id=1)])

    zones = load_reference(out_path)

    # write_tps formats coordinates to 4 decimal places -- round-trip
    # tolerance matches that, not exact float equality.
    assert zones == pytest.approx(consensus, abs=1e-4)


def test_load_reference_rejects_multi_specimen_tps(tmp_path):
    path = tmp_path / "not_a_reference.tps"
    _write_ground_truth_tps(path, [BASE_SHAPE, BASE_SHAPE])
    with pytest.raises(SystemExit):
        load_reference(path)


def test_load_reference_missing_file_raises(tmp_path):
    with pytest.raises(SystemExit):
        load_reference(tmp_path / "missing.tps")


def test_main_writes_plain_tps_and_drop_removes_one_landmark(tmp_path):
    ref_path = tmp_path / "ground_truth.tps"
    _write_ground_truth_tps(ref_path, [BASE_SHAPE, BASE_SHAPE * 1.1 + 0.2])
    out_path = tmp_path / "shapes" / "reference.tps"

    main(["--ref", str(ref_path), "--drop", "3", "--out", str(out_path)])

    assert out_path.exists()
    specimens, errors = parse_tps(out_path, strict=False)
    assert not errors
    assert len(specimens) == 1
    assert specimens[0].n_points == 7  # 8 - 1 dropped

    zones = load_reference(out_path)
    assert zones.shape == (7, 2)
