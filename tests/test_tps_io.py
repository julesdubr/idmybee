"""Tests for core.tps_io."""
import numpy as np
import pytest

from core.tps_io import (
    ImageLandmarks,
    TpsParseException,
    assign_sequential_ids,
    parse_tps,
    write_tps,
)

SAMPLE = [
    ImageLandmarks(
        n_points=3,
        landmarks=np.array([[1.0, 2.0], [3.5, 4.5], [-1.0, 0.0]]),
        image_path="crops/a.jpg",
        tps_id=1,
        photo_id="ARLY_0001_P_1",
        inv_id="ARLY_0001",
    ),
    ImageLandmarks(
        n_points=3,
        landmarks=np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
        image_path="crops/b.jpg",
        tps_id=42,
    ),
]


def test_write_then_parse_round_trip(tmp_path):
    path = tmp_path / "out.tps"
    write_tps(path, SAMPLE)

    parsed, errors = parse_tps(path)

    assert errors == []
    assert len(parsed) == 2
    np.testing.assert_allclose(parsed[0].landmarks, SAMPLE[0].landmarks)
    assert parsed[0].image_path == "crops/a.jpg"
    assert parsed[0].photo_id == "ARLY_0001_P_1"
    assert parsed[0].inv_id == "ARLY_0001"
    # entry without photo_id/inv_id: no COMMENT= written, stays None on reread
    assert parsed[1].photo_id is None
    assert parsed[1].inv_id is None
    assert parsed[1].tps_id == 42


def test_assign_sequential_ids_orders_by_photo_id():
    specimens = [
        ImageLandmarks(n_points=1, landmarks=np.zeros((1, 2)), image_path="c.jpg", tps_id=99, photo_id="B_0001_P_1"),
        ImageLandmarks(n_points=1, landmarks=np.zeros((1, 2)), image_path="a.jpg", tps_id=99, photo_id="A_0001_P_1"),
    ]

    result = assign_sequential_ids(specimens)

    assert [sp.photo_id for sp in result] == ["A_0001_P_1", "B_0001_P_1"]
    assert [sp.tps_id for sp in result] == [1, 2]


def test_parse_tps_strict_raises_on_malformed_block(tmp_path):
    path = tmp_path / "bad.tps"
    path.write_text("LM=2\n1.0 2.0\nIMAGE=a.jpg\nID=0\n", encoding="utf-8")  # only 1 of 2 points

    with pytest.raises(TpsParseException):
        parse_tps(path, strict=True)


def test_parse_tps_non_strict_collects_errors_and_skips_bad_block(tmp_path):
    path = tmp_path / "mixed.tps"
    path.write_text(
        "LM=2\n1.0 2.0\n3.0 4.0\nIMAGE=a.jpg\nID=notanumber\n"  # malformed: bad ID=
        "LM=2\n1.0 2.0\n3.0 4.0\nIMAGE=b.jpg\nID=1\n",  # valid
        encoding="utf-8",
    )

    parsed, errors = parse_tps(path, strict=False)

    assert len(parsed) == 1
    assert parsed[0].image_path == "b.jpg"
    assert len(errors) >= 1


def test_parse_tps_non_strict_recovers_block_after_truncated_one(tmp_path):
    """Regression test: a block with FEWER coordinate lines than its LM=
    header claims must not desync the reader into the next block's own
    LM=/points -- only the truncated block is dropped, the next valid one
    is still parsed (see _DIRECTIVE_PREFIXES in tps_io.py)."""
    path = tmp_path / "truncated.tps"
    path.write_text(
        "LM=2\n1.0 2.0\nIMAGE=a.jpg\nID=0\n"  # truncated: only 1 of 2 points
        "LM=2\n1.0 2.0\n3.0 4.0\nIMAGE=b.jpg\nID=1\n",  # otherwise valid
        encoding="utf-8",
    )

    parsed, errors = parse_tps(path, strict=False)

    assert len(parsed) == 1
    assert parsed[0].image_path == "b.jpg"
    assert len(errors) == 1
    assert "expected 2 landmarks, found 1" in errors[0].message


def test_parse_tps_missing_id_is_an_error(tmp_path):
    path = tmp_path / "no_id.tps"
    path.write_text("LM=1\n1.0 2.0\nIMAGE=a.jpg\n", encoding="utf-8")

    parsed, errors = parse_tps(path, strict=False)

    assert parsed == []
    assert len(errors) == 1
