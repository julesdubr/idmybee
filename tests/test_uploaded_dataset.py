"""Tests for utils.uploaded_dataset.join_specimens_to_bio -- the two ways a
TPS specimen gets an inv_id before write_dataset_root/load_dataset can use
it: COMMENT= (a TPS exported by this codebase) or, when the TPS carries
none, positionally via a `tps_id` column in the CSV (see module
docstring)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.tps_io import ImageLandmarks
from utils.uploaded_dataset import DatasetMergeError, join_specimens_to_bio


def _specimen(tps_id: int, inv_id: str | None = None) -> ImageLandmarks:
    return ImageLandmarks(
        n_points=3, landmarks=np.zeros((3, 2)), image_path=f"img_{tps_id}.jpg",
        tps_id=tps_id, photo_id=f"photo_{tps_id}", inv_id=inv_id,
    )


def test_join_by_comment_inv_id_unchanged_when_present():
    specimens = [_specimen(1, "IND1"), _specimen(2, "IND2")]
    bio_df = pd.DataFrame({"inv_id": ["IND1", "IND2"], "species": ["a", "b"], "caste": ["w", "w"]})
    result = join_specimens_to_bio(specimens, bio_df)
    assert [sp.inv_id for sp in result] == ["IND1", "IND2"]


def test_join_by_comment_inv_id_raises_when_no_overlap():
    specimens = [_specimen(1, "IND1")]
    bio_df = pd.DataFrame({"inv_id": ["OTHER"], "species": ["a"], "caste": ["w"]})
    with pytest.raises(DatasetMergeError, match="No inv_id in common"):
        join_specimens_to_bio(specimens, bio_df)


def test_join_by_tps_id_when_no_comment():
    specimens = [_specimen(1), _specimen(2), _specimen(3)]
    bio_df = pd.DataFrame({
        "tps_id": [1, 2, 3], "inv_id": ["IND1", "IND1", "IND2"],
        "species": ["a", "a", "b"], "caste": ["w", "w", "w"],
    })
    result = join_specimens_to_bio(specimens, bio_df)
    assert [sp.inv_id for sp in result] == ["IND1", "IND1", "IND2"]


def test_join_by_tps_id_raises_when_csv_has_no_tps_id_column():
    specimens = [_specimen(1)]
    bio_df = pd.DataFrame({"inv_id": ["IND1"], "species": ["a"], "caste": ["w"]})
    with pytest.raises(DatasetMergeError, match="tps_id"):
        join_specimens_to_bio(specimens, bio_df)


def test_join_by_tps_id_raises_when_a_specimen_has_no_matching_row():
    specimens = [_specimen(1), _specimen(2)]
    bio_df = pd.DataFrame({"tps_id": [1], "inv_id": ["IND1"], "species": ["a"], "caste": ["w"]})
    with pytest.raises(DatasetMergeError, match="no matching 'tps_id'"):
        join_specimens_to_bio(specimens, bio_df)


def test_join_by_tps_id_used_when_only_some_specimens_have_inv_id():
    # Mixed COMMENT=-present/absent across specimens (e.g. concatenated
    # uploads of mixed origin) falls back to the tps_id join uniformly
    # rather than trusting the partial COMMENT= data.
    specimens = [_specimen(1, "IND1"), _specimen(2)]
    bio_df = pd.DataFrame({
        "tps_id": [1, 2], "inv_id": ["IND1", "IND2"], "species": ["a", "b"], "caste": ["w", "w"],
    })
    result = join_specimens_to_bio(specimens, bio_df)
    assert [sp.inv_id for sp in result] == ["IND1", "IND2"]
