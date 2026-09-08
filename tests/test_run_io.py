"""Tests for utils.run_io."""
from pathlib import Path

import pytest

from utils.run_io import (
    build_eval_tag,
    build_run_id,
    build_variance_id,
    run_id_from_model_path,
    slugify,
    tag_from_tps,
)


@pytest.mark.parametrize("raw,expected", [
    ("train", "train"),
    ("Bombus rupestris", "Bombus-rupestris"),
    ("a---b", "a-b"),
    ("  leading/trailing  ", "leading-trailing"),
    ("", "run"),
    ("___", "run"),
])
def test_slugify(raw, expected):
    assert slugify(raw) == expected


def test_tag_from_tps_none_when_no_path():
    assert tag_from_tps(None) is None


def test_tag_from_tps_uses_stem():
    assert tag_from_tps(Path("data/annotations/tancrede_19lm.tps")) == "tancrede_19lm"


def test_build_run_id_composes_parts_in_order():
    run_id = build_run_id("species", "collection", devices=["P1", "S1"], run_label="tancrede19lm")
    assert run_id == "species_collection_P1-S1_tancrede19lm"


def test_build_run_id_without_optional_parts():
    assert build_run_id("caste", "terrain") == "caste_terrain"


def test_build_eval_tag_has_no_level_prefix():
    tag = build_eval_tag("terrain", devices=["P2"])
    assert tag == "terrain_P2"


def test_build_variance_id_joins_levels():
    vid = build_variance_id(["species", "caste"], "collection", devices=["P1", "P2", "S1", "S2"])
    assert vid == "species-caste_collection_P1-P2-S1-S2"


def test_run_id_from_model_path_parses_convention():
    p = Path("data/models/lda/species_train/train/model.joblib")
    family, run_id = run_id_from_model_path(p)
    assert family == "lda"
    assert run_id == "species_train"


def test_run_id_from_model_path_rejects_other_layouts():
    with pytest.raises(ValueError):
        run_id_from_model_path(Path("data/models/lda/species_train/model.joblib"))
