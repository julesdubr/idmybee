"""Tests for manifest.identification."""
import pandas as pd

from manifest.identification import (
    MAPPING_COLUMNS,
    attach_canonical_inv_id,
    build_device_type_table,
    build_specimen_device_table,
    build_specimen_table,
    extend_mapping,
    resolve_identification,
    restrict_to_present_images,
)

COMPARE_COLUMNS = ["genus", "species", "caste", "dd", "mm", "yyyy"]


def _empty_mapping() -> pd.DataFrame:
    return pd.DataFrame(columns=MAPPING_COLUMNS)


def _mapping(source_type: str, original_ids: list[str], inv_ids: list[str]) -> pd.DataFrame:
    n = len(original_ids)
    return pd.DataFrame({
        "source_type": [source_type] * n,
        "original_id": original_ids,
        "inv_num": list(range(1, n + 1)),
        "inv_id": inv_ids,
        "inv_name": ["TEST"] * n,
        "resolved_conflict": [False] * n,
        "frozen_at": ["2026-01-01T00:00:00"] * n,
    })


def test_attach_canonical_inv_id_drops_legacy_and_substitutes_canonical():
    mapping = _mapping("collection", ["CD1", "CD2"], ["TEST_0001", "TEST_0002"])
    # Raw source ships a legacy, device-suffixed `inv_id` placeholder column,
    # distinct from the real key column (`raw_id`) -- see module docstring.
    identification_df = pd.DataFrame({
        "raw_id": ["CD1", "CD1", "CD2"],
        "device_type": ["P", "S", "P"],
        "inv_id": ["CD1_P", "CD1_S", "CD2_P"],
        "genus": ["Bombus"] * 3,
    })

    result = attach_canonical_inv_id(identification_df, mapping, "raw_id", "collection")

    assert list(result["inv_id"]) == ["TEST_0001", "TEST_0001", "TEST_0002"]
    assert result.columns[0] == "inv_id"
    # key_column's raw values are kept, always under the "original_id" name
    assert list(result["original_id"]) == ["CD1", "CD1", "CD2"]


def test_attach_canonical_inv_id_renames_key_column_on_collision():
    mapping = _mapping("terrain", ["WB1_23_0007"], ["WB1_23_0007"])
    # Terrain sources already name their raw key column "inv_id".
    identification_df = pd.DataFrame({"inv_id": ["WB1_23_0007"], "genus": ["Bombus"]})

    result = attach_canonical_inv_id(identification_df, mapping, "inv_id", "terrain")

    assert list(result["inv_id"]) == ["WB1_23_0007"]
    assert "original_id" in result.columns
    assert list(result["original_id"]) == ["WB1_23_0007"]


def test_attach_canonical_inv_id_drops_key_column_when_not_kept():
    mapping = _mapping("collection", ["CD1"], ["TEST_0001"])
    identification_df = pd.DataFrame({"original_id": ["CD1"], "genus": ["Bombus"]})

    result = attach_canonical_inv_id(identification_df, mapping, "original_id", "collection", keep_original_id=False)

    assert "original_id" not in result.columns
    assert list(result["inv_id"]) == ["TEST_0001"]


def test_attach_canonical_inv_id_raises_on_original_id_collision():
    # If df already has its own, distinct "original_id" column, blindly
    # renaming key_column to "original_id" would silently overwrite it --
    # this must fail loud instead (regression for the bug where
    # identification_conflicts.csv's first column ended up holding the raw
    # id under the "original_id" name meant for the canonical inv_id).
    mapping = _mapping("collection", ["CD1"], ["TEST_0001"])
    identification_df = pd.DataFrame({
        "specimen_code": ["CD1"], "original_id": ["some other value"], "genus": ["Bombus"],
    })

    try:
        attach_canonical_inv_id(identification_df, mapping, "specimen_code", "collection")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_build_specimen_table_keeps_key_column_as_raw_id_reference():
    mapping = _mapping("collection", ["CD1", "CD2"], ["TEST_0001", "TEST_0002"])
    # The raw CSV's key column (its own "original_id") is kept in the
    # output next to the canonical inv_id -- it's the source's own
    # human-readable raw-id reference, not redundant clutter.
    identification_df = pd.DataFrame({
        "original_id": ["CD1", "CD2"],
        "genus": ["Bombus"] * 2,
    })

    result = build_specimen_table(mapping, identification_df, "original_id", "collection")

    assert list(result["original_id"]) == ["CD1", "CD2"]
    assert list(result["inv_id"]) == ["TEST_0001", "TEST_0002"]


def test_build_specimen_table_keeps_renamed_key_column():
    mapping = _mapping("terrain", ["WB1_23_0007"], ["WB1_23_0007"])
    identification_df = pd.DataFrame({"inv_id": ["WB1_23_0007"], "genus": ["Bombus"]})

    result = build_specimen_table(mapping, identification_df, "inv_id", "terrain")

    # original_id is the raw-id reference here -- terrain's raw key column
    # is itself named "inv_id", so it gets renamed to "original_id" to avoid
    # colliding with the canonical inv_id column.
    assert "original_id" in result.columns
    assert list(result["original_id"]) == ["WB1_23_0007"]
    assert list(result["inv_id"]) == ["WB1_23_0007"]


def test_build_specimen_table_drops_original_id_when_not_kept():
    # Embedded numbering (no --origin-codes) sets inv_id = raw key value
    # verbatim -- keeping a separate original_id column would just
    # duplicate inv_id, so callers opt out via keep_original_id=False.
    mapping = _mapping("terrain", ["WB1_23_0007"], ["WB1_23_0007"])
    identification_df = pd.DataFrame({"inv_id": ["WB1_23_0007"], "genus": ["Bombus"]})

    result = build_specimen_table(mapping, identification_df, "inv_id", "terrain", keep_original_id=False)

    assert "original_id" not in result.columns
    assert list(result["inv_id"]) == ["WB1_23_0007"]


def test_build_specimen_table_still_pivots_n_photos_per_device():
    mapping = _mapping("collection", ["CD1", "CD2"], ["TEST_0001", "TEST_0002"])
    identification_df = pd.DataFrame({
        "raw_id": ["CD1", "CD1", "CD2"],
        "device_type": ["P", "S", "P"],
        "n_photos": [3, 5, 2],
        "genus": ["Bombus"] * 3,
    })

    result = build_specimen_table(mapping, identification_df, "raw_id", "collection", "device_type")

    assert len(result) == 2  # one row per specimen, not per device
    cd1 = result[result["inv_id"] == "TEST_0001"].iloc[0]
    assert cd1["n_photos_P"] == 3
    assert cd1["n_photos_S"] == 5
    cd2 = result[result["inv_id"] == "TEST_0002"].iloc[0]
    assert cd2["n_photos_P"] == 2


def test_build_device_type_table_returns_distinct_pairs():
    identification_df = pd.DataFrame({
        "device_type": ["P", "S", "P", "S"],
        "device": ["Canon EOS 70D - 40mm", "Smartphone Samsung s10",
                   "Canon EOS 70D - 40mm", "Smartphone Samsung s10"],
    })

    result = build_device_type_table(identification_df, "device_type", "device")

    assert list(result.columns) == ["device_type", "device"]
    assert len(result) == 2
    assert dict(zip(result["device_type"], result["device"])) == {
        "P": "Canon EOS 70D - 40mm", "S": "Smartphone Samsung s10",
    }


def test_build_device_type_table_keeps_every_device_name_per_type():
    # device_type is a coarse code -- a collection source's "P" can cover
    # several camera bodies/lenses used over time, not just one.
    identification_df = pd.DataFrame({
        "device_type": ["P", "P", "S"],
        "device": ["Canon EOS 70D - 40mm", "Canon EOS 70D - 100mm macro", "Smartphone Samsung s10"],
    })

    result = build_device_type_table(identification_df, "device_type", "device")

    assert len(result) == 3
    assert sorted(result.loc[result["device_type"] == "P", "device"]) == [
        "Canon EOS 70D - 100mm macro", "Canon EOS 70D - 40mm",
    ]


def test_build_specimen_device_table_resolves_actual_device_per_photo_group():
    mapping = _mapping("collection", ["CD1", "CD2"], ["TEST_0001", "TEST_0002"])
    # CD1 was shot with one specific P camera and one specific phone; CD2
    # only with a *different* P camera -- the point of this table is that a
    # photo's device_type ("P") alone doesn't say which one.
    identification_df = pd.DataFrame({
        "raw_id": ["CD1", "CD1", "CD2"],
        "device_type": ["P", "S", "P"],
        "device": ["Canon EOS 70D - 40mm", "Smartphone Samsung s10", "Canon PowerShot G16"],
        "genus": ["Bombus"] * 3,
    })

    result = build_specimen_device_table(mapping, identification_df, "raw_id", "collection", "device_type", "device")

    assert list(result.columns) == ["inv_id", "device_type", "device"]
    assert len(result) == 3
    lookup = {(r.inv_id, r.device_type): r.device for r in result.itertuples()}
    assert lookup[("TEST_0001", "P")] == "Canon EOS 70D - 40mm"
    assert lookup[("TEST_0001", "S")] == "Smartphone Samsung s10"
    assert lookup[("TEST_0002", "P")] == "Canon PowerShot G16"


def _make_conflict_fixture() -> pd.DataFrame:
    """CD1: one clean specimen (P+S, no images issue). CD2: no image at
    all (P+S). CD3: original_id reused for two distinct identities (a
    soroeensis drone and a sichelii worker), each with a P+S pair -- a
    genuine identity conflict."""
    return pd.DataFrame({
        "raw_id": ["CD1", "CD1", "CD2", "CD2", "CD3", "CD3", "CD3", "CD3"],
        "device_type": ["P", "S", "P", "S", "P", "S", "P", "S"],
        # Legacy, device-suffixed placeholder id shipped by the raw source.
        "inv_id": ["CD1_P", "CD1_S", "CD2_P", "CD2_S", "CD3_P1", "CD3_S1", "CD3_P2", "CD3_S2"],
        "inv_name": ["TEST"] * 8,
        "genus": ["Bombus"] * 8,
        "species": ["terrestris", "terrestris", "lapidarius", "lapidarius",
                    "soroeensis", "soroeensis", "sichelii", "sichelii"],
        "caste": ["worker", "worker", "worker", "worker", "drone", "drone", "worker", "worker"],
        "dd": [pd.NA] * 8, "mm": [pd.NA] * 8, "yyyy": [pd.NA] * 8,
    })


def test_reports_are_one_row_per_specimen_with_canonical_inv_id():
    """Regression for the reports showing one line per device (device-
    suffixed inv_id) instead of one per specimen -- see
    tools/ingestion/export_clean_dataset.py, which follows this exact sequence."""
    identification_df = _make_conflict_fixture()
    manifest_df = pd.DataFrame({"original_id": ["CD1", "CD1"]})  # only CD1 was photographed

    resolved_all, conflicts = resolve_identification(
        identification_df, "raw_id", "device_type", COMPARE_COLUMNS,
    )
    conflicted_keys = set(conflicts["raw_id"])

    specimens = resolved_all.drop_duplicates("raw_id")[["raw_id", "inv_name", "dd", "mm", "yyyy"]].copy()
    specimens = specimens.rename(columns={"raw_id": "original_id"})
    specimens["source_type"] = "collection"
    specimens["resolved_conflict"] = specimens["original_id"].isin(conflicted_keys)

    mapping = extend_mapping(_empty_mapping(), specimens, "2026-01-01T00:00:00", embedded_numbering=False)

    _, excluded_no_image = restrict_to_present_images(resolved_all, manifest_df, "raw_id")

    conflicts = attach_canonical_inv_id(conflicts, mapping, "raw_id", "collection")
    excluded_no_image = attach_canonical_inv_id(excluded_no_image, mapping, "raw_id", "collection")
    excluded_no_image = excluded_no_image.drop_duplicates("inv_id").reset_index(drop=True)

    cd2_id = mapping.loc[mapping["original_id"] == "CD2", "inv_id"].iloc[0]
    cd3_id = mapping.loc[mapping["original_id"] == "CD3", "inv_id"].iloc[0]

    # CD2 (no image, P+S) and CD3 (identity conflict, P+S) each collapse to
    # a single report row -- not one per device.
    assert len(excluded_no_image) == 2
    assert set(excluded_no_image["inv_id"]) == {cd2_id, cd3_id}

    # Two distinct identities were found under CD3 -- still two rows, but
    # both correctly labelled with CD3's one real inv_id, not two different
    # device-suffixed placeholder values.
    assert len(conflicts) == 2
    assert set(conflicts["inv_id"]) == {cd3_id}
