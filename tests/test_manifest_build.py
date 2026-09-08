"""Tests for manifest.build."""
import pandas as pd

from manifest.build import (
    assign_sequential_photo_ids,
    biological_columns,
    build_biological_data,
    build_manifest_table,
    check_biological_consistency,
    flag_duplicate_photo_ids,
    resolve_and_verify_images,
    validate_required_columns,
)


def test_validate_required_columns_reports_all_missing():
    df = pd.DataFrame({"inv_id": ["A_0001"], "path": ["a.jpg"]})
    assert sorted(validate_required_columns(df, "path")) == ["caste", "species"]


def test_validate_required_columns_empty_when_all_present():
    df = pd.DataFrame({"inv_id": ["A_0001"], "species": ["terrestris"], "caste": ["worker"], "path": ["a.jpg"]})
    assert validate_required_columns(df, "path") == []


def test_biological_columns_excludes_photo_level_and_path():
    df = pd.DataFrame({
        "inv_id": ["A_0001"], "species": ["terrestris"], "caste": ["worker"],
        "path": ["a.jpg"], "photo_id": ["A_0001_S_1"], "device_type": ["S"], "genus": ["Bombus"],
    })
    assert sorted(biological_columns(df, "path")) == ["caste", "genus", "species"]


def test_resolve_and_verify_images_ok_for_readable_file(tmp_path):
    image = tmp_path / "a.jpg"
    image.write_bytes(b"fake image content")
    df = pd.DataFrame({"path": ["a.jpg"]})

    result = resolve_and_verify_images(df, "path", tmp_path)

    assert list(result["status"]) == ["OK"]
    assert result["ext"].iloc[0] == ".jpg"
    assert result["file_size_bytes"].iloc[0] == len(b"fake image content")
    assert result["content_hash"].iloc[0]


def test_resolve_and_verify_images_failed_for_missing_file(tmp_path):
    df = pd.DataFrame({"path": ["missing.jpg"]})

    result = resolve_and_verify_images(df, "path", tmp_path)

    assert list(result["status"]) == ["FAILED"]
    assert "unreadable_image_or_missing_file" in result["status_reason"].iloc[0]
    assert pd.isna(result["content_hash"].iloc[0])


def test_flag_duplicate_photo_ids_fails_every_occurrence_after_the_first():
    df = pd.DataFrame({
        "photo_id": ["A_0001_S_1", "A_0001_S_1", "A_0002_S_1"],
        "status": ["OK", "OK", "OK"],
        "status_reason": ["", "", ""],
    })

    result = flag_duplicate_photo_ids(df)

    assert list(result["status"]) == ["OK", "FAILED", "OK"]
    assert result["status_reason"].iloc[1] == "duplicate photo_id in input"


def test_flag_duplicate_photo_ids_never_downgrades_an_already_failed_row():
    df = pd.DataFrame({
        "photo_id": ["A_0001_S_1", "A_0001_S_1"],
        "status": ["FAILED", "OK"],
        "status_reason": ["unreadable_image_or_missing_file: boom", ""],
    })

    result = flag_duplicate_photo_ids(df)

    assert result["status_reason"].iloc[0] == "unreadable_image_or_missing_file: boom"
    assert result["status"].iloc[1] == "FAILED"


def test_check_biological_consistency_flags_disagreeing_rows_as_suspect():
    df = pd.DataFrame({
        "inv_id": ["A_0001", "A_0001", "A_0002"],
        "species": ["terrestris", "lapidarius", "terrestris"],
        "status": ["OK", "OK", "OK"],
        "status_reason": ["", "", ""],
    })

    result, conflicts = check_biological_consistency(df, ["species"])

    assert list(result["status"]) == ["SUSPECT", "SUSPECT", "OK"]
    assert len(conflicts) == 2  # one row per distinct variant for A_0001


def test_check_biological_consistency_ignores_agreeing_rows():
    df = pd.DataFrame({
        "inv_id": ["A_0001", "A_0001"],
        "species": ["terrestris", "terrestris"],
        "status": ["OK", "OK"],
        "status_reason": ["", ""],
    })

    result, conflicts = check_biological_consistency(df, ["species"])

    assert list(result["status"]) == ["OK", "OK"]
    assert len(conflicts) == 0


def test_check_biological_consistency_never_downgrades_a_failed_row():
    df = pd.DataFrame({
        "inv_id": ["A_0001", "A_0001"],
        "species": ["terrestris", "lapidarius"],
        "status": ["FAILED", "OK"],
        "status_reason": ["unreadable_image_or_missing_file: boom", ""],
    })

    result, _ = check_biological_consistency(df, ["species"])

    assert list(result["status"]) == ["FAILED", "SUSPECT"]


def test_assign_sequential_photo_ids_derives_missing_fields():
    df = pd.DataFrame({"inv_id": ["A_0001", "A_0001", "A_0002"], "path": ["c.jpg", "a.jpg", "b.jpg"]})

    result = assign_sequential_photo_ids(df, "path", default_device_type="S")

    assert list(result["device_type"]) == ["S", "S", "S"]
    # sorted by path within (inv_id, device_type): a.jpg -> 1, c.jpg -> 2
    a_row = result[result["path"] == "a.jpg"].iloc[0]
    c_row = result[result["path"] == "c.jpg"].iloc[0]
    assert a_row["photo_index"] == 1
    assert c_row["photo_index"] == 2
    assert a_row["photo_id"] == "A_0001_S_1"
    assert c_row["photo_id"] == "A_0001_S_2"


def test_assign_sequential_photo_ids_never_overwrites_an_existing_photo_id():
    df = pd.DataFrame({
        "inv_id": ["A_0001"], "path": ["a.jpg"], "photo_id": ["A_0001_P_7"], "device_type": ["P"],
    })

    result = assign_sequential_photo_ids(df, "path", default_device_type="S")

    assert result["photo_id"].iloc[0] == "A_0001_P_7"
    assert result["device_type"].iloc[0] == "P"


def test_build_biological_data_dedups_and_pivots_photo_counts():
    df = pd.DataFrame({
        "inv_id": ["A_0001", "A_0001", "A_0002"],
        "species": ["terrestris", "terrestris", "lapidarius"],
        "device_type": ["P", "S", "P"],
    })

    result = build_biological_data(df, ["species"])

    assert len(result) == 2
    a1 = result[result["inv_id"] == "A_0001"].iloc[0]
    assert a1["n_photos_P"] == 1
    assert a1["n_photos_S"] == 1


def test_build_manifest_table_keeps_only_present_optional_columns():
    df = pd.DataFrame({
        "photo_id": ["A_0001_S_1"], "inv_id": ["A_0001"], "device_type": ["S"],
        "photo_index": [1], "ext": [".jpg"], "content_hash": ["abc"],
        "file_size_bytes": [100], "path": ["a.jpg"], "status": ["OK"], "status_reason": [""],
        "species": ["terrestris"],  # biological column, should be dropped here
    })

    result = build_manifest_table(df)

    assert "species" not in result.columns
    assert "device" not in result.columns  # not present in input, correctly omitted
    assert list(result.columns) == [
        "photo_id", "inv_id", "device_type", "photo_index", "ext",
        "content_hash", "file_size_bytes", "path", "status", "status_reason",
    ]
