"""Tests for utils.review -- synthetic crops.csv/landmarks_numbered.csv,
no real dataset needed."""
import pandas as pd
import pytest

from utils.review import (
    build_crop_review_df,
    build_landmark_review_df,
    crops_reviewed_path,
    landmarks_reviewed_csv_path,
    read_review_csv,
    review_audit_path,
    write_crop_review,
    write_landmarks_review,
)


def _make_crops_csv(dataset, mode="light"):
    path = dataset / "extraction" / mode / "crops.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"photo_id": "A_P_1", "inv_id": "A", "status": "OK", "error_reason": "",
         "aspect_ratio": 2.0, "output_path": "extraction/light/images/A_P_1.jpg",
         "processing_time_s": 0.1, "processed_at": "2026-01-01T00:00:00"},
        {"photo_id": "A_P_2", "inv_id": "A", "status": "OK", "error_reason": "",
         "aspect_ratio": 2.0, "output_path": "extraction/light/images/A_P_2.jpg",
         "processing_time_s": 0.1, "processed_at": "2026-01-01T00:00:00"},
        {"photo_id": "B_P_1", "inv_id": "B", "status": "FAILED", "error_reason": "normalization_failed",
         "aspect_ratio": "", "output_path": "", "processing_time_s": 0.1, "processed_at": "2026-01-01T00:00:00"},
    ]).to_csv(path, index=False)
    return path


def _make_landmarks_numbered_csv(dataset):
    path = dataset / "landmarks" / "landmarks_numbered.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"photo_id": "A_P_1", "inv_id": "A", "tps_id": 1, "image_path": "x1.jpg", "status": "OK",
         "registration_cost": 0.01, "n_outlier_landmarks": 0, "error_reason": "", "processing_time_s": 0.1},
        {"photo_id": "A_P_2", "inv_id": "A", "tps_id": 2, "image_path": "x2.jpg", "status": "SUSPECT",
         "registration_cost": 0.5, "n_outlier_landmarks": 4, "error_reason": "outlier landmarks", "processing_time_s": 0.1},
    ]).to_csv(path, index=False)
    return path


def test_build_crop_review_df_keeps_latest_row_per_photo_id(tmp_path):
    path = tmp_path / "extraction" / "light" / "crops.csv"
    path.parent.mkdir(parents=True)
    # append-only log: A_P_1 was FAILED then retried OK -- keep the latest.
    pd.DataFrame([
        {"photo_id": "A_P_1", "inv_id": "A", "status": "FAILED", "error_reason": "x",
         "aspect_ratio": "", "output_path": "", "processing_time_s": 0.1, "processed_at": "t1"},
        {"photo_id": "A_P_1", "inv_id": "A", "status": "OK", "error_reason": "",
         "aspect_ratio": 2.0, "output_path": "img.jpg", "processing_time_s": 0.1, "processed_at": "t2"},
    ]).to_csv(path, index=False)

    df = build_crop_review_df(tmp_path, mode="light")
    assert len(df) == 1
    assert df.iloc[0]["auto_status"] == "OK"
    assert df.iloc[0]["reviewed_status"] == "OK"


def test_build_crop_review_df_missing_crops_csv_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_crop_review_df(tmp_path, mode="light")


def test_write_crop_review_overrides_status_and_reason(tmp_path):
    _make_crops_csv(tmp_path)
    df = build_crop_review_df(tmp_path, mode="light")
    df.loc[df["photo_id"] == "A_P_2", "reviewed_status"] = "FAILED"

    out_path, audit_path = write_crop_review(tmp_path, "light", df)
    assert out_path == crops_reviewed_path(tmp_path, "light")
    assert audit_path == review_audit_path(tmp_path, "crops")

    reconciled = pd.read_csv(out_path).set_index("photo_id")
    assert reconciled.loc["A_P_2", "status"] == "FAILED"
    assert reconciled.loc["A_P_2", "error_reason"] == "excluded_by_manual_review"
    # untouched rows keep their original status/reason
    assert reconciled.loc["A_P_1", "status"] == "OK"
    assert reconciled.loc["B_P_1", "status"] == "FAILED"
    assert reconciled.loc["B_P_1", "error_reason"] == "normalization_failed"


def test_build_landmark_review_df_and_write_landmarks_review(tmp_path):
    _make_landmarks_numbered_csv(tmp_path)
    df = build_landmark_review_df(tmp_path)
    assert set(df["photo_id"]) == {"A_P_1", "A_P_2"}

    # human decides the SUSPECT outlier is actually fine
    df.loc[df["photo_id"] == "A_P_2", "reviewed_status"] = "OK"
    out_path, audit_path = write_landmarks_review(tmp_path, df)
    assert out_path == landmarks_reviewed_csv_path(tmp_path)

    reconciled = pd.read_csv(out_path).set_index("photo_id")
    assert reconciled.loc["A_P_2", "status"] == "OK"
    assert "manual review override" in reconciled.loc["A_P_2", "error_reason"]
    assert reconciled.loc["A_P_1", "status"] == "OK"  # untouched


def test_read_review_csv_rejects_invalid_status(tmp_path):
    path = tmp_path / "bad_review.csv"
    pd.DataFrame([{"photo_id": "A", "auto_status": "OK", "reviewed_status": "MAYBE"}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="invalid reviewed_status"):
        read_review_csv(path)


def test_read_review_csv_rejects_missing_columns(tmp_path):
    path = tmp_path / "bad_review.csv"
    pd.DataFrame([{"photo_id": "A", "reviewed_status": "OK"}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing column"):
        read_review_csv(path)


def test_write_crop_review_when_every_error_reason_is_empty(tmp_path):
    """Regression: when every row's error_reason is empty, pandas infers
    the column as float64 (NaN) rather than object/string on read --
    assigning a string override used to raise a pandas TypeError."""
    path = tmp_path / "extraction" / "light" / "crops.csv"
    path.parent.mkdir(parents=True)
    pd.DataFrame([
        {"photo_id": "A_P_1", "inv_id": "A", "status": "OK", "error_reason": "",
         "aspect_ratio": 2.0, "output_path": "img.jpg", "processing_time_s": 0.1, "processed_at": "t"},
    ]).to_csv(path, index=False)

    df = build_crop_review_df(tmp_path, mode="light")
    df.loc[df["photo_id"] == "A_P_1", "reviewed_status"] = "FAILED"

    out_path, _audit = write_crop_review(tmp_path, "light", df)
    reconciled = pd.read_csv(out_path).set_index("photo_id")
    assert reconciled.loc["A_P_1", "error_reason"] == "excluded_by_manual_review"


def test_write_landmarks_review_when_every_error_reason_is_empty(tmp_path):
    path = tmp_path / "landmarks" / "landmarks_numbered.csv"
    path.parent.mkdir(parents=True)
    pd.DataFrame([
        {"photo_id": "A_P_1", "inv_id": "A", "tps_id": 1, "image_path": "x1.jpg", "status": "OK",
         "registration_cost": 0.01, "n_outlier_landmarks": 0, "error_reason": "", "processing_time_s": 0.1},
    ]).to_csv(path, index=False)

    df = build_landmark_review_df(tmp_path)
    df.loc[df["photo_id"] == "A_P_1", "reviewed_status"] = "FAILED"

    out_path, _audit = write_landmarks_review(tmp_path, df)
    reconciled = pd.read_csv(out_path).set_index("photo_id")
    assert "manual review override" in reconciled.loc["A_P_1", "error_reason"]


def test_read_review_csv_roundtrips_with_write_crop_review(tmp_path):
    _make_crops_csv(tmp_path)
    df = build_crop_review_df(tmp_path, mode="light")
    df.loc[df["photo_id"] == "A_P_1", "reviewed_status"] = "SUSPECT"
    _out_path, audit_path = write_crop_review(tmp_path, "light", df)

    reloaded = read_review_csv(audit_path)
    out_path2, _audit2 = write_crop_review(tmp_path, "light", reloaded)
    reconciled = pd.read_csv(out_path2).set_index("photo_id")
    assert reconciled.loc["A_P_1", "status"] == "SUSPECT"
