"""Tests for core.dataset."""
import csv

import numpy as np

from core.tps_io import ImageLandmarks, write_tps
from core.dataset import load_dataset


def _write_csv(path, fieldnames, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_dataset(root, with_device_name=True, with_second_photo=False):
    (root / "landmarks").mkdir(parents=True)

    specimens = [
        ImageLandmarks(
            n_points=2, landmarks=np.array([[0.0, 0.0], [1.0, 1.0]]),
            image_path="crops/ARLY_0001_P_1.jpg", tps_id=1,
            photo_id="ARLY_0001_P_1", inv_id="ARLY_0001",
        ),
        ImageLandmarks(
            n_points=2, landmarks=np.array([[2.0, 2.0], [3.0, 3.0]]),
            image_path="crops/ARLY_0002_S_1.jpg", tps_id=2,
            photo_id="ARLY_0002_S_1", inv_id="ARLY_0002",
        ),
    ]
    if with_second_photo:
        specimens.append(ImageLandmarks(
            n_points=2, landmarks=np.array([[4.0, 4.0], [5.0, 5.0]]),
            image_path="crops/ARLY_0001_S_1.jpg", tps_id=3,
            photo_id="ARLY_0001_S_1", inv_id="ARLY_0001",
        ))
    write_tps(root / "landmarks" / "landmarks_numbered.tps", specimens)

    _write_csv(root / "biological_data.csv", ["inv_id", "species", "caste"], [
        {"inv_id": "ARLY_0001", "species": "terrestris", "caste": "worker"},
        {"inv_id": "ARLY_0002", "species": "lapidarius", "caste": "queen"},
    ])

    manifest_fields = ["photo_id", "inv_id", "device_type", "photo_index", "status"]
    manifest_rows = [
        {"photo_id": "ARLY_0001_P_1", "inv_id": "ARLY_0001", "device_type": "P", "photo_index": "1", "status": "OK"},
        {"photo_id": "ARLY_0002_S_1", "inv_id": "ARLY_0002", "device_type": "S", "photo_index": "1", "status": "OK"},
    ]
    if with_second_photo:
        manifest_rows.append(
            {"photo_id": "ARLY_0001_S_1", "inv_id": "ARLY_0001", "device_type": "S", "photo_index": "1", "status": "OK"},
        )
    if with_device_name:
        manifest_fields.append("device")
        for row, device in zip(manifest_rows, ["Canon EOS 70D - 40mm", "Smartphone Samsung s10", "Smartphone Samsung s10"]):
            row["device"] = device
    _write_csv(root / "manifest.csv", manifest_fields, manifest_rows)

    return specimens


def test_load_dataset_joins_via_photo_id_and_inv_id(tmp_path):
    _make_dataset(tmp_path)

    specimens, meta_df = load_dataset(tmp_path)

    assert len(specimens) == 2
    assert set(meta_df["inv_id"]) == {"ARLY_0001", "ARLY_0002"}
    row1 = meta_df[meta_df["inv_id"] == "ARLY_0001"].iloc[0]
    assert row1["species"] == "terrestris"
    assert row1["caste"] == "worker"
    assert row1["device_type"] == "P"
    assert row1["device"] == "Canon EOS 70D - 40mm"
    assert row1["device_tag"] == "P1"


def test_load_dataset_device_type_and_device_are_distinct_columns(tmp_path):
    # Regression: device_type (coarse code) and device (real camera/phone
    # name) must not collide -- a device_type can cover several devices.
    _make_dataset(tmp_path)

    _, meta_df = load_dataset(tmp_path)

    row2 = meta_df[meta_df["inv_id"] == "ARLY_0002"].iloc[0]
    assert row2["device_type"] == "S"
    assert row2["device"] == "Smartphone Samsung s10"


def test_load_dataset_without_device_column_leaves_device_missing(tmp_path):
    _make_dataset(tmp_path, with_device_name=False)

    _, meta_df = load_dataset(tmp_path)

    assert "device_type" in meta_df.columns
    assert meta_df["device"].isna().all()


def test_load_dataset_multiple_photos_per_specimen(tmp_path):
    _make_dataset(tmp_path, with_second_photo=True)

    specimens, meta_df = load_dataset(tmp_path)

    assert len(specimens) == 3  # one row per PHOTO, not per specimen
    assert (meta_df["inv_id"] == "ARLY_0001").sum() == 2
