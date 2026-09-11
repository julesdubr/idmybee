"""Tests for core.dataset_config."""
import pandas as pd

from core.dataset_config import (
    read_dataset_config,
    resolve_dataset_name,
    sanitize_dataset_name,
    write_dataset_config,
)


def _write_dataset(root, n_photos=3, n_specimens=2):
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"photo_id": [f"p{i}" for i in range(n_photos)]}).to_csv(root / "manifest.csv", index=False)
    pd.DataFrame({"inv_id": [f"inv{i}" for i in range(n_specimens)]}).to_csv(root / "biological_data.csv", index=False)


def test_sanitize_dataset_name_replaces_path_separators():
    assert sanitize_dataset_name("Bombus/collection") == "Bombus_collection"


def test_sanitize_dataset_name_strips_whitespace():
    assert sanitize_dataset_name("  collection  ") == "collection"


def test_resolve_dataset_name_falls_back_to_folder_name(tmp_path):
    root = tmp_path / "collection"
    root.mkdir()
    assert resolve_dataset_name(root) == "collection"


def test_resolve_dataset_name_reads_config_when_present(tmp_path):
    root = tmp_path / "collection"
    _write_dataset(root)
    write_dataset_config(root, "Bombus/collection")
    assert resolve_dataset_name(root) == "Bombus_collection"


def test_write_dataset_config_counts_photos_and_specimens(tmp_path):
    root = tmp_path / "collection"
    _write_dataset(root, n_photos=3, n_specimens=2)
    write_dataset_config(root, "collection")
    config = read_dataset_config(root)
    assert config["n_photos"] == 3
    assert config["n_specimens"] == 2


def test_read_dataset_config_none_when_missing(tmp_path):
    root = tmp_path / "no_config_yet"
    root.mkdir()
    assert read_dataset_config(root) is None
