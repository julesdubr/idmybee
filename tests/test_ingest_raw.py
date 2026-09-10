"""tests/test_ingest_raw.py
Covers run_for_folder() -- the per-source, no-roots-config entry point
tools.ingestion.prepare_dataset uses (one images folder in, its own hidden
manifest scan out, reused instead of rescanned if kept from an earlier
run). The multi-root run()/main() CLI path is exercised manually (see
their own docstrings), not here.
"""
from __future__ import annotations

from tools.ingestion import ingest_raw


def _make_photo(images_dir, name: str) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    (images_dir / name).write_bytes(b"fake-image-bytes")


def test_run_for_folder_scans_and_writes_hidden_manifest(tmp_path):
    images_dir = tmp_path / "collection"
    _make_photo(images_dir, "42_S1.jpg")
    _make_photo(images_dir, "42_S2.jpg")

    manifest_path, ingest_dir, reused = ingest_raw.run_for_folder(str(images_dir), "collection")

    assert reused is False
    assert ingest_dir == images_dir / ".idmybee_ingest"
    assert manifest_path == ingest_dir / "manifest.csv"
    rows = manifest_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 3  # header + 2 photos


def test_run_for_folder_reuses_a_kept_manifest(tmp_path):
    images_dir = tmp_path / "collection"
    _make_photo(images_dir, "42_S1.jpg")

    manifest_path, ingest_dir, reused = ingest_raw.run_for_folder(str(images_dir), "collection")
    assert reused is False

    # simulate a second photo dropped in after the first scan -- a kept
    # manifest must NOT pick it up, proving it's reused rather than rescanned.
    _make_photo(images_dir, "43_S1.jpg")
    manifest_path_2, ingest_dir_2, reused_2 = ingest_raw.run_for_folder(str(images_dir), "collection")

    assert reused_2 is True
    assert manifest_path_2 == manifest_path
    assert ingest_dir_2 == ingest_dir
    rows = manifest_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 2  # header + only the first photo
