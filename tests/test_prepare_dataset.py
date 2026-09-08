"""tests/test_prepare_dataset.py
Synthetic tests for tools.ingestion.prepare_dataset's ORCHESTRATION logic
(which script gets called with which argv, in what order, and what happens
when a source's build_manifest fails) -- the four wrapped scripts
(ingest_raw/export_clean_dataset/build_manifest/combine_manifests) are
monkeypatched, each already covered on its own elsewhere.
"""
from __future__ import annotations

import json

import pytest

from tools.ingestion import prepare_dataset as pd


def _write_config(tmp_path, config: dict) -> str:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return str(path)


def test_compliant_source_skips_ingest_and_export(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(pd.build_manifest, "main", lambda argv: (calls.append(("build_manifest", argv)), (
        (tmp_path / "clean" / "manifest.csv").parent.mkdir(parents=True, exist_ok=True),
        (tmp_path / "clean" / "manifest.csv").write_text(""),
    ))[0])
    monkeypatch.setattr(pd.combine_manifests, "main", lambda argv: calls.append(("combine_manifests", argv)))
    monkeypatch.setattr(pd.export_clean_dataset, "main", lambda argv: pytest.fail("should not be called"))
    monkeypatch.setattr(pd.ingest_raw, "main", lambda argv: pytest.fail("should not be called"))

    config = _write_config(tmp_path, {
        "output_dir": str(tmp_path / "combined"),
        "sources": [{
            "type": "compliant",
            "dataset_csv": str(tmp_path / "third_party" / "dataset.csv"),
            "manifest_output_dir": str(tmp_path / "clean"),
        }],
    })

    pd.main([config])

    kinds = [c[0] for c in calls]
    assert kinds == ["build_manifest", "combine_manifests"]
    build_argv = calls[0][1]
    assert build_argv[0] == str(tmp_path / "third_party" / "dataset.csv")
    assert "--output-dir" in build_argv and str(tmp_path / "clean") in build_argv
    combine_argv = calls[1][1]
    assert str(tmp_path / "clean") in combine_argv
    assert "--output-dir" in combine_argv and str(tmp_path / "combined") in combine_argv


def test_raw_source_runs_export_then_build_manifest(tmp_path, monkeypatch):
    calls = []

    def fake_export(argv):
        calls.append(("export_clean_dataset", argv))
        out_dir = tmp_path / "clean" / "collection"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "dataset.csv").write_text("")

    def fake_build_manifest(argv):
        calls.append(("build_manifest", argv))
        out_dir = tmp_path / "clean" / "collection"
        (out_dir / "manifest.csv").write_text("")

    monkeypatch.setattr(pd.export_clean_dataset, "main", fake_export)
    monkeypatch.setattr(pd.build_manifest, "main", fake_build_manifest)
    monkeypatch.setattr(pd.combine_manifests, "main", lambda argv: calls.append(("combine_manifests", argv)))

    raw_manifest = tmp_path / "raw" / "manifest.csv"
    raw_manifest.parent.mkdir(parents=True)
    raw_manifest.write_text("")

    config = _write_config(tmp_path, {
        "mapping_file": str(tmp_path / "mapping.csv"),
        "output_dir": str(tmp_path / "combined"),
        "sources": [{
            "type": "raw",
            "source_type": "collection",
            "raw_manifest": str(raw_manifest),
            "identification_csv": str(tmp_path / "ident.csv"),
            "key_column": "inv_id",
            "clean_output_dir": str(tmp_path / "clean" / "collection"),
            "manifest_output_dir": str(tmp_path / "clean" / "collection"),
        }],
    })

    pd.main([config])

    kinds = [c[0] for c in calls]
    assert kinds == ["export_clean_dataset", "build_manifest", "combine_manifests"]
    export_argv = calls[0][1]
    assert export_argv[0] == str(raw_manifest)
    assert "--identification-csv" in export_argv
    assert "--mapping-file" in export_argv and str(tmp_path / "mapping.csv") in export_argv
    build_argv = calls[1][1]
    assert build_argv[0] == str(tmp_path / "clean" / "collection" / "dataset.csv")


def test_raw_source_without_mapping_file_raises(tmp_path, monkeypatch):
    config = _write_config(tmp_path, {
        "output_dir": str(tmp_path / "combined"),
        "sources": [{
            "type": "raw", "source_type": "collection",
            "raw_manifest": str(tmp_path / "raw.csv"),
            "identification_csv": str(tmp_path / "ident.csv"), "key_column": "inv_id",
            "clean_output_dir": str(tmp_path / "clean"), "manifest_output_dir": str(tmp_path / "clean"),
        }],
    })
    with pytest.raises(SystemExit, match="mapping_file"):
        pd.main([config])


def test_failed_source_excluded_from_combine(tmp_path, monkeypatch):
    """build_manifest.py writes manifest_raw.csv (not manifest.csv) on
    failure -- that source must be skipped, not fed to combine_manifests."""
    calls = []

    def fake_build_manifest(argv):
        calls.append(("build_manifest", argv))
        out_dir = tmp_path / "bad"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "manifest_raw.csv").write_text("")  # no manifest.csv written

    monkeypatch.setattr(pd.build_manifest, "main", fake_build_manifest)
    monkeypatch.setattr(pd.combine_manifests, "main", lambda argv: pytest.fail("should not be called"))

    config = _write_config(tmp_path, {
        "output_dir": str(tmp_path / "combined"),
        "sources": [{
            "type": "compliant",
            "dataset_csv": str(tmp_path / "bad" / "dataset.csv"),
            "manifest_output_dir": str(tmp_path / "bad"),
        }],
    })

    with pytest.raises(SystemExit, match="No source produced"):
        pd.main([config])


def test_skip_ingest_reuses_existing_manifest_without_rescanning(tmp_path, monkeypatch):
    monkeypatch.setattr(pd.ingest_raw, "main", lambda argv: pytest.fail("should not rescan with --skip-ingest"))

    calls = []

    def fake_export(argv):
        calls.append(argv)
        out_dir = tmp_path / "clean"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "dataset.csv").write_text("")

    monkeypatch.setattr(pd.export_clean_dataset, "main", fake_export)
    monkeypatch.setattr(pd.build_manifest, "main", lambda argv: (tmp_path / "clean" / "manifest.csv").write_text(""))
    monkeypatch.setattr(pd.combine_manifests, "main", lambda argv: None)

    config = _write_config(tmp_path, {
        "ingest": {"roots_json": "config/roots.json", "name": "bombus_raw", "out_dir": str(tmp_path / "data")},
        "mapping_file": str(tmp_path / "mapping.csv"),
        "output_dir": str(tmp_path / "combined"),
        "sources": [{
            "type": "raw", "source_type": "collection",
            "identification_csv": str(tmp_path / "ident.csv"), "key_column": "inv_id",
            "clean_output_dir": str(tmp_path / "clean"), "manifest_output_dir": str(tmp_path / "clean"),
        }],
    })

    pd.main([config, "--skip-ingest"])

    expected_raw_manifest = str(tmp_path / "data" / "bombus_raw" / "manifest.csv")
    assert calls[0][0] == expected_raw_manifest
