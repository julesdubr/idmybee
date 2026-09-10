"""tests/test_prepare_dataset.py
Synthetic tests for tools.ingestion.prepare_dataset's ORCHESTRATION logic
(which script gets called with which argv, in what order, and what happens
when a source's build_manifest fails) -- the wrapped scripts
(ingest_raw.run_for_folder/export_clean_dataset/build_manifest/
combine_manifests) are monkeypatched, each already covered on its own
elsewhere.
"""
from __future__ import annotations

import json

import pytest

from tools.ingestion import prepare_dataset as pd


def _write_config(tmp_path, config: dict) -> str:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return str(path)


def _patch_ingest(monkeypatch, tmp_path, *, reused: bool = False):
    calls = []

    def fake_run_for_folder(images_dir, source_type, *, photographer_subfolder=False, naming=None):
        calls.append((images_dir, source_type, photographer_subfolder, naming))
        ingest_dir = tmp_path / "raw" / "ingest"
        ingest_dir.mkdir(parents=True, exist_ok=True)
        manifest = ingest_dir / "manifest.csv"
        manifest.write_text("")
        return manifest, ingest_dir, reused

    monkeypatch.setattr(pd.ingest_raw, "run_for_folder", fake_run_for_folder)
    return calls


def test_source_runs_scan_then_export_then_build_manifest(tmp_path, monkeypatch):
    calls = []
    ingest_calls = _patch_ingest(monkeypatch, tmp_path)

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

    config = _write_config(tmp_path, {
        "mapping_file": str(tmp_path / "mapping.csv"),
        "output_dir": str(tmp_path / "combined"),
        "sources": [{
            "images_dir": str(tmp_path / "raw" / "collection"),
            "identification_csv": str(tmp_path / "ident.csv"),
            "source_type": "collection",
            "key_column": "inv_id",
            "output_dir": str(tmp_path / "clean" / "collection"),
        }],
    })

    pd.main([config])

    assert ingest_calls == [(str(tmp_path / "raw" / "collection"), "collection", False, None)]
    kinds = [c[0] for c in calls]
    assert kinds == ["export_clean_dataset", "build_manifest", "combine_manifests"]
    export_argv = calls[0][1]
    assert "--identification-csv" in export_argv
    assert "--mapping-file" in export_argv and str(tmp_path / "mapping.csv") in export_argv
    build_argv = calls[1][1]
    assert build_argv[0] == str(tmp_path / "clean" / "collection" / "dataset.csv")


def test_missing_output_dir_raises(tmp_path, monkeypatch):
    config = _write_config(tmp_path, {
        "sources": [{
            "images_dir": str(tmp_path / "raw" / "collection"),
            "identification_csv": str(tmp_path / "ident.csv"), "source_type": "collection", "key_column": "inv_id",
            "output_dir": str(tmp_path / "clean"),
        }],
    })
    with pytest.raises(SystemExit, match="output_dir"):
        pd.main([config])


def test_omitted_mapping_file_defaults_inside_output_dir(tmp_path, monkeypatch):
    """No 'mapping_file' in the config -- it must default to
    <output_dir>/inv_id_mapping.csv (the combined dataset root this run
    itself produces), not a separate shared location."""
    _patch_ingest(monkeypatch, tmp_path)
    export_calls = []
    monkeypatch.setattr(pd.export_clean_dataset, "main", lambda argv: (
        export_calls.append(argv),
        (tmp_path / "clean").mkdir(parents=True, exist_ok=True),
        (tmp_path / "clean" / "dataset.csv").write_text(""),
    ))
    monkeypatch.setattr(pd.build_manifest, "main", lambda argv: (
        (tmp_path / "clean" / "manifest.csv").write_text(""),
    ))
    monkeypatch.setattr(pd.combine_manifests, "main", lambda argv: None)

    output_dir = tmp_path / "combined"
    config = _write_config(tmp_path, {
        "output_dir": str(output_dir),
        "sources": [{
            "images_dir": str(tmp_path / "raw" / "collection"),
            "identification_csv": str(tmp_path / "ident.csv"), "source_type": "collection", "key_column": "inv_id",
            "output_dir": str(tmp_path / "clean"),
        }],
    })

    pd.main([config])

    export_argv = export_calls[0]
    assert "--mapping-file" in export_argv
    mapping_arg = export_argv[export_argv.index("--mapping-file") + 1]
    assert mapping_arg == str(output_dir / "inv_id_mapping.csv")


def test_failed_source_excluded_from_combine(tmp_path, monkeypatch):
    """build_manifest.py writes manifest_raw.csv (not manifest.csv) on
    failure -- that source must be skipped, not fed to combine_manifests."""
    _patch_ingest(monkeypatch, tmp_path)
    monkeypatch.setattr(pd.export_clean_dataset, "main", lambda argv: (
        (tmp_path / "clean").mkdir(parents=True, exist_ok=True),
        (tmp_path / "clean" / "dataset.csv").write_text(""),
    ))

    def fake_build_manifest(argv):
        out_dir = tmp_path / "clean"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "manifest_raw.csv").write_text("")  # no manifest.csv written

    monkeypatch.setattr(pd.build_manifest, "main", fake_build_manifest)
    monkeypatch.setattr(pd.combine_manifests, "main", lambda argv: pytest.fail("should not be called"))

    config = _write_config(tmp_path, {
        "mapping_file": str(tmp_path / "mapping.csv"),
        "output_dir": str(tmp_path / "combined"),
        "sources": [{
            "images_dir": str(tmp_path / "raw" / "bad"),
            "identification_csv": str(tmp_path / "ident.csv"), "source_type": "collection", "key_column": "inv_id",
            "output_dir": str(tmp_path / "clean"),
        }],
    })

    with pytest.raises(SystemExit, match="No source produced"):
        pd.main([config])


def test_kept_raw_manifest_is_not_deleted(tmp_path, monkeypatch):
    rmtree_calls = []
    monkeypatch.setattr(pd.shutil, "rmtree", lambda *a, **k: rmtree_calls.append(a))

    _patch_ingest(monkeypatch, tmp_path)
    monkeypatch.setattr(pd.export_clean_dataset, "main", lambda argv: (
        (tmp_path / "clean").mkdir(parents=True, exist_ok=True),
        (tmp_path / "clean" / "dataset.csv").write_text(""),
    ))
    monkeypatch.setattr(pd.build_manifest, "main", lambda argv: (tmp_path / "clean" / "manifest.csv").write_text(""))
    monkeypatch.setattr(pd.combine_manifests, "main", lambda argv: None)

    config = _write_config(tmp_path, {
        "mapping_file": str(tmp_path / "mapping.csv"),
        "output_dir": str(tmp_path / "combined"),
        "sources": [{
            "images_dir": str(tmp_path / "raw" / "collection"),
            "identification_csv": str(tmp_path / "ident.csv"), "source_type": "collection", "key_column": "inv_id",
            "output_dir": str(tmp_path / "clean"), "keep_raw_manifest": True,
        }],
    })

    pd.main([config])

    assert rmtree_calls == []


def test_unkept_raw_manifest_is_deleted(tmp_path, monkeypatch):
    rmtree_calls = []
    monkeypatch.setattr(pd.shutil, "rmtree", lambda *a, **k: rmtree_calls.append(a))

    _patch_ingest(monkeypatch, tmp_path)
    monkeypatch.setattr(pd.export_clean_dataset, "main", lambda argv: (
        (tmp_path / "clean").mkdir(parents=True, exist_ok=True),
        (tmp_path / "clean" / "dataset.csv").write_text(""),
    ))
    monkeypatch.setattr(pd.build_manifest, "main", lambda argv: (tmp_path / "clean" / "manifest.csv").write_text(""))
    monkeypatch.setattr(pd.combine_manifests, "main", lambda argv: None)

    config = _write_config(tmp_path, {
        "mapping_file": str(tmp_path / "mapping.csv"),
        "output_dir": str(tmp_path / "combined"),
        "sources": [{
            "images_dir": str(tmp_path / "raw" / "collection"),
            "identification_csv": str(tmp_path / "ident.csv"), "source_type": "collection", "key_column": "inv_id",
            "output_dir": str(tmp_path / "clean"),
        }],
    })

    pd.main([config])

    assert len(rmtree_calls) == 1
