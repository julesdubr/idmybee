"""prepare_dataset.py
Step 0 orchestrator: chains raw ingestion through to one combined, clean
dataset root -- the gap flagged since PIPELINE.md's first version ("no
orchestrator covers stage 1"). Driven by a single JSON config rather than a
long flag list, because stage 1 genuinely has one config block per source
(collection, terrain, ...), each with its own identification CSV/columns --
see "Config file" below. Wraps the exact same tools already documented in
PIPELINE.md section 1, in-process (each stage's own main(argv), no
subprocess, no duplicated logic -- same pattern as utils.landmarking_pipeline
for stages 2-6):

    tools.ingestion.ingest_raw.run_for_folder  (once per source -- scans that
                                                 source's own images folder)
    tools.ingestion.export_clean_dataset       (once per source -- identity
                                                 resolution against the shared
                                                 mapping_file)
    tools.ingestion.build_manifest             (once per source)
    tools.ingestion.combine_manifests          (once, over however many
                                                 sources -- a single source
                                                 works too)

A source is exactly what a person starting from raw data actually has: a
folder of photos (whatever filename convention) plus an identification CSV
with the biological data (species, caste, ...) and a column matching each
photo's parsed id -- no separate "roots" config, and no hand-prepared
per-photo CSV to build first. The raw scan of a source's images folder is
a throwaway byproduct of preparing THAT source (see run_for_folder's
docstring): it's written inside the images folder itself and deleted
afterwards unless the source sets "keep_raw_manifest": true.

Config file (JSON):
{
  "mapping_file": null,                  // optional -- frozen original_id -> inv_id,
                                          // shared across sources to keep inv_id unique
                                          // between them. Defaults to
                                          // <output_dir>/inv_id_mapping.csv (read-modify-
                                          // written across runs) -- pass an explicit path
                                          // only to share one mapping across several
                                          // separately-prepared dataset roots.
  "output_dir": "data/Bombus/collection",  // final combined manifest.csv/biological_data.csv
  "dataset_name": "Bombus_collection",   // optional (default: Path(output_dir).name) -- human-facing
                                          // name written to dataset_config.json alongside n_photos/
                                          // n_specimens (see core.dataset_config.write_dataset_config).
                                          // Sanitized (no "/", it becomes a folder name) -- read back by
                                          // classifiers.train/predict to label runs/<family>/<run_id>/
                                          // {train,predict}/<dataset_name>/ instead of the raw folder
                                          // basename.
  "sources": [
    {
      "images_dir": "/Volumes/EXT DATA/IDMB/images/Bombus/collection",
      "identification_csv": "/Volumes/EXT DATA/IDMB/IDMB_Bombus_collect.csv",
      "source_type": "collection",
      "key_column": "inv_id",
      "output_dir": "data/clean/collection",  // this source's own clean
                                               // dataset.csv + manifest.csv
      "photographer_subfolder": false,    // optional (default: false)
      "device_column": "device_type",     // optional
      "device_name_column": "device",     // optional (default: "device")
      "compare_columns": null,            // optional, comma-separated (default:
                                           // manifest.identification.DEFAULT_COMPARE_COLUMNS)
      "origin_codes": "data/clean/collection/collection_origin_codes.csv",  // optional --
                                           // origin -> inv_name lookup, see manifest.origin_table.
                                           // Typically written by app/setup_dataset.py's
                                           // interactive table rather than hand-authored.
      "origin_column": "collection_origin",   // optional (default: "collection_origin")
      "image_group_by": "genus,species,caste",  // optional, comma-separated
      "default_device_type": "S",         // optional (default: "S")
      "no_copy_images": false,            // optional
      "keep_raw_manifest": false          // optional -- keep <images_dir>/.idmybee_ingest/
                                           // instead of deleting it once this source is
                                           // prepared; a kept scan is reused (not rescanned)
                                           // next run
    }
  ]
}

A source's own output_dir is kept (not a throwaway temp dir) -- useful on
its own for inspecting one source in isolation, same spirit as
tools.ingestion.export_clean_dataset's per-source outputs.

Usage:
    python -m tools.ingestion.prepare_dataset config/prepare_collection.json
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

from tools.ingestion import build_manifest, combine_manifests, export_clean_dataset, ingest_raw
from utils.cli import add_logging_args, log_level_from_args, verbosity_argv
from core.dataset_config import write_dataset_config
from core.run_io import setup_console_logging

logger = logging.getLogger(__name__)


def _optional_flags(source: dict) -> list[str]:
    """export_clean_dataset.py flags that only apply to a 'raw' source and
    have a real default in that script's own parser -- only pass them
    through when the config actually sets them, so export_clean_dataset's
    own defaults stay the single source of truth."""
    argv: list[str] = []
    if source.get("device_column"):
        argv += ["--device-column", source["device_column"]]
    if source.get("device_name_column"):
        argv += ["--device-name-column", source["device_name_column"]]
    if source.get("compare_columns"):
        argv += ["--compare-columns", source["compare_columns"]]
    if source.get("origin_codes"):
        argv += ["--origin-codes", source["origin_codes"]]
    if source.get("origin_column"):
        argv += ["--origin-column", source["origin_column"]]
    if source.get("image_group_by"):
        argv += ["--image-group-by", source["image_group_by"]]
    if source.get("default_device_type"):
        argv += ["--default-device-type", source["default_device_type"]]
    if source.get("no_copy_images"):
        argv += ["--no-copy-images"]
    return argv


def run_source(source: dict, mapping_file: str, verbosity: list[str]) -> Path:
    """Scans this source's own images folder, resolves identity against
    `mapping_file`, then builds its canonical manifest -- see module
    docstring. Returns the source's own output_dir -- caller checks
    whether manifest.csv actually landed there (success) or only
    manifest_raw.csv (failure, see build_manifest.py)."""
    output_dir = Path(source["output_dir"])

    print(f"\n=== Step 1a: raw scan ({source['images_dir']}) ===")
    raw_manifest, ingest_dir, reused = ingest_raw.run_for_folder(
        source["images_dir"], source["source_type"],
        photographer_subfolder=source.get("photographer_subfolder", False),
    )
    print(f"{'reusing kept scan' if reused else 'scanned'}: {raw_manifest}")

    print(f"\n=== Step 1b: clean export ({source['source_type']}) ===")
    export_clean_dataset.main([
        str(raw_manifest),
        "--identification-csv", source["identification_csv"],
        "--source-type", source["source_type"],
        "--key-column", source["key_column"],
        "--mapping-file", mapping_file,
        "--output-dir", str(output_dir),
        *_optional_flags(source), *verbosity,
    ])

    if not source.get("keep_raw_manifest"):
        shutil.rmtree(ingest_dir, ignore_errors=True)

    print(f"\n=== Step 1c: build manifest ({output_dir}) ===")
    build_manifest.main([str(output_dir / "dataset.csv"), "--output-dir", str(output_dir), *verbosity])

    return output_dir


def run(config: dict, verbosity: list[str] | None = None) -> dict:
    """Runs the config exactly as `main()` does, but takes an already-
    parsed dict rather than a JSON file path -- the in-process entry point
    for a caller that already has the config in memory (e.g.
    app/setup_dataset.py's dataset-prep wizard), same pattern as every
    other stage here already offers both a CLI main(argv) and a direct
    Python call. Returns {"output_dir", "manifest_dirs", "failed_sources", "dataset_config"}.
    """
    verbosity = verbosity or []
    sources = config["sources"]
    if not sources:
        raise SystemExit("config: 'sources' is empty -- nothing to do.")
    if not config.get("output_dir"):
        raise SystemExit("config: 'output_dir' is required.")

    # Defaults to living inside the combined dataset root this run itself
    # produces (read-modify-written by export_clean_dataset.py across
    # sources/runs, same as before) -- no separate, shared external file to
    # point at.
    mapping_file = config.get("mapping_file") or str(Path(config["output_dir"]) / "inv_id_mapping.csv")

    manifest_dirs: list[Path] = []
    failed_sources: list[str] = []
    for source in sources:
        label = source.get("source_type", "?")
        manifest_dir = run_source(source, mapping_file, verbosity)
        if (manifest_dir / "manifest.csv").exists():
            manifest_dirs.append(manifest_dir)
        else:
            failed_sources.append(label)
            print(f"\n{label}: build_manifest.py did not produce manifest.csv -- see {manifest_dir / 'manifest_raw.csv'}")

    if not manifest_dirs:
        raise SystemExit("\nNo source produced a valid manifest.csv -- nothing to combine, aborting.")

    print(f"\n=== Step 1d: combine {len(manifest_dirs)} source(s) -> {config['output_dir']} ===")
    combine_manifests.main([*(str(d) for d in manifest_dirs), "--output-dir", config["output_dir"], *verbosity])

    dataset_name = config.get("dataset_name") or Path(config["output_dir"]).name
    config_path = write_dataset_config(config["output_dir"], dataset_name)

    print(f"\nDone. Clean dataset -> {config['output_dir']}")
    print(f"Dataset config -> {config_path} (dataset_name={dataset_name!r})")
    if failed_sources:
        print(f"Skipped ({len(failed_sources)}, see manifest_raw.csv above): {failed_sources}")
    print(
        f"Next: python -m tools.pipeline.train_dataset {config['output_dir']} "
        "--unet-model <weights.pt>   (see PIPELINE.md scenario 1)"
    )
    return {
        "output_dir": config["output_dir"], "manifest_dirs": manifest_dirs,
        "failed_sources": failed_sources, "dataset_config": str(config_path),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="JSON config file (see module docstring).")
    add_logging_args(parser)
    args = parser.parse_args(argv)
    setup_console_logging(log_level_from_args(args))

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    run(config, verbosity=verbosity_argv(args))


if __name__ == "__main__":
    main()
