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

    tools.ingestion.ingest_raw            (once, optional -- only if raw data
                                            isn't already a per-photo CSV)
    tools.ingestion.export_clean_dataset  (once per "raw" source, optional --
                                            only if that source needs identity
                                            resolution)
    tools.ingestion.build_manifest        (once per source, always)
    tools.ingestion.combine_manifests     (once, over however many sources --
                                            works with a single source too)

Config file (JSON):
{
  // "ingest": omit entirely if every source is already a per-photo CSV, or
  // if ingest_raw was already run by hand. Otherwise, one of:
  //   "roots": {...}       -- the roots config inline (base_root/roots,
  //                            see tools.ingestion.ingest_raw's docstring
  //                            for the shape) -- one JSON, end to end.
  //   "roots_json": "..."  -- path to a separate roots file instead, for
  //                            the few roots configs actually reused
  //                            across several prepare runs (e.g. one per
  //                            external drive -- see config/roots_*.json).
  "ingest": {
    "roots": {
      "base_root": {"darwin": "/Volumes/EXT DATA/"},
      "roots": [
        {"path": "IDMB/images/Bombus/collection", "source_type": "collection"}
      ]
    },
    "name": "bombus_raw",
    "out_dir": "data"                       // default: "data"
  },
  "mapping_file": "data/identification/inv_id_mapping.csv",  // required iff
                                                               // any source is "raw"
  "output_dir": "data/Bombus/collection",  // final combined manifest.csv/biological_data.csv
  "sources": [
    {
      "type": "raw",                      // needs identity resolution -- see
                                           // tools.ingestion.export_clean_dataset
      "source_type": "collection",
      "raw_manifest": null,               // path to a raw manifest.csv -- omit to use
                                           // <ingest.out_dir>/<ingest.name>/manifest.csv
      "identification_csv": "data/identification/IDMB_Bombus_collect.csv",
      "key_column": "inv_id",
      "device_column": "device_type",     // optional
      "device_name_column": "device",     // optional (default: "device")
      "compare_columns": null,            // optional, comma-separated (default:
                                           // manifest.identification.DEFAULT_COMPARE_COLUMNS)
      "origin_codes": "data/identification/collection_origin_codes.csv",  // optional
      "origin_column": "collection_origin",   // optional (default: "collection_origin")
      "image_group_by": "genus,species,caste",  // optional, comma-separated
      "no_copy_images": false,
      "clean_output_dir": "data/clean/collection",
      "manifest_output_dir": "data/clean/collection",
      "path_column": "path",              // build_manifest's --path-column (default: "path")
      "default_device_type": "S"          // build_manifest's --default-device-type
    },
    {
      "type": "compliant",                // already a compliant per-photo CSV --
                                           // skip identity resolution entirely
      "dataset_csv": "data/some_third_party_dataset/dataset.csv",
      "manifest_output_dir": "data/clean/third_party",
      "path_column": "path",
      "base_dir": null,
      "default_device_type": "S"
    }
  ]
}

A source's own manifest_output_dir is kept (not a throwaway temp dir) --
useful on its own for inspecting one source in isolation, same spirit as
tools.ingestion.export_clean_dataset's per-source outputs.

Usage:
    python -m tools.ingestion.prepare_dataset config/prepare_collection.json
    python -m tools.ingestion.prepare_dataset config/prepare_collection.json --skip-ingest
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from tools.ingestion import build_manifest, combine_manifests, export_clean_dataset, ingest_raw
from utils.cli import add_logging_args, log_level_from_args, verbosity_argv
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


def run_ingest(config: dict) -> Path:
    """Runs tools.ingestion.ingest_raw once, returns the raw manifest.csv it
    wrote. ingest_cfg['roots'] (inline) is the common case -- one config
    end to end; ingest_cfg['roots_json'] (a separate file) is kept for the
    few roots configs actually reused across several prepare runs."""
    ingest_cfg = config["ingest"]
    out_dir = ingest_cfg.get("out_dir", "data")
    if "roots" in ingest_cfg:
        roots, label = ingest_cfg["roots"], "inline"
    else:
        roots, label = ingest_raw.load_roots_config(ingest_cfg["roots_json"]), ingest_cfg["roots_json"]
    print(f"\n=== Step 0a: raw ingestion ({label}) ===")
    return ingest_raw.run(roots, ingest_cfg["name"], out_dir)


def run_source(source: dict, default_raw_manifest: Path | None, mapping_file: str | None, verbosity: list[str]) -> Path:
    """Runs export_clean_dataset.py (if source['type'] == 'raw') then
    build_manifest.py for one source. Returns its manifest_output_dir --
    caller checks whether manifest.csv actually landed there (success) or
    only manifest_raw.csv (failure, see build_manifest.py)."""
    manifest_output_dir = Path(source["manifest_output_dir"])

    if source["type"] == "raw":
        raw_manifest = Path(source["raw_manifest"]) if source.get("raw_manifest") else default_raw_manifest
        if raw_manifest is None:
            raise SystemExit(
                f"source {source.get('source_type', '?')!r}: type='raw' needs either its own "
                "'raw_manifest', or a top-level 'ingest' block to derive one from."
            )
        if mapping_file is None:
            raise SystemExit(f"source {source['source_type']!r}: type='raw' needs a top-level 'mapping_file'.")

        print(f"\n=== Step 1a: clean export ({source['source_type']}) ===")
        export_clean_dataset.main([
            str(raw_manifest),
            "--identification-csv", source["identification_csv"],
            "--source-type", source["source_type"],
            "--key-column", source["key_column"],
            "--mapping-file", mapping_file,
            "--output-dir", source["clean_output_dir"],
            *_optional_flags(source), *verbosity,
        ])
        dataset_csv = Path(source["clean_output_dir"]) / "dataset.csv"
    elif source["type"] == "compliant":
        dataset_csv = Path(source["dataset_csv"])
    else:
        raise SystemExit(f"source: unknown type {source['type']!r} (expected 'raw' or 'compliant').")

    print(f"\n=== Step 1b: build manifest ({manifest_output_dir}) ===")
    build_manifest_argv = [str(dataset_csv), "--output-dir", str(manifest_output_dir), *verbosity]
    if source.get("path_column"):
        build_manifest_argv += ["--path-column", source["path_column"]]
    if source.get("base_dir"):
        build_manifest_argv += ["--base-dir", source["base_dir"]]
    if source.get("default_device_type") and source["type"] == "compliant":
        # for a 'raw' source, --default-device-type already applied at the
        # export_clean_dataset step above -- passing it again here is
        # harmless (every row already has a device_type by then) but
        # redundant, so only forwarded for 'compliant' sources.
        build_manifest_argv += ["--default-device-type", source["default_device_type"]]
    build_manifest.main(build_manifest_argv)

    return manifest_output_dir


def run(config: dict, skip_ingest: bool = False, verbosity: list[str] | None = None) -> dict:
    """Runs the config exactly as `main()` does, but takes an already-
    parsed dict rather than a JSON file path -- the in-process entry point
    for a caller that already has the config in memory (e.g.
    app/build_dataset.py's dataset-prep wizard), same pattern as every
    other stage here already offers both a CLI main(argv) and a direct
    Python call. Returns {"output_dir", "manifest_dirs", "failed_sources"}.
    """
    verbosity = verbosity or []
    sources = config["sources"]
    if not sources:
        raise SystemExit("config: 'sources' is empty -- nothing to do.")

    default_raw_manifest = None
    if "ingest" in config and not skip_ingest:
        default_raw_manifest = run_ingest(config)
    elif "ingest" in config:
        default_raw_manifest = Path(config["ingest"].get("out_dir", "data")) / config["ingest"]["name"] / "manifest.csv"
        print(f"\n--skip-ingest: reusing {default_raw_manifest}")

    manifest_dirs: list[Path] = []
    failed_sources: list[str] = []
    for source in sources:
        label = source.get("source_type") or source.get("dataset_csv", "?")
        manifest_dir = run_source(source, default_raw_manifest, config.get("mapping_file"), verbosity)
        if (manifest_dir / "manifest.csv").exists():
            manifest_dirs.append(manifest_dir)
        else:
            failed_sources.append(label)
            print(f"\n{label}: build_manifest.py did not produce manifest.csv -- see {manifest_dir / 'manifest_raw.csv'}")

    if not manifest_dirs:
        raise SystemExit("\nNo source produced a valid manifest.csv -- nothing to combine, aborting.")

    print(f"\n=== Step 1c: combine {len(manifest_dirs)} source(s) -> {config['output_dir']} ===")
    combine_manifests.main([*(str(d) for d in manifest_dirs), "--output-dir", config["output_dir"], *verbosity])

    print(f"\nDone. Clean dataset -> {config['output_dir']}")
    if failed_sources:
        print(f"Skipped ({len(failed_sources)}, see manifest_raw.csv above): {failed_sources}")
    print(
        f"Next: python -m tools.pipeline.train_dataset {config['output_dir']} "
        "--unet-model <weights.pt>   (see PIPELINE.md scenario 1)"
    )
    return {"output_dir": config["output_dir"], "manifest_dirs": manifest_dirs, "failed_sources": failed_sources}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="JSON config file (see module docstring).")
    parser.add_argument("--skip-ingest", action="store_true",
                         help="Skip tools.ingestion.ingest_raw even if the config has an 'ingest' block "
                              "-- reuse whatever raw manifest.csv is already on disk (faster than a full "
                              "rescan when re-running after fixing a source's own config).")
    add_logging_args(parser)
    args = parser.parse_args(argv)
    setup_console_logging(log_level_from_args(args))

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    run(config, skip_ingest=args.skip_ingest, verbosity=verbosity_argv(args))


if __name__ == "__main__":
    main()
