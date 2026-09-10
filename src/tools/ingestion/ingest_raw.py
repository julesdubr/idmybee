"""ingest_raw.py
Scans one or more local RAW image roots (organized/terrain/loose, see
config/roots.json), parses original_id/device_type/shot_index from
filenames (shape auto-detected per file, see parse_filename -- no naming
convention to declare or select), and writes manifest.csv (+
manifest/duplicates.csv) under <out-dir>/<name>/.

This is the first of two steps for messy raw data:
    tools/ingestion/ingest_raw.py          (this file)  raw folder -> raw manifest.csv
    tools/ingestion/export_clean_dataset.py             raw manifest.csv + raw
                                               identification CSV -> clean,
                                               independent, canonically-named
                                               dataset (its own manifest.csv
                                               + biological_data.csv, no
                                               reference back to raw files)

If a dataset is already clean (canonical `<inv_id>_<device_type>_<n>`
naming from the start), skip both -- there's nothing to ingest or export.

No identification/biological data here on purpose -- resolving specimen
identity (conflicts, `inv_id` assignment) is `tools/ingestion/export_clean_dataset.py`'s
job, not this one's. This script only ever answers "what image files exist,
and can their filename be parsed".

config/roots.json:
    {
      "base_root": {"linux": "/mnt/drive", "darwin": "/Volumes/drive"},
      "roots": [
        {"path": "organized", "source_type": "collection"},
        {"path": "terrain", "source_type": "terrain", "photographer_subfolder": true}
      ]
    }
`source_type` is later passed as-is to `tools/ingestion/export_clean_dataset.py
--source-type` -- if this manifest ever combines several source_types (as
in the example above), export_clean_dataset.py filters it down to the one
it's processing, so it doesn't need to be pre-split by hand.

Usage:
    python -m tools.ingestion.ingest_raw config/roots.json --name bombus_raw

`tools.ingestion.prepare_dataset` doesn't go through this CLI/`run()` at
all -- it scans one source's own images folder at a time via
`run_for_folder()` below (no roots config, no per-platform `base_root`),
since a source there already IS one images folder. `run()`/this CLI stay
useful standalone for scanning several roots from an external drive by
hand in one pass (e.g. cross-source duplicate-content detection, which
`run_for_folder()`'s one-folder-at-a-time scans can't do).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Optional

from utils.cli import add_logging_args, log_level_from_args
from core.pipeline_io import RunCounter
from core.run_io import setup_console_logging

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".tif", ".tiff", ".bmp"}

# organized/loose suffix: e.g. "S1", "P12"
_ORGANIZED_SUFFIX = re.compile(r"^([SP])(\d+)$", re.IGNORECASE)
# terrain suffix: e.g. "3"
_TERRAIN_SUFFIX = re.compile(r"^(\d+)$")


def _parse_organized_underscore(stem: str):
    """num_inv_[S|P]<n>  (letter+number glued together, separator = underscore)"""
    if "_" not in stem:
        return None
    original_id, suffix = stem.rsplit("_", 1)
    if not original_id:
        return None
    m = _ORGANIZED_SUFFIX.match(suffix)
    if not m:
        return None
    return original_id, m.group(1).upper(), int(m.group(2))


def _parse_terrain_underscore(stem: str):
    """num_inv_<n>  (no device letter, separator = underscore)"""
    if "_" not in stem:
        return None
    original_id, suffix = stem.rsplit("_", 1)
    if not original_id:
        return None
    m = _TERRAIN_SUFFIX.match(suffix)
    if not m:
        return None
    return original_id, None, int(m.group(1))


def _parse_organized_hyphen(stem: str):
    """num_inv-[S|P]-<n>  (letter and number separated by a hyphen, NOT
    glued together like in the underscore scheme -- convention seen on the
    external drive)."""
    parts = stem.split("-")
    if len(parts) < 3:
        return None
    *specimen_parts, device, shot = parts
    if device.upper() not in ("S", "P") or not shot.isdigit():
        return None
    original_id = "-".join(specimen_parts)
    if not original_id:
        return None
    return original_id, device.upper(), int(shot)


# Known raw-filename shapes, tried in turn by parse_filename() below --
# adding a new shape means one more function here, nothing else to touch.
# Which one matched is used to parse the row and then discarded -- not
# stored in the output (see module docstring): downstream steps only ever
# need the parsed result. No convention to declare/select per root anymore:
# each file's shape is auto-detected on its own, so a root can freely mix
# sources that were photographed under different naming habits.
_FILENAME_SHAPES = (_parse_organized_hyphen, _parse_organized_underscore, _parse_terrain_underscore)


@dataclass
class ImageRecord:
    original_id: Optional[str]
    source_type: str
    photographer: Optional[str]
    device_type: Optional[str]   # "S" / "P" / None (terrain -> None, see photographer)
    shot_index: Optional[int]
    raw_path: str
    ext: str
    file_size_bytes: int
    content_hash: str
    status: str          # OK | SUSPECT | FAILED (see core/pipeline_io.py; SKIPPED unused, always a full rescan)
    status_reason: str   # explains SUSPECT/FAILED, empty for OK


def compute_hash(path: Path, chunk_size: int = 1 << 20) -> str:
    """Hashes the raw CONTENT (no image decoding) -- also works on HEIC,
    and serves as the dedup key when the same photo exists twice (external
    drive + local copy, for example)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()[:16]


def parse_filename(stem: str):
    """Returns (original_id, device_type, shot_index, ok) by trying each
    known filename shape in turn (see _FILENAME_SHAPES) and using the first
    one that matches. Makes no assumption about the format of the inventory
    number itself (can contain underscores/hyphens): each shape only
    consumes its own suffix and leaves the rest as original_id."""
    for parser in _FILENAME_SHAPES:
        result = parser(stem)
        if result is not None:
            original_id, device_type, shot_index = result
            return original_id, device_type, shot_index, True
    return None, None, None, False


def scan_root(root_cfg: dict, base_dir: Path, seen_hash: set[str]) -> list[ImageRecord]:
    """`seen_hash` is shared across all roots of the run, so content
    duplicates ACROSS roots (e.g. the same photo present both in an
    external-drive root and a local backup root) are caught too, not just
    within one root."""
    root_path = base_dir / Path(root_cfg["path"])
    source_type = root_cfg["source_type"]
    photographer_subfolder = bool(root_cfg.get("photographer_subfolder", False))

    if not root_path.exists():
        logger.warning("root not found, skipped: %s", root_path)
        return []

    records: list[ImageRecord] = []
    for path in sorted(root_path.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue

        photographer = None
        if photographer_subfolder:
            try:
                rel = path.relative_to(root_path)
                photographer = rel.parts[0] if len(rel.parts) > 1 else None
            except ValueError:
                photographer = None

        original_id, device_type, shot_index, name_ok = parse_filename(path.stem)

        try:
            content_hash = compute_hash(path)
            size = path.stat().st_size
        except OSError as exc:
            records.append(ImageRecord(
                original_id=original_id, source_type=source_type, photographer=photographer,
                device_type=device_type, shot_index=shot_index, raw_path=str(path),
                ext=path.suffix.lower(), file_size_bytes=-1, content_hash="",
                status="FAILED", status_reason=f"unreadable: {exc}",
            ))
            logger.warning("unreadable: %s (%s)", path, exc)
            continue

        if not name_ok:
            records.append(ImageRecord(
                original_id=None, source_type=source_type, photographer=photographer,
                device_type=None, shot_index=None, raw_path=str(path), ext=path.suffix.lower(),
                file_size_bytes=size, content_hash=content_hash,
                status="FAILED", status_reason="filename format not recognized",
            ))
            continue

        is_duplicate = content_hash in seen_hash
        seen_hash.add(content_hash)
        records.append(ImageRecord(
            original_id=original_id, source_type=source_type, photographer=photographer,
            device_type=device_type, shot_index=shot_index, raw_path=str(path), ext=path.suffix.lower(),
            file_size_bytes=size, content_hash=content_hash,
            status="SUSPECT" if is_duplicate else "OK",
            status_reason="duplicate content of an earlier file in this run" if is_duplicate else "",
        ))
    return records


def load_roots_config(path: Optional[str]) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        logger.warning("roots file not found, skipped: %s", p)
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def build_duplicates_report(records: list[ImageRecord]) -> list[dict]:
    """One row per group of >=2 files sharing the same content hash --
    cross-reference report, complementary to each row's own SUSPECT status
    in manifest.csv (this groups them, that flags them individually)."""
    by_hash: dict[str, list[ImageRecord]] = defaultdict(list)
    for r in records:
        if r.content_hash:
            by_hash[r.content_hash].append(r)
    return [
        {"content_hash": h, "n_copies": len(recs), "paths": " | ".join(r.raw_path for r in recs)}
        for h, recs in by_hash.items() if len(recs) > 1
    ]


def write_csv(rows: list[dict], out_path: Path, fieldnames: Optional[list[str]] = None):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        # still writes an empty file with a header if known, so downstream
        # steps don't have to handle a missing file.
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            if fieldnames:
                csv.writer(f).writerow(fieldnames)
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_for_folder(
    images_dir: str, source_type: str, *, photographer_subfolder: bool = False,
) -> tuple[Path, Path, bool]:
    """Scans ONE raw images folder and writes its manifest.csv (+
    manifest/duplicates.csv) into a hidden subfolder inside it
    (`<images_dir>/.idmybee_ingest/`) rather than some separately-configured
    output location -- the scan is a byproduct of preparing that one
    source, not a shared artifact spanning several sources/roots the way
    `run()` above is. This is what `tools.ingestion.prepare_dataset` calls,
    once per source (see its module docstring) -- a source there IS one
    images folder, no separate roots config needed.

    If that hidden subfolder already holds a manifest.csv from an earlier
    run (kept on purpose -- see prepare_dataset's `keep_raw_manifest`),
    it's reused as-is instead of rescanning.

    Returns (manifest.csv path, the hidden subfolder itself -- caller's to
    delete unless the source asked to keep it, whether it was reused
    rather than freshly scanned).
    """
    ingest_dir = Path(images_dir) / ".idmybee_ingest"
    manifest_path = ingest_dir / "manifest.csv"
    if manifest_path.exists():
        return manifest_path, ingest_dir, True

    root_cfg = {
        "path": str(Path(images_dir).resolve()), "source_type": source_type,
        "photographer_subfolder": photographer_subfolder,
    }

    # root_cfg["path"] is already absolute, so the base_dir passed to
    # scan_root() here is irrelevant (Path.__truediv__ discards the left
    # side when the right side is absolute) -- there's no per-platform
    # base_root to resolve for a single folder the caller already picked
    # on this machine.
    records = scan_root(root_cfg, Path("."), seen_hash=set())
    logger.info("%-60s [%-10s] -> %d images", images_dir, source_type, len(records))

    duplicate_rows = build_duplicates_report(records)
    image_fields = [f.name for f in fields(ImageRecord)]
    write_csv([asdict(r) for r in records], manifest_path, image_fields)
    write_csv(duplicate_rows, ingest_dir / "duplicates.csv", ["content_hash", "n_copies", "paths"])
    return manifest_path, ingest_dir, False


def run(roots: dict, name: str, out_dir: str = "data") -> Path:
    """Scans every root in `roots` (an already-parsed roots config -- see
    module docstring for the shape) and writes manifest.csv (+
    manifest/duplicates.csv) under <out_dir>/<name>/. In-process entry
    point for a caller that already has the config in memory (e.g.
    tools.ingestion.prepare_dataset, or app/setup_dataset.py's dataset-prep
    wizard, which builds `roots` straight from its own widgets) -- same
    pattern as every other stage here already offers both a CLI main(argv)
    and a direct Python call. Returns the manifest.csv path."""
    if not roots:
        logger.error("No valid root to scan.")
        raise SystemExit(1)

    base_dir = Path(roots["base_root"][sys.platform])

    seen_hash: set[str] = set()
    counter = RunCounter()
    all_records: list[ImageRecord] = []
    for root_cfg in roots["roots"]:
        recs = scan_root(root_cfg, base_dir, seen_hash)
        logger.info("%-60s [%-10s] -> %d images", root_cfg["path"], root_cfg["source_type"], len(recs))
        for r in recs:
            counter.add(r.status)
        all_records.extend(recs)

    duplicate_rows = build_duplicates_report(all_records)

    out_path = Path(out_dir) / name
    image_fields = [f.name for f in fields(ImageRecord)]
    write_csv([asdict(r) for r in all_records], out_path / "manifest.csv", image_fields)
    write_csv(duplicate_rows, out_path / "manifest/duplicates.csv", ["content_hash", "n_copies", "paths"])

    print("\n--- Summary ---")
    print(f"manifest.csv   : {len(all_records)} row(s) -- {counter}")
    print(f"duplicates.csv : {len(duplicate_rows)} group(s) of identical content")
    print(f"\nWritten to: {out_path.resolve()}")
    print("Next: tools/ingestion/export_clean_dataset.py to resolve identity and produce a clean, independent dataset.")
    return out_path / "manifest.csv"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roots", type=str, help="JSON file describing local roots (organized/terrain/loose), e.g. config/roots.json.")
    parser.add_argument("--name", required=True, help="Output goes to <out-dir>/<name>/.")
    parser.add_argument("--out-dir", default="data")
    add_logging_args(parser)
    args = parser.parse_args(argv)
    setup_console_logging(log_level_from_args(args))

    roots = load_roots_config(args.roots)
    run(roots, args.name, args.out_dir)


if __name__ == "__main__":
    main()