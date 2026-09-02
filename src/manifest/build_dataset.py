"""build_dataset.py
Scans one or more local image roots (organized/terrain/loose, see
config/roots.json), parses specimen_id/device_type/shot_index from
filenames, deduplicates by content hash, and writes manifest.csv +
specimens.csv (+ manifest/unparsed.csv, manifest/duplicates.csv) under
<out-dir>/<name>/.

Usage:
    python -m manifest.build_dataset config/roots.json --name Bombus \\
        --identification data/identification/species.csv
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from utils.cli import add_logging_args, log_level_from_args
from utils.run_io import setup_console_logging

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
    specimen_id, suffix = stem.rsplit("_", 1)
    if not specimen_id:
        return None
    m = _ORGANIZED_SUFFIX.match(suffix)
    if not m:
        return None
    return specimen_id, m.group(1).upper(), int(m.group(2))


def _parse_terrain_underscore(stem: str):
    """num_inv_<n>  (no device letter, separator = underscore)"""
    if "_" not in stem:
        return None
    specimen_id, suffix = stem.rsplit("_", 1)
    if not specimen_id:
        return None
    m = _TERRAIN_SUFFIX.match(suffix)
    if not m:
        return None
    return specimen_id, None, int(m.group(1))


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
    specimen_id = "-".join(specimen_parts)
    if not specimen_id:
        return None
    return specimen_id, device.upper(), int(shot)


# registry of known naming conventions -> adding a new convention means one
# more function here, nothing else to touch.
NAMING_PARSERS = {
    "organized_underscore": _parse_organized_underscore,
    "terrain_underscore": _parse_terrain_underscore,
    "organized_hyphen": _parse_organized_hyphen,
}


def resolve_naming(root_cfg: dict) -> str:
    """Naming convention to use for this root: explicit via
    root_cfg['naming'] if present, otherwise inferred from
    collector_subfolder to stay compatible with roots.json files written
    before the hyphen scheme was added."""
    if "naming" in root_cfg:
        return root_cfg["naming"]
    return "terrain_underscore" if root_cfg.get("collector_subfolder") else "organized_underscore"


@dataclass
class ImageRecord:
    image_id: str  # = truncated content hash: stable even if the file is moved/copied
    specimen_id: Optional[str]
    split: str
    collector: Optional[str]
    device_type: Optional[str]   # "S" / "P" / None (terrain -> None, see collector)
    shot_index: Optional[int]
    raw_path: str
    source_root: str
    naming: str
    ext: str
    file_size_bytes: int
    content_hash: str
    status_ingest: str  # parsed_ok | unparsed_name | unreadable


def compute_hash(path: Path, chunk_size: int = 1 << 20) -> str:
    """Hashes the raw CONTENT (no image decoding) -- also works on HEIC,
    and serves as the dedup key when the same photo exists twice (external
    drive + local copy, for example)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()[:16]


def parse_filename(stem: str, naming: str):
    """Returns (specimen_id, device_type, shot_index, ok) by applying the
    `naming` convention (see NAMING_PARSERS). Makes no assumption about the
    format of the inventory number itself (can contain underscores/hyphens):
    each parser only consumes its own suffix and leaves the rest as
    specimen_id."""
    parser = NAMING_PARSERS.get(naming)
    if parser is None:
        raise ValueError(
            f"unknown naming convention: {naming!r} "
            f"(known: {list(NAMING_PARSERS)})"
        )
    result = parser(stem)
    if result is None:
        return None, None, None, False
    specimen_id, device_type, shot_index = result
    return specimen_id, device_type, shot_index, True


def scan_root(root_cfg: dict, base_dir: Path) -> list[ImageRecord]:
    root_path = base_dir / Path(root_cfg["path"])
    split = root_cfg["split"]
    collector_subfolder = bool(root_cfg.get("collector_subfolder", False))
    naming = resolve_naming(root_cfg)

    if not root_path.exists():
        logger.warning("root not found, skipped: %s", root_path)
        return []

    records: list[ImageRecord] = []
    for path in root_path.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue

        collector = None
        if collector_subfolder:
            try:
                rel = path.relative_to(root_path)
                collector = rel.parts[0] if len(rel.parts) > 1 else None
            except ValueError:
                collector = None

        specimen_id, device_type, shot_index, ok = parse_filename(
            path.stem, naming=naming
        )

        try:
            content_hash = compute_hash(path)
            size = path.stat().st_size
            status = "parsed_ok" if ok else "unparsed_name"
        except OSError as e:
            content_hash = ""
            size = -1
            status = "unreadable"
            logger.warning("unreadable: %s (%s)", path, e)

        image_id = content_hash if content_hash else f"unreadable:{path}"

        records.append(
            ImageRecord(
                image_id=image_id,
                specimen_id=specimen_id,
                split=split,
                collector=collector,
                device_type=device_type,
                shot_index=shot_index,
                raw_path=str(path),
                source_root=str(root_path),
                naming=naming,
                ext=path.suffix.lower(),
                file_size_bytes=size,
                content_hash=content_hash,
                status_ingest=status,
            )
        )
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


def build_images_table(all_records: list[ImageRecord]) -> tuple[list[dict], list[dict], list[dict]]:
    """Splits records into (images_ok, unparsed, duplicate_groups)."""
    by_hash: dict[str, list[ImageRecord]] = defaultdict(list)
    for r in all_records:
        if r.content_hash:
            by_hash[r.content_hash].append(r)

    seen_hash: set[str] = set()
    images_rows: list[dict] = []
    unparsed_rows: list[dict] = []

    for r in all_records:
        row = asdict(r)
        if r.status_ingest != "parsed_ok":
            unparsed_rows.append(row)
            continue
        # flags content duplicates (same bytes, different paths) without
        # excluding them: everything is kept, just an indicator is added.
        row["is_duplicate_content"] = r.content_hash in seen_hash
        seen_hash.add(r.content_hash)
        images_rows.append(row)

    duplicate_rows = []
    for content_hash, recs in by_hash.items():
        if len(recs) > 1:
            duplicate_rows.append(
                {
                    "content_hash": content_hash,
                    "n_copies": len(recs),
                    "paths": " | ".join(r.raw_path for r in recs),
                }
            )

    return images_rows, unparsed_rows, duplicate_rows


def build_specimens_table(
    images_rows: list[dict],
    identification: Optional[str],
    id_col: str,
    species_col: str,
    caste_col: Optional[str],
) -> list[dict]:
    # specimens seen in the images (source of truth for "which num_inv exist")
    by_specimen: dict[str, dict] = defaultdict(lambda: {"n_images": 0, "splits": set()})
    for row in images_rows:
        sid = row.get("specimen_id")
        if not sid:
            continue
        by_specimen[sid]["n_images"] += 1
        by_specimen[sid]["splits"].add(row["split"])

    identif: dict[str, dict] = {}
    if identification:
        p = Path(identification)
        if not p.exists():
            logger.warning("identification CSV not found, skipped: %s", p)
        else:
            with open(p, encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                if id_col not in (reader.fieldnames or []):
                    logger.warning(
                        "id column '%s' missing from %s (columns found: %s)",
                        id_col, p, reader.fieldnames,
                    )
                else:
                    for row in reader:
                        identif[row[id_col]] = row

    all_ids = set(by_specimen.keys()) | set(identif.keys())
    out = []

    for sid in sorted(all_ids):
        info = by_specimen.get(sid, {"n_images": 0, "splits": set()})
        ident = identif.get(sid, {})
        species = ident.get(species_col) if species_col else None
        caste = ident.get(caste_col) if caste_col else None

        out.append(
            {
                "specimen_id": sid,
                "species": species or "",
                "caste": caste or "",
                "is_labeled": bool(species),
                "n_images": info["n_images"],
                "splits_present": ",".join(sorted(info["splits"])),
                "in_identification_csv": sid in identif,
                "in_images": sid in by_specimen,
            }
        )
    return out


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


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Scan local image roots and build manifest.csv + specimens.csv.")
    parser.add_argument("roots", type=str, help="JSON file describing local roots (organized/terrain/loose), e.g. config/roots.json.")
    parser.add_argument("--name", required=True, help="Dataset name (output goes to <out-dir>/<name>/).")
    parser.add_argument("--identification", default=None, help="Species/caste identification CSV, keyed by num_inv.")
    parser.add_argument("--id-col", default="num_inv")
    parser.add_argument("--species-col", default="species")
    parser.add_argument("--caste-col", default="caste")
    parser.add_argument("--out-dir", default="data")
    add_logging_args(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    setup_console_logging(log_level_from_args(args))

    roots = load_roots_config(args.roots)
    if not roots:
        logger.error("No valid root to scan.")
        sys.exit(1)

    base_dir = Path(roots["base_root"][sys.platform])

    all_records: list[ImageRecord] = []
    for root_cfg in roots["splits"]:
        recs = scan_root(root_cfg, base_dir)
        logger.info("%-60s [%-10s] -> %d images", root_cfg["path"], root_cfg["split"], len(recs))
        all_records.extend(recs)

    images_rows, unparsed_rows, duplicate_rows = build_images_table(all_records)
    specimens_rows = build_specimens_table(
        images_rows, args.identification, args.id_col, args.species_col, args.caste_col
    )

    out_dir = Path(args.out_dir + "/" + args.name)
    image_fields = [
        "image_id", "specimen_id", "split", "collector", "device_type", "shot_index",
        "raw_path", "source_root", "naming", "ext", "file_size_bytes", "content_hash",
        "status_ingest", "is_duplicate_content",
    ]
    write_csv(images_rows, out_dir / "manifest.csv", image_fields)
    write_csv(unparsed_rows, out_dir / "manifest/unparsed.csv", image_fields[:-1])
    write_csv(duplicate_rows, out_dir / "manifest/duplicates.csv", ["content_hash", "n_copies", "paths"])
    write_csv(
        specimens_rows,
        out_dir / "specimens.csv",
        ["specimen_id", "species", "caste", "is_labeled", "n_images",
         "splits_present", "in_identification_csv", "in_images"],
    )

    n_unlabeled = sum(1 for r in specimens_rows if not r["is_labeled"])
    print("\n--- Summary ---")
    print(f"manifest.csv   : {len(images_rows)} row(s)")
    print(f"unparsed.csv   : {len(unparsed_rows)} row(s) (unrecognized names -- for manual review)")
    print(f"duplicates.csv : {len(duplicate_rows)} group(s) of identical content")
    print(f"specimens.csv  : {len(specimens_rows)} specimen(s) ({n_unlabeled} without species -- prediction pool)")
    print(f"\nWritten to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
