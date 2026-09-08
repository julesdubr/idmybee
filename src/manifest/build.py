"""Turns ANY per-photo dataset CSV -- produced by `tools/ingestion/export_clean_dataset.py`
or not -- into `manifest.csv`/`biological_data.csv`, the exact contract
every downstream pipeline stage reads via `core.dataset.load_dataset()`.

Unlike `manifest/identification.py` (identity resolution: frozen mapping,
conflict arbitration between rows claiming the same specimen), this module
does structural validation only: required columns present, referenced
image files exist and are readable, no duplicate `photo_id`, biological
data consistent per `inv_id`. It never decides which of two disagreeing
rows is right -- that stays `tools/ingestion/export_clean_dataset.py`'s job, for
messy raw data. See `tools/ingestion/build_manifest.py` for the CLI wrapping this.

Input contract (one row per photo):
- mandatory: `inv_id`, `species`, `caste`, plus a path column (default
  `path`) giving each photo's image file.
- optional "photo-level" columns (kept in `manifest.csv`, not
  `biological_data.csv`): `photo_id`, `device_type`, `device`,
  `photo_index`, `photographer`, `source_type`. `device_type`/
  `photo_index`/`photo_id` are derived automatically where absent (see
  `assign_sequential_photo_ids`) -- never re-derived when already present,
  since an already-copied image's on-disk filename is `<photo_id><ext>`.
- everything else: treated as biological/specimen-level data, carried into
  `biological_data.csv`.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from core.pipeline_io import resolve_path

REQUIRED_COLUMNS = ["inv_id", "species", "caste"]
PHOTO_LEVEL_COLUMNS = ["photo_id", "device_type", "device", "photo_index", "photographer", "source_type"]
MANIFEST_COLUMNS = [
    "photo_id", "inv_id", "source_type", "device_type", "device", "photo_index",
    "photographer", "ext", "content_hash", "file_size_bytes", "path", "status", "status_reason",
]


def _compute_hash(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()[:16]


def validate_required_columns(df: pd.DataFrame, path_column: str) -> list[str]:
    """Mandatory columns missing from `df`, if any -- `inv_id`/`species`/
    `caste` plus `path_column` itself. Empty means the input's SHAPE is
    usable; row-level problems are checked separately (see
    `resolve_and_verify_images`/`check_biological_consistency`)."""
    required = REQUIRED_COLUMNS + [path_column]
    return [c for c in required if c not in df.columns]


def biological_columns(df: pd.DataFrame, path_column: str) -> list[str]:
    """Every column that isn't `inv_id`, photo-level, or the path column --
    carried into `biological_data.csv`. Call this on the CSV as read, before
    `resolve_and_verify_images`/`assign_sequential_photo_ids` add their own
    technical columns (`ext`, `content_hash`, ... -- those would otherwise
    be mistaken for biological data too)."""
    excluded = set(PHOTO_LEVEL_COLUMNS) | {path_column, "inv_id"}
    return [c for c in df.columns if c not in excluded]


def resolve_and_verify_images(df: pd.DataFrame, path_column: str, base_dir: Path | None) -> pd.DataFrame:
    """Resolves each row's `path_column` against `base_dir`, hashes/sizes
    the file it points to, and sets `path`/`ext`/`content_hash`/
    `file_size_bytes`/`status`/`status_reason` -- one source of truth for
    "is this file OK", independent of whatever a producer (or a
    third-party CSV) claims about it. `FAILED` (no hash/size) for a
    missing/unreadable file; never raises."""
    df = df.copy()
    paths, exts, hashes, sizes, statuses, reasons = [], [], [], [], [], []
    for raw_path in df[path_column]:
        resolved = resolve_path(raw_path, base_dir)
        try:
            content_hash = _compute_hash(resolved)
            size = resolved.stat().st_size
        except OSError as exc:
            paths.append(str(resolved))
            exts.append(resolved.suffix.lower())
            hashes.append(None)
            sizes.append(None)
            statuses.append("FAILED")
            reasons.append(f"unreadable_image_or_missing_file: {exc}")
            continue
        paths.append(str(resolved))
        exts.append(resolved.suffix.lower())
        hashes.append(content_hash)
        sizes.append(size)
        statuses.append("OK")
        reasons.append("")
    return df.assign(
        path=paths, ext=exts, content_hash=hashes, file_size_bytes=sizes,
        status=statuses, status_reason=reasons,
    )


def flag_duplicate_photo_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Marks every row after the first occurrence of a repeated `photo_id`
    as `FAILED` -- a compliant dataset has one row per photo. Never
    downgrades a row already `FAILED` for another reason (e.g. an
    unreadable file)."""
    if "photo_id" not in df.columns:
        return df
    df = df.copy()
    is_dup = df.duplicated("photo_id", keep="first") & (df["status"] != "FAILED")
    df.loc[is_dup, "status"] = "FAILED"
    df.loc[is_dup, "status_reason"] = "duplicate photo_id in input"
    return df


def check_biological_consistency(df: pd.DataFrame, bio_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flags every not-already-`FAILED` row of an `inv_id` group as
    `SUSPECT` when its rows disagree on any `bio_columns` value, and
    returns a report of the distinct variants found per conflicting
    `inv_id` (one row per variant, first occurrence). Reporting only -- no
    harmonization: this is a structural sanity check, not identity
    arbitration (see `tools/ingestion/export_clean_dataset.py` for that)."""
    empty_report = df.iloc[0:0].copy()
    if not bio_columns or "inv_id" not in df.columns:
        return df, empty_report

    df = df.copy()
    compare = df[bio_columns].fillna("__NA__").astype(str)
    conflict_reports = []
    for inv_id, idx in df.groupby("inv_id").groups.items():
        idx = list(idx)
        if len(idx) < 2:
            continue
        variants = compare.loc[idx].drop_duplicates()
        if len(variants) == 1:
            continue
        flaggable = [i for i in idx if df.loc[i, "status"] != "FAILED"]
        df.loc[flaggable, "status"] = "SUSPECT"
        df.loc[flaggable, "status_reason"] = f"biological columns disagree within inv_id: {bio_columns}"
        first_seen_per_variant = compare.loc[idx].drop_duplicates(keep="first")
        conflict_reports.append(df.loc[first_seen_per_variant.index])

    conflicts = pd.concat(conflict_reports, ignore_index=True) if conflict_reports else empty_report
    return df, conflicts


def assign_sequential_photo_ids(df: pd.DataFrame, path_column: str, default_device_type: str) -> pd.DataFrame:
    """Fills a missing/blank `device_type` with `default_device_type`, and
    derives `photo_index`/`photo_id` for rows missing them (sorted by
    `inv_id`/`device_type`/`path_column`, 1..n per group -- same pattern as
    `identification.assign_photo_ids`, generalized to not need
    `shot_index`/a mapping file). Rows that already have a `photo_id` keep
    it untouched: an already-copied image's on-disk filename is
    `<photo_id><ext>`, so re-deriving it with a different sort key would
    stop matching the actual file."""
    df = df.copy()

    if "device_type" not in df.columns:
        df["device_type"] = default_device_type
    else:
        blank = df["device_type"].isna() | (df["device_type"] == "")
        df.loc[blank, "device_type"] = default_device_type

    if "photo_index" not in df.columns:
        df["photo_index"] = pd.NA
    if "photo_id" not in df.columns:
        df["photo_id"] = pd.NA

    needs_index = df["photo_index"].isna() | (df["photo_index"] == "")
    if needs_index.any():
        ordered = df.loc[needs_index].sort_values(["inv_id", "device_type", path_column])
        df.loc[ordered.index, "photo_index"] = (ordered.groupby(["inv_id", "device_type"]).cumcount() + 1).values

    needs_id = df["photo_id"].isna() | (df["photo_id"] == "")
    if needs_id.any():
        df.loc[needs_id, "photo_id"] = (
            df.loc[needs_id, "inv_id"].astype(str) + "_" + df.loc[needs_id, "device_type"].astype(str)
            + "_" + df.loc[needs_id, "photo_index"].astype(int).astype(str)
        )
    return df


def build_biological_data(df: pd.DataFrame, bio_columns: list[str]) -> pd.DataFrame:
    """One row per `inv_id` (first occurrence) with `bio_columns`, plus an
    `n_photos_<device_type>` pivot -- a generic version of
    `identification.build_specimen_table`'s device pivot, computed here for
    any dataset rather than only `export_clean_dataset.py`'s output."""
    specimens = df[["inv_id"] + bio_columns].drop_duplicates("inv_id").reset_index(drop=True)
    if "device_type" in df.columns:
        counts = (
            df.groupby(["inv_id", "device_type"]).size().unstack(fill_value=0)
            .add_prefix("n_photos_").reset_index()
        )
        specimens = specimens.merge(counts, on="inv_id", how="left")
    return specimens


def build_manifest_table(df: pd.DataFrame) -> pd.DataFrame:
    """Final `manifest.csv`/`manifest_raw.csv` column order -- identical
    shape either way, only present optional columns (`device`,
    `photographer`, `source_type`) included."""
    columns = [c for c in MANIFEST_COLUMNS if c in df.columns]
    return df[columns]
