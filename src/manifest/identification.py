"""Normalization of raw specimen identification data into a clean,
specimen-level dataset with stable `inv_id` values.

`tools/ingestion/export_clean_dataset.py` runs this module once per source (e.g. once
for collection, once for terrain, once for a future other-pollinator
dataset) -- nothing here combines several sources into one table. Sources
share identity space only through the frozen mapping file passed to
`extend_mapping`/`build_specimen_table` (kept apart internally by
`source_type`).

Terminology:
- `original_id`: specimen identifier as found in the raw data for this
  source -- the identification CSV's `--key-column` and the raw ingest
  manifest's own `original_id` column (see `tools/ingestion/ingest_raw.py`) are the
  same value, just read from two different files. May collide across
  physically different specimens -- see `resolve_identification`. Never
  the canonical identifier -- see `inv_id`.
- `inv_name`: short code identifying the inventory a specimen belongs to.
  Two derivation strategies, chosen by the caller based on whether an
  origin lookup table is available -- see `assign_inv_name`.
- `inv_id`: canonical specimen id, `<inv_name>_<inv_num:04d>`.
  Frozen once assigned -- see `load_frozen_mapping`/`extend_mapping`.
- `photo_id`: `<inv_id>_<device_type>_<photo_index>`, one per photo.

Known caveat: sources with a real capture date may still have rows missing
`dd`/`mm`/`yyyy` (e.g. about half of the collection rows). Rows with an
incomplete date sort after all dated rows within their `inv_name`
group, tie-broken by `original_id` -- deterministic, but not chronological.
See `extend_mapping`.
"""
from __future__ import annotations

import logging
import re

import pandas as pd

from manifest.origin_table import resolve_origin_codes

logger = logging.getLogger(__name__)

DEFAULT_COMPARE_COLUMNS = ["genus", "species", "caste", "dd", "mm", "yyyy"]

MAPPING_COLUMNS = [
    "source_type", "original_id", "inv_num", "inv_id", "inv_name",
    "resolved_conflict", "frozen_at",
]

_EMBEDDED_ID_RE = re.compile(r"^(.+)_(\d+)$")


def _lead_with(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Reorder `df` so any of `columns` present in it come first, in the
    given order, followed by everything else untouched."""
    lead = [c for c in columns if c in df.columns]
    return df[lead + [c for c in df.columns if c not in lead]]


def load_origin_codes(path: str) -> dict[str, str]:
    """Load the origin -> `inv_name` lookup table. A row left with a blank
    `inv_name` keeps its `collection_origin` value as-is -- see
    `manifest.origin_table.resolve_origin_codes`."""
    df = pd.read_csv(path)
    return resolve_origin_codes(df["collection_origin"], df["inv_name"])


def restrict_to_present_images(
    identification_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    key_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split identification rows into those with at least one image in the
    manifest (kept) and those with none (dropped, for reporting).

    `key_column` is the identification_df column that matches
    `manifest_df["original_id"]`.
    """
    present = set(manifest_df["original_id"].unique())
    has_image = identification_df[key_column].isin(present)
    kept, dropped = identification_df[has_image], identification_df[~has_image]
    if len(dropped):
        logger.info(
            "%d identification row(s) dropped: no image in manifest (%s)",
            len(dropped), sorted(dropped[key_column].unique())[:10],
        )
    return kept, dropped


def find_orphan_images(manifest_df: pd.DataFrame, known_original_ids: set[str]) -> pd.DataFrame:
    """Manifest rows whose specimen has no identification row in this
    source's identification CSV.

    These have no biological data to attach and must be excluded from the
    clean export -- returned here only so they can be reported, not
    silently dropped.
    """
    orphan_mask = ~manifest_df["original_id"].isin(known_original_ids)
    orphans = manifest_df[orphan_mask]
    if len(orphans):
        logger.warning(
            "%d image(s) have no identification row (original_id: %s)",
            orphans["original_id"].nunique(),
            sorted(orphans["original_id"].unique())[:10],
        )
    return orphans


def resolve_identification(
    identification_df: pd.DataFrame,
    key_column: str,
    device_column: str | None = None,
    compare_columns: list[str] = DEFAULT_COMPARE_COLUMNS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve rows sharing the same `key_column` (original_id) value.

    Two distinct things happen:
    - Literal duplicate rows for the same (`key_column`, `device_column`)
      pair collapse to the first one (e.g. a copy-pasted row).
    - If the surviving rows for a `key_column` value disagree on
      `compare_columns` -- a real identity conflict (e.g. the same raw id
      reused across two physically different specimens), not a data-entry
      duplicate -- `compare_columns` are overwritten on every row with the
      first row's values (file order), so the whole group agrees. Columns
      outside `compare_columns` (including `device_column`) are left
      untouched.

    Conflicts are reported once per distinct identity variant
      (`matches_first_occurrence` column), never once per raw row/device --
      device is not a meaningful conflict axis, since two device rows of
      the same real specimen always agree on identity by construction.

    Returns (resolved_df, conflicts_report_df); the latter is empty (right
    columns, no rows) if there were no conflicts.
    """
    compare_full = identification_df[compare_columns].fillna("__NA__").astype(str)
    harmonized = identification_df.copy()
    conflict_reports = []

    for key, idx in identification_df.groupby(key_column).groups.items():
        idx = list(idx)
        if len(idx) < 2:
            continue
        variants = compare_full.loc[idx].drop_duplicates()
        if len(variants) == 1:
            continue  # rows agree -- not a conflict, may still be literal duplicates (handled below)

        # Conflict: identity fields disagree somewhere in this specimen's
        # rows. One report row per distinct variant (first occurrence of
        # each, in original file order) -- not one per raw row, or a
        # conflict spanning several devices would be reported once per
        # device instead of once per actual disagreement.
        canonical_idx = idx[0]  # first row in the file for this specimen
        logger.warning(
            "%s: %d distinct identity found across %d row(s) -- using first "
            "occurrence as canonical, see conflicts report", key, len(variants), len(idx),
        )
        first_seen_per_variant = compare_full.loc[idx].drop_duplicates(keep="first")
        report = identification_df.loc[first_seen_per_variant.index].copy()
        report["matches_first_occurrence"] = (
            compare_full.loc[first_seen_per_variant.index].eq(compare_full.loc[canonical_idx]).all(axis=1)
        )
        conflict_reports.append(report)
        harmonized.loc[idx, compare_columns] = identification_df.loc[canonical_idx, compare_columns].values

    # Now that every row of a given specimen agrees on identity fields,
    # collapse to one row per (specimen, device): first occurrence wins,
    # for genuine duplicates and resolved conflicts alike.
    group_cols = [key_column] + ([device_column] if device_column else [])
    resolved = harmonized.drop_duplicates(subset=group_cols, keep="first")

    conflicts_report = (
        pd.concat(conflict_reports, ignore_index=True) if conflict_reports
        else identification_df.iloc[0:0].assign(matches_first_occurrence=pd.Series(dtype=bool))
    )
    return resolved, conflicts_report


def assign_inv_name(
    df: pd.DataFrame,
    key_column: str,
    origin_codes: dict[str, str] | None = None,
    origin_column: str = "collection_origin",
) -> pd.Series:
    """Derive the `inv_name` code for every row of `df`.

    Two strategies, chosen by whether `origin_codes` is given:
    - Lookup (`origin_codes` provided): maps `df[origin_column]` through the
      table (e.g. `<source_output_dir>/collection_origin_codes.csv` --
      see `manifest.origin_table` for how the interactive setup tool
      builds one). Raises if a value has no entry -- an unmapped origin
      must be added to that table explicitly, never silently defaulted.
    - Embedded (`origin_codes=None`): `key_column` is already a canonical
      `<inv_name>_<number>` id (e.g. a terrain campaign id such as
      `WB1_23_0007`) -- only the `inv_name` half is used here, see
      `parse_inv_name_embedded` for the matching `inv_num` half.
    """
    if origin_codes is not None:
        unmapped = set(df[origin_column].unique()) - set(origin_codes)
        if unmapped:
            raise ValueError(
                f"{origin_column} value(s) missing from origin codes table: "
                f"{sorted(unmapped)}. Add them to the origin codes CSV."
            )
        return df[origin_column].map(origin_codes)
    return df[key_column].map(lambda value: parse_inv_name_embedded(value)[0])


def parse_inv_name_embedded(original_id: str) -> tuple[str, int]:
    """Split an already-canonical id (`WB1_23_0007`) into
    (inv_name, inv_num). Raises if the id doesn't match the expected
    `<name>_<digits>` shape. Used for sources with no origin lookup table
    (`--origin-codes` omitted), e.g. terrain campaign ids."""
    match = _EMBEDDED_ID_RE.match(str(original_id))
    if not match:
        raise ValueError(f"id does not match <name>_<digits>: {original_id!r}")
    return match.group(1), int(match.group(2))


def load_frozen_mapping(path: str) -> pd.DataFrame:
    """Load the frozen original_id -> inv_id mapping, or an empty table
    with the right schema if it doesn't exist yet (first run)."""
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        logger.info("No existing mapping at %s, starting a new one", path)
        return pd.DataFrame(columns=MAPPING_COLUMNS)


def extend_mapping(
    existing_mapping: pd.DataFrame,
    specimens_df: pd.DataFrame,
    frozen_at: str,
    embedded_numbering: bool,
) -> pd.DataFrame:
    """Assign `inv_num`/`inv_id` for every specimen in `specimens_df` not
    already in `existing_mapping`, and return the full updated mapping.

    `specimens_df` is one run's worth of specimens -- a single
    `source_type` (one `export_clean_dataset` run processes one source at
    a time, see `tools/ingestion/export_clean_dataset.py`) -- with columns
    `source_type`, `original_id`, `inv_name`, `yyyy`, `mm`, `dd`,
    `resolved_conflict`.

    Existing rows in `existing_mapping` are never modified -- this is the
    freeze mechanism, and lets several sources share the same mapping file
    (kept apart by `source_type`, since an `original_id` string is only
    guaranteed unique within its own source).

    `embedded_numbering=True`: reuse the numeric suffix already present in
    `original_id` (see `parse_inv_name_embedded`) instead of
    assigning a fresh one, so `inv_id` ends up equal to `original_id`
    verbatim -- for sources whose ids are already canonical (e.g. terrain
    campaigns); see `attach_canonical_inv_id`'s `keep_original_id`, which
    relies on this to drop the now-redundant raw column downstream.
    `embedded_numbering=False`: assign a fresh `inv_num`, appended after
    the current max of its `inv_name` group, ordered by
    (yyyy, mm, dd); rows with an incomplete date sort last within the
    group, tie-broken by `original_id` -- for sources needing lookup-based
    inventory codes (e.g. collection specimens, see `assign_inv_name`).
    """
    existing_keys = set(zip(existing_mapping["source_type"], existing_mapping["original_id"]))
    is_new = ~specimens_df.apply(lambda r: (r["source_type"], r["original_id"]) in existing_keys, axis=1)
    new_rows = specimens_df[is_new].copy()
    if new_rows.empty:
        logger.info("No new specimens -- mapping unchanged")
        return existing_mapping

    source_type = new_rows["source_type"].iloc[0]
    new_entries = []

    if embedded_numbering:
        for _, row in new_rows.iterrows():
            inv_name, inv_num = parse_inv_name_embedded(row["original_id"])
            new_entries.append({
                "source_type": source_type, "original_id": row["original_id"],
                "inv_num": inv_num, "inv_id": row["original_id"], "inv_name": inv_name,
                "resolved_conflict": row["resolved_conflict"], "frozen_at": frozen_at,
            })
    else:
        max_num_by_group = (
            existing_mapping[existing_mapping["source_type"] == source_type]
            .groupby("inv_name")["inv_num"].max()
            .to_dict()
        )
        new_rows["_has_date"] = new_rows[["yyyy", "mm", "dd"]].notna().all(axis=1)
        for inv_name, group in new_rows.groupby("inv_name"):
            dated = group[group["_has_date"]].sort_values(["yyyy", "mm", "dd", "original_id"])
            undated = group[~group["_has_date"]].sort_values("original_id")
            ordered = pd.concat([dated, undated])
            next_num = max_num_by_group.get(inv_name, 0) + 1
            for offset, (_, row) in enumerate(ordered.iterrows()):
                inv_num = next_num + offset
                new_entries.append({
                    "source_type": source_type, "original_id": row["original_id"],
                    "inv_num": inv_num, "inv_id": f"{inv_name}_{inv_num:04d}", "inv_name": inv_name,
                    "resolved_conflict": row["resolved_conflict"],
                    "frozen_at": frozen_at,
                })

    logger.info("Assigned %d new inv_id (mapping frozen for the rest)", len(new_entries))
    return pd.concat([existing_mapping, pd.DataFrame(new_entries)], ignore_index=True)


def attach_canonical_inv_id(
    df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    key_column: str,
    source_type: str,
    keep_original_id: bool = True,
) -> pd.DataFrame:
    """Join the frozen canonical `inv_id` onto `df` by `key_column`.

    Drops any legacy `inv_id`/`inv_num` column already present in `df`
    (pre-refactor placeholder, often device-suffixed) in favour of the
    freshly-assigned canonical one.

    `keep_original_id`: whether `key_column`'s raw values are kept in the
    result, always under the name `original_id` regardless of what
    `key_column` itself was called (a raw identification CSV may name its
    own key column `inv_id`, which would otherwise collide with the
    freshly-assigned canonical one). Pass `False` when `inv_id` was
    assigned via embedded numbering (no `--origin-codes`, see
    `extend_mapping`) -- there, `inv_id` *is* the raw value verbatim, so
    keeping both is pure duplication. Raises if `df` already has its own,
    distinct `original_id` column -- that would silently collide with the
    one being created here.

    `inv_id` (then `inv_name`, if present) leads the result's columns.
    Rows of `df` whose `key_column` value has no entry in `mapping_df` for
    `source_type` are dropped by the inner join -- should not happen if
    `df` was derived from a `specimens_df` already passed through
    `extend_mapping`.
    """
    # Merge helper columns are prefixed/private (`_...`) rather than reusing
    # the real `original_id`/`inv_id` names here -- `df` may already carry
    # a genuine `original_id`/`inv_id` column of its own, which would
    # otherwise silently collide with this function's internal one (pandas
    # suffixes both to `_x`/`_y`, breaking the drop below).
    keys = mapping_df.loc[mapping_df["source_type"] == source_type, ["original_id", "inv_id"]] \
        .rename(columns={"original_id": "_original_id_key", "inv_id": "_clean_inv_id"})
    merged = df.merge(keys, left_on=key_column, right_on="_original_id_key", how="inner")
    merged = merged.drop(columns=["_original_id_key"])

    legacy_placeholder_cols = [c for c in ("inv_id", "inv_num") if c in merged.columns and c != key_column]
    merged = merged.drop(columns=legacy_placeholder_cols)
    if not keep_original_id:
        merged = merged.drop(columns=[key_column])
    elif key_column != "original_id":
        if "original_id" in merged.columns:
            raise ValueError(
                f"cannot keep key column {key_column!r} as 'original_id': "
                "df already has a distinct 'original_id' column"
            )
        merged = merged.rename(columns={key_column: "original_id"})
    merged = merged.rename(columns={"_clean_inv_id": "inv_id"})
    return _lead_with(merged, ["inv_id", "inv_name"])


def build_specimen_table(
    mapping_df: pd.DataFrame,
    identification_df: pd.DataFrame,
    key_column: str,
    source_type: str,
    device_column: str | None = None,
    keep_original_id: bool = True,
) -> pd.DataFrame:
    """Join the frozen `inv_id` onto resolved identification rows (see
    `attach_canonical_inv_id`) and collapse to one row per specimen.

    `keep_original_id` is forwarded to `attach_canonical_inv_id` -- pass
    `False` for sources using embedded numbering, where the raw id column
    would just duplicate `inv_id`.

    If `device_column` is given and the merged rows carry an `n_photos`
    column, per-device photo counts are pivoted into `n_photos_<device>`
    columns (e.g. `n_photos_P`/`n_photos_S` for a collection source with
    P/S devices). Sources with no device distinction (`device_column=None`)
    just collapse to one row per specimen as-is.
    """
    merged = attach_canonical_inv_id(identification_df, mapping_df, key_column, source_type, keep_original_id)

    if device_column and "n_photos" in merged.columns:
        photo_counts = (
            merged.pivot_table(index="inv_id", columns=device_column, values="n_photos", aggfunc="first")
            .add_prefix("n_photos_")
        )
        bio = (
            merged.drop(columns=[c for c in ("device", device_column, "n_photos") if c in merged.columns])
            .drop_duplicates("inv_id").set_index("inv_id")
        )
        result = bio.join(photo_counts).reset_index()
    else:
        result = merged.drop_duplicates("inv_id").reset_index(drop=True)

    return _lead_with(result, ["inv_id", "inv_name"])


def build_device_type_table(
    identification_df: pd.DataFrame, device_column: str, device_name_column: str,
) -> pd.DataFrame:
    """One row per distinct (`device_type`, device name) pair seen in this
    source's identification CSV.

    Photo-level outputs downstream (e.g. `manifest.csv`) only carry
    `device_type` (a coarse code like `P`/`S`) -- this table is the
    reference to look back up which actual device(s)/camera(s) it stands
    for, now that `build_specimen_table` no longer keeps it per specimen.
    `device_type` is not necessarily 1:1 with `device_name_column`: e.g. a
    collection source's `P` may cover several distinct camera bodies/lenses
    used over time, all kept here as separate rows sharing that
    `device_type`. See also `build_specimen_device_table` for the
    per-specimen (not just dataset-wide) version, used to attach the
    actual device used to each photo.
    """
    pairs = identification_df[[device_column, device_name_column]].drop_duplicates()
    return (
        pairs.rename(columns={device_column: "device_type", device_name_column: "device"})
        .sort_values("device_type").reset_index(drop=True)
    )


def build_specimen_device_table(
    mapping_df: pd.DataFrame,
    identification_df: pd.DataFrame,
    key_column: str,
    source_type: str,
    device_column: str,
    device_name_column: str,
) -> pd.DataFrame:
    """One row per (`inv_id`, `device_type`) pair actually used by a
    specimen in this source, with the specific device/camera name recorded
    for it.

    Unlike `build_device_type_table` (dataset-wide, doesn't say which
    device took a given photo when a `device_type` covers several), this
    is specimen-level: within one (specimen, device_type) group, the raw
    identification CSV already pins down one specific camera/phone name.
    Feeds `tools/ingestion/export_clean_dataset.py`'s `manifest.csv` `device` column
    -- photo-level `device_type` alone isn't enough once a code covers more
    than one physical device dataset-wide.

    If a specimen's rows for one device_type somehow disagree on the
    device name (data-entry inconsistency, not expected), the first
    occurrence wins silently -- same convention as the rest of this
    module's duplicate handling.
    """
    merged = attach_canonical_inv_id(identification_df, mapping_df, key_column, source_type)
    pairs = merged[["inv_id", device_column, device_name_column]].drop_duplicates(["inv_id", device_column])
    return (
        pairs.rename(columns={device_column: "device_type", device_name_column: "device"})
        .reset_index(drop=True)
    )


def assign_photo_ids(
    manifest_df: pd.DataFrame, mapping_df: pd.DataFrame
) -> pd.DataFrame:
    """One row per photo with a clean `photo_id`.

    `mapping_df` should already be restricted to this run's `source_type`
    (see `tools/ingestion/export_clean_dataset.py`) -- an `original_id` string is
    only guaranteed unique within its own source, so merging against an
    unfiltered multi-source mapping could match the wrong specimen.

    `manifest_df["shot_index"]` is not guaranteed to be contiguous from 1
    (original filenames may have gaps) -- it is used only to order photos
    within a (specimen, device) group; the clean `photo_index` is a fresh
    1..n re-numbering over that order.
    """
    keys = mapping_df[["original_id", "inv_id"]]
    merged = manifest_df.merge(keys, on="original_id", how="inner")
    merged = merged.sort_values(["inv_id", "device_type", "shot_index", "content_hash"])
    merged["photo_index"] = merged.groupby(["inv_id", "device_type"]).cumcount() + 1
    merged["photo_id"] = (
        merged["inv_id"] + "_" + merged["device_type"] + "_" + merged["photo_index"].astype(str)
    )
    return merged[["photo_id", "inv_id"] + [c for c in merged.columns if c not in ("photo_id", "inv_id")]]
