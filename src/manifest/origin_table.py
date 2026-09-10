"""origin_table.py
Interactive counterpart to `manifest.identification.assign_inv_name`'s
lookup mode, for `app/setup_dataset.py`: instead of requiring a
pre-authored origin_codes CSV, the setup tool can build one from an
editable table pre-filled with the values actually found in a source's
identification CSV origin column -- or, if that column doesn't even
exist, fall back to a single `inv_name` applied to every row of the
source.

The written CSV always uses the literal column name `collection_origin`
(matching `manifest.identification.load_origin_codes`, which reads that
column name regardless of what `--origin-column` a source's identification
CSV itself used) -- `origin_column` here only says which column of the
identification CSV to pull distinct values from.
"""
from __future__ import annotations

import pandas as pd

ORIGIN_CODES_COLUMN = "collection_origin"

# Synthetic origin column added to a copy of the identification CSV when it
# has no real origin column at all -- every row gets the same value, so a
# one-row origin_codes table (this constant -> the user's single inv_name)
# reuses assign_inv_name's existing lookup code path instead of a third mode.
CONSTANT_ORIGIN_VALUE = "_all"


def unique_origins(identification_df: pd.DataFrame, origin_column: str) -> list[str]:
    """Distinct values in `origin_column`, for pre-filling an editable
    origin -> inv_name table. Empty if the column doesn't exist -- see
    `assign_constant_origin` for that case."""
    if origin_column not in identification_df.columns:
        return []
    return sorted(identification_df[origin_column].dropna().unique().tolist())


def build_origin_table(identification_df: pd.DataFrame, origin_column: str) -> pd.DataFrame:
    """One row per distinct value found in `origin_column`, with an empty
    `inv_name` column for the user to fill in (e.g. via st.data_editor)."""
    origins = unique_origins(identification_df, origin_column)
    return pd.DataFrame({ORIGIN_CODES_COLUMN: origins, "inv_name": [""] * len(origins)})


def resolve_origin_codes(origins: pd.Series, inv_names: pd.Series) -> dict[str, str]:
    """{origin: inv_name}, defaulting a blank/missing inv_name to its own
    origin value -- a row left blank is accepted as-is rather than
    erroring downstream. Shared by `origin_codes_from_table` (the
    interactive table, still in memory) and
    `manifest.identification.load_origin_codes` (an imported CSV, already
    on disk) -- same blank-handling either way an origin_codes table
    reaches the pipeline."""
    origins = origins.astype(str)
    filled = inv_names.fillna("").astype(str).str.strip()
    resolved = filled.where(filled != "", origins)
    return dict(zip(origins, resolved))


def origin_codes_from_table(table: pd.DataFrame) -> dict[str, str]:
    """`table` (as returned/edited from `build_origin_table`) ->
    {collection_origin value: inv_name}, the same dict[str, str] shape
    `manifest.identification.assign_inv_name` accepts as `origin_codes`.
    See `resolve_origin_codes` for the blank-row default."""
    return resolve_origin_codes(table[ORIGIN_CODES_COLUMN], table["inv_name"])


def assign_constant_origin(
    identification_df: pd.DataFrame, inv_name: str,
) -> tuple[pd.DataFrame, str, pd.DataFrame]:
    """For a source whose identification CSV has no origin column at all:
    every row gets the same `inv_name`. Returns (a copy of
    `identification_df` with a constant origin column added, that column's
    name, and the one-row origin table for it) -- the column name and
    table are what `export_clean_dataset.main` and `write_origin_codes`
    below expect, same as the real-origin-column path."""
    df = identification_df.copy()
    df[CONSTANT_ORIGIN_VALUE] = CONSTANT_ORIGIN_VALUE
    table = pd.DataFrame({ORIGIN_CODES_COLUMN: [CONSTANT_ORIGIN_VALUE], "inv_name": [inv_name]})
    return df, CONSTANT_ORIGIN_VALUE, table


def write_origin_codes(table: pd.DataFrame, path) -> None:
    """Writes a filled-in origin table to `path` (typically
    `<source_output_dir>/collection_origin_codes.csv`) -- provenance, and
    reusable as `--origin-codes`/config `origin_codes` on a later run."""
    table.to_csv(path, index=False)
