"""Tests for manifest.origin_table."""
import pandas as pd

from manifest.identification import assign_inv_name
from manifest.origin_table import (
    ORIGIN_CODES_COLUMN,
    assign_constant_origin,
    build_origin_table,
    origin_codes_from_table,
    unique_origins,
)


def test_unique_origins_sorted_distinct_values():
    df = pd.DataFrame({"collection_origin": ["WB", "EE", "WB", "EE", None]})
    assert unique_origins(df, "collection_origin") == ["EE", "WB"]


def test_unique_origins_empty_when_column_missing():
    df = pd.DataFrame({"other": [1, 2]})
    assert unique_origins(df, "collection_origin") == []


def test_build_origin_table_prefills_blank_inv_name():
    df = pd.DataFrame({"collection_origin": ["WB", "EE"]})
    table = build_origin_table(df, "collection_origin")
    assert list(table[ORIGIN_CODES_COLUMN]) == ["EE", "WB"]
    assert list(table["inv_name"]) == ["", ""]


def test_origin_codes_from_table_keeps_unfilled_rows_as_is():
    table = pd.DataFrame({ORIGIN_CODES_COLUMN: ["EE", "WB"], "inv_name": ["ESQ", "  "]})
    assert origin_codes_from_table(table) == {"EE": "ESQ", "WB": "WB"}


def test_origin_codes_from_table_feeds_assign_inv_name():
    identification_df = pd.DataFrame({
        "inv_id": ["A1", "A2"], "collection_origin": ["WB", "EE"],
    })
    table = pd.DataFrame({ORIGIN_CODES_COLUMN: ["WB", "EE"], "inv_name": ["ESQ", "REN"]})
    origin_codes = origin_codes_from_table(table)

    result = assign_inv_name(identification_df, "inv_id", origin_codes, "collection_origin")

    assert list(result) == ["ESQ", "REN"]


def test_assign_constant_origin_adds_synthetic_column_and_table():
    identification_df = pd.DataFrame({"inv_id": ["A1", "A2"]})

    df, origin_column, table = assign_constant_origin(identification_df, "MYSITE")

    assert origin_column in df.columns
    assert df[origin_column].unique().tolist() == [df[origin_column].iloc[0]]
    origin_codes = origin_codes_from_table(table)
    inv_names = assign_inv_name(df, "inv_id", origin_codes, origin_column)
    assert list(inv_names) == ["MYSITE", "MYSITE"]
