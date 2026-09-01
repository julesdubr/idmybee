"""
Shared I/O helpers for the manifest's consolidated CSV tables (images.csv,
specimens.csv, crops.csv, future landmarks.csv, ...).

Every pipeline step (extraction, numbering, classification) reads and
writes its tables through these same functions, so the same resume/
compatibility logic isn't reimplemented each time -- and so a behavior
change (e.g. how an incompatible schema is detected) happens in one place.

`should_skip` lives in utils.pipeline_io (shared by every step, not just
the manifest tables) -- re-exported here for convenience.
"""

import csv
from pathlib import Path

from utils.pipeline_io import should_skip  # noqa: F401 -- re-exported

__all__ = ["parse_bool", "read_table", "check_schema", "load_existing_by_key", "should_skip"]


def parse_bool(value) -> bool:
    """A Python bool written by csv.DictWriter comes back as the string
    'True'/'False' on read -- must be reinterpreted explicitly."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def read_table(path) -> list[dict]:
    """Read a CSV table into a list of dicts. Missing table -> empty list,
    so a downstream step doesn't crash if the upstream step hasn't run yet
    (it will just process 0 rows, with a clear message left to the caller)."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def check_schema(path, expected_fields: list[str]):
    """Refuses to continue if an existing table has a different column
    schema than what the current code expects (e.g. written by an earlier
    version of the script) -- better to stop outright than corrupt the
    table in append mode with misaligned columns."""
    path = Path(path)
    if not path.exists():
        return
    with open(path, newline="", encoding="utf-8") as f:
        header = next(csv.reader(f), [])
    if header and header != expected_fields:
        raise SystemExit(
            f"{path} has a different schema than what the current code expects.\n"
            f"  expected: {expected_fields}\n"
            f"  found:    {header}\n"
            f"Rename/move the old file before rerunning (nothing was changed)."
        )


def load_existing_by_key(path, key_field: str) -> dict:
    """Reload an existing table (already checked compatible via
    check_schema) indexed by `key_field`, to resume a run without
    duplicating or losing rows already computed."""
    return {row[key_field]: row for row in read_table(path)}
