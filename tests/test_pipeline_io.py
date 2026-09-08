"""Tests for core.pipeline_io."""
import csv
from pathlib import Path

import pytest

from core.pipeline_io import RunCounter, append_rows, format_duration, read_csv_rows, resolve_path, should_skip


@pytest.mark.parametrize("seconds,expected", [
    (3.2, "3.2s"),
    (59.99, "60.0s"),
    (65.4, "1m05.4s"),
    (3600, "1h00m00.0s"),
    (3725.5, "1h02m05.5s"),
])
def test_format_duration(seconds, expected):
    assert format_duration(seconds) == expected


def test_run_counter_counts_known_and_unknown_statuses():
    counter = RunCounter()
    for status in ["OK", "OK", "SUSPECT", "FAILED", "OK"]:
        counter.add(status)

    counts = counter.as_dict()
    assert counts == {"total": 5, "ok": 3, "suspect": 1, "skipped": 0, "failed": 1}


def test_resolve_path_absolute_is_unchanged():
    abs_path = Path("/tmp/x/y.jpg")
    assert resolve_path(str(abs_path), base_dir=Path("/other")) == abs_path


def test_resolve_path_relative_joins_base_dir():
    result = resolve_path("crops/a.jpg", base_dir=Path("/data/root"))
    assert result == Path("/data/root/crops/a.jpg")


def test_resolve_path_normalizes_windows_separators():
    result = resolve_path("crops\\sub\\a.jpg", base_dir=Path("/data/root"))
    assert result == Path("/data/root/crops/sub/a.jpg")


def test_resolve_path_no_base_dir_returns_as_is():
    assert resolve_path("crops/a.jpg", base_dir=None) == Path("crops/a.jpg")


def test_append_rows_writes_header_once_then_appends(tmp_path):
    path = tmp_path / "out.csv"
    fields = ["a", "b"]

    append_rows(path, [{"a": "1", "b": "2"}], fields, write_header=True)
    append_rows(path, [{"a": "3", "b": "4"}], fields, write_header=False)

    rows = read_csv_rows(path)
    assert rows == [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]


def test_append_rows_noop_on_empty_list(tmp_path):
    path = tmp_path / "out.csv"
    append_rows(path, [], ["a"], write_header=True)
    assert not path.exists()


@pytest.mark.parametrize("status,overwrite,retry_failed,output_exists,expected", [
    (None, False, False, True, False),          # no history -> always process
    ("OK", False, False, True, True),            # OK + output still there -> skip
    ("OK", False, False, False, False),          # OK but output gone -> reprocess (auto-heal)
    ("OK", True, False, True, False),            # --overwrite -> always reprocess
    ("SUSPECT", False, False, True, True),       # SUSPECT behaves like OK
    ("FAILED", False, False, True, True),        # FAILED -> skip by default
    ("FAILED", False, True, True, False),        # --retry-failed -> reprocess
    ("UNKNOWN_STATUS", False, False, True, False),  # unrecognized status -> reprocess, not silently skipped
])
def test_should_skip(status, overwrite, retry_failed, output_exists, expected):
    prev_row = None if status is None else {"status": status}
    assert should_skip(prev_row, overwrite, retry_failed, output_exists) is expected
