"""pipeline_io.py
Generic run-tracking utilities shared by ALL pipeline steps (detection,
normalization, landmarks, renumbering): status counter, incremental CSV
writing, path resolution, and `pipeline_stats.csv` -- a single file at the
dataset root summarizing each (step, approach) so steps and approaches can
be compared against each other (output count per status, mean/total time
per image).

Nothing here is step-specific: CSV fields specific to a step
(DETECTION_FIELDS, CROP_FIELDS, LANDMARKS_FIELDS, ...) stay defined locally
in that step's own module (see extraction/extraction_io.py,
landmarks/predict.py).
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

PIPELINE_STATS_FIELDS = [
    "step", "approach", "total", "ok", "suspect", "skipped", "failed",
    "mean_time_s", "total_time_s", "updated_at",
]

# Canonical location of tools/pipeline/export_final_landmarks.py's package, relative
# to a dataset root. Distinct from the root's own biological_data.csv
# (specimen-level, from tools/ingestion/export_clean_dataset.py): the file of the
# same name in this folder is photo-level, row-aligned to the exported TPS.
DATASET_EXPORT_DIRNAME = "exports"

# The four files tools/pipeline/export_final_landmarks.py writes, minus their
# shared "<name>-" prefix (see export_filename/parse_export_name below) --
# `name` is normally the dataset's own resolve_dataset_name(), so the
# package stays self-describing wherever it's copied or re-uploaded
# (session du 11 sept. 2026: previously named landmarks_<n>lm_crop.tps etc,
# identical across every dataset, which is why app/train_model.py/
# app/predict_dataset.py had no way to recover a meaningful dataset/model
# name from an ad hoc upload -- see PIPELINE.md "Nommage des modeles").
EXPORT_CROP_SUFFIX = "landmarks_crop.tps"
EXPORT_RAW_SUFFIX = "landmarks_raw.tps"
EXPORT_BIO_SUFFIX = "biological_data.csv"
EXPORT_FAILED_SUFFIX = "failed.csv"
EXPORT_SUFFIXES = [EXPORT_BIO_SUFFIX, EXPORT_FAILED_SUFFIX, EXPORT_CROP_SUFFIX, EXPORT_RAW_SUFFIX]


def dataset_export_dir(dataset: Path) -> Path:
    """R-facing landmarks package for this dataset: <dataset>/exports/."""
    return Path(dataset) / DATASET_EXPORT_DIRNAME


def export_lm_dir(export_dir: Path, n_points: int) -> Path:
    """<export_dir>/<n>lm/ -- keeps exports of different landmark schemes
    (19lm, 18lm, ...) from a single dataset root sorted into their own
    subfolder, since filenames no longer carry the point count themselves
    (see EXPORT_SUFFIXES)."""
    return export_dir / f"{n_points}lm"


def export_filename(name: str, suffix: str) -> str:
    """"<name>-<suffix>" for one of tools.pipeline.export_final_landmarks's
    four output files (suffix is one of EXPORT_SUFFIXES)."""
    return f"{name}-{suffix}"


def parse_export_name(filename: str) -> str | None:
    """Recovers the `name` export_filename() embedded in one of these four
    files -- None if `filename` doesn't end in any known suffix (e.g. a
    TPS/CSV that didn't come from tools.pipeline.export_final_landmarks,
    still accepted by app/train_model.py/app/predict_dataset.py via their
    own fallback -- see utils.uploaded_dataset.infer_dataset_label)."""
    for suffix in EXPORT_SUFFIXES:
        marker = f"-{suffix}"
        if filename.endswith(marker) and len(filename) > len(marker):
            return filename[:-len(marker)]
    return None


def read_csv_rows(path: Path) -> list[dict]:
    """Load any CSV into a list of dicts."""
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def append_rows(path: Path, rows: list[dict], fields: list[str], write_header: bool) -> None:
    """Append rows to a CSV, writing the header if needed.

    `write_header=True` only on a run's very first flush (overwrites any
    previous file); `False` for subsequent flushes.
    """
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if write_header else "a"
    with path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def resolve_path(raw_path: str, base_dir: Path | None = None) -> Path:
    """Resolve a path, absolute or relative to `base_dir`.

    Also normalizes `\\` separators: a CSV produced on Windows (e.g.
    crops.csv's output_path) contains backslash paths, unreadable as-is as
    relative paths on Linux/macOS.
    """
    path = Path(str(raw_path).replace("\\", "/"))
    if path.is_absolute() or base_dir is None:
        return path
    return base_dir / path


def format_duration(seconds: float) -> str:
    """Format a duration as `1h05m30.0s` / `5m12.3s` / `3.2s`."""
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{int(hours)}h{int(minutes):02d}m{secs:04.1f}s"
    if minutes:
        return f"{int(minutes)}m{secs:04.1f}s"
    return f"{secs:.1f}s"


def should_skip(
    prev_row: dict | None, overwrite: bool, retry_failed: bool, output_exists: bool = True,
) -> bool:
    """Generic resume decision, reusable by any step that produces one
    result per image/specimen with an OK/SUSPECT/FAILED status.

    OK/SUSPECT: skipped if `output_exists` (whatever "the output" means for
    this step -- a file on disk, an entry in an in-memory working set
    reloaded from the actual output file, ...) and `--overwrite` wasn't
    requested. `output_exists` defaults to True (trust the log) for a
    caller with no cheaper way to check; pass an actual check to auto-heal
    if the output was partially deleted after being logged.
    FAILED: skipped by default, otherwise a repeated run replays and
    re-logs the same failures indefinitely; `retry_failed` to retry them
    explicitly.
    """
    if prev_row is None:
        return False
    status = prev_row.get("status")
    if status in ("OK", "SUSPECT"):
        return output_exists and not overwrite
    if status == "FAILED":
        return not retry_failed
    return False


class RunCounter:
    """Counts statuses (OK/SUSPECT/SKIPPED/FAILED) over the course of a run,
    for display + pipeline_stats.csv.

    Generic and open-ended: a status that's never added stays at 0 rather
    than being filed elsewhere by mistake (the old version, specific to
    extraction, counted any unknown status as FAILED -- correct as long as
    only OK/SKIPPED/FAILED existed, no longer true since SUSPECT).
    """

    def __init__(self):
        self._counts: dict[str, int] = {}

    def add(self, status: str) -> None:
        self._counts[status] = self._counts.get(status, 0) + 1

    def get(self, status: str) -> int:
        return self._counts.get(status, 0)

    @property
    def total(self) -> int:
        return sum(self._counts.values())

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "ok": self.get("OK"),
            "suspect": self.get("SUSPECT"),
            "skipped": self.get("SKIPPED"),
            "failed": self.get("FAILED"),
        }

    def __str__(self) -> str:
        return " ".join(f"{k}={v}" for k, v in sorted(self._counts.items()))


def update_pipeline_stats(
    stats_path: Path, step: str, approach: str, counts: dict, total_time_s: float,
) -> None:
    """Update pipeline_stats.csv: one row per (step, approach).

    `counts` comes from RunCounter.as_dict() (total/ok/suspect/skipped/failed).
    The existing row for this (step, approach) is replaced; others are kept
    -- lets several approaches for the same step (e.g. light vs heavy, or a
    future numbering method) and several steps be compared against each
    other, in a single per-dataset file.
    """
    rows = read_csv_rows(stats_path) if stats_path.exists() else []
    rows = [r for r in rows if not (r.get("step") == step and r.get("approach") == approach)]

    total = counts["total"]
    mean_time_s = total_time_s / total if total else 0.0
    rows.append({
        "step": step,
        "approach": approach,
        "total": str(total),
        "ok": str(counts["ok"]),
        "suspect": str(counts["suspect"]),
        "skipped": str(counts["skipped"]),
        "failed": str(counts["failed"]),
        "mean_time_s": f"{mean_time_s:.4f}",
        "total_time_s": f"{total_time_s:.4f}",
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PIPELINE_STATS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
