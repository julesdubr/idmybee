"""extraction_io.py
CSV reading and extraction-specific (detection + normalization) helpers:
detection.csv/crops.csv column schema, manifest image selection.

Generic tracking utilities (status counter, stats, duration, path
resolution) live in utils.pipeline_io -- reused by every pipeline step, not
just extraction (formerly mixed in here under the name detection_io.py).

Input: manifest.csv (manifest) or detection.csv (depending on the caller).
Output: none (pure helper functions).
"""
from __future__ import annotations

from pathlib import Path

from utils.pipeline_io import read_csv_rows

# Output of detect_wing.py (batch mode): extraction/{mode}/detection.csv
DETECTION_FIELDS = [
    "image_id",
    "specimen_id",
    "split",
    "status",
    "error_reason",
    "confidence",
    "n_detections",
    "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4",
    "processing_time_s",
    "processed_at",
]

# Output of normalize_crop.py (batch mode): extraction/{mode}/crops.csv
CROP_FIELDS = [
    "image_id",
    "specimen_id",
    "split",
    "status",
    "error_reason",
    "aspect_ratio",
    "output_path",
    "processing_time_s",
    "processed_at",
]


def read_images_csv(path: Path) -> list[dict]:
    """Load manifest.csv and validate its minimal columns."""
    required = {"image_id", "raw_path"}
    rows = read_csv_rows(path)
    if not rows:
        raise ValueError(f"manifest.csv is empty: {path}")

    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")

    return rows


def select_images(
    rows: list[dict],
    split: str | None = None,
    image_ids: set[str] | None = None,
) -> list[dict]:
    """Apply the generic filters to manifest.csv rows."""
    selected = []

    for row in rows:
        if split and row.get("split") != split:
            continue
        if image_ids is not None and row.get("image_id") not in image_ids:
            continue
        if row.get("status_ingest") and row.get("status_ingest") != "parsed_ok":
            continue
        selected.append(row)

    return selected
