"""Wing detection on an image (heavy YOLOE mode or light YOLO-OBB mode).

Two uses:
- `detect_one_image(mode, image, args)`: one already-loaded image -> box +
  status. Useful for a "one photo at a time" pipeline (e.g. capture from a
  device).
- CLI (`python -m extraction.detect_wing <dataset> --mode ...`): processes
  a whole dataset from its manifest, writes `extraction/{mode}/detection.csv`.

Input (CLI): manifest.csv (manifest).
Output (CLI): extraction/{mode}/detection.csv, <dataset>/pipeline_stats.csv.

To normalize crops from detection.csv, see `normalize_crop.py`.
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from extraction.extraction_io import DETECTION_FIELDS, read_images_csv, select_images
from extraction.normalize_crop import read_image
from utils.cli import add_dataset_positional, add_logging_args, log_level_from_args
from core.pipeline_io import RunCounter, append_rows, format_duration, resolve_path, update_pipeline_stats
from core.run_io import setup_console_logging

logger = logging.getLogger(__name__)

BATCH_SIZE = 50


def get_backend(mode: str):
    """Import the backend (heavy or light) matching `mode`."""
    if mode == "heavy":
        from extraction.heavy import detection as backend
    else:
        from extraction.light import detection as backend
    return backend


def detect_one_image(mode: str, ctx: dict, image: np.ndarray, args) -> dict:
    """Detect the wing on an already-loaded image. Does no I/O.

    `ctx` comes from `backend.load_model(args)`. Returns the dict produced
    by `backend.detect_one`: status, error_reason, confidence,
    n_detections, box.
    """
    return get_backend(mode).detect_one(ctx, image, args)


def box_to_row_fields(box: np.ndarray | None) -> dict:
    """Flattens a 4x2 box (or None) into x1..y4 fields for the CSV."""
    if box is None:
        return {f"{axis}{i}": "" for i in range(1, 5) for axis in ("x", "y")}

    values = box.reshape(-1)
    return {
        f"{axis}{i}": f"{values[2 * (i - 1) + (0 if axis == 'x' else 1)]:.8f}"
        for i in range(1, 5)
        for axis in ("x", "y")
    }


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Wing detection on a dataset, heavy (YOLOE) or light (YOLO-OBB) mode."
    )
    add_dataset_positional(parser)
    parser.add_argument("--mode", default="light", choices=["heavy", "light"])

    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--device", default=None)

    parser.add_argument("--base-dir", default=None)
    add_logging_args(parser)

    known_args, _ = parser.parse_known_args(argv)
    backend = get_backend(known_args.mode)
    backend.add_arguments(parser)

    args = parser.parse_args(argv)
    args.backend = backend
    return args


def new_row(source: dict) -> dict:
    return {
        "photo_id": source.get("photo_id", ""),
        "inv_id": source.get("inv_id", ""),
        "status": "FAILED",
        "error_reason": "",
        "confidence": "",
        "n_detections": "0",
        **box_to_row_fields(None),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    setup_console_logging(log_level_from_args(args))

    images = read_images_csv(Path(args.dataset / "manifest.csv"))
    targets = select_images(images)

    logger.info("Mode: %s", args.mode)
    logger.info("Images to process: %d", len(targets))

    if not targets:
        logger.info("No image selected, nothing to do.")
        return

    ctx = args.backend.load_model(args)

    extraction_root = Path(args.dataset / "extraction")
    output_path = extraction_root / args.mode / "detection.csv"
    stats_path = Path(args.dataset) / "pipeline_stats.csv"
    write_header = True

    base_dir = Path(args.base_dir) if args.base_dir else None
    pipeline_start = time.perf_counter()
    batch = []
    counter = RunCounter()

    for index, source in enumerate(targets, start=1):
        item_start = time.perf_counter()
        row = new_row(source)

        image_path = resolve_path(source["path"], base_dir)
        image = read_image(image_path)

        if image is None:
            row["error_reason"] = "unreadable_image_or_unsupported_format"
        else:
            detection = detect_one_image(args.mode, ctx, image, args)
            row["status"] = detection["status"]
            row["error_reason"] = detection["error_reason"]
            row["n_detections"] = str(detection["n_detections"])

            if detection["confidence"] is not None:
                row["confidence"] = f"{detection['confidence']:.4f}"

            row.update(box_to_row_fields(detection["box"]))

        row["processing_time_s"] = f"{time.perf_counter() - item_start:.4f}"
        row["processed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        counter.add(row["status"])
        batch.append(row)

        is_last = index == len(targets)
        if len(batch) >= BATCH_SIZE or is_last:
            append_rows(output_path, batch, DETECTION_FIELDS, write_header)
            write_header = False
            batch = []

            elapsed = time.perf_counter() - pipeline_start
            logger.info(
                "[%d/%d] elapsed: %s -- average: %.3f s/image -- %s",
                index, len(targets), format_duration(elapsed), elapsed / index, counter,
            )

    total_time_s = time.perf_counter() - pipeline_start
    update_pipeline_stats(stats_path, "detection", args.mode, counter.as_dict(), total_time_s)
    print(f"CSV -> {output_path}")
    print(f"Stats -> {stats_path}")


if __name__ == "__main__":
    main()
