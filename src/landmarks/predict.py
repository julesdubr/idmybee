"""predict.py
Phase 2: automatic landmark placement (UNet) on normalized crops (output of
extraction/normalize_crop.py).

Two uses, same prediction logic:
- `predict_landmarks(image, model, device, n_landmarks)`: one already-loaded
  crop (BGR) -> landmarks (x, y) + status. Does no I/O -- useful for a "one
  photo at a time" pipeline (field use). See also
  landmarks/renumber.py::numerate_one to chain through to canonical
  numbering on that same image.
- CLI (`python -m landmarks.predict <dataset> --mode ...`): processes a
  whole dataset from `extraction/{mode}/crops.csv`, writes
  `landmarks/<tps>` + `landmarks/landmarks.csv`.

Input (CLI): extraction/{mode}/crops.csv (Phase 1), or --crops-csv to read a
reviewed override instead (see utils.review.write_crop_review).
Output (CLI): landmarks/<tps>, landmarks/landmarks.csv, <dataset>/pipeline_stats.csv.

A crop SKIPPED in Phase 1 (file already present on disk, not a failed
detection) is as valid a crop as any other: treated here as an OK, not
ignored. The previous version of this script only kept status=="OK",
silently losing the SKIPPED crops of a previous Phase 1 run (5 images on
this dataset).

Statuses (see also core.pipeline_io.RunCounter):
  - OK      : the expected `n_landmarks` points were found.
  - SUSPECT : fewer points than expected (but at least one) -- renumbering
    (Phase 3, landmarks/renumber.py) will automatically fail for these
    specimens, since their point count no longer matches the reference.
    SUSPECT here does NOT mean "worth watching", but "will become FAILED
    in Phase 3, and here's why".
  - FAILED  : no peak found, or unreadable crop.

Two substantive fixes relative to the original notebook:

  1. `local_maxima()` can flag SEVERAL adjacent pixels for the same peak
     (a near-identical-value plateau). Treating them as separate points
     then keeping "the N highest values" can drop a genuinely distinct
     landmark in favor of duplicates of the same peak. Each peak is
     grouped by connected component (scipy.ndimage.label) before
     ranking/truncating.
  2. If the model doesn't produce enough distinct peaks, the old code
     silently truncated (`maximas[-18:,:]` returns everything there is if
     fewer than 18 rows -- no error). Here, a result with fewer landmarks
     than expected is explicitly marked SUSPECT with the reason, never
     just written through unflagged.

`core.tps_io.ImageLandmarks.tps_id` must be an integer, unique per photo
WITHIN THE FILE, but carries no identity across runs -- `photo_id` (already
a stable, unique string per photo, see tools/ingestion/export_clean_dataset.py) is
what this script actually tracks; `tps_id` is only assigned, sequentially,
at checkpoint time (see `core.tps_io.assign_sequential_ids`).
`ImageLandmarks.from_image()` persists `photo_id`/`inv_id` in the TPS
(COMMENT=): later steps no longer need to join against crops.csv/
manifest.csv to recover these identifiers.

Since write_tps rewrites everything, this script works by "checkpoint": at
startup, the existing TPS is reparsed to know what's already there; each
processed image updates (or removes, if it now fails after having
succeeded before) the corresponding entry in an in-memory dict (keyed by
`photo_id`); the TPS and landmarks.csv are rewritten in full every
`--log-every` (and once more at the end).

Usage:
    python -m landmarks.predict data/Bombus --mode light \\
        --model models/unet_landmarks/<run_id>/weights.pth

    # only retry a previous run's failures:
    python -m landmarks.predict data/Bombus --mode light --model ... --retry-failed
"""
from __future__ import annotations

import argparse
import csv
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import ndimage
from skimage.morphology import local_maxima

from landmarks_trainer.model import load_weights
from utils.cli import add_dataset_positional, add_logging_args, log_level_from_args
from core.pipeline_io import RunCounter, format_duration, read_csv_rows, resolve_path, should_skip, update_pipeline_stats
from core.run_io import setup_console_logging
from core.tps_io import ImageLandmarks, assign_sequential_ids, parse_tps, write_tps

logger = logging.getLogger(__name__)

LANDMARKS_FIELDS = [
    "photo_id", "inv_id", "status", "error_reason",
    "n_landmarks_found", "model_name", "processing_time_s", "processed_at",
]


# ---------------------------------------------------------------------------
# Extracting landmarks from the UNet's predicted heatmap
# ---------------------------------------------------------------------------

def extract_top_landmarks(heatmap: np.ndarray, n_landmarks: int) -> tuple[np.ndarray, int]:
    """Returns (points_yx, n_found): up to `n_landmarks` points (y, x), one
    per distinct heatmap peak.

    Groups by connected component (one peak = one component, see point 1 of
    the module docstring) and keeps the highest-value pixel of each group,
    rather than treating every pixel of a plateau as its own landmark.

    n_found can be < n_landmarks if the model didn't produce enough
    distinct peaks on this image -- left to the caller to decide what to do
    (here: SUSPECT rather than a silently truncated landmark set).
    """
    mask = local_maxima(heatmap)
    labeled, n_components = ndimage.label(mask)
    if n_components == 0:
        return np.empty((0, 2), dtype=float), 0

    peaks = []
    for label_id in range(1, n_components + 1):
        ys, xs = np.where(labeled == label_id)
        values = heatmap[ys, xs]
        best = np.argmax(values)
        peaks.append((ys[best], xs[best], values[best]))

    peaks.sort(key=lambda p: p[2], reverse=True)
    top = peaks[:n_landmarks]
    coords_yx = np.array([(y, x) for y, x, _ in top], dtype=float)
    return coords_yx, len(top)


def predict_landmarks(
    image_bgr: np.ndarray, model, device, n_landmarks: int,
) -> tuple[str, str, np.ndarray | None, int]:
    """Predicts the landmarks of ONE already-loaded crop (BGR). Does no I/O
    (see `predict_landmarks_from_path` for the file-based version, used by
    the CLI). Returns (status, error_reason, coords_xy_or_None, n_found);
    coords_xy is in (x, y) order, ready for ImageLandmarks."""
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_t = torch.tensor(img.transpose(2, 0, 1) / 255.0, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_t).cpu().squeeze(0).numpy().transpose(1, 2, 0).squeeze(axis=2)

    coords_yx, n_found = extract_top_landmarks(output, n_landmarks)
    if n_found == 0:
        return "FAILED", "no_local_maximum", None, 0

    coords_xy = coords_yx[:, [1, 0]]
    if n_found < n_landmarks:
        return "SUSPECT", f"only_{n_found}_of_{n_landmarks}_expected_maxima", coords_xy, n_found
    return "OK", "", coords_xy, n_found


def predict_landmarks_from_path(
    crop_path: Path, model, device, n_landmarks: int,
) -> tuple[str, str, np.ndarray | None, int]:
    """File-based version of `predict_landmarks`, for the CLI (reads the crop)."""
    img = cv2.imread(str(crop_path))
    if img is None or img.size == 0:
        return "FAILED", "unreadable_crop", None, 0
    return predict_landmarks(img, model, device, n_landmarks)


# ---------------------------------------------------------------------------
# Selecting the crops to process (Phase 1 -> Phase 2)
# ---------------------------------------------------------------------------

def load_target_crops(crops_path: Path) -> list[dict]:
    """Reads crops.csv and keeps only the LATEST occurrence of each
    photo_id (crops.csv is append-only -- a Phase 1 retry/overwrite may
    have added a more recent row, different status, for the same
    photo_id). The status filter applies AFTER, on this latest known state.

    A SKIPPED crop (file already present on disk at Phase 1, not a failed
    detection) is treated as an OK: the file is valid, only how it was
    produced differs. FAILED (no file written) is the only one excluded."""
    if not crops_path.exists():
        raise SystemExit(
            f"{crops_path} not found -- run detect_wing.py and normalize_crop.py "
            f"(Phase 1) for this mode first."
        )
    crops = read_csv_rows(crops_path)
    if not crops:
        raise SystemExit(f"{crops_path} is empty -- nothing to process.")

    latest_by_id = {row["photo_id"]: row for row in crops}
    return [row for row in latest_by_id.values() if row.get("status") in ("OK", "SKIPPED")]


# ---------------------------------------------------------------------------
# In-memory TPS + landmarks.csv state, with regular checkpoints
# ---------------------------------------------------------------------------

def load_previous_status(landmarks_path: Path) -> dict[str, dict]:
    """Reloads a previous run's landmarks.csv (resume), if any. A file with
    a different schema (older script version) is flagged explicitly rather
    than mixed in with the current format."""
    if not landmarks_path.exists():
        return {}
    rows = read_csv_rows(landmarks_path)
    if rows and set(rows[0].keys()) != set(LANDMARKS_FIELDS):
        raise SystemExit(
            f"{landmarks_path} exists with a different schema -- move or delete it "
            f"before rerunning (expected columns: {LANDMARKS_FIELDS})."
        )
    return {row["photo_id"]: row for row in rows}


def load_working_tps(tps_path: Path) -> dict[str, ImageLandmarks]:
    """Reparses the existing TPS (if any) into a {photo_id: ImageLandmarks}
    dict -- photo_id, not tps_id: the TPS ID= field carries no identity
    across runs (see core.tps_io module docstring), only photo_id does.
    Malformed blocks are explicitly reported (never silently swallowed),
    the rest is carried over as-is."""
    if not tps_path.exists():
        return {}
    specimens, errors = parse_tps(tps_path, strict=False)
    if errors:
        logger.warning("%d unreadable block(s) in %s (ignored, not lost):", len(errors), tps_path)
        for e in errors[:10]:
            logger.warning("  specimen #%d, line %d: %s", e.specimen_index, e.line_no, e.message)
        if len(errors) > 10:
            logger.warning("  ... and %d more.", len(errors) - 10)
    return {sp.photo_id: sp for sp in specimens}


def checkpoint(tps_path: Path, working_tps: dict, landmarks_path: Path, landmarks_status: dict) -> None:
    tps_path.parent.mkdir(parents=True, exist_ok=True)
    write_tps(tps_path, assign_sequential_ids(list(working_tps.values())))

    landmarks_path.parent.mkdir(parents=True, exist_ok=True)
    with landmarks_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LANDMARKS_FIELDS)
        writer.writeheader()
        writer.writerows(landmarks_status.values())


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Automatic landmark placement (UNet, Phase 2).")
    add_dataset_positional(parser)
    parser.add_argument("--mode", default="light", choices=["heavy", "light"],
                         help="Phase 1 backend whose crops are read (extraction/{mode}/crops.csv).")
    parser.add_argument("--crops-csv", type=Path, default=None,
                         help="Read this CSV instead of the default extraction/{mode}/crops.csv -- "
                              "e.g. a crops_reviewed.csv written by utils.review.write_crop_review "
                              "after a manual validation pass, to skip a human-rejected crop here "
                              "without mutating the original crops.csv.")
    parser.add_argument("--model", required=True, help="Path to the UNet .pth model.")
    parser.add_argument("--base-dir", default=None, help="Root to resolve crops.csv's relative output_path values.")
    parser.add_argument("--tps", default="landmarks.tps", help="Output TPS filename (in <dataset>/landmarks/).")
    parser.add_argument("--n-landmarks", type=int, default=19,
                         help="19 = Tancrede's full blueprint (LM3 included, current default). "
                              "Pass 18 to run an older/legacy model that doesn't predict LM3.")
    parser.add_argument("--device", default=None, help="'cpu', 'cuda', etc. Empty = auto-detect.")
    parser.add_argument("--overwrite", action="store_true", help="Reprocess even if already logged.")
    parser.add_argument("--retry-failed", action="store_true", help="Retry images logged FAILED in a previous run.")
    parser.add_argument("--log-every", type=int, default=50, help="Checkpoint and progress-message frequency.")
    add_logging_args(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    setup_console_logging(log_level_from_args(args))

    crops_path = args.crops_csv or (args.dataset / "extraction" / args.mode / "crops.csv")
    tps_path = args.dataset / "landmarks" / args.tps
    landmarks_path = args.dataset / "landmarks" / "landmarks.csv"
    stats_path = args.dataset / "pipeline_stats.csv"
    base_dir = Path(args.base_dir) if args.base_dir else None

    targets = load_target_crops(crops_path)
    logger.info("Mode: %s", args.mode)
    logger.info("%d crop(s) to consider (%s)", len(targets), crops_path)
    if not targets:
        logger.info("No crop selected, nothing to do.")
        return

    landmarks_status = load_previous_status(landmarks_path)
    if landmarks_status:
        logger.info("%d entry/entries already present in %s (resuming previous run).", len(landmarks_status), landmarks_path)

    working_tps = load_working_tps(tps_path)
    if working_tps:
        logger.info("%d specimen(s) already present in %s (carried over as-is if skipped this run).", len(working_tps), tps_path)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device: %s", device)
    model = load_weights(args.model, device=device)  # state_dict, not the old full-pickle format -- see landmarks_trainer/migrate_legacy_weights.py for existing .pth files
    model.eval()
    model_name = Path(args.model).stem

    pipeline_start = time.perf_counter()
    counter = RunCounter()
    n_done = 0
    n_resumed = 0

    for row in targets:
        photo_id = row["photo_id"]

        prev_row = landmarks_status.get(photo_id)
        if should_skip(prev_row, args.overwrite, args.retry_failed, output_exists=photo_id in working_tps):
            n_resumed += 1
            continue

        crop_path = resolve_path(row["output_path"], base_dir)
        item_start = time.perf_counter()
        try:
            status, error_reason, coords_xy, n_found = predict_landmarks_from_path(
                crop_path, model, device, args.n_landmarks,
            )
        except Exception as e:
            status, error_reason, coords_xy, n_found = "FAILED", f"exception: {e}", None, 0
        processing_time_s = time.perf_counter() - item_start

        if coords_xy is not None:
            working_tps[photo_id] = ImageLandmarks.from_image(
                n_points=len(coords_xy), landmarks=coords_xy, image_path=str(crop_path),
                tps_id=0, photo_id=photo_id, inv_id=row.get("inv_id"),
            )
        elif photo_id in working_tps:
            # succeeded in a previous run, fails this time (--overwrite):
            # drop the stale entry rather than leave a TPS that no longer
            # matches the logged status.
            del working_tps[photo_id]

        landmarks_status[photo_id] = dict(
            photo_id=photo_id, inv_id=row.get("inv_id"),
            status=status, error_reason=error_reason or "", n_landmarks_found=n_found,
            model_name=model_name, processing_time_s=f"{processing_time_s:.4f}",
            processed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        counter.add(status)
        n_done += 1

        if n_done % args.log_every == 0:
            checkpoint(tps_path, working_tps, landmarks_path, landmarks_status)
            elapsed = time.perf_counter() - pipeline_start
            print(
                f"[{n_done}/{len(targets) - n_resumed}] elapsed: {format_duration(elapsed)} -- "
                f"average: {elapsed / n_done:.3f} s/image -- {counter}"
                f"{f'  ({n_resumed} resumed)' if n_resumed else ''}"
            )

    checkpoint(tps_path, working_tps, landmarks_path, landmarks_status)

    total_time_s = time.perf_counter() - pipeline_start
    update_pipeline_stats(stats_path, "landmarks", model_name, counter.as_dict(), total_time_s)

    print()
    print("Done.")
    print(counter, f" (+ {n_resumed} resumed from a previous run)" if n_resumed else "")
    print(f"Statuses    -> {landmarks_path}")
    print(f"Coordinates -> {tps_path}")
    print(f"Stats       -> {stats_path}")


if __name__ == "__main__":
    main()
