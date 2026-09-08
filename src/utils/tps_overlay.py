"""tps_overlay.py
Renders overlays (image + numbered landmarks) from any .tps file --
reusable anywhere the pipeline produces a TPS with IMAGE= pointing to a
readable image: outputs of landmarks/predict.py (landmarks.tps), of
landmarks_trainer/reproject_reference.py, or any other TPS following the
format read by core.tps_io.parse_tps.

    python -m utils.tps_overlay data/Bombus/landmarks/landmarks.tps \\
        --output-dir data/Bombus/landmarks/overlays \\
        --csv data/Bombus/landmarks/landmarks.csv

Sorting into subfolders: if the TPS has COMMENT= (photo_id) AND a --csv is
given, each image goes into <output-dir>/<status>/<photo_id>.png, `status`
being read from the CSV (--status-col column, "status" by default) joined
on --photo-id-col ("photo_id" by default). A photo_id present in the TPS
but missing from the CSV goes into <output-dir>/_unmatched/ rather than
being silently dropped or mixed in with the rest.

Without --csv (or a TPS with no COMMENT=): all images go straight into
<output-dir>/, flat, named by photo_id if known, otherwise by ID= (tps_id),
otherwise by specimen index.

Doesn't modify or depend on anything other than core.tps_io -- this file
has no opinion on WHO produces the TPS, only on how to draw it.
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from utils.cli import add_logging_args, log_level_from_args
from core.pipeline_io import read_csv_rows, resolve_path
from core.run_io import setup_console_logging
from core.tps_io import parse_tps

logger = logging.getLogger(__name__)

UNMATCHED_DIR = "_unmatched"


def draw_landmarks(image: np.ndarray, points_xy: np.ndarray, radius: int = 4,
                    point_color: tuple = (0, 0, 255), text_color: tuple = (0, 255, 0),
                    font_scale: float = 0.35) -> np.ndarray:
    """Draws numbered points (0, 1, 2, ...) on a copy of `image`. Does not
    modify `image` in place."""
    annotated = image.copy()
    for i, (x, y) in enumerate(points_xy):
        xi, yi = int(round(x)), int(round(y))
        cv2.circle(annotated, (xi, yi), radius, point_color, -1)
        cv2.putText(annotated, str(i), (xi + 5, yi - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, text_color, 1, cv2.LINE_AA)
    return annotated


def load_status_by_photo_id(csv_path: str, photo_id_col: str, status_col: str) -> dict:
    rows = read_csv_rows(Path(csv_path))
    missing = [c for c in (photo_id_col, status_col) if rows and c not in rows[0]]
    if missing:
        raise ValueError(f"{csv_path} has no {missing} column (columns present: {list(rows[0].keys())})")
    return {row[photo_id_col]: row[status_col] for row in rows}


def output_name(specimen, index: int) -> str:
    if specimen.photo_id:
        return f"{specimen.photo_id}.png"
    if specimen.tps_id is not None:
        return f"tps{specimen.tps_id}.png"
    return f"specimen_{index:04d}.png"


def render_tps_overlays(
    tps_path, output_dir, base_dir=None, csv_path=None,
    photo_id_col: str = "photo_id", status_col: str = "status", flip_y=False
) -> dict:
    """Callable directly from Python (not just from the CLI) -- see
    reproject_reference.py for an example call after writing a TPS + its
    companion CSV in the same run."""
    specimens, tps_errors = parse_tps(Path(tps_path), strict=False)

    status_by_id = load_status_by_photo_id(csv_path, photo_id_col, status_col) if csv_path else {}
    sortable = bool(csv_path)  # condition: COMMENT= AND csv provided

    output_dir = Path(output_dir)
    written, unmatched, skipped = 0, 0, []

    for idx, sp in tqdm(enumerate(specimens), total=len(specimens)):
        image_path = resolve_path(sp.image_path, base_dir)
        image = cv2.imread(str(image_path))
        if image is None:
            skipped.append({"tps_id": sp.tps_id, "photo_id": sp.photo_id or "", "reason": f"image not found: {image_path}"})
            continue

        points_raw = sp.landmarks.astype(np.float64).copy()
        if flip_y:
            # Tancrede's TPS appears to use tpsDig's Y-up convention (Y=0 at
            # the bottom of the image) rather than the standard pixel
            # convention (Y=0 at the top) used by detection.csv/the OBB.
            # This MUST be corrected HERE, in raw-image space, before
            # rotation -- flipping Y afterwards, in final crop space, does
            # NOT work once the wing is rotated (reflection and rotation
            # don't commute): verified, a point flipped after the fact
            # doesn't land in the right place as soon as the OBB's rotation
            # angle isn't 0.
            points_raw[:, 1] = image.shape[0] - points_raw[:, 1]

        annotated = draw_landmarks(image, points_raw)
        name = output_name(sp, idx)

        subdir = output_dir
        if sortable and sp.photo_id:
            status = status_by_id.get(sp.photo_id)
            if status is None:
                subdir = output_dir / UNMATCHED_DIR
                unmatched += 1
            else:
                subdir = output_dir / status
        elif sortable and not sp.photo_id:
            subdir = output_dir / UNMATCHED_DIR
            unmatched += 1

        subdir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(subdir / name), annotated)
        written += 1

    if skipped:
        skipped_path = output_dir / "skipped.csv"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(skipped_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["tps_id", "photo_id", "reason"])
            writer.writeheader()
            writer.writerows(skipped)

    return {
        "n_specimens": len(specimens),
        "n_tps_errors": len(tps_errors),
        "written": written,
        "unmatched": unmatched,
        "skipped": len(skipped),
    }


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Render numbered-landmark overlays from a TPS file.")
    parser.add_argument("tps", type=Path, help="Path to the TPS file.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to write overlay images to.")
    parser.add_argument("--base-dir", default=None, help="Root to resolve IMAGE= if relative, same meaning as resolve_path.")
    parser.add_argument("--csv", default=None, help="Companion CSV (photo_id + status) to sort into subfolders -- optional.")
    parser.add_argument("--photo-id-col", default="photo_id")
    parser.add_argument("--status-col", default="status")
    parser.add_argument("--flip-y", action="store_true")
    add_logging_args(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    setup_console_logging(log_level_from_args(args))

    base_dir = Path(args.base_dir) if args.base_dir else None
    summary = render_tps_overlays(
        args.tps, args.output_dir, base_dir=base_dir, csv_path=args.csv,
        photo_id_col=args.photo_id_col, status_col=args.status_col, flip_y=args.flip_y
    )

    print("Done.")
    print(f"{summary['n_specimens']} specimen(s), {summary['n_tps_errors']} TPS parsing error(s)")
    print(f"written={summary['written']} unmatched={summary['unmatched']} skipped={summary['skipped']}")
    print(f"Overlays -> {args.output_dir}")


if __name__ == "__main__":
    main()
