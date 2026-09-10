"""build_reference.py
Freezes, ONCE, the reference shape (GPA consensus of the ground-truth
template, e.g. Tancrede) used by landmarks/renumber.py to number specimens
-- in batch mode as well as single-image (field) mode.

Why a separate artifact rather than recomputing the consensus on every
renumbering run (old behavior): the reference must never move silently. If
the ground-truth TPS is completed later (new specimens digitized, for
example), an on-the-fly recompute would slightly change the consensus --
and therefore EVERY future numbering -- without anything flagging it, and
without guaranteeing that batch mode and a future field mode (one image at
a time) use exactly the same reference. A frozen, versioned artifact makes
this choice explicit: this script must be rerun by hand to change the
reference, never a side effect of some other run.

Input: a reference TPS (ORDERED landmarks, ground truth, e.g. Tancrede --
19 landmarks, one of which is absent from the UNet detector's scheme, see
--drop).
Output: a plain .tps, one specimen block, the reference shape ready to be
passed to `landmarks.methods.*.numerate()` -- deliberately just landmarks,
no provenance metadata attached (no source/drop/date bookkeeping): a
reference shape is any single-specimen .tps a person can also produce by
hand (e.g. digitizing directly in TPSdig), not something only this script
can create or that requires a companion sidecar to use.

Usage:
    python -m landmarks.build_reference --ref references/ref-landmarks.tps \\
        --drop 3 --out references/shapes/reference_shape.tps
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from utils.cli import add_logging_args, log_level_from_args
from core.gpa import gpagen
from core.run_io import setup_console_logging
from core.tps_io import ImageLandmarks, parse_tps, write_tps

logger = logging.getLogger(__name__)


def build_reference_shape(ref_specimens: list, expected_lm: int) -> np.ndarray:
    """GPA consensus of the reference template, on specimens with
    `expected_lm` landmarks only (a ground-truth TPS may contain a few
    incomplete specimens)."""
    matching = [s for s in ref_specimens if s.n_points == expected_lm]
    if not matching:
        raise SystemExit(
            f"No specimen with {expected_lm} landmarks in the reference TPS "
            f"(try --n-landmarks to force a different count)."
        )
    return gpagen([s.landmarks for s in matching]).mean_shape


def load_reference(path: Path) -> np.ndarray:
    """Loads a reference shape -- any plain .tps with exactly one specimen
    block, frozen by this script or hand-made. Used by
    landmarks/renumber.py, in batch mode as well as single-image mode."""
    if not path.exists():
        raise SystemExit(
            f"{path} not found -- run this first:\n"
            f"  python -m landmarks.build_reference --ref <tps> --drop <n> --out {path}"
        )
    specimens, errors = parse_tps(path, strict=False)
    if len(specimens) != 1:
        raise SystemExit(
            f"{path} must contain exactly one specimen block (found {len(specimens)}, "
            f"{len(errors)} parsing error(s)) -- a reference shape is one consensus/hand-digitized "
            "landmark set, not a dataset."
        )
    return specimens[0].landmarks


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ref", type=Path, required=True, help="Reference TPS (ground truth).")
    parser.add_argument("--drop", type=int, default=None,
                         help="Landmark(s) in --ref with no counterpart in the detector (e.g. 3).")
    parser.add_argument("--n-landmarks", type=int, default=None,
                         help="Expected landmark count in --ref (default: the most frequent one found).")
    parser.add_argument("--out", type=Path, required=True, help="Output .tps file.")
    add_logging_args(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    setup_console_logging(log_level_from_args(args))

    ref_specimens, ref_errors = parse_tps(args.ref, strict=False)
    if not ref_specimens:
        raise SystemExit(
            f"No valid specimen in {args.ref} ({len(ref_errors)} parsing "
            f"error(s)). Check that this file is really a .tps, not a metadata CSV."
        )

    if args.n_landmarks is not None:
        expected_lm = args.n_landmarks
    else:
        counts = np.bincount([s.n_points for s in ref_specimens])
        expected_lm = int(np.argmax(counts))

    consensus = build_reference_shape(ref_specimens, expected_lm)
    n_used = sum(1 for s in ref_specimens if s.n_points == expected_lm)

    zones = np.delete(consensus, args.drop, axis=0) if args.drop is not None else consensus

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_tps(args.out, [ImageLandmarks(n_points=len(zones), landmarks=zones, image_path="", tps_id=1)])

    print(f"Reference: {n_used} specimen(s) with {expected_lm} landmarks in {args.ref}")
    print(f"Frozen shape: {len(zones)} zones" + (f" (LM{args.drop} dropped)" if args.drop is not None else ""))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
