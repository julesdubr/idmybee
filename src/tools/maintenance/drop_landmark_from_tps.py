"""drop_landmark_from_tps.py
Removes one landmark from every specimen of a TPS -- useful to compare
Tancrede (19 LM) against a reconstructed file that's missing that landmark.

Usage: python -m tools.maintenance.drop_landmark_from_tps tancrede.tps tancrede_minus_lm3.tps --drop 3
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import replace

import numpy as np

from utils.cli import add_logging_args, log_level_from_args
from core.run_io import setup_console_logging
from core.tps_io import parse_tps, write_tps

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    parser.add_argument("--drop", type=int, default=3, help="0-indexed landmark to drop.")
    add_logging_args(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    setup_console_logging(log_level_from_args(args))

    specimens, errors = parse_tps(args.input_path, strict=False)
    if errors:
        logger.warning("%d unreadable block(s) in %s (skipped)", len(errors), args.input_path)

    out = [replace(s, landmarks=np.delete(s.landmarks, args.drop, axis=0), n_points=s.n_points - 1)
           for s in specimens]
    write_tps(args.output_path, out)
    print(f"Written {len(out)} specimen(s) ({out[0].n_points} landmarks) -> {args.output_path}")


if __name__ == "__main__":
    main()
