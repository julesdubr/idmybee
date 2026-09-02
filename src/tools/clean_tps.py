"""clean_tps.py
Renumbers specimen IDs in a .tps file so they're consecutive starting at 0.

Usage:
    python -m tools.clean_tps --tps annotations.tps --out cleaned.tps
"""
from __future__ import annotations

import argparse
import logging

from utils.cli import add_logging_args, log_level_from_args
from utils.run_io import setup_console_logging
from utils.tps_io import parse_tps

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tps", required=True)
    parser.add_argument("--out", default="cleaned.tps")
    add_logging_args(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    setup_console_logging(log_level_from_args(args))

    specimens, errors = parse_tps(args.tps, strict=False)
    if errors:
        logger.warning("%d unreadable block(s) in %s (skipped)", len(errors), args.tps)
    logger.info("%d specimen(s) found in %s", len(specimens), args.tps)

    with open(args.out, "wb") as f:
        for idx, spec in enumerate(specimens):
            f.write(f"LM={spec.landmarks.shape[0]}\n".encode())
            for x, y in spec.landmarks:
                f.write(f"{x:.5f} {y:.5f}\n".encode())
            f.write(f"IMAGE={spec.image_path}\n".encode())
            f.write(f"ID={idx}\n".encode())

    print(f"{len(specimens)} specimen(s) written -> {args.out}")


if __name__ == "__main__":
    main()
