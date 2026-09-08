"""verify_tps.py
Checks that a .tps file parses correctly by overlaying its points on the
original image.

Usage:
    python -m tools.verify_tps --tps annotations.tps --sid 0 --out out/check.png --flip-y
"""
from __future__ import annotations

import argparse
import logging

import cv2
import matplotlib.pyplot as plt

from utils.cli import add_logging_args, log_level_from_args
from utils.run_io import setup_console_logging
from core.tps_io import parse_tps

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tps", required=True)
    parser.add_argument("--sid", type=int, default=0)
    parser.add_argument("--flip-y", action="store_true")
    parser.add_argument("--out", default=None)
    add_logging_args(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    setup_console_logging(log_level_from_args(args))

    specimens, errors = parse_tps(args.tps, strict=False)
    if errors:
        logger.warning("%d unreadable block(s) in %s (skipped)", len(errors), args.tps)
    logger.info("%d specimen(s) found in %s", len(specimens), args.tps)

    spec = specimens[args.sid]
    img_path = spec.image_path
    logger.info("Image for specimen #%d: %s", args.sid, img_path)

    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    landmarks = spec.landmarks.copy()
    if args.flip_y:
        landmarks[:, 1] = h - landmarks[:, 1]

    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], c="red", s=20)
    plt.title(f"ID={spec.tps_id} -- inv_id {spec.inv_id} ({w}x{h})")
    plt.axis("off")
    plt.show()

    if args.out is not None:
        plt.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"Check saved to {args.out}")


if __name__ == "__main__":
    main()
