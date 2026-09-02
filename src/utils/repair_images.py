"""repair_images.py
Scans a directory of images and rewrites each recoverable JPEG in place
(applies EXIF orientation, re-encodes at quality=100/subsampling=0) --
useful before a pipeline step that reads pixels directly and would
otherwise silently inherit a wrong orientation or a corrupt-but-openable
file.

Files that fail to open/save are reported, never dropped silently.

Usage:
    python -m utils.repair_images data/Bombus/images
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from PIL import Image, ImageOps
from tqdm import tqdm

from utils.cli import add_dataset_positional, add_logging_args, log_level_from_args
from utils.run_io import setup_console_logging

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".JPG", ".JPEG"}


def iter_images(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix in IMAGE_EXTENSIONS]


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Repair recoverable JPEG files in place (EXIF orientation, re-encode).")
    add_dataset_positional(parser, help="Root directory to scan for images.")
    add_logging_args(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    setup_console_logging(log_level_from_args(args))

    paths = iter_images(args.dataset)
    logger.info("%d image(s) found under %s", len(paths), args.dataset)

    failed = []
    for image_path in tqdm(paths):
        try:
            ImageOps.exif_transpose(Image.open(image_path)).save(
                image_path, "JPEG", subsampling=0, quality=100
            )
        except Exception as e:
            failed.append((image_path, e))

    if failed:
        logger.warning("%d file(s) could not be repaired:", len(failed))
        for path, e in failed[:10]:
            logger.warning("  %s: %s", path, e)
        if len(failed) > 10:
            logger.warning("  ... and %d more.", len(failed) - 10)

    print(f"Done. {len(paths) - len(failed)}/{len(paths)} image(s) repaired.")


if __name__ == "__main__":
    main()
