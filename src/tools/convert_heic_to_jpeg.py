"""convert_heic_to_jpeg.py
Converts HEIC images to JPEG and places them in the parent role folder.

Expected layout:
  organized/species/role/images.jpg
  organized/species/role/heic/images.heic

HEIC images are converted and stored directly in the role folder, then the
now-empty heic/ folders are removed.

Dependencies:
  pip install pillow pillow-heif

Usage:
    python -m tools.convert_heic_to_jpeg organized
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

from PIL import Image

from utils.cli import add_logging_args, log_level_from_args
from utils.run_io import setup_console_logging

logger = logging.getLogger(__name__)

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    print("pillow-heif is not installed.")
    print("Install with: pip install pillow-heif")
    sys.exit(1)


def find_and_convert_heic(root_dir: Path) -> dict:
    """Walks the tree and converts HEIC to JPEG, then removes the emptied
    heic folders. Returns a dict of conversion statistics."""
    if not root_dir.exists():
        logger.error("Directory %r does not exist.", str(root_dir))
        return {"error": f"Directory {root_dir} not found"}

    stats = {"converted": 0, "failed": 0, "skipped": 0, "total_found": 0, "deleted_dirs": 0}
    heic_dirs_to_delete = []

    # Walk: organized/species/role/heic/
    for species_dir in root_dir.iterdir():
        if not species_dir.is_dir():
            continue

        for role_dir in species_dir.iterdir():
            if not role_dir.is_dir():
                continue

            heic_dir = role_dir / "heic"
            if not heic_dir.exists():
                continue

            heic_files = list(heic_dir.glob("*.heic")) + list(heic_dir.glob("*.HEIC"))
            if not heic_files:
                continue

            heic_dirs_to_delete.append(heic_dir)
            logger.info("%s/%s/heic (%d file(s))", species_dir.name, role_dir.name, len(heic_files))

            for heic_file in heic_files:
                stats["total_found"] += 1
                jpeg_filename = heic_file.stem + ".jpg"
                jpeg_path = role_dir / jpeg_filename

                if jpeg_path.exists():
                    logger.info("  skipped: %s -> %s (already exists)", heic_file.name, jpeg_filename)
                    stats["skipped"] += 1
                    continue

                try:
                    img = Image.open(heic_file)
                    if img.mode in ("RGBA", "LA", "P"):
                        rgb_img = Image.new("RGB", img.size, (255, 255, 255))
                        rgb_img.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
                        img = rgb_img

                    img.save(jpeg_path, "JPEG", quality=95)
                    logger.info("  converted: %s -> %s", heic_file.name, jpeg_filename)
                    stats["converted"] += 1

                except Exception as e:
                    logger.warning("  failed: %s: %s", heic_file.name, e)
                    stats["failed"] += 1

    if stats["failed"] == 0:
        logger.info("Removing heic folders...")
        for heic_dir in heic_dirs_to_delete:
            try:
                shutil.rmtree(heic_dir)
                logger.info("  removed: %s/", heic_dir.relative_to(root_dir))
                stats["deleted_dirs"] += 1
            except Exception as e:
                logger.warning("  could not remove %s/: %s", heic_dir.relative_to(root_dir), e)
    else:
        logger.warning("Failed conversion(s) detected -- heic folders kept for inspection.")

    return stats


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Convert HEIC images to JPEG in an organized/ tree (see module docstring).")
    parser.add_argument("dataset", type=Path, nargs="?", default=Path("organized"),
                         help="Root directory to scan (default: organized).")
    add_logging_args(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    setup_console_logging(log_level_from_args(args))

    stats = find_and_convert_heic(args.dataset)
    if "error" in stats:
        sys.exit(1)

    print("\n" + "=" * 50)
    print("Conversion summary")
    print("=" * 50)
    print(f"Total found         : {stats['total_found']}")
    print(f"Converted           : {stats['converted']}")
    print(f"Skipped (existing)  : {stats['skipped']}")
    print(f"Errors              : {stats['failed']}")
    print(f"Folders removed     : {stats['deleted_dirs']}")
    print("=" * 50)

    if stats["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
