"""flatten_image_dirs.py
Flattens the photo tree: moves files nested in a subfolder that shares
their own name, so all photos of a role (male/worker/queen) sit directly
in its folder.

Before:
    species/Male/image_name/image_name_P1.jpg
After:
    species/male/image_name_P1.jpg

FEATURES:
  1. Renames French role folders to English:
     - mec -> male
     - Ouvriere -> worker
     - Fondatrice -> queen
     (renaming happens inside each "species" folder)
  2. Cleans up filenames:
     - Removes non-UTF8 characters (e.g. N°XXX -> NXXX)
     - Replaces spaces with underscores
     - Simplifies for pandas-safe use as IDs
  3. HEIC and "unknown" folder handling:
     - .heic files are moved to species/role/heic/
     - Images found in non-flat subfolders (parent folder name doesn't
       match the base filename) are moved to species/role/unknown/

Photos already flat (directly in role/) are left unchanged. Detection is
independent of the exact species/role depth.

SAFETY: by default the script only prints what it would do (dry-run mode).
Pass --apply to actually move files.

Usage:
    python -m tools.flatten_image_dirs data/images/groupe_images
    python -m tools.flatten_image_dirs data/images/groupe_images --apply
"""
from __future__ import annotations

import argparse
import logging
import re
import shutil
import unicodedata
from pathlib import Path

from utils.cli import add_dataset_positional, add_logging_args, log_level_from_args
from utils.run_io import setup_console_logging

logger = logging.getLogger(__name__)

# Photo-version suffixes used in the project (2 cameras + 3 smartphones)
PHOTO_RE = re.compile(
    r"^(?P<base>.+)_(?:P\d+|S\d+)\.(?:jpe?g|png|heic)$", re.IGNORECASE
)

# French -> English role folder mapping
ROLE_TRANSLATIONS = {
    "mec": "male",
    "Ouvriere": "worker",
    "Fondatrice": "queen",
}


def sanitize_filename(filename: str) -> str:
    """Cleans up a filename to make it pandas-safe and cross-platform.

    - Strips diacritics (e -> e)
    - Replaces special characters with underscore
    - Converts to ASCII
    - Handles cases like N°XXX -> NXXX
    """
    # Decomposes accented characters (NFKD), then keeps ASCII only
    normalized = unicodedata.normalize("NFKD", filename)
    ascii_str = "".join(c for c in normalized if ord(c) < 128)

    # Replaces spaces with underscores
    ascii_str = ascii_str.replace(" ", "_")

    # Strips or replaces special characters (keeps alphanumeric, underscore, dot, hyphen)
    # Handles cases like "N°XXX" (° disappears, N and XXX stay glued together)
    cleaned = re.sub(r"[^a-zA-Z0-9._\-]", "", ascii_str)

    # Avoids multiple consecutive underscores
    cleaned = re.sub(r"_{2,}", "_", cleaned)

    return cleaned


def rename_role_dirs(root: Path, apply: bool) -> dict[Path, Path]:
    """Renames French role folders to English inside each species folder.

    Returns a mapping {old_path: new_path} for the renamed roles."""
    renamed: dict[Path, Path] = {}

    for species in root.iterdir():
        if not species.is_dir():
            continue
        for old_name, new_name in ROLE_TRANSLATIONS.items():
            old_path = species / old_name
            new_path = species / new_name

            if old_path.exists() and old_path.is_dir():
                if new_path.exists():
                    print(f"[warning] {new_path} already exists, leaving {old_path} unchanged.")
                    continue

                if apply:
                    print(f"  Renamed: {old_path} -> {new_path}")
                    old_path.rename(new_path)
                    renamed[old_path] = new_path
                else:
                    print(f"  [dry-run] {old_path} -> {new_path}")
                    renamed[old_path] = new_path

    return renamed


def flatten(root: Path, apply: bool) -> None:
    root = Path(root)

    # Step 1: rename French role folders (per species)
    print("=== STEP 1: renaming role folders (French -> English) ===\n")
    role_renames = rename_role_dirs(root, apply)

    if not role_renames and not apply:
        print("  [dry-run] No French role folder found.\n")
    elif not role_renames and apply:
        print("  No French role folder found.\n")
    else:
        print()

    # Step 2: flatten the tree and clean up names
    print("=== STEP 2: flattening and cleaning up names ===\n")

    moved, conflicts, renamed_files = 0, [], []
    parent_dirs = set()

    # Walk: for each species, for each (real) role folder, process its files
    for species in root.iterdir():
        if not species.is_dir():
            continue
        for role_dir in species.iterdir():
            if not role_dir.is_dir():
                continue

            # Effective target path (if a rename was planned, even in dry-run)
            effective_role_dir = role_renames.get(role_dir, role_dir)

            # Walk everything under role_dir (recursive)
            for f in role_dir.rglob("*"):
                if not f.is_file():
                    continue

                # Skips files already flat (directly in role_dir)
                if f.parent == role_dir:
                    continue

                ext = f.suffix.lower()
                cleaned_name = sanitize_filename(f.name)

                # Determines the destination:
                # - .heic -> role/heic/
                # - nested matching base -> role/
                # - other images in subfolders -> role/unknown/
                m = PHOTO_RE.match(f.name)
                if ext == ".heic":
                    target_dir = effective_role_dir / "heic"
                elif m and f.parent.name.lower() == m.group("base").lower():
                    target_dir = effective_role_dir
                else:
                    # other files/images in non-flat subfolders
                    target_dir = effective_role_dir / "unknown"

                target = target_dir / cleaned_name
                parent_dirs.add(f.parent)

                if cleaned_name != f.name:
                    renamed_files.append((f, cleaned_name))

                if target.exists():
                    conflicts.append((f, target))
                    continue

                if apply:
                    target_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(f), str(target))
                    if cleaned_name != f.name:
                        print(f"  {f.parent.name}/{f.name}  ->  {target.parent.name}/{cleaned_name}")
                    else:
                        print(f"  {f.parent.name}/{f.name}  ->  {target.parent.name}/{f.name}")
                else:
                    # Dry-run: shows the planned destination (using effective_role_dir)
                    if cleaned_name != f.name:
                        print(f"  [dry-run] {f.parent.name}/{f.name}  ->  {target.parent.name}/{cleaned_name}")
                    else:
                        print(f"  [dry-run] {f.parent.name}/{f.name}  ->  {target.parent.name}/{f.name}")

                moved += 1

    if renamed_files:
        print(f"\n{len(renamed_files)} file(s) renamed:")
        for f, new in renamed_files[:20]:
            print(f"    {f}  ->  {new}")

    if conflicts:
        print(
            f"\n[warning] {len(conflicts)} conflict(s): a file already exists at the "
            "destination, these were NOT moved (check manually):"
        )
        for f, target in conflicts[:20]:
            print(f"    {f}  ->  {target} (already exists)")

    removed_dirs, non_empty_dirs = 0, []
    if apply:
        for d in parent_dirs:
            try:
                if d.exists():
                    remaining = list(d.iterdir())
                    if not remaining:
                        d.rmdir()
                        removed_dirs += 1
                    else:
                        non_empty_dirs.append((d, remaining))
            except OSError:
                pass

    if non_empty_dirs:
        print(
            f"\n[warning] {len(non_empty_dirs)} folder(s) not removed because other files "
            "remain inside (check manually):"
        )
        for d, remaining in non_empty_dirs[:10]:
            print(f"    {d}: {[p.name for p in remaining]}")

    print()
    if apply:
        print(
            f"{moved} file(s) moved and cleaned, {len(conflicts)} conflict(s) avoided, "
            f"{removed_dirs} empty folder(s) removed."
        )
        if role_renames:
            print(f"{len(role_renames)} role folder(s) renamed.")
    else:
        print(
            f"[DRY-RUN] {moved} file(s) would be moved and cleaned, {len(conflicts)} conflict(s) detected. "
            "Rerun with --apply to actually execute."
        )
        if role_renames:
            print(f"[DRY-RUN] {len(role_renames)} role folder(s) would be renamed.")


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Flatten a species/role photo tree (see module docstring).")
    add_dataset_positional(parser, help="Root to walk, e.g. data/images/groupe_images")
    parser.add_argument("--apply", action="store_true",
                         help="Actually apply the moves and renames (default: dry-run only)")
    add_logging_args(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    setup_console_logging(log_level_from_args(args))

    if not args.dataset.exists():
        raise SystemExit(f"Folder not found: {args.dataset}")

    flatten(args.dataset, apply=args.apply)


if __name__ == "__main__":
    main()
