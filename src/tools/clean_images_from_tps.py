import argparse
from pathlib import Path


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
}


def read_tps_images(tps_path: Path, source_dir: Path) -> set[Path]:
    """Extract IMAGE= paths from the TPS and make them relative to source_dir."""
    referenced = set()

    with tps_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()

            if not line.startswith("IMAGE="):
                continue

            image_path = Path(line[len("IMAGE="):].strip())

            try:
                relative_path = image_path.relative_to(source_dir)
            except ValueError:
                # Fallback if the absolute path in the TPS does not match
                # the provided source directory.
                relative_path = Path(image_path.name)

            referenced.add(relative_path)

    return referenced


def find_images(source_dir: Path) -> set[Path]:
    """Find all image files recursively."""
    return {
        path.relative_to(source_dir)
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def find_empty_directories(source_dir: Path) -> list[Path]:
    """
    Find empty directories recursively.

    Directories are returned from deepest to shallowest so that a directory
    containing only an empty subdirectory can also be removed.
    """
    directories = [
        path
        for path in source_dir.rglob("*")
        if path.is_dir()
    ]

    empty = []

    for path in sorted(directories, key=lambda p: len(p.parts), reverse=True):
        try:
            if not any(path.iterdir()):
                empty.append(path)
        except OSError:
            pass

    return empty


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Supprime les images absentes d'un fichier TPS "
            "et les sous-répertoires devenus vides."
        )
    )

    parser.add_argument(
        "source_dir",
        type=Path,
        help="Répertoire contenant les images",
    )

    parser.add_argument(
        "tps",
        type=Path,
        help="Fichier TPS de référence",
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help="Effectue réellement les suppressions. Sans cette option, dry-run.",
    )

    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    tps_path = args.tps.resolve()

    if not source_dir.is_dir():
        parser.error(f"Répertoire introuvable : {source_dir}")

    if not tps_path.is_file():
        parser.error(f"Fichier TPS introuvable : {tps_path}")

    print(f"Répertoire source : {source_dir}")
    print(f"Fichier TPS       : {tps_path}")
    print()

    referenced = read_tps_images(tps_path, source_dir)
    existing = find_images(source_dir)

    to_delete = existing - referenced
    not_found = referenced - existing

    print(f"Images présentes dans le dossier : {len(existing)}")
    print(f"Images référencées dans le TPS    : {len(referenced)}")
    print(f"Images à supprimer                : {len(to_delete)}")
    print(f"Images absentes mais référencées  : {len(not_found)}")
    print()

    if to_delete:
        print("=" * 70)
        print("IMAGES NON RÉFÉRENCÉES")
        print("=" * 70)

        for path in sorted(to_delete):
            print(path)

        print()


    if not_found:
            print("=" * 70)
            print("IMAGES RÉFÉRENCÉES DANS LE TPS MAIS INTROUVABLES")
            print("=" * 70)
    
            for path in sorted(not_found):
                print(path)
    
            print()


    # ------------------------------------------------------------------
    # Dry-run
    # ------------------------------------------------------------------

    if not args.apply:
        # Simulate the image deletion first, then determine which
        # directories would become empty.
        remaining_images = existing - to_delete

        # Determine directories that would contain no remaining files.
        directories = [
            path
            for path in source_dir.rglob("*")
            if path.is_dir()
        ]

        empty_dirs = []

        for directory in sorted(
            directories,
            key=lambda p: len(p.parts),
            reverse=True,
        ):
            relative = directory.relative_to(source_dir)

            # Check whether any remaining image belongs to this directory.
            contains_image = any(
                image.parent == relative
                or relative in image.parents
                for image in remaining_images
            )

            # A directory is considered empty only if it contains no files
            # and no non-empty subdirectories after deletion.
            if not contains_image:
                try:
                    children = list(directory.iterdir())

                    # Children that would remain after image deletion
                    remaining_children = []

                    for child in children:
                        if child.is_file():
                            if child.suffix.lower() not in IMAGE_EXTENSIONS:
                                remaining_children.append(child)
                            elif child.relative_to(source_dir) in remaining_images:
                                remaining_children.append(child)
                        elif child.is_dir():
                            child_rel = child.relative_to(source_dir)

                            # Keep the directory if it contains a remaining image
                            child_contains_image = any(
                                child_rel == image.parent
                                or child_rel in image.parents
                                for image in remaining_images
                            )

                            if child_contains_image:
                                remaining_children.append(child)

                    if not remaining_children:
                        empty_dirs.append(relative)

                except OSError:
                    pass

        print("=" * 70)
        print("RÉPERTOIRES QUI SERAIENT SUPPRIMÉS")
        print("=" * 70)

        if empty_dirs:
            for path in empty_dirs:
                print(path)
        else:
            print("Aucun")

        print()
        print("=" * 70)
        print("DRY-RUN")
        print("=" * 70)
        print(
            f"{len(to_delete)} image(s) seraient supprimée(s)."
        )
        print(
            f"{len(empty_dirs)} répertoire(s) vide(s) seraient supprimé(s)."
        )
        print("Aucun fichier n'a été modifié.")
        print()
        print("Pour effectuer réellement les suppressions :")
        print(
            f'python {Path(__file__).name} '
            f'"{source_dir}" "{tps_path}" --apply'
        )

        return

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------

    print("=" * 70)
    print("SUPPRESSION DES IMAGES")
    print("=" * 70)

    deleted = 0
    errors = 0

    for relative_path in sorted(to_delete):
        path = source_dir / relative_path

        try:
            path.unlink()
            print(f"Supprimé : {relative_path}")
            deleted += 1
        except OSError as e:
            print(f"ERREUR    : {relative_path} -> {e}")
            errors += 1

    # ------------------------------------------------------------------
    # Remove empty directories
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("SUPPRESSION DES RÉPERTOIRES VIDES")
    print("=" * 70)

    empty_dirs = find_empty_directories(source_dir)

    removed_dirs = 0

    for directory in empty_dirs:
        try:
            directory.rmdir()
            print(f"Supprimé : {directory.relative_to(source_dir)}")
            removed_dirs += 1
        except OSError as e:
            print(
                f"ERREUR    : "
                f"{directory.relative_to(source_dir)} -> {e}"
            )

    print()
    print("=" * 70)
    print("TERMINÉ")
    print("=" * 70)
    print(f"Images supprimées      : {deleted}")
    print(f"Répertoires supprimés  : {removed_dirs}")
    print(f"Erreurs                : {errors}")


if __name__ == "__main__":
    main()