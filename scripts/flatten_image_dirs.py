"""Aplatit l'arborescence des photos : déplace les fichiers imbriqués dans un
sous-dossier portant leur propre nom, pour que toutes les photos d'un rôle
(male/worker/queen) soient directement dans son dossier.

Avant :
    espece/role/nom_image/nom_image_P1.jpg
Après :
    espece/role/nom_image_P1.jpg

Les photos déjà à plat (directement dans role/) sont laissées inchangées.
Détection indépendante de la profondeur exacte espece/role (on ne se base
que sur : "le fichier est dans un dossier qui porte exactement son propre
nom de base").

SÉCURITÉ : par défaut, le script ne fait qu'afficher ce qu'il ferait (mode
simulation). Il faut passer --apply pour réellement déplacer les fichiers.

Usage :
    python scripts/flatten_image_dirs.py --root data/images/groupe_images
    python scripts/flatten_image_dirs.py --root data/images/groupe_images --apply
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

# Suffixes de version de photo utilisés dans le projet (2 appareils photo + 3 smartphones)
PHOTO_RE = re.compile(
    r"^(?P<base>.+)_(?:P1|P2|P3|S1|S2|S3)\.(?:jpe?g|png)$", re.IGNORECASE
)


def find_nested_photos(root: Path) -> list[Path]:
    """Retourne les fichiers situés dans un sous-dossier portant exactement
    leur propre nom de base (ex. nom_image/nom_image_P1.jpg)."""
    candidates = []
    for f in root.rglob("*"):
        if not f.is_file():
            continue
        m = PHOTO_RE.match(f.name)
        if not m:
            continue
        if f.parent.name.lower() == m.group("base").lower():
            candidates.append(f)
    return candidates


def flatten(root: Path, apply: bool) -> None:
    candidates = find_nested_photos(root)
    print(f"{len(candidates)} photo(s) imbriquée(s) trouvée(s) à aplatir.\n")

    moved, conflicts = 0, []
    parent_dirs = set()

    for f in candidates:
        target_dir = f.parent.parent
        target = target_dir / f.name
        parent_dirs.add(f.parent)

        if target.exists():
            conflicts.append((f, target))
            continue

        if apply:
            shutil.move(str(f), str(target))
        else:
            print(f"  [simulation] {f}  ->  {target}")
        moved += 1

    if conflicts:
        print(
            f"\n[avertissement] {len(conflicts)} conflit(s) : un fichier existe déjà à la "
            "destination, ceux-ci n'ont PAS été déplacés (à vérifier manuellement) :"
        )
        for f, target in conflicts[:20]:
            print(f"    {f}  ->  {target} (déjà existant)")

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
            f"\n[avertissement] {len(non_empty_dirs)} dossier(s) non supprimé(s) car il reste "
            "d'autres fichiers dedans (à vérifier manuellement) :"
        )
        for d, remaining in non_empty_dirs[:10]:
            print(f"    {d} : {[p.name for p in remaining]}")

    print()
    if apply:
        print(
            f"{moved} fichier(s) déplacé(s), {len(conflicts)} conflit(s) évité(s), "
            f"{removed_dirs} dossier(s) vide(s) supprimé(s)."
        )
    else:
        print(
            f"[SIMULATION] {moved} fichier(s) seraient déplacés, {len(conflicts)} conflit(s) détecté(s). "
            "Relancer avec --apply pour exécuter réellement."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        required=True,
        help="Racine à parcourir, ex. data/images/groupe_images",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Applique réellement les déplacements (sinon : simulation seule)",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Dossier introuvable : {root}")

    flatten(root, apply=args.apply)
