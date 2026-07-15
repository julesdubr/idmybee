"""Aplatit l'arborescence des photos : déplace les fichiers imbriqués dans un
sous-dossier portant leur propre nom, pour que toutes les photos d'un rôle
(male/worker/queen) soient directement dans son dossier.

Avant :
    espece/Mâle/nom_image/nom_image_P1.jpg
Après :
    espece/male/nom_image_P1.jpg

NOUVELLES FONCTIONNALITÉS :
  1. Renomme les dossiers rôles français en anglais :
     - Male → male
     - Ouvriere → worker
     - Fondatrice → queen
     (renommage effectué à l'intérieur de chaque dossier "espèce")
  2. Nettoie les noms de fichiers :
     - Supprime les caractères non-UTF8 (ex. N°XXX → NXXX)
     - Remplace les espaces par des underscores
     - Simplifie pour une utilisation pandas-safe en tant que IDs
  3. Gestion HEIC et dossiers "unknown" :
     - Les .heic sont déplacés vers espece/role/heic/
     - Les images présentes dans des sous-dossiers non-plats (non matching nom_parent == base)
       sont déplacées vers espece/role/unknown/

Les photos déjà à plat (directement dans role/) sont laissées inchangées.
Détection indépendante de la profondeur exacte espece/role.

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
import unicodedata
from pathlib import Path

# Suffixes de version de photo utilisés dans le projet (2 appareils photo + 3 smartphones)
PHOTO_RE = re.compile(
    r"^(?P<base>.+)_(?:P\d+|S\d+)\.(?:jpe?g|png|heic)$", re.IGNORECASE
)

# Mappage rôles français → anglais
ROLE_TRANSLATIONS = {
    "mec": "male",
    "Ouvriere": "worker",
    "Fondatrice": "queen",
}


def sanitize_filename(filename: str) -> str:
    """Nettoie un nom de fichier pour le rendre pandas-safe et cross-platform.

    - Supprime les caractères diacritiques (é → e)
    - Remplace les caractères spéciaux par underscore
    - Convertit en ASCII
    - Gère les cas comme N°XXX → NXXX
    """
    # Décompose les caractères accentués (NFKD), puis garde que l'ascii
    normalized = unicodedata.normalize("NFKD", filename)
    ascii_str = "".join(c for c in normalized if ord(c) < 128)

    # Remplace les espaces par des underscores
    ascii_str = ascii_str.replace(" ", "_")

    # Supprime ou remplace les caractères spéciaux (garde alphanumérique, underscore, point, tiret)
    # Exception : traite les cas comme "N°XXX" (° disparaît, N et XXX restent collés)
    cleaned = re.sub(r"[^a-zA-Z0-9._\-]", "", ascii_str)

    # Évite les underscores multiples consécutifs
    cleaned = re.sub(r"_{2,}", "_", cleaned)

    return cleaned


def rename_role_dirs(root: Path, apply: bool) -> dict[Path, Path]:
    """Renomme les dossiers rôles français en anglais à l'intérieur de chaque espèce.

    Retourne un mapping {ancienne_path: nouvelle_path} pour les rôles renommés.
    """
    renamed: dict[Path, Path] = {}

    for species in root.iterdir():
        if not species.is_dir():
            continue
        for old_name, new_name in ROLE_TRANSLATIONS.items():
            old_path = species / old_name
            new_path = species / new_name

            if old_path.exists() and old_path.is_dir():
                if new_path.exists():
                    print(f"[avertissement] {new_path} existe déjà, on garde {old_path} inchangé.")
                    continue

                if apply:
                    print(f"  Renommage : {old_path} → {new_path}")
                    old_path.rename(new_path)
                    renamed[old_path] = new_path
                else:
                    print(f"  [simulation] {old_path} → {new_path}")
                    renamed[old_path] = new_path

    return renamed


def flatten(root: Path, apply: bool) -> None:
    root = Path(root)

    # Étape 1 : Renommer les dossiers rôles français (par espèce)
    print("=== ÉTAPE 1 : Renommage des dossiers rôles (français → anglais) ===\n")
    role_renames = rename_role_dirs(root, apply)

    if not role_renames and not apply:
        print("  [simulation] Aucun dossier rôle français trouvé.\n")
    elif not role_renames and apply:
        print("  Aucun dossier rôle français trouvé.\n")
    else:
        print()

    # Étape 2 : Aplatir l'arborescence et nettoyer les noms
    print("=== ÉTAPE 2 : Aplatissement et nettoyage des noms ===\n")

    moved, conflicts, renamed_files = 0, [], []
    parent_dirs = set()

    # Parcours : pour chaque espèce, pour chaque dossier rôle (réel), traiter les fichiers dedans
    for species in root.iterdir():
        if not species.is_dir():
            continue
        for role_dir in species.iterdir():
            if not role_dir.is_dir():
                continue

            # Déterminer le chemin cible effectif (si en simulation un renommage a été prévu)
            effective_role_dir = role_renames.get(role_dir, role_dir)

            # Parcourir tout ce qui est sous role_dir (récursif)
            for f in role_dir.rglob("*"):
                if not f.is_file():
                    continue

                # Ignore les fichiers déjà à plat (directement dans role_dir)
                if f.parent == role_dir:
                    continue

                ext = f.suffix.lower()
                cleaned_name = sanitize_filename(f.name)

                # Déterminer la destination :
                # - .heic -> role/heic/
                # - nested matching base -> role/
                # - autres images dans sous-dossiers -> role/unknown/
                m = PHOTO_RE.match(f.name)
                if ext == ".heic":
                    target_dir = effective_role_dir / "heic"
                elif m and f.parent.name.lower() == m.group("base").lower():
                    target_dir = effective_role_dir
                else:
                    # autres fichiers/images dans des sous-dossiers non-plats
                    target_dir = effective_role_dir / "unknown"

                target = target_dir / cleaned_name
                parent_dirs.add(f.parent)

                if cleaned_name != f.name:
                    renamed_files.append((f, cleaned_name))

                if target.exists():
                    conflicts.append((f, target))
                    continue

                if apply:
                    # créer le dossier cible si nécessaire
                    target_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(f), str(target))
                    # Affichage concis
                    if cleaned_name != f.name:
                        print(f"  {f.parent.name}/{f.name}  →  {target.parent.name}/{cleaned_name}")
                    else:
                        print(f"  {f.parent.name}/{f.name}  →  {target.parent.name}/{f.name}")
                else:
                    # Simulation : afficher la destination prévue (en utilisant effective_role_dir)
                    if cleaned_name != f.name:
                        print(f"  [simulation] {f.parent.name}/{f.name}  →  {target.parent.name}/{cleaned_name}")
                    else:
                        print(f"  [simulation] {f.parent.name}/{f.name}  →  {target.parent.name}/{f.name}")

                moved += 1

    if renamed_files:
        print(f"\n{len(renamed_files)} fichier(s) renommé(s) :")
        for f, new in renamed_files[:20]:
            print(f"    {f}  →  {new}")

    if conflicts:
        print(
            f"\n[avertissement] {len(conflicts)} conflit(s) : un fichier existe déjà à la "
            "destination, ceux-ci n'ont PAS été déplacés (à vérifier manuellement) :"
        )
        for f, target in conflicts[:20]:
            print(f"    {f}  →  {target} (déjà existant)")

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
            f"{moved} fichier(s) déplacé(s) et nettoyé(s), {len(conflicts)} conflit(s) évité(s), "
            f"{removed_dirs} dossier(s) vide(s) supprimé(s)."
        )
        if role_renames:
            print(f"{len(role_renames)} dossier(s) rôle renommé(s).")
    else:
        print(
            f"[SIMULATION] {moved} fichier(s) seraient déplacés et nettoyé(s), {len(conflicts)} conflit(s) détecté(s). "
            "Relancer avec --apply pour exécuter réellement."
        )
        if role_renames:
            print(f"[SIMULATION] {len(role_renames)} dossier(s) rôle seraient renommé(s).")


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
        help="Applique réellement les déplacements et renommages (sinon : simulation seule)",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Dossier introuvable : {root}")

    flatten(root, apply=args.apply)
# ...existing code...