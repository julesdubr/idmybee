"""
Parseur de fichiers .tps (format TPSdig2 / TPSutil).

Format attendu, répété pour chaque spécimen :

    LM=19
    x1 y1
    x2 y2
    ...
    IMAGE=chemin/vers/image.jpg
    ID=00001
    SCALE=0.0123456                (optionnel, absent sur vos données)
    VARIABLES=OrigNum=14           (optionnel, présent sur vos données)

Points importants à vérifier sur VOS données (à valider avec
scripts/verify_tps_annotations.py avant d'aller plus loin) :

1. Convention d'axe Y : TPSdig2 peut stocker les coordonnées avec l'origine
   en bas à gauche (Y croissant vers le haut), alors que OpenCV/PIL utilisent
   l'origine en haut à gauche (Y croissant vers le bas). Si les points
   apparaissent "à l'envers" une fois superposés à l'image, il faut appliquer
   y_image = hauteur_image - y_tps. Le flag `flip_y` gère ce cas.

2. Chemin IMAGE= : peut être relatif au fichier .tps, ou absolu, ou juste un
   nom de fichier à retrouver dans un dossier donné. `resolve_image_path`
   gère ces trois cas.

3. ID != position dans la liste : les ID déclarés dans le fichier .tps ne
   sont PAS forcément dans un ordre croissant, ni contigus, ni uniques selon
   les versions du fichier. NE JAMAIS utiliser `specimens[int(id)]` pour
   retrouver un spécimen par son ID. Utiliser `index_by_id()` (ou
   `index_by_orig_num()`) qui construisent un dict de lookup explicite.

4. VARIABLES= : peut contenir une ou plusieurs paires clé=valeur (ex.
   `VARIABLES=OrigNum=14`), potentiellement séparées par une virgule s'il y
   en a plusieurs. `OrigNum` semble être le numéro d'origine du spécimen
   (probablement l'identifiant commun à travers les 5 photos P1/P2/S1/S2/S3
   d'un même individu -- à confirmer, mais c'est ce champ qu'il faudra
   utiliser plus tard pour regrouper les photos d'un même individu et éviter
   les fuites de données entre P1/P2/S1/S2/S3 lors des splits train/val).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _read_text_robust(path: Path) -> str:
    """Lit un fichier texte en essayant plusieurs encodages dans l'ordre.

    Les fichiers .tps produits par TPSutil/TPSdig2 sur Windows sont souvent
    encodés en cp1252 (Windows-1252), pas en UTF-8 : forcer UTF-8 corrompt
    silencieusement les caractères accentués/spéciaux (ex. "°" devient
    "┬░"). On essaie UTF-8 (strict) d'abord, puis cp1252, puis latin-1 en
    dernier recours (latin-1 ne lève jamais d'erreur, donc toujours un filet
    de sécurité, mais moins fiable que cp1252 pour du texte Windows/français).
    """
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")


def _normalize_filename(name: str) -> str:
    """Normalise un nom de fichier pour un matching robuste aux problèmes
    d'encodage : ne garde que les caractères alphanumériques, en minuscule.
    Ainsi 'N°407_P1.JPG' et sa version corrompue 'N┬░407_P1.JPG' donnent la
    même clé ('n407p1jpg'), donc se retrouvent même si l'encodage du .tps
    était incorrect à un moment de la chaîne.
    """
    normalized = unicodedata.normalize("NFKD", name)
    return re.sub(r"[^a-zA-Z0-9]", "", normalized).lower()


@dataclass
class TpsSpecimen:
    landmarks: np.ndarray  # shape (n_points, 2), dtype float
    image_path: str  # tel que déclaré dans le fichier .tps
    specimen_id: str = ""
    scale: float | None = None
    variables: dict = field(default_factory=dict)  # ex. {"OrigNum": "14"}
    extra: dict = field(default_factory=dict)

    @property
    def orig_num(self) -> str | None:
        return self.variables.get("OrigNum")


def _parse_variables(raw: str) -> dict:
    """Parse le contenu d'un champ VARIABLES=... en dict.

    Gère le cas simple `OrigNum=14` ainsi que plusieurs paires séparées par
    une virgule ou un point-virgule, ex. `OrigNum=14,Sex=F`.
    """
    variables = {}
    for part in re.split(r"[;,]", raw):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            k, _, v = part.partition("=")
            variables[k.strip()] = v.strip()
        else:
            # valeur seule sans clé explicite : on la garde quand même
            variables[part] = ""
    return variables


def parse_tps_file(tps_path: str | Path) -> list[TpsSpecimen]:
    """Parse un fichier .tps et retourne la liste des spécimens."""
    tps_path = Path(tps_path)
    text = _read_text_robust(tps_path)
    lines = [l.strip() for l in text.splitlines() if l.strip() != ""]

    specimens: list[TpsSpecimen] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"LM3?=(\d+)", line, re.IGNORECASE)
        if not m:
            i += 1
            continue

        n_points = int(m.group(1))
        coords = np.zeros((n_points, 2), dtype=float)
        i += 1
        for p in range(n_points):
            parts = lines[i].replace(",", " ").split()
            coords[p, 0] = float(parts[0])
            coords[p, 1] = float(parts[1])
            i += 1

        image_path, specimen_id, scale = "", "", None
        variables = {}
        extra = {}
        # Les champs suivants (IMAGE=, ID=, SCALE=, VARIABLES=, ...) continuent
        # jusqu'au prochain bloc LM= ou la fin du fichier.
        while i < len(lines) and not re.match(r"LM3?=(\d+)", lines[i], re.IGNORECASE):
            if "=" in lines[i]:
                key, _, val = lines[i].partition("=")
                key = key.strip().upper()
                val = val.strip()
                if key == "IMAGE":
                    image_path = val
                elif key == "ID":
                    specimen_id = val
                elif key == "SCALE":
                    try:
                        scale = float(val)
                    except ValueError:
                        scale = None
                elif key == "VARIABLES":
                    variables = _parse_variables(val)
                else:
                    extra[key] = val
            i += 1

        specimens.append(
            TpsSpecimen(
                landmarks=coords,
                image_path=image_path,
                specimen_id=specimen_id,
                scale=scale,
                variables=variables,
                extra=extra,
            )
        )

    duplicate_ids = {
        sid
        for sid in (s.specimen_id for s in specimens)
        if sid and sum(1 for s in specimens if s.specimen_id == sid) > 1
    }
    if duplicate_ids:
        print(
            f"[avertissement] {len(duplicate_ids)} ID(s) en doublon dans {tps_path.name} : "
            f"{sorted(duplicate_ids)[:10]}{'...' if len(duplicate_ids) > 10 else ''} "
            "-> index_by_id() écrasera les doublons, préférer une autre clé si besoin."
        )

    return specimens


def index_by_id(specimens: list[TpsSpecimen]) -> dict[str, TpsSpecimen]:
    """Construit un dict {specimen_id: TpsSpecimen} pour un lookup fiable.

    À utiliser systématiquement à la place de `specimens[int(id)]` : l'ordre
    dans la liste ne correspond PAS à l'ID déclaré dans le fichier .tps.

    ATTENTION : si votre .tps agrège plusieurs espèces/castes (ex. dossiers
    Bombus_pratorum/worker, Bombus_pratorum/queen, ...), l'ID semble se
    répéter d'un sous-dossier à l'autre (numérotation locale, pas globale).
    Dans ce cas cette fonction écrase les doublons et n'est PAS le bon choix
    de clé -- utiliser `index_by_path()` (garanti unique, une entrée par
    photo) ou construire une clé composite (ex. espèce + caste + ID).
    """
    return {s.specimen_id: s for s in specimens if s.specimen_id}


def index_by_path(specimens: list[TpsSpecimen]) -> dict[str, TpsSpecimen]:
    """Construit un dict {image_path déclaré: TpsSpecimen}. Contrairement à
    l'ID, le chemin d'image est normalement unique par photo -- clé à
    privilégier tant que l'unicité de l'ID n'est pas confirmée sur vos données.
    """
    return {s.image_path: s for s in specimens if s.image_path}


def index_by_orig_num(specimens: list[TpsSpecimen]) -> dict[str, TpsSpecimen]:
    """Construit un dict {OrigNum: TpsSpecimen}, utile pour retrouver le
    même individu à travers les différents types de photos (P1/P2/S1/S2/S3)
    une fois que ces fichiers seront combinés."""
    return {s.orig_num: s for s in specimens if s.orig_num}


def build_image_index(images_root: str | Path) -> dict[str, Path]:
    """Scanne une fois `images_root` et construit un index
    {chemin_relatif_normalisé: chemin_réel}, pour un matching robuste face
    aux problèmes d'encodage.

    IMPORTANT : la clé est calculée sur le CHEMIN RELATIF complet (dossiers
    inclus), pas sur le seul nom de fichier. Sur ce jeu de données, la
    numérotation des photos (N<numéro>_P1.JPG) se répète d'une espèce/caste
    à l'autre (ex. Bombus_pratorum/male/N126_P1.JPG et .../worker/N126_P1.JPG
    peuvent coexister) : indexer par nom seul provoquerait des collisions
    massives et un mauvais fichier pourrait être résolu silencieusement.
    """
    images_root = Path(images_root)
    index: dict[str, Path] = {}
    collisions = 0
    for p in images_root.rglob("*"):
        if p.is_file():
            rel = p.relative_to(images_root)
            key = _normalize_filename(str(rel))
            if key in index and index[key] != p:
                collisions += 1
            index[key] = p
    if collisions:
        print(
            f"[avertissement] {collisions} collision(s) de chemins après "
            "normalisation (deux fichiers différents donnent la même clé) : "
            "le matching risque d'être ambigu pour ces fichiers."
        )
    return index


def resolve_image_path(
    specimen: TpsSpecimen,
    images_root: str | Path,
    image_index: dict[str, Path] | None = None,
) -> Path:
    """Résout le chemin réel de l'image sur disque.

    Essaie, dans l'ordre :
    1. chemin absolu déclaré (s'il existe tel quel)
    2. chemin relatif à images_root (jointure directe -- le cas normal)
    3. chemin relatif normalisé via `image_index` (filet de sécurité en cas
       de caractères mal encodés dans le .tps, ex. "°")
    4. en tout dernier recours, recherche par nom de fichier seul -- REFUSÉE
       si le nom existe dans plusieurs dossiers différents (ambigu sur ce
       jeu de données où la numérotation se répète par espèce/caste), pour
       éviter de résoudre silencieusement vers le mauvais fichier.
    """
    images_root = Path(images_root)
    declared = Path(specimen.image_path)

    if declared.is_absolute() and declared.exists():
        return declared

    candidate = images_root / declared
    if candidate.exists():
        return candidate

    if image_index is not None:
        key = _normalize_filename(str(declared))
        if key in image_index:
            return image_index[key]

    matches = list(images_root.rglob(declared.name))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        return matches[0]
        raise FileNotFoundError(
            f"Nom de fichier {declared.name!r} ambigu pour le spécimen "
            f"{specimen.specimen_id!r} : {len(matches)} fichiers différents "
            f"trouvés sous {images_root} (chemin déclaré : {specimen.image_path!r}). "
            "Résolution refusée pour éviter de charger la mauvaise image."
        )

    raise FileNotFoundError(
        f"Image introuvable pour le spécimen {specimen.specimen_id!r} "
        f"(déclarée : {specimen.image_path!r}), recherché sous {images_root}"
    )


def flip_y_coordinates(
    specimens: list[TpsSpecimen], image_heights: dict[str, float]
) -> None:
    """Applique y_image = hauteur - y_tps en place, si la convention TPS est
    origine en bas à gauche. `image_heights` doit mapper image_path -> hauteur.
    """
    for spec in specimens:
        h = image_heights.get(spec.image_path)
        if h is None:
            continue
        spec.landmarks[:, 1] = h - spec.landmarks[:, 1]


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python tps_parser.py fichier.tps")
        sys.exit(1)

    specs = parse_tps_file(sys.argv[1])
    print(f"{len(specs)} spécimens parsés.")
    for s in specs[:3]:
        print(
            f"  ID={s.specimen_id!r} OrigNum={s.orig_num!r} "
            f"image={s.image_path!r} n_points={len(s.landmarks)}"
        )

    by_id = index_by_id(specs)
    print(f"Index par ID construit : {len(by_id)} entrées.")
