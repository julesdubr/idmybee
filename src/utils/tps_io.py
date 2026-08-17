"""tps_io.py
Lecture/écriture de fichiers .tps (landmarks 2D).

Format, un bloc par spécimen :
    LM=19
    123.40 567.80
    ...              (n_points lignes "x y")
    IMAGE=relative/path/to/image.jpg
    ID=0
    COMMENT=image_id=...;specimen_id=...     (optionnel, notre convention)

Un ImageLandmarks représente les landmarks d'UNE PHOTO (une ligne de
crops.csv/landmarks.csv), pas d'un spécimen biologique : un même
specimen_id peut avoir plusieurs photos, donc plusieurs entrées TPS.
`tps_id` (le champ ID=) doit être un entier unique par photo -- voir
image_id_to_sid() plus bas.

`image_id` (la vraie clé, une chaîne hex) et `specimen_id` ne sont pas des
champs standards du format tps. On les persiste dans un COMMENT= -- un
champ tps prévu pour du texte libre, explicitement ignoré par
geomorph::readland.tps ("all other information... comments, variables,
radii, etc. is ignored") donc sans risque pour la compatibilité R -- pour
éviter d'avoir à rejoindre images.csv/specimens.csv à chaque lecture. Un
TPS écrit avant ce champ (ou par un outil tiers) n'aura pas de COMMENT= :
image_id/specimen_id restent alors None après parse_tps, et l'appelant
rejoint via utils.dataset comme avant.

parse_tps ne lève jamais d'exception en mode non-strict : les blocs
malformés sont sautés et retournés dans `errors` (jamais avalés en
silence). En mode strict (défaut), la première anomalie lève
TpsParseException.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ImageLandmarks:
    n_points: int
    landmarks: np.ndarray           # (n_points, 2)
    image_path: str
    tps_id: int                     # valeur brute du TPS ID= -- entier unique par PHOTO
    image_id: str | None = None     # identifiant canonique (hash hex, Phase 0) si connu
    specimen_id: str | None = None  # idem -- None si à rejoindre via images.csv/specimens.csv

    @classmethod
    def from_image(
        cls, n_points: int, landmarks: np.ndarray, image_path: str,
        image_id: str, specimen_id: str | None = None,
    ) -> "ImageLandmarks":
        """Construit à partir d'un image_id connu (ex: predict_unet.py, qui
        l'a déjà via crops.csv) : calcule tps_id automatiquement et persiste
        image_id/specimen_id dans le fichier (COMMENT=) pour que les
        lectures futures n'aient plus besoin de rejoindre le manifest."""
        return cls(
            n_points=n_points, landmarks=landmarks, image_path=image_path,
            tps_id=image_id_to_sid(image_id), image_id=image_id, specimen_id=specimen_id,
        )


@dataclass
class TpsParseError:
    specimen_index: int
    line_no: int
    message: str


class TpsParseException(Exception):
    pass


def image_id_to_sid(image_id: str) -> int:
    """Encode un image_id (hash hex, ex: sha256 tronqué) en entier utilisable
    comme ImageLandmarks.tps_id / TPS ID=. Utilisé à l'écriture et pour
    rejoindre un TPS à images.csv quand image_id n'est pas déjà connu (pas
    de COMMENT=) -- toujours dans ce sens (image_id -> entier), jamais
    l'inverse pour une comparaison (voir sid_to_image_id)."""
    return int(image_id, 16)


def sid_to_image_id(sid: int) -> str:
    """Tentative d'inverse de image_id_to_sid, pour l'affichage/debug
    UNIQUEMENT -- jamais pour rejoindre des données. `hex(sid)[2:]` ne
    restitue pas les zéros initiaux éventuels de l'image_id d'origine, donc
    peut différer du vrai image_id même quand sid est correct. Pour
    retrouver un image_id fiable : COMMENT= s'il est présent, sinon une
    jointure via images.csv (image_id_to_sid appliqué à chaque ligne,
    jamais l'inverse)."""
    return hex(sid)[2:]


def _parse_comment(comment: str) -> dict[str, str]:
    """Décode notre mini-format 'clé=valeur;clé=valeur' d'un COMMENT=.
    Ignore silencieusement ce qui n'y ressemble pas : COMMENT= est du texte
    libre selon le format tps, un fichier tiers (ou plus ancien) peut y
    mettre autre chose, ou rien."""
    fields: dict[str, str] = {}
    for part in comment.split(";"):
        if "=" not in part:
            continue
        key, _, value = part.partition("=")
        key, value = key.strip(), value.strip()
        if key:
            fields[key] = value
    return fields


def _format_comment(image_id: str | None, specimen_id: str | None) -> str | None:
    parts = []
    if image_id is not None:
        parts.append(f"image_id={image_id}")
    if specimen_id is not None:
        parts.append(f"specimen_id={specimen_id}")
    return ";".join(parts) if parts else None


def parse_tps(path: str | Path, strict: bool = True) -> tuple[list[ImageLandmarks], list[TpsParseError]]:
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    specimens: list[ImageLandmarks] = []
    errors: list[TpsParseError] = []
    i, specimen_index, n_lines = 0, 0, len(lines)

    def fail_or_record(msg: str, line_no: int) -> None:
        errors.append(TpsParseError(specimen_index, line_no, msg))
        if strict:
            raise TpsParseException(f"{msg} (ligne {line_no})")

    while i < n_lines:
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if not line.upper().startswith("LM="):
            fail_or_record(f"'LM=' attendu, trouvé : {line!r}", i + 1)
            i += 1
            continue

        block_start_line = i + 1
        try:
            n_points = int(line.split("=", 1)[1].strip())
        except (IndexError, ValueError) as e:
            fail_or_record(f"Nombre de points illisible dans {line!r} ({e})", block_start_line)
            i += 1
            specimen_index += 1
            continue

        i += 1
        coords: list[tuple[float, float]] = []
        block_ok = True
        while len(coords) < n_points and i < n_lines:
            coord_line = lines[i].strip()
            i += 1
            if not coord_line:
                continue
            parts = coord_line.replace(",", " ").split()
            if len(parts) < 2:
                fail_or_record(f"Landmark malformé : {coord_line!r}", i)
                block_ok = False
                continue
            try:
                coords.append((float(parts[0]), float(parts[1])))
            except ValueError as e:
                fail_or_record(f"Coordonnée non numérique : {coord_line!r} ({e})", i)
                block_ok = False

        if len(coords) != n_points:
            fail_or_record(f"{n_points} landmarks attendus, {len(coords)} trouvés", block_start_line)
            block_ok = False

        image_path, tps_id = "", None
        image_id, specimen_id = None, None
        while i < n_lines:
            meta = lines[i].strip()
            if not meta:
                i += 1
                continue
            upper = meta.upper()
            if upper.startswith("LM="):
                break
            if upper.startswith("IMAGE="):
                image_path = meta.split("=", 1)[1].strip()
            elif upper.startswith("ID="):
                try:
                    tps_id = int(meta.split("=", 1)[1].strip())
                except ValueError:
                    fail_or_record(f"ID non entier : {meta!r}", i + 1)
            elif upper.startswith("COMMENT="):
                fields = _parse_comment(meta.split("=", 1)[1].strip())
                image_id = fields.get("image_id", image_id)
                specimen_id = fields.get("specimen_id", specimen_id)
            i += 1

        if tps_id is None:
            fail_or_record("Aucun ID= trouvé pour ce spécimen", block_start_line)
            block_ok = False

        if block_ok:
            specimens.append(ImageLandmarks(
                n_points, np.array(coords, dtype=float), image_path, tps_id,
                image_id=image_id, specimen_id=specimen_id,
            ))
        specimen_index += 1

    if errors:
        logger.warning("parse_tps(%s): %d spécimens ok, %d erreur(s)", path, len(specimens), len(errors))
    return specimens, errors


def write_tps(path: str | Path, specimens: list[ImageLandmarks]) -> None:
    """Écrit une liste de ImageLandmarks au format .tps (CRLF). Un COMMENT=
    est ajouté si image_id et/ou specimen_id sont renseignés (ignoré par
    geomorph::readland.tps, donc sans risque pour la compatibilité R)."""
    with open(path, "w", newline="\r\n") as f:
        for sp in specimens:
            f.write(f"LM={sp.n_points}\n")
            for x, y in sp.landmarks:
                f.write(f"{x:.4f} {y:.4f}\n")
            f.write(f"IMAGE={sp.image_path}\n")
            f.write(f"ID={sp.tps_id}\n")
            comment = _format_comment(sp.image_id, sp.specimen_id)
            if comment:
                f.write(f"COMMENT={comment}\n")