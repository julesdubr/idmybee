"""Parseur pour les fichiers .tps (landmarks 2D).

Format attendu (un bloc par spécimen, blocs concaténés dans le fichier) :

    LM=19
    123.40 567.80
    ...                 (n_points lignes "x y")
    IMAGE=relative/path/to/image.jpg
    ID=0

Ce parseur est strict par défaut : un bloc malformé lève une exception
plutôt que d'être ignoré silencieusement. En mode non strict, les blocs
en erreur sont sautés mais TOUJOURS retournés dans la liste d'erreurs
(jamais avalés en silence) et un résumé est loggé.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Specimen:
    n_points: int
    landmarks: np.ndarray  # shape = (n_points, 2), dtype float
    image_path: str
    sid: int


@dataclass
class TpsParseError:
    specimen_index: int  # index (0-based) du bloc dans le fichier
    line_no: int  # numéro de ligne (1-based) où le problème a été détecté
    message: str


class TpsParseException(Exception):
    """Levée en mode strict dès qu'un bloc ne peut pas être parsé proprement."""


def parse_tps(
    path: str | Path, strict: bool = True
) -> tuple[list[Specimen], list[TpsParseError]]:
    """Parse un fichier .tps en liste de Specimen.

    Parameters
    ----------
    path:
        Chemin vers le fichier .tps.
    strict:
        Si True (défaut), lève TpsParseException à la première anomalie.
        Si False, saute le bloc en erreur, continue le parsing, et
        retourne toutes les erreurs rencontrées (le spécimen concerné est
        alors absent de la liste retournée, mais l'erreur, elle, ne
        disparaît jamais).

    Returns
    -------
    (specimens, errors)
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    specimens: list[Specimen] = []
    errors: list[TpsParseError] = []

    i = 0
    specimen_index = 0
    n_lines = len(lines)

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
            fail_or_record(f"Ligne inattendue hors bloc, 'LM=' attendu : {line!r}", i + 1)
            i += 1
            continue

        block_start_line = i + 1
        try:
            n_points = int(line.split("=", 1)[1].strip())
        except (IndexError, ValueError) as e:
            fail_or_record(f"Impossible de lire le nombre de points depuis {line!r} ({e})", block_start_line)
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
                fail_or_record(f"Ligne de landmark malformée : {coord_line!r}", i)
                block_ok = False
                continue
            try:
                x, y = float(parts[0]), float(parts[1])
            except ValueError as e:
                fail_or_record(f"Coordonnée non numérique : {coord_line!r} ({e})", i)
                block_ok = False
                continue
            coords.append((x, y))

        if len(coords) != n_points:
            fail_or_record(
                f"{n_points} landmarks attendus, {len(coords)} trouvés", block_start_line
            )
            block_ok = False

        image_path = ""
        sid: int | None = None
        while i < n_lines:
            meta_line = lines[i].strip()
            if not meta_line:
                i += 1
                continue
            upper = meta_line.upper()
            if upper.startswith("LM="):
                break  # début du bloc suivant
            if upper.startswith("IMAGE="):
                image_path = meta_line.split("=", 1)[1].strip()
                i += 1
            elif upper.startswith("ID="):
                id_str = meta_line.split("=", 1)[1].strip()
                try:
                    sid = int(id_str)
                except ValueError:
                    fail_or_record(f"ID non entier : {id_str!r}", i + 1)
                i += 1
            else:
                # clé inconnue (SCALE=, COMMENT=, ...) : tolérée, on saute
                i += 1

        if sid is None:
            fail_or_record("Aucun champ ID= trouvé pour ce spécimen", block_start_line)
            block_ok = False

        if block_ok:
            specimens.append(
                Specimen(
                    n_points=n_points,
                    landmarks=np.array(coords, dtype=float),
                    image_path=image_path,
                    sid=sid,  # type: ignore[arg-type]
                )
            )

        specimen_index += 1

    if errors:
        logger.warning(
            "parse_tps(%s): %d spécimen(s) parsé(s), %d erreur(s) rencontrée(s) (strict=%s)",
            path,
            len(specimens),
            len(errors),
            strict,
        )
    else:
        logger.info("parse_tps(%s): %d spécimens parsés sans erreur", path, len(specimens))

    return specimens, errors