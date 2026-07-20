"""Lecture/écriture de fichiers .tps (landmarks 2D).

Format, un bloc par spécimen :
    LM=19
    123.40 567.80
    ...              (n_points lignes "x y")
    IMAGE=relative/path/to/image.jpg
    ID=0

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
class Specimen:
    n_points: int
    landmarks: np.ndarray  # (n_points, 2)
    image_path: str
    sid: int


@dataclass
class TpsParseError:
    specimen_index: int
    line_no: int
    message: str


class TpsParseException(Exception):
    pass


def parse_tps(path: str | Path, strict: bool = True) -> tuple[list[Specimen], list[TpsParseError]]:
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    specimens: list[Specimen] = []
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

        image_path, sid = "", None
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
                    sid = int(meta.split("=", 1)[1].strip())
                except ValueError:
                    fail_or_record(f"ID non entier : {meta!r}", i + 1)
            i += 1

        if sid is None:
            fail_or_record("Aucun ID= trouvé pour ce spécimen", block_start_line)
            block_ok = False

        if block_ok:
            specimens.append(Specimen(n_points, np.array(coords, dtype=float), image_path, sid))
        specimen_index += 1

    if errors:
        logger.warning("parse_tps(%s): %d spécimens ok, %d erreur(s)", path, len(specimens), len(errors))
    return specimens, errors


def write_tps(path: str | Path, specimens: list[Specimen]) -> None:
    """Écrit une liste de Specimen au format .tps (CRLF)."""
    with open(path, "w", newline="\r\n") as f:
        for sp in specimens:
            f.write(f"LM={sp.n_points}\n")
            for x, y in sp.landmarks:
                f.write(f"{x:.4f} {y:.4f}\n")
            f.write(f"IMAGE={sp.image_path}\n")
            f.write(f"ID={sp.sid}\n")
