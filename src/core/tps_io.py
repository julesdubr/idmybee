"""tps_io.py
Read/write .tps landmark files (2D).

Format, one block per specimen:
    LM=19
    123.40 567.80
    ...              (n_points lines "x y")
    IMAGE=relative/path/to/image.jpg
    ID=0
    COMMENT=photo_id=...;inv_id=...     (required, our own convention)

An ImageLandmarks represents the landmarks of ONE PHOTO (one row of
crops.csv/landmarks.csv), not of a biological specimen: a single inv_id
can have several photos, hence several TPS entries.

`tps_id` (the ID= field) is just a TPS-format requirement (must be a
unique integer per photo *within one file*) -- it carries no identity
across files/runs. `photo_id` (the real, human-readable, stable key -- see
tools/export_clean_dataset.py) and `inv_id` are not standard tps fields.
We persist them in a COMMENT= -- a tps field meant for free text,
explicitly ignored by geomorph::readland.tps ("all other information...
comments, variables, radii, etc. is ignored"), so it's safe for R
compatibility. Unlike an earlier, hash-based `image_id` scheme, this
COMMENT= is the SOLE join mechanism now (see assign_sequential_ids() below
for how `ID=` is assigned) -- a TPS missing it (written before this
convention, or by a third-party tool) simply can't be joined back to a
manifest by this codebase.

parse_tps never raises in non-strict mode: malformed blocks are skipped
and returned in `errors` (never silently swallowed). In strict mode
(default), the first anomaly raises TpsParseException.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Lines starting with one of these (case-insensitive) end a specimen's
# coordinate block even if fewer than n_points valid pairs were read --
# without this check, a truncated block makes the coordinate reader consume
# subsequent metadata/directive lines (and potentially the next block's
# own LM=/points) while still looking for enough numeric pairs, corrupting
# the parse of a specimen that was otherwise perfectly valid.
_DIRECTIVE_PREFIXES = ("LM=", "IMAGE=", "ID=", "COMMENT=", "SCALE=")


@dataclass
class ImageLandmarks:
    n_points: int
    landmarks: np.ndarray           # (n_points, 2)
    image_path: str
    tps_id: int                     # raw TPS ID= value -- unique integer per photo WITHIN THIS FILE only
    photo_id: str | None = None     # canonical per-photo identifier if known
    inv_id: str | None = None       # canonical specimen identifier -- None if still needs joining

    @classmethod
    def from_image(
        cls, n_points: int, landmarks: np.ndarray, image_path: str,
        tps_id: int, photo_id: str, inv_id: str | None = None,
    ) -> "ImageLandmarks":
        """Build from a known photo_id (e.g. landmarks/predict.py, which
        already has it via crops.csv): persists photo_id/inv_id in the file
        (COMMENT=) so future reads no longer need to join against the
        manifest. `tps_id` is caller-supplied (a placeholder is fine --
        see assign_sequential_ids() for how it's finalized at write time)."""
        return cls(
            n_points=n_points, landmarks=landmarks, image_path=image_path,
            tps_id=tps_id, photo_id=photo_id, inv_id=inv_id,
        )


@dataclass
class TpsParseError:
    specimen_index: int
    line_no: int
    message: str


class TpsParseException(Exception):
    pass


def assign_sequential_ids(specimens: list[ImageLandmarks]) -> list[ImageLandmarks]:
    """Returns a copy of `specimens` with `tps_id` reassigned 1..N, sorted
    by `photo_id`.

    `ID=` is purely a TPS-format requirement (a unique int per block in
    the file) -- this is the only place it gets assigned. Callers that
    need a different row order for their TPS (e.g.
    tools/export_final_landmarks.py, which numbers by biological_data.csv
    row order) should not use this helper and assign `tps_id` themselves.
    """
    ordered = sorted(specimens, key=lambda sp: sp.photo_id or "")
    return [replace(sp, tps_id=i) for i, sp in enumerate(ordered, start=1)]


def _parse_comment(comment: str) -> dict[str, str]:
    """Decode our small 'key=value;key=value' format from a COMMENT=.
    Silently ignores anything that doesn't look like it: COMMENT= is free
    text per the tps format, a third-party (or older) file may put
    something else there, or nothing."""
    fields: dict[str, str] = {}
    for part in comment.split(";"):
        if "=" not in part:
            continue
        key, _, value = part.partition("=")
        key, value = key.strip(), value.strip()
        if key:
            fields[key] = value
    return fields


def _format_comment(photo_id: str | None, inv_id: str | None) -> str | None:
    parts = []
    if photo_id is not None:
        parts.append(f"photo_id={photo_id}")
    if inv_id is not None:
        parts.append(f"inv_id={inv_id}")
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
            raise TpsParseException(f"{msg} (line {line_no})")

    while i < n_lines:
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if not line.upper().startswith("LM="):
            fail_or_record(f"expected 'LM=', found: {line!r}", i + 1)
            i += 1
            continue

        block_start_line = i + 1
        try:
            n_points = int(line.split("=", 1)[1].strip())
        except (IndexError, ValueError) as e:
            fail_or_record(f"unreadable point count in {line!r} ({e})", block_start_line)
            i += 1
            specimen_index += 1
            continue

        i += 1
        coords: list[tuple[float, float]] = []
        block_ok = True
        while len(coords) < n_points and i < n_lines:
            coord_line = lines[i].strip()
            if not coord_line:
                i += 1
                continue
            if coord_line.upper().startswith(_DIRECTIVE_PREFIXES):
                # truncated block: stop consuming lines here instead of
                # reaching into the next directive/block looking for more
                # coordinate pairs (see _DIRECTIVE_PREFIXES docstring above).
                break
            i += 1
            parts = coord_line.replace(",", " ").split()
            if len(parts) < 2:
                fail_or_record(f"malformed landmark: {coord_line!r}", i)
                block_ok = False
                continue
            try:
                coords.append((float(parts[0]), float(parts[1])))
            except ValueError as e:
                fail_or_record(f"non-numeric coordinate: {coord_line!r} ({e})", i)
                block_ok = False

        if len(coords) != n_points:
            fail_or_record(f"expected {n_points} landmarks, found {len(coords)}", block_start_line)
            block_ok = False

        image_path, tps_id = "", None
        photo_id, inv_id = None, None
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
                    fail_or_record(f"non-integer ID: {meta!r}", i + 1)
            elif upper.startswith("COMMENT="):
                fields = _parse_comment(meta.split("=", 1)[1].strip())
                photo_id = fields.get("photo_id", photo_id)
                inv_id = fields.get("inv_id", inv_id)
            i += 1

        if tps_id is None:
            fail_or_record("no ID= found for this specimen", block_start_line)
            block_ok = False

        if block_ok:
            specimens.append(ImageLandmarks(
                n_points, np.array(coords, dtype=float), image_path, tps_id,
                photo_id=photo_id, inv_id=inv_id,
            ))
        specimen_index += 1

    if errors:
        logger.warning("parse_tps(%s): %d specimen(s) ok, %d error(s)", path, len(specimens), len(errors))
    return specimens, errors


def write_tps(path: str | Path, specimens: list[ImageLandmarks]) -> None:
    """Write a list of ImageLandmarks to .tps format (CRLF). A COMMENT= is
    added if photo_id and/or inv_id are set (ignored by
    geomorph::readland.tps, so safe for R compatibility)."""
    with open(path, "w", newline="\r\n") as f:
        for sp in specimens:
            f.write(f"LM={sp.n_points}\n")
            for x, y in sp.landmarks:
                f.write(f"{x:.4f} {y:.4f}\n")
            f.write(f"IMAGE={sp.image_path}\n")
            f.write(f"ID={sp.tps_id}\n")
            comment = _format_comment(sp.photo_id, sp.inv_id)
            if comment:
                f.write(f"COMMENT={comment}\n")
