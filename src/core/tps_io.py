"""tps_io.py
Read/write .tps landmark files (2D).

Format, one block per specimen:
    LM=19
    123.40 567.80
    ...              (n_points lines "x y")
    IMAGE=relative/path/to/image.jpg
    ID=0
    COMMENT=image_id=...;specimen_id=...     (optional, our own convention)

An ImageLandmarks represents the landmarks of ONE PHOTO (one row of
crops.csv/landmarks.csv), not of a biological specimen: a single
specimen_id can have several photos, hence several TPS entries.
`tps_id` (the ID= field) must be an integer, unique per photo -- see
image_id_to_sid() below.

`image_id` (the real key, a hex string) and `specimen_id` are not standard
tps fields. We persist them in a COMMENT= -- a tps field meant for free
text, explicitly ignored by geomorph::readland.tps ("all other
information... comments, variables, radii, etc. is ignored"), so it's safe
for R compatibility -- to avoid having to join images.csv/specimens.csv on
every read. A TPS written before this field existed (or by a third-party
tool) won't have a COMMENT=: image_id/specimen_id then stay None after
parse_tps, and the caller joins via utils.dataset as before.

parse_tps never raises in non-strict mode: malformed blocks are skipped
and returned in `errors` (never silently swallowed). In strict mode
(default), the first anomaly raises TpsParseException.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
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
    tps_id: int                     # raw TPS ID= value -- unique integer per PHOTO
    image_id: str | None = None     # canonical identifier (hex hash, Phase 0) if known
    specimen_id: str | None = None  # idem -- None if it still needs joining via images.csv/specimens.csv

    @classmethod
    def from_image(
        cls, n_points: int, landmarks: np.ndarray, image_path: str,
        image_id: str, specimen_id: str | None = None,
    ) -> "ImageLandmarks":
        """Build from a known image_id (e.g. predict_unet.py, which already
        has it via crops.csv): computes tps_id automatically and persists
        image_id/specimen_id in the file (COMMENT=) so future reads no
        longer need to join against the manifest."""
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
    """Encode an image_id (hex hash, e.g. truncated sha256) as an integer
    usable as ImageLandmarks.tps_id / TPS ID=. Used when writing, and to
    join a TPS to images.csv when image_id isn't already known (no
    COMMENT=) -- always in this direction (image_id -> integer), never the
    reverse for comparison (see sid_to_image_id)."""
    return int(image_id, 16)


def sid_to_image_id(sid: int) -> str:
    """Best-effort inverse of image_id_to_sid, for display/debugging ONLY --
    never for joining data. `hex(sid)[2:]` does not restore any leading
    zeros the original image_id may have had, so it can differ from the
    real image_id even when sid is correct. To recover a reliable
    image_id: COMMENT= if present, otherwise a join via images.csv
    (image_id_to_sid applied to each row, never the reverse)."""
    return hex(sid)[2:]


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
                    fail_or_record(f"non-integer ID: {meta!r}", i + 1)
            elif upper.startswith("COMMENT="):
                fields = _parse_comment(meta.split("=", 1)[1].strip())
                image_id = fields.get("image_id", image_id)
                specimen_id = fields.get("specimen_id", specimen_id)
            i += 1

        if tps_id is None:
            fail_or_record("no ID= found for this specimen", block_start_line)
            block_ok = False

        if block_ok:
            specimens.append(ImageLandmarks(
                n_points, np.array(coords, dtype=float), image_path, tps_id,
                image_id=image_id, specimen_id=specimen_id,
            ))
        specimen_index += 1

    if errors:
        logger.warning("parse_tps(%s): %d specimen(s) ok, %d error(s)", path, len(specimens), len(errors))
    return specimens, errors


def write_tps(path: str | Path, specimens: list[ImageLandmarks]) -> None:
    """Write a list of ImageLandmarks to .tps format (CRLF). A COMMENT= is
    added if image_id and/or specimen_id are set (ignored by
    geomorph::readland.tps, so safe for R compatibility)."""
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
