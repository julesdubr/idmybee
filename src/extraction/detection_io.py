"""Lecture de CSV et fonctions communes à la détection et la normalisation.

Entrée : images.csv (manifest) ou detection.csv (selon l'appelant).
Sortie : aucune (fonctions utilitaires pures).
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path


# Sortie de detect_wing.py (mode dataset) : extraction/{mode}/detection.csv
DETECTION_FIELDS = [
    "image_id",
    "specimen_id",
    "dataset",
    "status",
    "error_reason",
    "confidence",
    "n_detections",
    "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4",
    "processing_time_s",
    "processed_at",
]

# Sortie de normalize_crop.py (mode dataset) : extraction/{mode}/crops.csv
CROP_FIELDS = [
    "image_id",
    "specimen_id",
    "dataset",
    "status",
    "error_reason",
    "aspect_ratio",
    "output_path",
    "processing_time_s",
    "processed_at",
]

STATS_FIELDS = ["mode", "step", "total", "ok", "skipped", "failed", "updated_at"]


def read_images_csv(path: Path) -> list[dict]:
    """Charge images.csv et valide les colonnes minimales."""
    required = {"image_id", "raw_path"}

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"images.csv est vide : {path}")

    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"Colonnes manquantes dans {path}: {sorted(missing)}")

    return rows


def read_csv_rows(path: Path) -> list[dict]:
    """Charge un CSV quelconque en liste de dicts."""
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def select_images(
    rows: list[dict],
    dataset: str | None = None,
    image_ids: set[str] | None = None,
    skip_duplicates: bool = True,
) -> list[dict]:
    """Applique les filtres génériques aux lignes de images.csv."""
    selected = []

    for row in rows:
        if dataset and row.get("dataset") != dataset:
            continue
        if image_ids is not None and row.get("image_id") not in image_ids:
            continue
        if row.get("status_ingest") and row.get("status_ingest") != "parsed_ok":
            continue
        if skip_duplicates and str(row.get("is_duplicate_content", "")).lower() == "true":
            continue
        selected.append(row)

    return selected


def resolve_raw_path(raw_path: str, base_dir: Path | None = None) -> Path:
    """Résout un raw_path absolu ou relatif."""
    path = Path(raw_path)
    if path.is_absolute() or base_dir is None:
        return path
    return base_dir / path


def append_rows(path: Path, rows: list[dict], fields: list[str], write_header: bool) -> None:
    """Ajoute des lignes à un CSV, en écrivant l'entête si nécessaire.

    `write_header=True` uniquement lors du tout premier flush d'un run
    (écrase un éventuel fichier précédent) ; `False` pour les flush suivants.
    """
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if write_header else "a"
    with path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def update_stats(stats_path: Path, mode: str, step: str, counts: dict) -> None:
    """Met à jour extraction_stats.csv : une ligne par (mode, step).

    `counts` doit contenir `total`, `ok`, `skipped`, `failed`. La ligne
    existante pour ce (mode, step) est remplacée ; les autres sont conservées.
    """
    rows = read_csv_rows(stats_path) if stats_path.exists() else []
    rows = [r for r in rows if not (r.get("mode") == mode and r.get("step") == step)]

    rows.append({
        "mode": mode,
        "step": step,
        "total": str(counts["total"]),
        "ok": str(counts["ok"]),
        "skipped": str(counts["skipped"]),
        "failed": str(counts["failed"]),
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=STATS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def format_duration(seconds: float) -> str:
    """Formate une durée en `1h05m30.0s` / `5m12.3s` / `3.2s`."""
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{int(hours)}h{int(minutes):02d}m{secs:04.1f}s"
    if minutes:
        return f"{int(minutes)}m{secs:04.1f}s"
    return f"{secs:.1f}s"


class RunCounter:
    """Compte OK/SKIPPED/FAILED au fil d'un run, pour affichage + stats."""

    def __init__(self):
        self.ok = 0
        self.skipped = 0
        self.failed = 0

    def add(self, status: str) -> None:
        if status == "OK":
            self.ok += 1
        elif status == "SKIPPED":
            self.skipped += 1
        else:
            self.failed += 1

    @property
    def total(self) -> int:
        return self.ok + self.skipped + self.failed

    def as_dict(self) -> dict:
        return {"total": self.total, "ok": self.ok, "skipped": self.skipped, "failed": self.failed}

    def __str__(self) -> str:
        return f"OK={self.ok} SKIPPED={self.skipped} FAILED={self.failed}"