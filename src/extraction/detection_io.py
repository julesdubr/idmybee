"""Lecture de images.csv et fonctions communes aux détecteurs."""

from __future__ import annotations

import csv
from pathlib import Path


COMMON_FIELDS = [
    "image_id",
    "specimen_id",
    "dataset",
    "status",
    "error_reason",
    "confidence",
    "selection_score",
    "n_detections",
    "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4",
    "processing_time_s",
    "processed_at",
]

# Sortie combinée de extract_wings.py : détection + normalisation en une seule
# ligne par image (une seule lecture d'image, un seul passage).
EXTRACT_FIELDS = [
    "image_id",
    "specimen_id",
    "dataset",
    "detection_status",
    "normalization_status",
    "error_reason",
    "confidence",
    "selection_score",
    "n_detections",
    "aspect_ratio",
    "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4",
    "output_path",
    "processing_time_s",
    "processed_at",
]


def read_images_csv(path: Path) -> list[dict]:
    """Charge images.csv et valide les colonnes minimales."""
    required = {"image_id", "raw_path"}

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"images.csv est vide : {path}")

    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(
            f"Colonnes manquantes dans {path}: {sorted(missing)}"
        )

    return rows


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


def write_rows(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    """Écrit un CSV complet, non append-only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Aucune ligne à écrire.")

    fields = fields or list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def append_rows(path: Path, rows: list[dict], fields: list[str], write_header: bool) -> None:
    """Ajoute des lignes à un CSV, en écrivant l'entête si nécessaire.

    Utilisé pour l'écriture incrémentale par lots : `write_header` doit valoir
    True uniquement lors du tout premier flush (fichier absent ou tronqué).
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