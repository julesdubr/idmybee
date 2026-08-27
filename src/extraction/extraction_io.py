"""extraction_io.py
Lecture de CSV et fonctions propres à l'extraction (détection + normalisation) :
schéma des colonnes de detection.csv/crops.csv, sélection des images du
manifest.

Les utilitaires génériques de suivi (compteur de statuts, stats, durée,
résolution de chemin) sont dans utils.pipeline_io -- réutilisés par toutes
les étapes du pipeline, pas seulement l'extraction (anciennement mélangés
ici sous le nom detection_io.py).

Entrée : manifest.csv (manifest) ou detection.csv (selon l'appelant).
Sortie : aucune (fonctions utilitaires pures).
"""
from __future__ import annotations

from pathlib import Path

from utils.pipeline_io import read_csv_rows

# Sortie de detect_wing.py (mode split) : extraction/{mode}/detection.csv
DETECTION_FIELDS = [
    "image_id",
    "specimen_id",
    "split",
    "status",
    "error_reason",
    "confidence",
    "n_detections",
    "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4",
    "processing_time_s",
    "processed_at",
]

# Sortie de normalize_crop.py (mode split) : extraction/{mode}/crops.csv
CROP_FIELDS = [
    "image_id",
    "specimen_id",
    "split",
    "status",
    "error_reason",
    "aspect_ratio",
    "output_path",
    "processing_time_s",
    "processed_at",
]


def read_images_csv(path: Path) -> list[dict]:
    """Charge manifest.csv et valide les colonnes minimales."""
    required = {"image_id", "raw_path"}
    rows = read_csv_rows(path)
    if not rows:
        raise ValueError(f"manifest.csv est vide : {path}")

    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"Colonnes manquantes dans {path}: {sorted(missing)}")

    return rows


def select_images(
    rows: list[dict],
    split: str | None = None,
    image_ids: set[str] | None = None,
) -> list[dict]:
    """Applique les filtres génériques aux lignes de manifest.csv."""
    selected = []

    for row in rows:
        if split and row.get("split") != split:
            continue
        if image_ids is not None and row.get("image_id") not in image_ids:
            continue
        if row.get("status_ingest") and row.get("status_ingest") != "parsed_ok":
            continue
        selected.append(row)

    return selected
