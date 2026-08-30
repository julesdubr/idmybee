"""dataset.py
Chargement des landmarks + métadonnées biologiques pour classifiers/* et
analysis/variance_report.py.

Requiert, sous `root` (ex: data/Bombus/) :
    specimens.csv                       specimen_id, species, caste, is_labeled
    manifest.csv                        image_id, specimen_id, split, device_type, shot_index
    landmarks/landmarks_numbered.tps    landmarks, tous spécimens
    landmarks/landmarks_numbered.csv    statut OK/SUSPECT/FAILED par photo (tps_id, status)

Le TPS et son CSV de statut sont surchargeables (landmarks_tps,
landmarks_status_csv) pour évaluer une autre source de landmarks sur les
mêmes specimens.csv/manifest.csv. Jointure via COMMENT= (image_id/
specimen_id) si présent dans le TPS, sinon via ID=/tps_id dans manifest.csv.

Une ligne de sortie = une photo, pas un spécimen (un individu a souvent
plusieurs photos). meta_df : specimen_id, species, caste, groupe
(species_caste), device, device_tag, split.

Usage :
    specimens, meta_df = load_dataset("data/Bombus", split="train")
"""
from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Sequence

import pandas as pd

from utils.tps_io import ImageLandmarks, image_id_to_sid, parse_tps

logger = logging.getLogger(__name__)


def add_groupe_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute la colonne composée 'groupe' (species + '_' + caste), utilisée par --level=caste."""
    df = df.copy()
    df["groupe"] = df["species"].astype(str) + "_" + df["caste"].astype(str)
    return df


def target_groupe(meta_df: pd.DataFrame, level: str) -> pd.Series:
    """Colonne de regroupement pour la classification ('species' ou 'caste')."""
    if level == "caste":
        return meta_df["groupe"]
    if level not in meta_df.columns:
        raise ValueError(f"--level {level!r} inconnu (attendu : 'species' ou 'caste')")
    return meta_df[level]


def load_unlabeled_tps(tps_path: str | Path, strict: bool = True) -> list[ImageLandmarks]:
    """Lecture seule d'un TPS, sans jointure biologique (ex: une photo terrain, hors specimens.csv)."""
    specimens, errors = parse_tps(tps_path, strict=strict)
    if errors:
        logger.warning("%d erreur(s) de parsing TPS (voir ci-dessus)", len(errors))
    return specimens


def _drop_invalid_landmark_counts(
    specimens: list[ImageLandmarks], meta_df: pd.DataFrame
) -> tuple[list[ImageLandmarks], pd.DataFrame]:
    """Écarte les spécimens dont le nombre de landmarks diffère du schéma majoritaire (requis par la GPA)."""
    if not specimens:
        return specimens, meta_df
    n_points, _ = Counter(sp.n_points for sp in specimens).most_common(1)[0]
    keep_mask = [sp.n_points == n_points for sp in specimens]
    n_dropped = sum(not k for k in keep_mask)
    if n_dropped:
        dropped_ids = [sp.tps_id for sp, keep in zip(specimens, keep_mask) if not keep]
        logger.warning(
            "%d spécimen(s) écarté(s) : nombre de landmarks incohérent (%d points attendus) -- tps_id: %s%s",
            n_dropped, n_points, dropped_ids[:10], ", ..." if len(dropped_ids) > 10 else "",
        )
    specimens = [sp for sp, keep in zip(specimens, keep_mask) if keep]
    meta_df = meta_df[keep_mask].reset_index(drop=True)
    return specimens, meta_df


def _device_tag(device_type: str, shot_index) -> str:
    """Étiquette appareil+prise, ex: "P1", "S2" (voir --devices)."""
    device_type = "" if pd.isna(device_type) else str(device_type)
    shot_index = "" if pd.isna(shot_index) else str(int(shot_index))
    return device_type + shot_index


def _apply_mask(
    specimens: list[ImageLandmarks], meta_df: pd.DataFrame, mask: list[bool], label: str
) -> tuple[list[ImageLandmarks], pd.DataFrame]:
    specimens = [sp for sp, keep in zip(specimens, mask) if keep]
    meta_df = meta_df[mask].reset_index(drop=True)
    print(f"Filtré sur {label} : {len(specimens)} photo(s) restante(s)")
    return specimens, meta_df


def load_dataset(
    root: str | Path,
    split: str = "all",
    devices: Sequence[str] | None = None,
    species: Sequence[str] | None = None,
    castes: Sequence[str] | None = None,
    exclude_outliers: bool = False,
    labeled_only: bool = True,
    strict: bool = True,
    landmarks_tps: str | Path | None = None,
    landmarks_status_csv: str | Path | None = None,
) -> tuple[list[ImageLandmarks], pd.DataFrame]:
    """Charge un dataset (TPS + specimens.csv + manifest.csv) et applique les filtres.

    split : valeur de la colonne 'split' de manifest.csv, ou "all".
    devices : device_tag à garder (ex: ["P1", "S1"]). None = tout garder.
    species / castes : liste blanche de valeurs à garder. None = tout garder.
    exclude_outliers : exclut les photos SUSPECT/FAILED (voir landmarks_status_csv).
        Si aucun CSV de statut n'est disponible, l'exclusion est sautée avec un avertissement.
    labeled_only : écarte les photos sans espèce connue (défaut: True).
    landmarks_tps / landmarks_status_csv : remplacent
        root/landmarks/landmarks_numbered.{tps,csv} (ex: pour évaluer une
        autre source de landmarks sur les mêmes specimens.csv/manifest.csv).

    Écarte aussi, systématiquement, les spécimens dont le nombre de
    landmarks diffère du schéma majoritaire (la GPA exige un nombre de
    points homogène).
    """
    root = Path(root)
    tps_path = Path(landmarks_tps) if landmarks_tps is not None else root / "landmarks" / "landmarks_numbered.tps"
    specimens = load_unlabeled_tps(tps_path, strict=strict)

    specimens_df = pd.read_csv(root / "specimens.csv")
    required = {"specimen_id", "species", "caste"}
    missing = required - set(specimens_df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans {root / 'specimens.csv'} : {missing}")
    specimens_df = specimens_df.set_index("specimen_id", drop=False)

    manifest_df = pd.read_csv(root / "manifest.csv")
    manifest_df["_device_tag"] = [
        _device_tag(d, s) for d, s in zip(manifest_df["device_type"], manifest_df["shot_index"])
    ]
    manifest_df["_tps_id"] = manifest_df["image_id"].apply(image_id_to_sid)
    manifest_df = manifest_df.set_index("_tps_id", drop=True)

    exclude_set: set[int] = set()
    if exclude_outliers:
        if landmarks_status_csv is not None:
            status_path = Path(landmarks_status_csv)
        elif landmarks_tps is None:
            status_path = root / "landmarks" / "landmarks_numbered.csv"
        else:
            status_path = None  # tps custom sans --landmarks-status-csv : pas de statut par défaut

        if status_path is None or not status_path.exists():
            logger.warning(
                "--exclude-outliers demandé mais aucun CSV de statut disponible%s -- exclusion sautée "
                "(passer --landmarks-status-csv si un statut existe pour ce TPS).",
                f" ({status_path} introuvable)" if status_path is not None else "",
            )
            status_path = None

        if status_path is not None:
            status_df = pd.read_csv(status_path)
            if "tps_id" not in status_df.columns or "status" not in status_df.columns:
                raise ValueError(f"{status_path} : colonnes 'tps_id'+'status' attendues pour --exclude-outliers.")
            exclude_set = set(status_df.loc[status_df["status"] != "OK", "tps_id"])

    kept_specimens: list[ImageLandmarks] = []
    kept_rows: list[pd.Series] = []
    unmatched, excluded = 0, 0

    for sp in specimens:
        if sp.tps_id in exclude_set:
            excluded += 1
            continue

        specimen_id = sp.specimen_id
        img_row = manifest_df.loc[sp.tps_id] if sp.tps_id in manifest_df.index else None
        if specimen_id is None and img_row is not None:
            specimen_id = img_row["specimen_id"]
        if specimen_id is None or specimen_id not in specimens_df.index:
            unmatched += 1
            continue

        row = specimens_df.loc[specimen_id].copy()
        if img_row is not None:
            row["device"] = img_row["device_type"]
            row["device_tag"] = img_row["_device_tag"]
            row["split"] = img_row["split"]
        kept_specimens.append(sp)
        kept_rows.append(row)

    if excluded:
        print(f"Exclus {excluded} photo(s) SUSPECT/FAILED via --exclude-outliers")
    if unmatched:
        logger.warning("%d photo(s) sans specimen_id résolu ou absent de specimens.csv, ignorée(s)", unmatched)
    if not kept_specimens:
        raise ValueError(f"Aucun spécimen chargé depuis {root} -- vérifier les filtres et les fichiers.")

    meta_df = pd.DataFrame(kept_rows).reset_index(drop=True)
    meta_df = add_groupe_column(meta_df)
    kept_specimens, meta_df = _drop_invalid_landmark_counts(kept_specimens, meta_df)

    if labeled_only:
        mask = meta_df["species"].notna().tolist()
        kept_specimens, meta_df = _apply_mask(kept_specimens, meta_df, mask, "labeled_only=True")

    if split and split != "all":
        mask = (meta_df["split"] == split).tolist()
        kept_specimens, meta_df = _apply_mask(kept_specimens, meta_df, mask, f"split={split}")

    if devices:
        mask = meta_df["device_tag"].isin(devices).tolist()
        kept_specimens, meta_df = _apply_mask(kept_specimens, meta_df, mask, f"devices={list(devices)}")

    if species:
        mask = meta_df["species"].isin(species).tolist()
        kept_specimens, meta_df = _apply_mask(kept_specimens, meta_df, mask, f"species={list(species)}")

    if castes:
        mask = meta_df["caste"].isin(castes).tolist()
        kept_specimens, meta_df = _apply_mask(kept_specimens, meta_df, mask, f"castes={list(castes)}")

    if not kept_specimens:
        raise ValueError("Aucun spécimen restant après filtrage -- vérifier split/devices/species/castes.")

    logger.info("%d photo(s) chargée(s) depuis %s", len(kept_specimens), root)
    return kept_specimens, meta_df