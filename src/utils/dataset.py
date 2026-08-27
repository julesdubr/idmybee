"""dataset.py
Chargement unifié landmarks + métadonnées, pour classifiers/* (train.py,
predict.py) et analysis/variance_report.py.

Point d'entrée unique : load_dataset(root). `root` est un dossier comme
data/Bombus/, qui doit contenir :
    root/specimens.csv                       (specimen_id, species, caste, is_labeled)
    root/manifest.csv                        (image_id, specimen_id, dataset, device_type, shot_index)
    root/landmarks/landmarks_numbered.csv    (statut OK/SUSPECT/FAILED par photo, colonnes tps_id+status)
    root/landmarks/landmarks_numbered.tps    (landmarks, tous spécimens)

Une ligne de sortie (specimens[i] <-> meta_df.iloc[i]) = UNE PHOTO, pas un
spécimen biologique (un même specimen_id a souvent plusieurs photos, voir
utils.tps_io). meta_df a les colonnes : specimen_id, species, caste, groupe
(species_caste), device (P/S), device_tag (P1/S2/... -- device + shot_index,
voir _device_tag), dataset (valeur brute de manifest.csv, ex: train/test/
vrac/basile_m1 -- c'est aussi la valeur attendue par --split, pas de mapping
séparé : "train"/"test" ne sont pas des alias, ce sont les vraies valeurs).

Elle écarte aussi automatiquement, toujours, les spécimens dont le nombre de
landmarks diffère du schéma majoritaire (échec de détection/numérotation en
amont) : la GPA exige un nombre de points identique partout.
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
    """Ajoute la colonne composée 'groupe' (species_caste) -- pour
    --level=caste de train.py (la caste seule mélangerait des espèces
    différentes sous un même label "worker"/"queen"/"male")."""
    df = df.copy()
    df["groupe"] = df["species"].astype(str) + "_" + df["caste"].astype(str)
    return df


def target_groupe(meta_df: pd.DataFrame, level: str) -> pd.Series:
    """Colonne de regroupement pour la classification (train.py --level)."""
    if level == "caste":
        return meta_df["groupe"]
    if level not in meta_df.columns:
        raise ValueError(f"--level {level!r} inconnu (attendu : 'species' ou 'caste')")
    return meta_df[level]


def load_unlabeled_tps(tps_path: str | Path, strict: bool = True) -> list[ImageLandmarks]:
    """Lecture seule d'un TPS, sans jointure biologique (ex: predict.py
    single sur une photo fraîche, pas encore dans specimens.csv)."""
    specimens, errors = parse_tps(tps_path, strict=strict)
    if errors:
        logger.warning("%d erreur(s) de parsing TPS (voir ci-dessus)", len(errors))
    return specimens


def _drop_invalid_landmark_counts(
    specimens: list[ImageLandmarks], meta_df: pd.DataFrame
) -> tuple[list[ImageLandmarks], pd.DataFrame]:
    """Écarte les spécimens dont le nombre de landmarks diffère du schéma
    majoritaire -- toujours appliqué, sans option : un nombre de points
    incohérent ne peut jamais entrer dans une GPA."""
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
    """Étiquette combinée appareil+prise, ex: "P1", "S2" -- utilisée par
    --devices pour cibler une photo précise (ex: toujours P1 et S1 comme
    représentants d'un individu, pour ne pas mélanger plusieurs reprises
    dans une même analyse -- voir variance_report.py)."""
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
) -> tuple[list[ImageLandmarks], pd.DataFrame]:
    """Charge un dossier `root` (ex: data/Bombus) et applique les filtres
    demandés. Un seul appel remplace la jointure TPS<->specimens.csv<->
    manifest.csv + tous les filtres (avant : load_labeled_dataset +
    apply_filters, deux étapes séparées).

    split   : valeur brute de la colonne 'dataset' de manifest.csv (ex:
              "train", "test", "vrac", "basile_m1"), ou "all" (pas de
              filtre). Pas de mapping séparé -- ce que rend manifest.csv est
              utilisé tel quel.
    devices : étiquettes device_tag à garder (ex: ["P1", "S1"] -- voir
              _device_tag). None = tout garder.
    species/castes : ne garder QUE ces valeurs (liste blanche). None = tout
              garder.
    exclude_outliers : exclut les photos SUSPECT/FAILED de
              landmarks_numbered.csv (colonnes tps_id+status). Les FAILED ne
              sont normalement déjà plus dans le TPS (voir reconstruct_tps.py
              en amont) ; ce flag rattrape surtout les SUSPECT.
    labeled_only : True (défaut) -> écarte les photos sans espèce connue
              (species NaN dans specimens.csv), nécessaire pour train.py et
              pour évaluer predict.py batch. False -> garde tout, y compris
              non déterminé (prédiction pure sans évaluation possible).
    """
    root = Path(root)
    specimens = load_unlabeled_tps(root / "landmarks" / "landmarks_light_numbered.tps", strict=strict)

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
        status_path = root / "landmarks" / "landmarks_light_numbered.csv"
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