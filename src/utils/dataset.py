"""dataset.py
Chargement partagé landmarks + métadonnées pour classifiers/* (lda.py,
predict.py) et analysis/report_variance.py.

load_labeled_dataset() résout specimen_id pour chaque entrée du TPS par un
chemin rapide ou un repli, jamais les deux mélangés à l'aveugle :
  1. ImageLandmarks.specimen_id déjà connu (COMMENT= du TPS, écrit par une
     version à jour de predict_unet.py) -- pas besoin d'images.csv.
  2. sinon, repli via images_csv (image_id_to_sid(image_id) == tps_id, puis
     image_id -> specimen_id) -- nécessaire pour les TPS écrits avant ce
     champ, ou produits par un outil tiers.
`device` (device_type) est résolu séparément, systématiquement via
images_csv si fourni (colonne PAR PHOTO, indépendante de la façon dont
specimen_id a été résolu -- un même spécimen a souvent des photos prises
avec des appareils différents, voir images.csv).
Elle écarte aussi automatiquement, toujours, les spécimens dont le nombre
de landmarks diffère du schéma majoritaire (échec de detection/numérotation
en amont) : la GPA exige un nombre de points identique partout, ce n'est
jamais une option laissée à l'appelant.

target_groupe() choisit ensuite la colonne de regroupement pour la
classification : "species" seule, ou le composé "species_caste" pour
--level=caste.

apply_filters() accepte un CSV de qualité en --exclude-ids
(landmarks_numbered.csv de numbering/reconstruct_tps.py -- statut
OK/SUSPECT/FAILED par spécimen, SUSPECT/FAILED exclus), une restriction à
un appareil (--device) ou un jeu de données (--dataset, ex: organized/
terrain/vrac/basile_m1 -- colonne 'datasets_present' de specimens.csv), et
une exclusion d'espèce(s) entière(s) (--exclude-species).
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
    """Ajoute la colonne composée 'groupe' (species_caste) utilisée par
    target_groupe(level="caste") -- factorisé pour rester identique partout
    où une vérité terrain est construite (load_labeled_dataset ici,
    classifiers/predict.py pour l'évaluation --specimens-csv)."""
    df = df.copy()
    df["groupe"] = df["species"].astype(str) + "_" + df["caste"].astype(str)
    return df


def load_unlabeled_tps(tps_path: str | Path, strict: bool = True) -> list[ImageLandmarks]:
    """Lecture seule d'un TPS, sans jointure biologique (nouveaux spécimens
    à classer via predict.py, pas encore dans specimens.csv)."""
    specimens, errors = parse_tps(tps_path, strict=strict)
    if errors:
        logger.warning("%d erreur(s) de parsing TPS (voir ci-dessus)", len(errors))
    return specimens


def _drop_invalid_landmark_counts(
    specimens: list[ImageLandmarks], meta_df: pd.DataFrame
) -> tuple[list[ImageLandmarks], pd.DataFrame]:
    """Écarte les spécimens dont le nombre de landmarks diffère du schéma
    majoritaire (ex: un crop UNet incomplet qui aurait échappé au filtrage
    de numbering/reconstruct_tps.py). Toujours appliqué, sans option --
    un nombre de points incohérent ne peut jamais entrer dans une GPA."""
    if not specimens:
        return specimens, meta_df
    n_points, _ = Counter(sp.n_points for sp in specimens).most_common(1)[0]
    keep_mask = [sp.n_points == n_points for sp in specimens]
    n_dropped = sum(not k for k in keep_mask)
    if n_dropped:
        dropped_ids = [sp.tps_id for sp, keep in zip(specimens, keep_mask) if not keep]
        logger.warning(
            "%d spécimen(s) écarté(s) : nombre de landmarks incohérent avec le schéma attendu "
            "(%d points) -- tps_id: %s%s",
            n_dropped, n_points, dropped_ids[:10], ", ..." if len(dropped_ids) > 10 else "",
        )
    specimens = [sp for sp, keep in zip(specimens, keep_mask) if keep]
    meta_df = meta_df[keep_mask].reset_index(drop=True)
    return specimens, meta_df


def load_labeled_dataset(
    tps_path: str | Path,
    specimens_csv: str | Path,
    images_csv: str | Path | None = None,
    strict: bool = True,
) -> tuple[list[ImageLandmarks], pd.DataFrame]:
    """Charge un TPS + data/manifest/specimens.csv (species/caste), avec
    repli optionnel via images_csv, et écarte automatiquement les schémas
    de landmarks incohérents (voir docstring du module)."""
    specimens = load_unlabeled_tps(tps_path, strict=strict)

    specimens_df = pd.read_csv(specimens_csv)
    required_cols = {"specimen_id", "species", "caste"}
    missing = required_cols - set(specimens_df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans {specimens_csv} : {missing}")

    if "is_labeled" in specimens_df.columns:
        n_total = len(specimens_df)
        specimens_df = specimens_df[specimens_df["is_labeled"].astype(bool)]
        if len(specimens_df) < n_total:
            logger.info(
                "%d spécimen(s) non labellisé(s) (is_labeled=False) ignoré(s) dans %s",
                n_total - len(specimens_df), specimens_csv,
            )
    specimens_df = specimens_df.set_index("specimen_id", drop=False)

    images_df = None
    if images_csv is not None:
        images_df = pd.read_csv(images_csv)
        required_img_cols = {"image_id", "specimen_id"}
        missing_img = required_img_cols - set(images_df.columns)
        if missing_img:
            raise ValueError(f"Colonnes manquantes dans {images_csv} : {missing_img}")
        images_df["_tps_id"] = images_df["image_id"].apply(image_id_to_sid)
        if images_df["_tps_id"].duplicated().any():
            raise ValueError(
                f"{images_csv} : collision d'image_id une fois converti en entier -- "
                "inattendu pour un hash de contenu, vérifier le fichier."
            )
        images_df = images_df.set_index("_tps_id", drop=True)

    kept_specimens: list[ImageLandmarks] = []
    kept_rows: list[pd.Series] = []
    unmatched_image: list[int] = []
    unmatched_specimen: list[int] = []

    for sp in specimens:
        specimen_id = sp.specimen_id  # chemin rapide : déjà connu via COMMENT=

        # device_type est une propriété PAR PHOTO (image_id), indépendante de
        # la façon dont specimen_id a été résolu -- toujours tentée séparément
        # via tps_id, sinon un TPS avec COMMENT= (le cas courant depuis la
        # mise à jour de predict_unet.py) ne remonterait jamais de device,
        # même avec images_csv fourni (bug corrigé : auparavant device restait
        # toujours None dès que le chemin rapide specimen_id était pris).
        device = None
        if images_df is not None and sp.tps_id in images_df.index:
            device = images_df.loc[sp.tps_id].get("device_type")

        if specimen_id is None:
            if images_df is None or sp.tps_id not in images_df.index:
                unmatched_image.append(sp.tps_id)
                continue
            specimen_id = images_df.loc[sp.tps_id]["specimen_id"]

        if specimen_id not in specimens_df.index:
            unmatched_specimen.append(sp.tps_id)
            continue

        row = specimens_df.loc[specimen_id].copy()
        if device is not None:
            row["device"] = device
        kept_specimens.append(sp)
        kept_rows.append(row)

    if unmatched_image:
        hint = "" if images_df is not None else " -- fournir images_csv pour résoudre ces cas"
        logger.warning(
            "%d spécimen(s) du TPS sans specimen_id connu (COMMENT=) et sans correspondance "
            "dans images.csv (tps_id: %s%s)%s",
            len(unmatched_image), unmatched_image[:10],
            ", ..." if len(unmatched_image) > 10 else "", hint,
        )
    if unmatched_specimen:
        logger.warning(
            "%d spécimen(s) résolus en specimen_id mais absents de %s (tps_id: %s%s)",
            len(unmatched_specimen), specimens_csv, unmatched_specimen[:10],
            ", ..." if len(unmatched_specimen) > 10 else "",
        )

    if not kept_specimens:
        raise ValueError(
            "Aucun spécimen apparié entre le TPS et le CSV. Si le TPS n'a pas de COMMENT= "
            "(écrit avant la mise à jour de predict_unet.py/ImageLandmarks.from_image), "
            "fournir images_csv pour résoudre specimen_id via image_id."
        )

    meta_df = pd.DataFrame(kept_rows).reset_index(drop=True)
    meta_df = add_groupe_column(meta_df)

    kept_specimens, meta_df = _drop_invalid_landmark_counts(kept_specimens, meta_df)

    logger.info(
        "%d spécimens appariés TPS<->%s sur %d landmarks dans le TPS",
        len(kept_specimens), specimens_csv, len(specimens),
    )
    return kept_specimens, meta_df


def target_groupe(meta_df: pd.DataFrame, level: str) -> pd.Series:
    """Colonne de regroupement pour la classification.

    level="species" : discrimination par espèce seule.
    level="caste"    : discrimination par (species, caste) -- "groupe",
    déjà calculé par load_labeled_dataset -- car la caste seule mélangerait
    des espèces différentes sous un même label.
    """
    if level == "caste":
        return meta_df["groupe"]
    if level not in meta_df.columns:
        raise ValueError(f"--level {level!r} inconnu (attendu : 'species' ou 'caste')")
    return meta_df[level]


def apply_filters(
    specimens: list[ImageLandmarks],
    meta_df: pd.DataFrame,
    exclude_ids_paths: Sequence[str | Path] | str | Path | None = None,
    device: str | None = None,
    dataset: str | None = None,
    exclude_species: Sequence[str] | None = None,
) -> tuple[list[ImageLandmarks], pd.DataFrame]:
    """Filtres communs : exclusion de spécimens signalés par un ou plusieurs
    CSV de qualité (--exclude-ids, colonnes 'tps_id' + 'status', typiquement
    landmarks_numbered.csv -- tout ce qui n'est pas "OK" est exclu),
    restriction à un appareil (--device, nécessite une colonne 'device' dans
    meta_df -- voir load_labeled_dataset), restriction à un jeu de données
    d'origine (--dataset, ex: organized/terrain/vrac/basile_m1 -- colonne
    'datasets_present' de specimens.csv, utile pour entraîner/évaluer un
    modèle sur un sous-ensemble homogène), et exclusion d'espèce(s)
    entière(s) (--exclude-species -- utile le temps d'investiguer un
    problème de numérotation propre à une espèce).
    """
    if exclude_ids_paths:
        if isinstance(exclude_ids_paths, (str, Path)):
            exclude_ids_paths = [exclude_ids_paths]
        exclude_set: set = set()
        for path in exclude_ids_paths:
            exclude_df = pd.read_csv(path)
            if "tps_id" not in exclude_df.columns or "status" not in exclude_df.columns:
                raise ValueError(
                    f"{path} n'a pas les colonnes attendues ('tps_id', 'status') pour --exclude-ids "
                    "-- fichier au mauvais format, ou généré par une version obsolète de "
                    "numbering/reconstruct_tps.py (le régénérer)."
                )
            exclude_df = exclude_df[exclude_df["status"] != "OK"]
            exclude_set |= set(exclude_df["tps_id"])
        keep_mask = [sp.tps_id not in exclude_set for sp in specimens]
        specimens = [sp for sp, keep in zip(specimens, keep_mask) if keep]
        meta_df = meta_df[keep_mask].reset_index(drop=True)
        print(f"Exclus {sum(not k for k in keep_mask)} spécimen(s) via {len(exclude_ids_paths)} fichier(s) --exclude-ids")

    if device:
        if "device" not in meta_df.columns:
            raise ValueError(
                "--device demandé mais 'device' n'est pas disponible : images_csv n'a pas été "
                "fourni à load_labeled_dataset. Fournir --images-csv pour activer ce filtre."
            )
        keep_mask = (meta_df["device"] == device).tolist()
        specimens = [sp for sp, keep in zip(specimens, keep_mask) if keep]
        meta_df = meta_df[keep_mask].reset_index(drop=True)
        print(f"Filtré sur device={device} : {len(specimens)} spécimen(s) restants")

    if dataset:
        if "datasets_present" not in meta_df.columns:
            raise ValueError("--dataset demandé mais 'datasets_present' n'est pas dans specimens.csv.")
        present = meta_df["datasets_present"].astype(str).str.split(",")
        keep_mask = present.apply(lambda names: dataset in {n.strip() for n in names}).tolist()
        specimens = [sp for sp, keep in zip(specimens, keep_mask) if keep]
        meta_df = meta_df[keep_mask].reset_index(drop=True)
        print(f"Filtré sur dataset={dataset} : {len(specimens)} spécimen(s) restants")

    if exclude_species:
        keep_mask = (~meta_df["species"].isin(exclude_species)).tolist()
        n_dropped = sum(not k for k in keep_mask)
        specimens = [sp for sp, keep in zip(specimens, keep_mask) if keep]
        meta_df = meta_df[keep_mask].reset_index(drop=True)
        print(f"Exclus {n_dropped} spécimen(s) des espèces {list(exclude_species)}")

    return specimens, meta_df