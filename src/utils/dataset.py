"""dataset.py
Chargement partagé landmarks + métadonnées pour tout ce qui vit sous
classifiers/ (lda.py, predict.py, landmarks_knn.py à venir) et pour
tools/flag_outlier_specimens.py.

Sépare deux besoins distincts :
- load_labeled_dataset() : jointure TPS<->CSV pour l'entraînement/l'évaluation
  (nécessite espece/caste).
- load_unlabeled_tps() : lecture seule d'un TPS, sans jointure biologique,
  pour classer de nouveaux spécimens (predict.py) qui n'ont pas encore de CSV.

Les deux partagent apply_filters() (exclude-ids, device), pour que ce
comportement ne diverge pas silencieusement entre lda.py et predict.py au
fil des évolutions.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from utils.tps_io import Specimen, parse_tps

logger = logging.getLogger(__name__)


def load_unlabeled_tps(tps_path: str | Path, strict: bool = True) -> list[Specimen]:
    """Lecture seule d'un TPS, sans jointure biologique (ex: nouveaux
    spécimens à classer via predict.py, pas encore de CSV d'identification)."""
    specimens, errors = parse_tps(tps_path, strict=strict)
    if errors:
        logger.warning("%d erreur(s) de parsing TPS (voir ci-dessus)", len(errors))
    return specimens


def load_labeled_dataset(
    tps_path: str | Path, csv_path: str | Path, strict: bool = True
) -> tuple[list[Specimen], pd.DataFrame]:
    """Charge le TPS de référence + le CSV de métadonnées et les aligne par id/sid.

    Équivalent de la lecture de MyExcelFile + readland.tps(specID="imageID")
    dans le script R, mais la jointure ici se fait explicitement sur
    id (CSV) == sid (TPS ID=), au lieu de compter sur un ordre identique des
    deux fichiers. Les spécimens du TPS sans entrée CSV sont rapportés (pas
    d'exception) puis exclus du résultat.
    """
    specimens = load_unlabeled_tps(tps_path, strict=strict)

    meta = pd.read_csv(csv_path)
    required_cols = {"id", "espece", "caste"}
    missing = required_cols - set(meta.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV : {missing}")
    meta = meta.set_index("id", drop=False)

    kept_specimens: list[Specimen] = []
    kept_rows: list[pd.Series] = []
    unmatched: list[int] = []
    for sp in specimens:
        if sp.sid not in meta.index:
            unmatched.append(sp.sid)
            continue
        kept_specimens.append(sp)
        kept_rows.append(meta.loc[sp.sid])

    if unmatched:
        logger.warning(
            "%d spécimen(s) du TPS sans entrée CSV correspondante (id manquants: %s%s)",
            len(unmatched),
            unmatched[:10],
            ", ..." if len(unmatched) > 10 else "",
        )

    meta_df = pd.DataFrame(kept_rows).reset_index(drop=True)
    meta_df["groupe"] = meta_df["espece"].astype(str) + "_" + meta_df["caste"].astype(str)

    logger.info(
        "%d spécimens appariés TPS<->CSV sur %d landmarks dans le TPS",
        len(kept_specimens),
        len(specimens),
    )
    return kept_specimens, meta_df


def apply_filters(
    specimens: list[Specimen],
    meta_df: pd.DataFrame,
    exclude_ids_path: str | Path | None = None,
    device: str | None = None,
) -> tuple[list[Specimen], pd.DataFrame]:
    """Filtres communs entraînement/évaluation : exclusion d'outliers connus
    (CSV --exclude-ids, ex: out/outlier_specimens.csv) et restriction à un
    appareil donné (--device)."""
    if exclude_ids_path:
        exclude_df = pd.read_csv(exclude_ids_path)
        if "heavy" in exclude_df.columns:
            exclude_df = exclude_df[exclude_df["heavy"]]
        exclude_set = set(exclude_df["id"])
        keep_mask = [sp.sid not in exclude_set for sp in specimens]
        specimens = [sp for sp, keep in zip(specimens, keep_mask) if keep]
        meta_df = meta_df[keep_mask].reset_index(drop=True)
        print(f"Exclus {sum(not k for k in keep_mask)} spécimen(s) via {exclude_ids_path}")

    if device:
        keep_mask = (meta_df["device"] == device).tolist()
        specimens = [sp for sp, keep in zip(specimens, keep_mask) if keep]
        meta_df = meta_df[keep_mask].reset_index(drop=True)
        print(f"Filtré sur device={device} : {len(specimens)} spécimen(s) restants")

    return specimens, meta_df