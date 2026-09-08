"""dataset.py
Loads landmarks + biological metadata for classifiers/* and
analysis/variance_report.py.

Requires, under `root` (e.g. data/clean/collection/, or a
`tools/combine_manifests.py` output combining several sources):
    biological_data.csv                 inv_id, species, caste, ... (see
                                         tools/export_clean_dataset.py)
    manifest.csv                        photo_id, inv_id, device_type,
                                         device, photo_index, path, status
    landmarks/landmarks_numbered.tps    landmarks, all specimens
    landmarks/landmarks_numbered.csv    OK/SUSPECT/FAILED status per photo (tps_id, status)

The TPS and its status CSV can be overridden (landmarks_tps,
landmarks_status_csv) to evaluate a different landmark source on the same
biological_data.csv/manifest.csv. Joined via COMMENT= (photo_id/inv_id) --
required, see core.tps_io module docstring: a TPS with no COMMENT= (e.g.
from a third-party tool) simply can't be joined by this function.

One output row = one photo, not one specimen (an individual often has
several photos). meta_df: inv_id, species, caste, groupe (species_caste),
device_type, device, device_tag.

Usage:
    specimens, meta_df = load_dataset("data/clean/collection")
"""
from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Sequence

import pandas as pd

from core.tps_io import ImageLandmarks, parse_tps

logger = logging.getLogger(__name__)


def add_groupe_column(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the composite 'groupe' column (species + '_' + caste), used by --level=caste."""
    df = df.copy()
    df["groupe"] = df["species"].astype(str) + "_" + df["caste"].astype(str)
    return df


def target_groupe(meta_df: pd.DataFrame, level: str) -> pd.Series:
    """Grouping column for classification ('species' or 'caste')."""
    if level == "caste":
        return meta_df["groupe"]
    if level not in meta_df.columns:
        raise ValueError(f"--level {level!r} unknown (expected: 'species' or 'caste')")
    return meta_df[level]


def load_unlabeled_tps(tps_path: str | Path, strict: bool = True) -> list[ImageLandmarks]:
    """Reads a TPS with no biological join (e.g. a field photo, outside biological_data.csv)."""
    specimens, errors = parse_tps(tps_path, strict=strict)
    if errors:
        logger.warning("%d TPS parsing error(s) (see above)", len(errors))
    return specimens


def _drop_invalid_landmark_counts(
    specimens: list[ImageLandmarks], meta_df: pd.DataFrame
) -> tuple[list[ImageLandmarks], pd.DataFrame]:
    """Drops specimens whose landmark count differs from the majority scheme (required by GPA)."""
    if not specimens:
        return specimens, meta_df
    n_points, _ = Counter(sp.n_points for sp in specimens).most_common(1)[0]
    keep_mask = [sp.n_points == n_points for sp in specimens]
    n_dropped = sum(not k for k in keep_mask)
    if n_dropped:
        dropped_ids = [sp.photo_id for sp, keep in zip(specimens, keep_mask) if not keep]
        logger.warning(
            "%d specimen(s) dropped: inconsistent landmark count (%d points expected) -- photo_id: %s%s",
            n_dropped, n_points, dropped_ids[:10], ", ..." if len(dropped_ids) > 10 else "",
        )
    specimens = [sp for sp, keep in zip(specimens, keep_mask) if keep]
    meta_df = meta_df[keep_mask].reset_index(drop=True)
    return specimens, meta_df


def _device_tag(device_type: str, photo_index) -> str:
    """Device+shot tag, e.g. "P1", "S2" (see --devices)."""
    device_type = "" if pd.isna(device_type) else str(device_type)
    photo_index = "" if pd.isna(photo_index) else str(int(photo_index))
    return device_type + photo_index


def _apply_mask(
    specimens: list[ImageLandmarks], meta_df: pd.DataFrame, mask: list[bool], label: str
) -> tuple[list[ImageLandmarks], pd.DataFrame]:
    specimens = [sp for sp, keep in zip(specimens, mask) if keep]
    meta_df = meta_df[mask].reset_index(drop=True)
    logger.info("Filtered on %s: %d photo(s) remaining", label, len(specimens))
    return specimens, meta_df


def restrict_to_complete_devices(
    specimens: list[ImageLandmarks], meta_df: pd.DataFrame, devices: Sequence[str]
) -> tuple[list[ImageLandmarks], pd.DataFrame]:
    """Keeps only specimens that have ALL of the `devices` device_tags (drops
    the entire individual -- all of its photos -- if it only has a subset,
    not just the missing photos).

    Used to balance a nested ANOVA species ⊃ caste ⊃ individual ⊃ device
    (see analysis/variance_report.py): without this filter, an individual
    with more photos (or a different device coverage) weighs more heavily
    in its biological group's average, and its "individual mean" is a
    P/S mix that differs from one individual to the next -- which biases
    both the biological levels and the device effect estimate.
    """
    required = set(devices)
    tag_sets = meta_df.groupby("inv_id")["device_tag"].agg(set)
    complete_ids = set(tag_sets[tag_sets.apply(required.issubset)].index)
    mask = meta_df["inv_id"].isin(complete_ids).tolist()
    return _apply_mask(specimens, meta_df, mask, f"complete devices coverage={sorted(required)}")


def load_dataset(
    root: str | Path,
    devices: Sequence[str] | None = None,
    species: Sequence[str] | None = None,
    castes: Sequence[str] | None = None,
    exclude_outliers: bool = False,
    labeled_only: bool = True,
    strict: bool = True,
    landmarks_tps: str | Path | None = None,
    landmarks_status_csv: str | Path | None = None,
) -> tuple[list[ImageLandmarks], pd.DataFrame]:
    """Loads a dataset (TPS + biological_data.csv + manifest.csv) and applies the filters.

    devices: device_tags to keep (e.g. ["P1", "S1"]). None = keep all.
    species / castes: whitelist of values to keep. None = keep all.
    exclude_outliers: excludes SUSPECT/FAILED photos (see landmarks_status_csv).
        If no status CSV is available, exclusion is skipped with a warning.
    labeled_only: drops photos with no known species (default: True).
    landmarks_tps / landmarks_status_csv: override
        root/landmarks/landmarks_numbered.{tps,csv} (e.g. to evaluate
        another landmark source on the same biological_data.csv/manifest.csv).

    Also systematically drops specimens whose landmark count differs from
    the majority scheme (GPA requires a homogeneous point count).
    """
    root = Path(root)
    tps_path = Path(landmarks_tps) if landmarks_tps is not None else root / "landmarks" / "landmarks_numbered.tps"
    specimens = load_unlabeled_tps(tps_path, strict=strict)

    specimens_df = pd.read_csv(root / "biological_data.csv")
    required = {"inv_id", "species", "caste"}
    missing = required - set(specimens_df.columns)
    if missing:
        raise ValueError(f"Missing columns in {root / 'biological_data.csv'}: {missing}")
    specimens_df = specimens_df.set_index("inv_id", drop=False)

    manifest_df = pd.read_csv(root / "manifest.csv")
    manifest_df["_device_tag"] = [
        _device_tag(d, p) for d, p in zip(manifest_df["device_type"], manifest_df["photo_index"])
    ]
    manifest_df = manifest_df.set_index("photo_id", drop=False)

    exclude_set: set[int] = set()
    if exclude_outliers:
        if landmarks_status_csv is not None:
            status_path = Path(landmarks_status_csv)
        elif landmarks_tps is None:
            status_path = root / "landmarks" / "landmarks_numbered.csv"
        else:
            status_path = None  # custom tps with no --landmarks-status-csv: no default status

        if status_path is None or not status_path.exists():
            logger.warning(
                "--exclude-outliers requested but no status CSV available%s -- exclusion skipped "
                "(pass --landmarks-status-csv if a status exists for this TPS).",
                f" ({status_path} not found)" if status_path is not None else "",
            )
            status_path = None

        if status_path is not None:
            status_df = pd.read_csv(status_path)
            if "tps_id" not in status_df.columns or "status" not in status_df.columns:
                raise ValueError(f"{status_path}: expected columns 'tps_id'+'status' for --exclude-outliers.")
            exclude_set = set(status_df.loc[status_df["status"] != "OK", "tps_id"])

    kept_specimens: list[ImageLandmarks] = []
    kept_rows: list[pd.Series] = []
    unmatched, excluded = 0, 0

    for sp in specimens:
        if sp.tps_id in exclude_set:
            excluded += 1
            continue

        inv_id = sp.inv_id
        img_row = manifest_df.loc[sp.photo_id] if sp.photo_id is not None and sp.photo_id in manifest_df.index else None
        if inv_id is None or inv_id not in specimens_df.index:
            unmatched += 1
            continue

        row = specimens_df.loc[inv_id].copy()
        if img_row is not None:
            row["device_type"] = img_row["device_type"]
            row["device"] = img_row.get("device")
            row["device_tag"] = img_row["_device_tag"]
        kept_specimens.append(sp)
        kept_rows.append(row)

    if excluded:
        logger.info("Excluded %d SUSPECT/FAILED photo(s) via --exclude-outliers", excluded)
    if unmatched:
        logger.warning("%d photo(s) with no resolved inv_id or missing from biological_data.csv, ignored", unmatched)
    if not kept_specimens:
        raise ValueError(f"No specimen loaded from {root} -- check filters and files.")

    meta_df = pd.DataFrame(kept_rows).reset_index(drop=True)
    meta_df = add_groupe_column(meta_df)
    kept_specimens, meta_df = _drop_invalid_landmark_counts(kept_specimens, meta_df)

    if labeled_only:
        mask = meta_df["species"].notna().tolist()
        kept_specimens, meta_df = _apply_mask(kept_specimens, meta_df, mask, "labeled_only=True")

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
        raise ValueError("No specimen remaining after filtering -- check devices/species/castes.")

    logger.info("%d photo(s) loaded from %s", len(kept_specimens), root)
    return kept_specimens, meta_df
