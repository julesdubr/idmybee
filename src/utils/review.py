"""review.py
Shared validation-review logic for the cropping and landmark-placement
checkpoints (see TODO.md Phase 2 "étape de validation"): build a DataFrame
from a stage's own auto status log, let a human override the status per
photo, then persist the result as a schema-compatible override file the
rest of the pipeline already knows how to consume -- no new mechanism on
the consuming side:

    crops_reviewed.csv       (same schema as crops.csv) -> landmarks.predict --crops-csv
    landmarks_reviewed.csv   (same schema as landmarks_numbered.csv) -> anything
                              reading via utils.cli.add_dataset_args's --landmarks-status-csv

Used identically by app/build_dataset.py's two validation steps and by the
CLI pair tools.pipeline.export_review (writes the editable review CSV) /
tools.pipeline.reconcile_review (re-applies a hand-edited one) -- one
implementation, see CONVENTIONS.md "Fonctions core réutilisables".

Never mutates crops.csv/landmarks_numbered.csv themselves: those stay the
auto pipeline's own log, for audit. Overrides live in a parallel file, and
in a companion `<dataset>/review/<name>_review.csv` documenting the
auto vs. reviewed status for every photo.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from core.tps_io import ImageLandmarks, parse_tps

logger = logging.getLogger(__name__)

REVIEW_DIRNAME = "review"
REVIEW_STATUSES = ("OK", "SUSPECT", "FAILED")


def crops_csv_path(dataset: Path, mode: str) -> Path:
    return Path(dataset) / "extraction" / mode / "crops.csv"


def crops_reviewed_path(dataset: Path, mode: str) -> Path:
    return Path(dataset) / "extraction" / mode / "crops_reviewed.csv"


def landmarks_numbered_csv_path(dataset: Path) -> Path:
    return Path(dataset) / "landmarks" / "landmarks_numbered.csv"


def landmarks_reviewed_csv_path(dataset: Path) -> Path:
    return Path(dataset) / "landmarks" / "landmarks_reviewed.csv"


def review_audit_path(dataset: Path, name: str) -> Path:
    return Path(dataset) / REVIEW_DIRNAME / f"{name}_review.csv"


def build_crop_review_df(dataset: Path, mode: str = "light") -> pd.DataFrame:
    """One row per photo processed by extraction.normalize_crop:
    photo_id, inv_id, auto_status, reviewed_status (defaults to
    auto_status), error_reason, output_path (crop image -- the review
    "picture" for this step IS the crop itself, no separate overlay
    needed)."""
    path = crops_csv_path(dataset, mode)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run extraction.normalize_crop first.")
    df = pd.read_csv(path)
    # crops.csv is append-only (checkpointed runs) -- keep only the latest
    # row per photo_id, same convention as landmarks.predict.load_target_crops.
    df = df.drop_duplicates("photo_id", keep="last").rename(columns={"status": "auto_status"})
    df["reviewed_status"] = df["auto_status"]
    return df[["photo_id", "inv_id", "auto_status", "reviewed_status", "error_reason", "output_path"]]


def build_landmark_review_df(dataset: Path) -> pd.DataFrame:
    """One row per photo processed by landmarks.renumber: photo_id, inv_id,
    tps_id, auto_status, reviewed_status (defaults to auto_status),
    registration_cost, n_outlier_landmarks, error_reason."""
    path = landmarks_numbered_csv_path(dataset)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run landmarks.renumber first.")
    df = pd.read_csv(path).rename(columns={"status": "auto_status"})
    df["reviewed_status"] = df["auto_status"]
    return df[[
        "photo_id", "inv_id", "tps_id", "auto_status", "reviewed_status",
        "registration_cost", "n_outlier_landmarks", "error_reason",
    ]]


def load_numbered_landmarks_by_photo_id(
    dataset: Path, tps_name: str = "landmarks_numbered.tps",
) -> dict[str, ImageLandmarks]:
    """For on-demand overlay preview during landmark review (see
    utils.tps_overlay.draw_landmarks) -- loaded once per review session,
    not per row, to stay light on a large dataset."""
    tps_path = Path(dataset) / "landmarks" / tps_name
    specimens, _errors = parse_tps(tps_path, strict=False)
    return {sp.photo_id: sp for sp in specimens if sp.photo_id}


def _write_audit(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df[["photo_id", "auto_status", "reviewed_status"]].to_csv(path, index=False)


def write_crop_review(dataset: Path, mode: str, df: pd.DataFrame) -> tuple[Path, Path]:
    """Persists edited crop statuses (df: photo_id, auto_status,
    reviewed_status, other columns ignored): a reconciled crops_reviewed.csv
    (same schema as crops.csv, status/error_reason overridden -- see
    landmarks.predict --crops-csv) and an audit trail
    (<dataset>/review/crops_review.csv). Returns (reconciled_path, audit_path).
    """
    original = pd.read_csv(crops_csv_path(dataset, mode)).drop_duplicates("photo_id", keep="last")
    overrides = dict(zip(df["photo_id"], df["reviewed_status"]))
    changed = set(df.loc[df["auto_status"] != df["reviewed_status"], "photo_id"])

    reconciled = original.copy()
    reconciled["status"] = reconciled["photo_id"].map(overrides).fillna(reconciled["status"])
    is_changed = reconciled["photo_id"].isin(changed)
    # error_reason reads back as float64 NaN (not object) when every row's
    # original value is empty (e.g. a dataset with no failures yet) --
    # cast explicitly, otherwise assigning a string here raises.
    reconciled["error_reason"] = reconciled["error_reason"].astype("object")
    reconciled.loc[is_changed, "error_reason"] = "excluded_by_manual_review"

    out_path = crops_reviewed_path(dataset, mode)
    reconciled.to_csv(out_path, index=False)
    audit_path = review_audit_path(dataset, "crops")
    _write_audit(df, audit_path)
    logger.info("%d crop status override(s) applied -- %s, %s", len(changed), out_path, audit_path)
    return out_path, audit_path


def write_landmarks_review(dataset: Path, df: pd.DataFrame) -> tuple[Path, Path]:
    """Persists edited landmark statuses (df: photo_id, auto_status,
    reviewed_status, other columns ignored): a reconciled
    landmarks_reviewed.csv (same schema as landmarks_numbered.csv -- pass
    as --landmarks-status-csv, see utils.cli.add_dataset_args, to export/
    train/predict on the reviewed statuses with no other code change) and
    an audit trail (<dataset>/review/landmarks_review.csv). Returns
    (reconciled_path, audit_path)."""
    original = pd.read_csv(landmarks_numbered_csv_path(dataset))
    overrides = dict(zip(df["photo_id"], df["reviewed_status"]))
    changed = set(df.loc[df["auto_status"] != df["reviewed_status"], "photo_id"])

    reconciled = original.copy()
    reconciled["status"] = reconciled["photo_id"].map(overrides).fillna(reconciled["status"])
    is_changed = reconciled["photo_id"].isin(changed)
    # see write_crop_review's comment: error_reason reads back as float64
    # NaN, not object, when every row's original value is empty.
    reconciled["error_reason"] = reconciled["error_reason"].astype("object")
    reconciled.loc[is_changed, "error_reason"] = (
        "manual review override (was: " + reconciled.loc[is_changed, "error_reason"].fillna("") + ")"
    )

    out_path = landmarks_reviewed_csv_path(dataset)
    reconciled.to_csv(out_path, index=False)
    audit_path = review_audit_path(dataset, "landmarks")
    _write_audit(df, audit_path)
    logger.info("%d landmark status override(s) applied -- %s, %s", len(changed), out_path, audit_path)
    return out_path, audit_path


def read_review_csv(path: Path) -> pd.DataFrame:
    """Reads back a review CSV (written by build_crop_review_df/
    build_landmark_review_df, possibly hand-edited in a spreadsheet in the
    meantime -- see tools.pipeline.export_review/reconcile_review).
    Validates just enough to fail loudly on a malformed hand-edit rather
    than silently reconciling garbage."""
    df = pd.read_csv(path)
    required = {"photo_id", "auto_status", "reviewed_status"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing column(s) {sorted(missing)}")
    invalid = set(df["reviewed_status"]) - set(REVIEW_STATUSES)
    if invalid:
        raise ValueError(f"{path}: invalid reviewed_status value(s) {sorted(invalid)} (expected {REVIEW_STATUSES})")
    return df
