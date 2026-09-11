"""uploaded_dataset.py
Turns TPS + biological-data CSV files picked ad hoc in a Streamlit
uploader (app/train_model.py, app/predict_dataset.py) into the dataset
root layout core.dataset.load_dataset expects (biological_data.csv,
manifest.csv, landmarks/landmarks_numbered.tps) -- these two tools still
never touch detection/landmark placement (see their own module
docstrings), they just accept the already-prepared TPS+CSV export as
direct uploads instead of a pre-arranged dataset folder.

Two ways to join a TPS specimen to its biological_data.csv row (see
join_specimens_to_bio): via COMMENT=...;inv_id=... (a TPS exported by this
codebase, e.g. tools/pipeline/export_final_landmarks.py -- see
core.tps_io module docstring), or, when the TPS carries no such COMMENT=
(e.g. landmarks placed by hand in a third-party tool, outside this
codebase entirely), positionally via a `tps_id` column in the CSV: one row
per specimen, in the same order as the TPS file(s) -- the ID= TPS assigns
each specimen 1..N per file (see core.tps_io.assign_sequential_ids),
mirrored by that CSV column.

parse_uploaded_tps_files()/parse_uploaded_bio_csv()/join_specimens_to_bio()
are pure (in-memory landmarks/dataframe in, validated in-memory result
out) so a caller can surface a DatasetMergeError to the user before
anything is written to disk. write_dataset_root() is the only function
that touches the filesystem, kept separate so a caller can materialize the
result into a throwaway directory right before handing it to a path-based
tool (classifiers.train.main/classifiers.predict.run_batch), then discard
it -- same "function in, disk I/O handled by the caller" split as
CONVENTIONS.md "Fonctions core reutilisables".

No --devices filter is possible on data merged this way: the synthesized
manifest.csv carries no real device_type/photo_index (an ad hoc upload
has no such metadata), only the photo_id column load_dataset's join needs.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from core.pipeline_io import parse_export_name
from core.run_io import slugify
from core.tps_io import ImageLandmarks, assign_sequential_ids, parse_tps, write_tps

REQUIRED_BIO_COLUMNS = {"inv_id", "species", "caste"}


def infer_dataset_label(tps_names: list[str], csv_names: list[str]) -> str:
    """Dataset label for write_dataset_root()'s throwaway root -- also this
    upload's default model/eval name, since classifiers.train/predict fall
    back to `core.dataset_config.resolve_dataset_name(dataset_root)` (which
    resolves to this exact label here, a throwaway root never getting its
    own dataset_config.json) when --model-name/--run-label aren't given
    (see core.run_io module docstring).

    Recovered from the uploaded files' own names when they came straight
    from tools.pipeline.export_final_landmarks: every one of that export's
    four files is named "<name>-<suffix>" (see core.pipeline_io.
    parse_export_name), so re-uploading an export here recovers the same
    `name` automatically instead of falling back to the export's own TPS
    filename -- previously always e.g. "landmarks_19lm_crop", identical
    across every dataset, which is why this label used to carry no real
    information (see PIPELINE.md "Nommage des modeles" and its step-6
    changelog). Falls back to the first TPS file's own stem, slugified, for
    a TPS/CSV pair not produced by that export (e.g. hand-digitized
    landmarks)."""
    for name in [*tps_names, *csv_names]:
        parsed_name = parse_export_name(name)
        if parsed_name:
            return parsed_name
    return slugify(Path(tps_names[0]).stem)


class DatasetMergeError(ValueError):
    """The uploaded TPS/CSV can't be merged into a usable dataset (missing
    columns, no specimen in common, ...) -- caught by the app to show a
    st.error instead of letting a low-level exception surface."""


def parse_uploaded_tps_files(paths: list[Path]) -> list[ImageLandmarks]:
    """Parses one or more uploaded .tps files and concatenates their
    specimens, renumbering `tps_id` sequentially across all of them (see
    core.tps_io.assign_sequential_ids) -- each upload keeps its own
    ID=/tps_id numbering on disk, which can collide once merged."""
    specimens: list[ImageLandmarks] = []
    for path in paths:
        parsed, errors = parse_tps(path, strict=False)
        if not parsed:
            raise DatasetMergeError(f"{path.name}: no valid specimen found ({len(errors)} parsing error(s)).")
        specimens.extend(parsed)
    return assign_sequential_ids(specimens)


def parse_uploaded_bio_csv(paths: list[Path]) -> pd.DataFrame:
    """Concatenates one or more uploaded biological_data.csv-style files
    and checks the columns core.dataset.load_dataset requires (inv_id,
    species, caste). Deduplicated on `tps_id` (one row per specimen) when
    that column is present -- needed for join_specimens_to_bio's
    positional join, where several rows legitimately share the same
    inv_id (one per photo of the same individual) -- otherwise on inv_id
    (keeping the first occurrence), since several uploaded exports may
    overlap on the same individual."""
    frames = [pd.read_csv(p) for p in paths]
    bio_df = pd.concat(frames, ignore_index=True)
    missing = REQUIRED_BIO_COLUMNS - set(bio_df.columns)
    if missing:
        raise DatasetMergeError(
            f"CSV file(s) missing required column(s): {sorted(missing)} "
            f"(expected at least {sorted(REQUIRED_BIO_COLUMNS)})."
        )
    dedup_col = "tps_id" if "tps_id" in bio_df.columns else "inv_id"
    return bio_df.drop_duplicates(subset=dedup_col, keep="first").reset_index(drop=True)


def join_specimens_to_bio(specimens: list[ImageLandmarks], bio_df: pd.DataFrame) -> list[ImageLandmarks]:
    """Resolves each specimen's inv_id against bio_df, so the result can be
    written by write_dataset_root and loaded normally afterwards (every
    specimen ends up with an inv_id found in bio_df, the only thing
    core.dataset.load_dataset actually needs to join). Raises
    DatasetMergeError if neither join is possible, or resolves to nothing.

    Two joins, tried in order:
    - COMMENT= inv_id (see core.tps_io module docstring): used whenever
      EVERY specimen already carries one (a TPS exported by this
      codebase) -- specimens are returned unchanged, only cross-checked
      against bio_df's own inv_id column.
    - positional, via a `tps_id` column in bio_df: used otherwise (e.g.
      landmarks placed by hand in a third-party tool that never wrote a
      COMMENT=) -- one CSV row per specimen, in the same order as the TPS
      file(s), matched on core.tps_io.ImageLandmarks.tps_id (the TPS ID=
      field, reassigned sequentially by parse_uploaded_tps_files). Returns
      specimens with inv_id filled in from the matching row.
    """
    if specimens and all(sp.inv_id for sp in specimens):
        bio_ids = set(bio_df["inv_id"].astype(str))
        tps_ids = {sp.inv_id for sp in specimens}
        if not (tps_ids & bio_ids):
            raise DatasetMergeError(
                "No inv_id in common between the TPS and the CSV -- check that these files come from "
                "the same dataset export."
            )
        return specimens

    if "tps_id" not in bio_df.columns:
        raise DatasetMergeError(
            "The TPS file(s) carry no inv_id in their COMMENT= field, and the CSV has no 'tps_id' "
            "column to join them positionally either -- either pass a TPS exported by this codebase "
            "(COMMENT=...;inv_id=...;, e.g. tools/pipeline/export_final_landmarks.py), or add a "
            "'tps_id' column to the CSV with one row per specimen, in the same order as the TPS "
            "file(s) (1, 2, 3, ...)."
        )
    try:
        tps_id_to_inv_id = dict(zip(bio_df["tps_id"].astype(int), bio_df["inv_id"].astype(str)))
    except (TypeError, ValueError) as exc:
        raise DatasetMergeError(f"CSV 'tps_id' column must contain only integers: {exc}") from exc

    missing = sorted({sp.tps_id for sp in specimens} - tps_id_to_inv_id.keys())
    if missing:
        raise DatasetMergeError(
            f"{len(missing)} TPS specimen(s) have no matching 'tps_id' in the CSV (expected one CSV "
            f"row per specimen, in the same order as the TPS file(s)) -- missing tps_id: {missing[:10]}"
            f"{', ...' if len(missing) > 10 else ''}."
        )
    return [replace(sp, inv_id=tps_id_to_inv_id[sp.tps_id]) for sp in specimens]


def write_dataset_root(specimens: list[ImageLandmarks], bio_df: pd.DataFrame, dest: Path) -> Path:
    """Materializes `dest` as a dataset root core.dataset.load_dataset can
    read: biological_data.csv, landmarks/landmarks_numbered.tps (the
    merged specimens), and a manifest.csv reduced to the columns
    load_dataset actually needs (photo_id, device_type, photo_index) --
    device_type/photo_index are left empty since an ad hoc upload carries
    no device metadata (only used for the --devices filter, not exposed by
    this upload flow).

    biological_data.csv drops bio_df's `tps_id` column (join-only, not
    part of the on-disk schema -- see join_specimens_to_bio) and is
    deduplicated to one row per inv_id: load_dataset indexes it by inv_id
    and expects a single row back (specimens_df.loc[inv_id]), whereas
    bio_df itself may still carry several rows per inv_id at this point
    (one per photo, from a positional tps_id join)."""
    dest.mkdir(parents=True, exist_ok=True)
    bio_out = bio_df.drop(columns=["tps_id"], errors="ignore")
    bio_out = bio_out.drop_duplicates(subset="inv_id", keep="first").reset_index(drop=True)
    bio_out.to_csv(dest / "biological_data.csv", index=False)

    # A hand-landmarked TPS (no COMMENT=, joined positionally via tps_id --
    # see join_specimens_to_bio) carries no photo_id at all: synthesize one
    # here and assign it back onto the specimen itself (not just this
    # manifest row) so it also lands in the TPS's own COMMENT= below --
    # otherwise core.dataset.load_dataset re-parses the TPS with
    # photo_id=None and every downstream predictions.csv/loocv_predictions
    # .csv ends up with an empty photo_id column for these specimens.
    specimens = [sp if sp.photo_id else replace(sp, photo_id=f"tps_{sp.tps_id}") for sp in specimens]

    manifest_df = pd.DataFrame({
        "photo_id": [sp.photo_id for sp in specimens],
        "device_type": None,
        "photo_index": None,
    })
    manifest_df.to_csv(dest / "manifest.csv", index=False)

    landmarks_dir = dest / "landmarks"
    landmarks_dir.mkdir(exist_ok=True)
    write_tps(landmarks_dir / "landmarks_numbered.tps", specimens)
    return dest
