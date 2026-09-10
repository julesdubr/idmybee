"""setup_dataset.py
Streamlit dataset setup/cleaning tool -- Tool 1 of 3 (see README.md
"Scenario 1"). Prepares a raw or existing dataset root, runs it through
detection -> crop -> UNet landmark placement -> renumbering -> export with
two human review checkpoints in between, and stops once the dataset is
ready. Fitting a model on it (app/train_model.py) or classifying it with
one (app/predict_dataset.py) are separate tools, run afterwards -- neither
runs any landmarking of its own, they just point at a dataset this tool has
already prepared.

    streamlit run app/setup_dataset.py

Reuses the exact same argument definitions as the CLI (utils.cli.
add_dataset_args/add_landmarking_args, utils.landmarking_pipeline.
dataset_filter_argv) to build the shared `args` Namespace from the widget
values in step 1, rather than hand-building one field by field -- one
parsing/validation/defaults implementation for both the CLI and this UI
(see CONVENTIONS.md "Fonctions core reutilisables"). Every pipeline stage
still runs through its own main(argv)/run_batch(args), in-process, exactly
as tools.pipeline.train_dataset/predict_dataset do -- this app only adds
the two validation checkpoints and the step-by-step pacing around them.

Preparing a dataset (raw images + a per-photo identification CSV, several
sources, ...) goes through tools.ingestion.prepare_dataset.run() -- the
exact same config-driven orchestrator the CLI uses (see that module's
docstring for the config shape), just built from the step 1 wizard's
widgets instead of a hand-written JSON file. inv_id_mapping.csv and (when
built interactively rather than supplied) collection_origin_codes.csv are
written inside the dataset this run produces -- see manifest.origin_table
and tools.ingestion.prepare_dataset's module docstring, no separate shared
identification/ folder anymore.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

# app/ itself isn't an installed package (unlike src/'s top-level packages,
# see pyproject.toml's src-layout) -- `streamlit run` doesn't reliably put
# this file's own directory on sys.path, so a sibling import needs an
# explicit assist.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    LOGO_PATH, discover, image_to_data_url, array_to_data_url, path_picker, plot_reference_shape, run_with_log,
)
from tools.ingestion import prepare_dataset  # noqa: E402
from tools.ingestion.ingest_raw import NAMING_PARSERS  # noqa: E402
from manifest.origin_table import (  # noqa: E402
    assign_constant_origin, build_origin_table, origin_codes_from_table, write_origin_codes,
)
from utils.cli import add_dataset_args, add_dataset_positional, add_logging_args  # noqa: E402
from utils.landmarking_pipeline import (  # noqa: E402
    DEFAULT_DETECTOR_MODEL,
    REFERENCE_SHAPES_DIR,
    add_landmarking_args,
    resolve_export_dir,
    run_detection_and_crop,
    run_export,
    run_landmark_placement,
)
from utils.review import (  # noqa: E402
    build_crop_review_df,
    build_landmark_review_df,
    load_numbered_landmarks_by_photo_id,
    write_crop_review,
    write_landmarks_review,
)
from utils.tps_overlay import draw_landmarks  # noqa: E402
from core.pipeline_io import resolve_path  # noqa: E402
from landmarks.build_reference import load_reference  # noqa: E402

STATUSES = ["OK", "SUSPECT", "FAILED"]
STEPS = ["Setup", "Detection & crop", "Crop review", "Landmark placement", "Landmark review", "Export"]

st.set_page_config(
    page_title="idmybee -- setup a dataset",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else None,
    layout="wide",
)


# ---------------------------------------------------------------------------
# Session state / small shared helpers
# ---------------------------------------------------------------------------

st.session_state.setdefault("step", 1)
st.session_state.setdefault("dataset_root", "")
st.session_state.setdefault("args", None)
st.session_state.setdefault("crop_review_df", None)
st.session_state.setdefault("landmark_review_df", None)
st.session_state.setdefault("prep_sources", [{"id": 0}])
st.session_state.setdefault("prep_next_id", 1)
st.session_state.setdefault("export_dir", None)


def restart() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


def advance(step: int) -> None:
    st.session_state.step = step
    st.rerun()


def landmarking_only_parser() -> argparse.ArgumentParser:
    """dataset + landmarking + dataset-filter flags only -- this tool never
    fits/applies a model itself (see app/train_model.py, app/predict_dataset.py)."""
    parser = argparse.ArgumentParser()
    add_dataset_positional(parser)
    add_landmarking_args(parser)
    add_dataset_args(parser)
    add_logging_args(parser)
    return parser


def header() -> None:
    col_logo, col_title = st.columns([1, 8])
    if LOGO_PATH.exists():
        col_logo.image(str(LOGO_PATH), width=100)
    with col_title:
        st.title("Set up a dataset")
        st.caption("Prepare a dataset and validate it -- detection, landmark placement, human review, export.")
    with st.sidebar:
        st.progress(st.session_state.step / len(STEPS), text=f"Step {st.session_state.step} of {len(STEPS)}")
        for i, name in enumerate(STEPS, start=1):
            if i < st.session_state.step:
                st.caption(f":material/check: ~~{name}~~")
            elif i == st.session_state.step:
                st.markdown(f"**:material/arrow_right: {name}**")
            else:
                st.caption(name)
        st.divider()
        st.button("Start over", icon=":material/restart_alt:", on_click=restart)


header()


# ---------------------------------------------------------------------------
# Step 1 -- setup: dataset source, origin codes, landmarking parameters
# ---------------------------------------------------------------------------

def _source_to_config(source: dict) -> dict:
    return {k: v for k, v in source.items() if k != "id" and not k.startswith("_") and v not in (None, "", False)}


def _render_origin_codes(sid, source: dict) -> None:
    """Origin-codes UI: an explicit CSV (power-user path, unchanged), or an
    interactive table built from the identification CSV's own origin
    column values -- or, if that column doesn't exist at all, a single
    inv_name applied to every row of this source. See
    manifest.origin_table and _resolve_source_for_submit()."""
    source["origin_codes"] = path_picker(
        "Origin codes CSV (optional -- overrides the automatic table below)",
        mode="file", key=f"src_{sid}_origin_codes",
    )
    source["origin_column"] = st.text_input(
        "Origin column in the identification CSV (optional, default: collection_origin)",
        key=f"src_{sid}_origin_column",
    )
    if source["origin_codes"]:
        return

    source["_auto_origin"] = st.checkbox(
        "Build inv_name codes automatically for this source", key=f"src_{sid}_auto_origin",
        help="Shows an editable table pre-filled with the origin values found in the identification "
             "CSV -- or, if it has no origin column, a single inv_name for the whole source.",
    )
    if not source["_auto_origin"]:
        return
    identification_csv = source.get("identification_csv")
    if not identification_csv or not Path(identification_csv).exists():
        st.caption("Pick an identification CSV above first.")
        return

    origin_column = source["origin_column"] or "collection_origin"
    try:
        identification_df = pd.read_csv(identification_csv)
    except (OSError, pd.errors.ParserError) as exc:
        st.error(f"Could not read {identification_csv}: {exc}")
        return

    if origin_column in identification_df.columns:
        table = build_origin_table(identification_df, origin_column)
        if table.empty:
            st.warning(f"No values found in column {origin_column!r}.")
            return
        st.caption(f"Fill in an inv_name for each {origin_column!r} value found:")
        source["_origin_table"] = st.data_editor(
            table, key=f"src_{sid}_origin_table", hide_index=True, num_rows="fixed",
        )
    else:
        st.caption(f"No {origin_column!r} column found in the identification CSV.")
        source["_constant_inv_name"] = st.text_input(
            "inv_name to use for every specimen in this source", key=f"src_{sid}_constant_inv_name",
        )


def _render_source(index: int, source: dict, n_sources: int) -> None:
    sid = source["id"]
    with st.container(border=True):
        col_title, col_remove = st.columns([5, 1])
        col_title.markdown(f"**Source {index + 1}**")
        if n_sources > 1 and col_remove.button(":material/delete:", key=f"src_{sid}_remove", help="Remove this source"):
            st.session_state.prep_sources.remove(source)
            st.rerun()

        source["images_dir"] = path_picker(
            "Raw images folder", mode="dir", key=f"src_{sid}_images_dir",
            help="Scanned for photos; each filename is parsed for an original id/device/shot index.",
        )
        source["identification_csv"] = path_picker(
            "Identification CSV (biological data)", mode="file", key=f"src_{sid}_id_csv",
        )
        col_a, col_b = st.columns(2)
        source["source_type"] = col_a.text_input("Source type", key=f"src_{sid}_source_type", placeholder="collection")
        source["key_column"] = col_b.text_input(
            "Key column", key=f"src_{sid}_key_column", placeholder="inv_id",
            help="Identification CSV column matching each photo's parsed original id.",
        )
        source["output_dir"] = path_picker(
            "Output dir (this source's own dataset.csv + manifest.csv)", mode="dir", key=f"src_{sid}_output_dir",
        )
        with st.expander("Origin codes (inv_name)", icon=":material/tag:"):
            _render_origin_codes(sid, source)
        with st.expander("Advanced options"):
            col_a, col_b = st.columns(2)
            source["photographer_subfolder"] = col_a.checkbox(
                "First-level subfolder is the photographer (terrain style)", key=f"src_{sid}_photog",
            )
            naming = col_b.selectbox(
                "Filename convention (blank = inferred from the checkbox above)",
                [""] + list(NAMING_PARSERS), index=0, key=f"src_{sid}_naming",
            )
            if naming:
                source["naming"] = naming
            else:
                source.pop("naming", None)
            source["device_column"] = st.text_input("Device column (optional)", key=f"src_{sid}_device_column")
            source["device_name_column"] = st.text_input(
                "Device name column (optional, default: device)", key=f"src_{sid}_device_name_column",
            )
            source["compare_columns"] = st.text_input(
                "Compare columns (comma-separated, optional, default: genus,species,caste,dd,mm,yyyy)",
                key=f"src_{sid}_compare_columns",
            )
            source["image_group_by"] = st.text_input(
                "Group copied images by columns (comma-separated, optional)", key=f"src_{sid}_image_group_by",
            )
            source["default_device_type"] = st.text_input("Default device type", value="S", key=f"src_{sid}_ddt")
            source["no_copy_images"] = st.checkbox("Don't copy image files (CSVs/reports only)", key=f"src_{sid}_no_copy")
            source["keep_raw_manifest"] = st.checkbox(
                "Keep the raw scan report after preparing (written inside the images folder as "
                ".idmybee_ingest/ -- also reused instead of rescanned on the next run)",
                key=f"src_{sid}_keep_raw",
            )


def _resolve_source_for_submit(source: dict) -> dict:
    """Returns a copy of `source` with origin_codes/origin_column/
    identification_csv finalized for tools.ingestion.prepare_dataset --
    writes the auto-built origin table (and, for the no-origin-column
    case, an adjusted identification CSV with a synthetic constant column)
    into this source's own output_dir. Raises ValueError if the user
    opted into auto-building but hasn't filled it in yet. See
    _render_origin_codes()/manifest.origin_table."""
    resolved = dict(source)
    if resolved.get("origin_codes") or not resolved.get("_auto_origin"):
        resolved.pop("_auto_origin", None)
        resolved.pop("_origin_table", None)
        resolved.pop("_constant_inv_name", None)
        return resolved

    output_dir = Path(resolved["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    origin_column = resolved.get("origin_column") or "collection_origin"
    identification_df = pd.read_csv(resolved["identification_csv"])
    label = resolved.get("source_type", "?")

    if origin_column in identification_df.columns:
        table = resolved.get("_origin_table")
        if table is None or not origin_codes_from_table(table):
            raise ValueError(f"Source {label!r}: fill in the origin codes table before preparing.")
        codes_path = output_dir / "collection_origin_codes.csv"
        write_origin_codes(table, codes_path)
        resolved["origin_codes"] = str(codes_path)
        resolved["origin_column"] = origin_column
    else:
        inv_name = (resolved.get("_constant_inv_name") or "").strip()
        if not inv_name:
            raise ValueError(f"Source {label!r}: enter an inv_name before preparing.")
        df_with_col, col, table = assign_constant_origin(identification_df, inv_name)
        adjusted_csv = output_dir / "identification_with_origin.csv"
        df_with_col.to_csv(adjusted_csv, index=False)
        codes_path = output_dir / "collection_origin_codes.csv"
        write_origin_codes(table, codes_path)
        resolved["identification_csv"] = str(adjusted_csv)
        resolved["origin_column"] = col
        resolved["origin_codes"] = str(codes_path)

    resolved.pop("_auto_origin", None)
    resolved.pop("_origin_table", None)
    resolved.pop("_constant_inv_name", None)
    return resolved


def _prepare_dataset_wizard() -> None:
    with st.expander("Expected format", icon=":material/info:"):
        st.markdown(
            "Each source is a folder of raw photos (several filename conventions supported -- "
            "the same raw label can be reused across specimens) plus an identification CSV with "
            "the biological data (species, caste, ...) and a column matching each photo's parsed "
            "id. Identity conflicts are resolved automatically into one clean, canonical dataset "
            "-- no per-photo CSV to hand-prepare first.\n"
            "- Add more than one source (e.g. collection + terrain) to combine them into one "
            "dataset root.\n"
            "- inv_id_mapping.csv is written inside the output dir below (shared across sources) -- "
            "no separate mapping file to point at unless you're deliberately sharing one across "
            "several dataset roots."
        )

    output_dir = path_picker(
        "Output dir (final combined manifest.csv/biological_data.csv)",
        mode="dir", key="prep_output_dir",
    )
    mapping_file = path_picker(
        "Identity mapping file (optional -- default: <output dir>/inv_id_mapping.csv)",
        mode="file", key="prep_mapping_file",
    )

    sources: list[dict] = st.session_state.prep_sources
    st.caption("Sources")
    for i, source in enumerate(sources):
        _render_source(i, source, len(sources))

    if st.button(":material/add: Add source", key="prep_add_source"):
        sources.append({"id": st.session_state.prep_next_id})
        st.session_state.prep_next_id += 1
        st.rerun()

    st.divider()
    if st.button("Prepare dataset", icon=":material/build:", type="primary", key="prep_submit"):
        errors = []
        if not output_dir:
            errors.append("Output dir is required.")
        required = ["images_dir", "identification_csv", "source_type", "key_column", "output_dir"]
        for i, source in enumerate(sources):
            if any(not source.get(field) for field in required):
                errors.append(f"Source {i + 1}: {', '.join(required)} are all required.")

        if errors:
            for error in errors:
                st.error(error)
            return

        try:
            resolved_sources = [_resolve_source_for_submit(s) for s in sources]
        except ValueError as exc:
            st.error(str(exc))
            return

        config: dict = {"output_dir": output_dir, "sources": [_source_to_config(s) for s in resolved_sources]}
        if mapping_file:
            config["mapping_file"] = mapping_file

        result = run_with_log("Preparing dataset...", prepare_dataset.run, config)

        if (Path(output_dir) / "manifest.csv").exists():
            st.session_state.dataset_root = output_dir
            n_photos = len(pd.read_csv(Path(output_dir) / "manifest.csv"))
            st.success(f"Dataset prepared -- {n_photos} photo(s) -> {output_dir}")
        if result["failed_sources"]:
            st.warning(f"{len(result['failed_sources'])} source(s) failed: {result['failed_sources']}")
            for source in resolved_sources:
                source_output_dir = Path(source.get("output_dir") or "")
                raw_path = source_output_dir / "manifest_raw.csv"
                if raw_path.exists():
                    st.caption(f"{source_output_dir}/manifest_raw.csv:")
                    st.dataframe(pd.read_csv(raw_path), hide_index=True)
                inconsistencies = source_output_dir / "reports" / "biological_inconsistencies.csv"
                if inconsistencies.exists():
                    st.caption(f"{source_output_dir}/reports/biological_inconsistencies.csv:")
                    st.dataframe(pd.read_csv(inconsistencies), hide_index=True)


def _unet_landmark_count(weights_path: str) -> int | None:
    """n_landmarks this UNet checkpoint was trained for, read from its
    sibling train_config.json (see landmarks_trainer/train.py) -- None if
    that file is missing or doesn't record it (e.g. a migrated legacy
    checkpoint), letting the caller fall back to manual entry."""
    config_path = Path(weights_path).parent / "train_config.json"
    if not config_path.exists():
        return None
    try:
        return json.loads(config_path.read_text(encoding="utf-8")).get("n_landmarks")
    except (OSError, json.JSONDecodeError):
        return None


def step_setup() -> None:
    st.header(STEPS[0])

    st.subheader("Dataset")
    source_mode = st.segmented_control(
        "Dataset source", options=["existing", "prepare"],
        format_func=lambda m: "Use an existing dataset root" if m == "existing" else "Prepare a new dataset",
        default="existing", required=True, key="source_mode_control",
    )

    if source_mode == "existing":
        dataset_root = path_picker(
            "Dataset root", mode="dir", value=st.session_state.dataset_root, key="dataset_root",
            help="Must already contain manifest.csv and biological_data.csv.",
        )
        st.session_state.dataset_root = dataset_root
        if dataset_root:
            root = Path(dataset_root)
            if (root / "manifest.csv").exists() and (root / "biological_data.csv").exists():
                n_photos = len(pd.read_csv(root / "manifest.csv"))
                n_specimens = len(pd.read_csv(root / "biological_data.csv"))
                st.success(f"{n_photos} photo(s), {n_specimens} specimen(s) found at {dataset_root}.")
            else:
                st.warning("manifest.csv/biological_data.csv not found at this path yet.")
    else:
        _prepare_dataset_wizard()

    dataset_root = st.session_state.dataset_root
    dataset_ready = bool(dataset_root) and (Path(dataset_root) / "manifest.csv").exists()
    if not dataset_ready:
        return

    st.subheader("Pipeline parameters")
    unet_choices = discover("models/unet_landmarks/*/weights.pt")
    with st.container(border=True):
        mode = st.segmented_control(
            "Detection backend", options=["light", "heavy"], default="light", required=True, key="lf_mode",
        )
        detector_model = path_picker(
            "Wing detector weights", mode="file", value=str(DEFAULT_DETECTOR_MODEL), key="lf_detector_model",
        )
        heavy_ref = path_picker(
            "YOLOE references JSON (--mode heavy only)", mode="file", key="lf_heavy_ref",
        ) if mode == "heavy" else ""

        if unet_choices:
            unet_model = st.selectbox(
                "UNet landmark weights", unet_choices, format_func=lambda p: Path(p).parent.name, key="lf_unet_choice",
            )
        else:
            unet_model = path_picker("UNet landmark weights path", mode="file", key="lf_unet_text")

        unet_n_landmarks = _unet_landmark_count(unet_model) if unet_model else None
        if unet_n_landmarks:
            st.caption(f"This UNet checkpoint predicts {unet_n_landmarks} landmark(s) (from its train_config.json).")
            n_landmarks = unet_n_landmarks
        else:
            n_landmarks = st.number_input(
                "Number of landmarks", min_value=1, value=19, step=1, key="lf_n_landmarks",
                help="No train_config.json n_landmarks found for this checkpoint -- set it manually "
                     "(19 = Tancrede's full blueprint, 18 for an older/legacy UNet model).",
            )

        reference_choices = discover(f"{REFERENCE_SHAPES_DIR}/*.tps")
        if reference_choices:
            reference = st.selectbox(
                "GPA reference shape", reference_choices, format_func=lambda p: Path(p).stem, key="lf_reference_choice",
                help="A plain .tps, one specimen block -- see app/train_model.py's "
                     "'Build a reference shape' to make a new one.",
            )
        else:
            reference = path_picker("GPA reference shape path", mode="file", key="lf_reference_text")
        if reference and Path(reference).exists():
            with st.expander("Preview reference shape"):
                try:
                    zones = load_reference(Path(reference))
                    st.caption(f"{len(zones)} landmark(s)")
                    st.pyplot(plot_reference_shape(zones))
                except SystemExit as exc:
                    st.error(str(exc))

        col_a, col_b = st.columns(2)
        imgsz = col_a.number_input("Detection image size", min_value=64, value=1024, step=32, key="lf_imgsz")
        conf = col_b.slider("Detection confidence threshold", 0.0, 1.0, 0.10, 0.01, key="lf_conf")
        padding = col_a.slider("Crop padding", 0.0, 0.5, 0.10, 0.01, key="lf_padding")
        device = col_b.selectbox("Device", ["cpu", "cuda"], index=0, key="lf_device")
        out_width = col_a.number_input("Crop width", min_value=32, value=512, step=32, key="lf_out_width")
        out_height = col_b.number_input("Crop height", min_value=32, value=256, step=32, key="lf_out_height")

        base_dir = path_picker(
            "Base dir (optional, resolves manifest.csv's path if relative)", mode="dir", key="lf_base_dir",
        )
        overwrite = st.checkbox("Overwrite crops/landmarks already logged (skip resuming a prior run)", key="lf_overwrite")
        retry_failed = st.checkbox("Retry photos previously logged FAILED", key="lf_retry_failed")

        submitted = st.button("Continue to detection & crop", icon=":material/arrow_forward:", type="primary", key="lf_submit")

    if submitted:
        if mode == "heavy" and not heavy_ref:
            st.error("--mode heavy requires a YOLOE references JSON.")
            return
        if not unet_model:
            st.error("A UNet landmark weights path is required.")
            return
        if not reference:
            st.error("A GPA reference shape is required.")
            return
        argv = [
            dataset_root, "--mode", mode, "--detector-model", detector_model,
            "--unet-model", unet_model, "--n-landmarks", str(int(n_landmarks)),
            "--reference", reference,
            "--imgsz", str(int(imgsz)), "--conf", str(conf),
            "--padding", str(padding), "--out-width", str(int(out_width)), "--out-height", str(int(out_height)),
            "--device", device,
        ]
        if heavy_ref:
            argv += ["--heavy-ref", heavy_ref]
        if base_dir:
            argv += ["--base-dir", base_dir]
        if overwrite:
            argv += ["--overwrite"]
        if retry_failed:
            argv += ["--retry-failed"]

        st.session_state.args = landmarking_only_parser().parse_args(argv)
        advance(2)


# ---------------------------------------------------------------------------
# Step 2 / 4 -- run a pipeline stage, with a "skip, already done" escape hatch
# ---------------------------------------------------------------------------

def step_run(title: str, run_label: str, fn, next_step: int, skip_hint: str) -> None:
    st.header(title)
    args = st.session_state.args
    st.caption(f"Dataset: {args.dataset}")

    slug = run_label.replace(" ", "_")
    col_run, col_skip = st.columns(2)
    if col_run.button(f"Run {run_label}", icon=":material/play_arrow:", type="primary", key=f"run_{slug}"):
        run_with_log(f"Running {run_label}...", fn, args)
        st.session_state.step = next_step
        st.rerun()
    if col_skip.button("Skip -- already done", icon=":material/skip_next:", help=skip_hint, key=f"skip_{slug}"):
        advance(next_step)


# ---------------------------------------------------------------------------
# Steps 3 / 5 -- validation review (shared rendering, different data source)
# ---------------------------------------------------------------------------

def _filtered_view(df: pd.DataFrame, state_key: str) -> pd.DataFrame:
    col_filter, col_search = st.columns([1, 2])
    status_filter = col_filter.segmented_control(
        "Filter by status", options=["All"] + STATUSES, default="All", key=f"{state_key}_filter", required=True,
    )
    search = col_search.text_input(
        "Search photo_id / inv_id", key=f"{state_key}_search", placeholder="Search photo_id / inv_id",
        label_visibility="collapsed",
    )
    view = df
    if status_filter and status_filter != "All":
        view = view[view["reviewed_status"] == status_filter]
    if search:
        mask = (
            view["photo_id"].astype(str).str.contains(search, case=False, na=False)
            | view["inv_id"].astype(str).str.contains(search, case=False, na=False)
        )
        view = view[mask]
    return view


def _thumbnail_width() -> int:
    size = st.segmented_control(
        "Thumbnail size", options=["medium", "large"], default="medium", required=True, key="thumb_size",
    )
    return 256 if size == "medium" else 512


def step_crop_review() -> None:
    st.header(STEPS[2])
    args = st.session_state.args

    if st.session_state.crop_review_df is None:
        try:
            st.session_state.crop_review_df = build_crop_review_df(args.dataset, args.mode)
        except FileNotFoundError as exc:
            st.error(str(exc))
            return

    full_df = st.session_state.crop_review_df
    counts = full_df["reviewed_status"].value_counts()
    st.caption(" | ".join(f"{s}: {counts.get(s, 0)}" for s in STATUSES))

    thumb_width = _thumbnail_width()
    view = _filtered_view(full_df, "crop")
    view = view.assign(thumbnail=[
        image_to_data_url(path, max_width=thumb_width) if isinstance(path, str) else ""
        for path in view["output_path"]
    ])
    display_cols = ["thumbnail", "photo_id", "inv_id", "auto_status", "reviewed_status", "error_reason"]
    edited = st.data_editor(
        view[display_cols], key="crop_editor", hide_index=True, row_height=thumb_width//2,
        column_config={
            "thumbnail": st.column_config.ImageColumn("Crop", width=thumb_width),
            "reviewed_status": st.column_config.SelectboxColumn(options=STATUSES, required=True),
        },
        disabled=[c for c in display_cols if c != "reviewed_status"],
    )
    full_df.loc[edited.index, "reviewed_status"] = edited["reviewed_status"]
    st.session_state.crop_review_df = full_df

    if st.button("Save crop review & continue", icon=":material/arrow_forward:", type="primary"):
        crops_reviewed_path, _audit = write_crop_review(args.dataset, args.mode, full_df)
        args.crops_csv = str(crops_reviewed_path)
        n_changed = int((full_df["auto_status"] != full_df["reviewed_status"]).sum())
        st.toast(f"{n_changed} override(s) saved -> {crops_reviewed_path}", icon=":material/check:")
        advance(4)


def step_landmark_review() -> None:
    st.header(STEPS[4])
    args = st.session_state.args

    if st.session_state.landmark_review_df is None:
        try:
            st.session_state.landmark_review_df = build_landmark_review_df(args.dataset)
        except FileNotFoundError as exc:
            st.error(str(exc))
            return

    full_df = st.session_state.landmark_review_df
    counts = full_df["reviewed_status"].value_counts()
    st.caption(" | ".join(f"{s}: {counts.get(s, 0)}" for s in STATUSES))

    thumb_width = _thumbnail_width()
    view = _filtered_view(full_df, "landmark")

    landmarks_by_photo = load_numbered_landmarks_by_photo_id(args.dataset)

    def _overlay_thumbnail(photo_id: str) -> str:
        specimen = landmarks_by_photo.get(photo_id)
        if specimen is None:
            return ""
        image_path = resolve_path(specimen.image_path)
        image = cv2.imread(str(image_path)) if image_path.exists() else None
        if image is None:
            return ""
        return array_to_data_url(draw_landmarks(image, specimen.landmarks), max_width=thumb_width)

    view = view.assign(thumbnail=[_overlay_thumbnail(photo_id) for photo_id in view["photo_id"]])
    display_cols = [
        "thumbnail", "photo_id", "inv_id", "auto_status", "reviewed_status",
        "registration_cost", "n_outlier_landmarks", "error_reason",
    ]
    edited = st.data_editor(
        view[display_cols], key="landmark_editor", hide_index=True, row_height=thumb_width//2,
        column_config={
            "thumbnail": st.column_config.ImageColumn("Landmarks", width=thumb_width),
            "reviewed_status": st.column_config.SelectboxColumn(options=STATUSES, required=True),
            "registration_cost": st.column_config.NumberColumn(format="%.4f"),
        },
        disabled=[c for c in display_cols if c != "reviewed_status"],
    )
    full_df.loc[edited.index, "reviewed_status"] = edited["reviewed_status"]
    st.session_state.landmark_review_df = full_df

    if st.button("Save landmark review & continue", icon=":material/arrow_forward:", type="primary"):
        landmarks_reviewed_path, _audit = write_landmarks_review(args.dataset, full_df)
        args.landmarks_status_csv = str(landmarks_reviewed_path)
        n_changed = int((full_df["auto_status"] != full_df["reviewed_status"]).sum())
        st.toast(f"{n_changed} override(s) saved -> {landmarks_reviewed_path}", icon=":material/check:")
        advance(6)


# ---------------------------------------------------------------------------
# Step 6 -- export (terminal: no further step, dataset is ready)
# ---------------------------------------------------------------------------

def step_export() -> None:
    st.header(STEPS[5])
    args = st.session_state.args
    st.caption(f"Dataset: {args.dataset}")

    col_run, col_skip = st.columns(2)
    if col_run.button("Run export", icon=":material/play_arrow:", type="primary", key="run_export"):
        st.session_state.export_dir = run_with_log("Exporting landmarks package...", run_export, args)
    if col_skip.button(
        "Skip -- already done", icon=":material/skip_next:",
        help="If <dataset>/export/ was already written.", key="skip_export",
    ):
        st.session_state.export_dir = resolve_export_dir(args)

    if st.session_state.export_dir:
        st.success(f"Dataset ready -> {args.dataset}")
        st.caption(f"Landmarks package -> {st.session_state.export_dir}")
        st.info(
            "Next: `streamlit run app/train_model.py` to fit a model on this dataset, "
            "or `streamlit run app/predict_dataset.py` to classify it with an existing model."
        )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

if st.session_state.args is not None:
    st.sidebar.caption(f"Dataset: {st.session_state.args.dataset}")

step = st.session_state.step
if step == 1:
    step_setup()
elif step == 2:
    step_run(STEPS[1], "detection & crop", run_detection_and_crop, next_step=3,
             skip_hint="If extraction/<mode>/crops.csv was already written for this dataset.")
elif step == 3:
    step_crop_review()
elif step == 4:
    step_run(STEPS[3], "landmark placement", run_landmark_placement, next_step=5,
             skip_hint="If landmarks/landmarks_numbered.csv was already written for this dataset.")
elif step == 5:
    step_landmark_review()
elif step == 6:
    step_export()
