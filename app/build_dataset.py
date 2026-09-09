"""build_dataset.py
Streamlit dataset-builder / reference-model workflow -- Scenario 1 (see
README.md "Scenario 1 -- prepare a dataset, build a reference model, and
make predictions on it"). Independent of app/single_image.py (the
field/single-photo tool): this one drives the DATASET-level orchestrators
(utils.landmarking_pipeline, classifiers.train/predict) a human can pause
mid-run to check and correct the automatic OK/SUSPECT/FAILED statuses (see
utils.review) before they feed a model, then either fit a new reference
model or classify the dataset with an existing one -- see the "goal" choice
in step 1.

    streamlit run app/build_dataset.py

Reuses the exact same argument definitions as the CLI (utils.cli.
add_dataset_args/add_landmarking_args, utils.landmarking_pipeline.
dataset_filter_argv) to build the shared `args` Namespace from the widget
values in step 1, rather than hand-building one field by field -- one
parsing/validation/defaults implementation for both the CLI and this UI
(see CONVENTIONS.md "Fonctions core reutilisables"). Every pipeline stage
still runs through its own main(argv)/run_batch(args), in-process, exactly
as tools.pipeline.train_dataset/predict_dataset do -- this app only adds
the two validation checkpoints and the step-by-step pacing around them.

Preparing a dataset (raw images + a per-photo CSV, several sources, ...)
goes through tools.ingestion.prepare_dataset.run() -- the exact same
config-driven orchestrator the CLI uses (see that module's docstring for
the config shape), just built from the step 1 wizard's widgets instead of
a hand-written JSON file.
"""
from __future__ import annotations

import argparse
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
from _common import LOGO_PATH, discover, image_to_data_url, array_to_data_url, path_picker, run_with_log  # noqa: E402
from classifiers.predict import run_batch as predict_run_batch
from classifiers.train import main as train_main
from tools.ingestion import prepare_dataset
from tools.ingestion.ingest_raw import NAMING_PARSERS
from utils.cli import add_dataset_args, add_dataset_positional, add_logging_args
from utils.landmarking_pipeline import (
    DEFAULT_DETECTOR_MODEL,
    add_landmarking_args,
    dataset_filter_argv,
    default_gpa_reference,
    run_detection_and_crop,
    run_export,
    run_landmark_placement,
)
from utils.review import (
    build_crop_review_df,
    build_landmark_review_df,
    load_numbered_landmarks_by_photo_id,
    write_crop_review,
    write_landmarks_review,
)
from utils.tps_overlay import draw_landmarks
from core.pipeline_io import resolve_path
from core.run_io import (
    FAMILY_LDA,
    build_eval_tag,
    build_run_id,
    model_display_name,
    read_metrics,
    result_path,
    run_id_from_model_path,
)

STATUSES = ["OK", "SUSPECT", "FAILED"]
STEPS = [
    "Setup", "Detection & crop", "Crop review",
    "Landmark placement", "Landmark review", "Export", "Build model / predict",
]

st.set_page_config(
    page_title="idmybee -- build a reference model",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else None,
    layout="wide",
)


# ---------------------------------------------------------------------------
# Session state / small shared helpers
# ---------------------------------------------------------------------------

st.session_state.setdefault("step", 1)
st.session_state.setdefault("goal", "train")
st.session_state.setdefault("dataset_root", "")
st.session_state.setdefault("args", None)
st.session_state.setdefault("crop_review_df", None)
st.session_state.setdefault("landmark_review_df", None)
st.session_state.setdefault("prep_sources", [{"id": 0}])
st.session_state.setdefault("prep_next_id", 1)
st.session_state.setdefault("prep_raw_roots", [{"id": 0}])
st.session_state.setdefault("prep_raw_roots_next_id", 1)


def restart() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


def advance(step: int) -> None:
    st.session_state.step = step
    st.rerun()


def landmarking_only_parser() -> argparse.ArgumentParser:
    """dataset + landmarking + dataset-filter flags only -- goal-specific
    flags (--level/--model-name for training, --model/--low-confidence-
    threshold for predicting) are added directly onto `args` in step 7,
    once the goal is known, rather than parsed here."""
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
        st.title("Build a reference model")
        st.caption("Prepare a dataset, validate it, then fit or apply a GPA-PCA-LDA model.")
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
# Step 1 -- setup: goal, dataset source, landmarking parameters
# ---------------------------------------------------------------------------

def _source_to_config(source: dict) -> dict:
    return {k: v for k, v in source.items() if k != "id" and v not in (None, "", False)}


def _render_source(index: int, source: dict, n_sources: int) -> None:
    sid = source["id"]
    with st.container(border=True):
        col_type, col_remove = st.columns([5, 1])
        source["type"] = col_type.segmented_control(
            f"Source {index + 1}", options=["compliant", "raw"],
            format_func=lambda t: "Compliant per-photo CSV" if t == "compliant" else "Raw (needs identity resolution)",
            default=source.get("type", "compliant"), required=True, key=f"src_{sid}_type",
        )
        if n_sources > 1 and col_remove.button(":material/delete:", key=f"src_{sid}_remove", help="Remove this source"):
            st.session_state.prep_sources.remove(source)
            st.rerun()

        if source["type"] == "compliant":
            source["dataset_csv"] = path_picker(
                "Per-photo CSV", mode="file", key=f"src_{sid}_dataset_csv",
                help="One row per photo -- see 'Expected format' above.",
            )
            col_a, col_b = st.columns(2)
            source["path_column"] = col_a.text_input("Path column", value="path", key=f"src_{sid}_path_column")
            source["default_device_type"] = col_b.text_input("Default device type", value="S", key=f"src_{sid}_ddt")
            source["base_dir"] = path_picker("Base dir (optional, resolves relative paths)", mode="dir", key=f"src_{sid}_base_dir")
            source["manifest_output_dir"] = path_picker("Manifest output dir", mode="dir", key=f"src_{sid}_manifest_out")
        else:
            col_a, col_b = st.columns(2)
            source["source_type"] = col_a.text_input("Source type label", key=f"src_{sid}_source_type", placeholder="collection")
            source["key_column"] = col_b.text_input(
                "Key column", key=f"src_{sid}_key_column", placeholder="inv_id",
                help="Identification CSV column matching the raw manifest's original_id.",
            )
            source["identification_csv"] = path_picker("Identification CSV", mode="file", key=f"src_{sid}_id_csv")
            source["raw_manifest"] = path_picker(
                "Raw manifest.csv (optional -- else derived from raw ingestion below)",
                mode="file", key=f"src_{sid}_raw_manifest",
            )
            source["clean_output_dir"] = path_picker("Clean export dir", mode="dir", key=f"src_{sid}_clean_out")
            source["manifest_output_dir"] = path_picker("Manifest output dir", mode="dir", key=f"src_{sid}_manifest_out")
            with st.expander("Advanced identity-resolution options"):
                source["device_column"] = st.text_input("Device column (optional)", key=f"src_{sid}_device_column")
                source["device_name_column"] = st.text_input(
                    "Device name column (optional, default: device)", key=f"src_{sid}_device_name_column",
                )
                source["compare_columns"] = st.text_input(
                    "Compare columns (comma-separated, optional, default: genus,species,caste,dd,mm,yyyy)",
                    key=f"src_{sid}_compare_columns",
                )
                source["origin_codes"] = path_picker("Origin codes CSV (optional)", mode="file", key=f"src_{sid}_origin_codes")
                source["origin_column"] = st.text_input(
                    "Origin column (optional, default: collection_origin)", key=f"src_{sid}_origin_column",
                )
                source["image_group_by"] = st.text_input(
                    "Group copied images by columns (comma-separated, optional)", key=f"src_{sid}_image_group_by",
                )
                source["default_device_type"] = st.text_input("Default device type", value="S", key=f"src_{sid}_ddt_raw")
                source["no_copy_images"] = st.checkbox("Don't copy image files (CSVs/reports only)", key=f"src_{sid}_no_copy")


def _render_raw_root(index: int, root: dict, n_roots: int) -> None:
    rid = root["id"]
    with st.container(border=True):
        col_path, col_remove = st.columns([5, 1])
        root["path"] = col_path.text_input(
            f"Root {index + 1} -- subfolder under the base dir", value=root.get("path", ""),
            key=f"root_{rid}_path", placeholder="IDMB/images/Bombus/collection",
        )
        if n_roots > 1 and col_remove.button(":material/delete:", key=f"root_{rid}_remove", help="Remove this root"):
            st.session_state.prep_raw_roots.remove(root)
            st.rerun()
        col_a, col_b = st.columns(2)
        root["source_type"] = col_a.text_input(
            "Source type", value=root.get("source_type", ""), key=f"root_{rid}_source_type", placeholder="collection",
        )
        root["photographer_subfolder"] = col_b.checkbox(
            "First-level subfolder is the photographer (terrain style)",
            value=root.get("photographer_subfolder", False), key=f"root_{rid}_photog",
        )
        naming = st.selectbox(
            "Filename convention (blank = inferred from the checkbox above)",
            [""] + list(NAMING_PARSERS), index=0, key=f"root_{rid}_naming",
        )
        if naming:
            root["naming"] = naming
        else:
            root.pop("naming", None)


def _prepare_dataset_wizard() -> None:
    with st.expander("Expected format", icon=":material/info:"):
        st.markdown(
            "- **Compliant per-photo CSV** (what `tools.ingestion.build_manifest` expects): one row "
            "per photo. Columns `inv_id`, `species`, `caste` are mandatory, plus a path column "
            "(default `path`) pointing at each photo's image file. Everything else becomes "
            "biological data; `photo_id`/`device_type`/`device`/`photo_index`/`photographer`/"
            "`source_type` are optional photo-level columns, auto-derived if left out.\n"
            "- **Raw source** (several filename conventions, the same raw label reused across "
            "specimens, ...): point at a raw identification CSV and the column that matches each "
            "photo's parsed original id -- identity conflicts get resolved into a compliant CSV "
            "first, automatically.\n"
            "- Add more than one source (e.g. collection + terrain) to combine them into one "
            "dataset root."
        )

    output_dir = path_picker(
        "Output dir (final combined manifest.csv/biological_data.csv)",
        mode="dir", key="prep_output_dir",
    )

    needs_ingest = st.checkbox(
        "Source(s) start as raw image folders (need raw ingestion first)", key="prep_needs_ingest",
    )
    ingest_cfg = None
    if needs_ingest:
        with st.container(border=True):
            st.caption("Raw ingestion")
            base_dir = path_picker(
                "Raw images base dir", mode="dir", key="prep_raw_base_dir",
                help="Local folder (or mounted drive) every root below is a subfolder of.",
            )
            raw_roots: list[dict] = st.session_state.prep_raw_roots
            for i, root in enumerate(raw_roots):
                _render_raw_root(i, root, len(raw_roots))
            if st.button(":material/add: Add root", key="prep_add_root"):
                raw_roots.append({"id": st.session_state.prep_raw_roots_next_id})
                st.session_state.prep_raw_roots_next_id += 1
                st.rerun()

            col_a, col_b = st.columns(2)
            ingest_name = col_a.text_input("Ingest name", key="prep_ingest_name", placeholder="bombus_raw")
            ingest_out_dir = col_b.text_input("Ingest out dir", value="data", key="prep_ingest_out_dir")
            ingest_cfg = {
                "roots": {
                    "base_root": {sys.platform: base_dir},
                    "roots": [{k: v for k, v in r.items() if k != "id"} for r in raw_roots],
                },
                "name": ingest_name, "out_dir": ingest_out_dir,
            }

    sources: list[dict] = st.session_state.prep_sources
    needs_mapping = any(s.get("type") == "raw" for s in sources)
    mapping_file = ""
    if needs_mapping:
        mapping_file = path_picker(
            "Identity mapping file (frozen original_id -> inv_id, shared across raw sources)",
            mode="file", key="prep_mapping_file",
        )

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
        if needs_ingest:
            if not (ingest_cfg["roots"]["base_root"][sys.platform] and ingest_cfg["name"]):
                errors.append("Raw ingestion needs a base dir and a name.")
            if any(not r.get("path") or not r.get("source_type") for r in ingest_cfg["roots"]["roots"]):
                errors.append("Raw ingestion: every root needs a path and a source type.")
        if needs_mapping and not mapping_file:
            errors.append("A mapping file is required when any source is raw.")
        for i, source in enumerate(sources):
            if source.get("type") == "raw":
                required = ["identification_csv", "key_column", "source_type", "manifest_output_dir", "clean_output_dir"]
            else:
                required = ["dataset_csv", "manifest_output_dir"]
            if any(not source.get(field) for field in required):
                errors.append(f"Source {i + 1}: {', '.join(required)} are all required.")

        if errors:
            for error in errors:
                st.error(error)
            return

        config: dict = {"output_dir": output_dir, "sources": [_source_to_config(s) for s in sources]}
        if ingest_cfg:
            config["ingest"] = ingest_cfg
        if mapping_file:
            config["mapping_file"] = mapping_file

        result = run_with_log("Preparing dataset...", prepare_dataset.run, config)

        if (Path(output_dir) / "manifest.csv").exists():
            st.session_state.dataset_root = output_dir
            n_photos = len(pd.read_csv(Path(output_dir) / "manifest.csv"))
            st.success(f"Dataset prepared -- {n_photos} photo(s) -> {output_dir}")
        if result["failed_sources"]:
            st.warning(f"{len(result['failed_sources'])} source(s) failed: {result['failed_sources']}")
            for source in sources:
                manifest_output_dir = Path(source.get("manifest_output_dir") or "")
                raw_path = manifest_output_dir / "manifest_raw.csv"
                if raw_path.exists():
                    st.caption(f"{manifest_output_dir}/manifest_raw.csv:")
                    st.dataframe(pd.read_csv(raw_path), hide_index=True)
                inconsistencies = manifest_output_dir / "reports" / "biological_inconsistencies.csv"
                if inconsistencies.exists():
                    st.caption(f"{manifest_output_dir}/reports/biological_inconsistencies.csv:")
                    st.dataframe(pd.read_csv(inconsistencies), hide_index=True)


def step_setup() -> None:
    st.header(STEPS[0])

    st.session_state.goal = st.segmented_control(
        "What do you want to do?",
        options=["train", "predict"],
        format_func=lambda g: "Build a reference model" if g == "train" else "Predict with an existing model",
        default=st.session_state.goal, required=True, key="goal_control",
    )

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
    unet_choices = discover("data/models/unet_landmarks/*/weights.pt")
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
        n_landmarks = st.number_input("Number of landmarks", min_value=1, value=19, step=1, key="lf_n_landmarks")
        reference = path_picker(
            "GPA reference shape (blank = default for the landmark count above)",
            mode="file", key="lf_reference", help=f"Default: {default_gpa_reference(19)}",
        )

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
        argv = [
            dataset_root, "--mode", mode, "--detector-model", detector_model,
            "--unet-model", unet_model, "--n-landmarks", str(int(n_landmarks)),
            "--imgsz", str(int(imgsz)), "--conf", str(conf),
            "--padding", str(padding), "--out-width", str(int(out_width)), "--out-height", str(int(out_height)),
            "--device", device,
        ]
        if heavy_ref:
            argv += ["--heavy-ref", heavy_ref]
        if reference:
            argv += ["--reference", reference]
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
# Step 6 -- export
# ---------------------------------------------------------------------------

def step_export() -> None:
    step_run(STEPS[5], "export", run_export, next_step=7, skip_hint="If <dataset>/export/ was already written.")


# ---------------------------------------------------------------------------
# Step 7 -- build model (train) or classify (predict)
# ---------------------------------------------------------------------------

def step_train() -> None:
    args = st.session_state.args
    with st.form("train_form", border=True):
        model_name = st.text_input(
            "Model name", placeholder="e.g. Red-rumped bumblebee identifier",
            help="Purely descriptive -- shown by a model picker (this app or app/single_image.py) "
                 "instead of the abstract run_id. Defaults to the run_id if left blank.",
            key="tf_model_name",
        )
        level = st.segmented_control(
            "Classify by", options=["species", "caste"], default="species", required=True, key="tf_level",
        )
        lda_components = st.number_input(
            "LDA components kept in the saved model", min_value=1, value=2, step=1, key="tf_lda_components",
        )
        submitted = st.form_submit_button("Build reference model", icon=":material/model_training:", type="primary")

    if submitted:
        train_argv = [
            str(args.dataset), "--level", level, "--lda-components", str(int(lda_components)),
            *dataset_filter_argv(args),
        ]
        if model_name:
            train_argv += ["--model-name", model_name]
        run_with_log("Fitting GPA -> PCA -> LDA...", train_main, train_argv)

        run_id = build_run_id(level, args.dataset.name, args.devices, args.landmarks_tps, args.run_label)
        out_dir = result_path(FAMILY_LDA, run_id, "train")
        metrics = read_metrics(out_dir)

        st.success(f"Model built: {model_display_name(out_dir / 'model.joblib')}")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Top-1 accuracy (LOOCV)", f"{metrics['accuracy_top1']:.1%}")
        col_b.metric("Top-3 accuracy (LOOCV)", f"{metrics['accuracy_top3']:.1%}")
        col_c.metric("Specimens", metrics["n_specimens"])
        st.caption(f"model.joblib -> {out_dir / 'model.joblib'}")


def step_predict() -> None:
    args = st.session_state.args
    model_choices = discover("data/models/lda/*/train/model.joblib")
    with st.form("predict_form", border=True):
        if model_choices:
            model_path = st.selectbox(
                "Classification model", model_choices, format_func=model_display_name, key="pf_model_choice",
            )
        else:
            model_path = st.text_input("Classification model path", key="pf_model_text")
        low_confidence_threshold = st.slider(
            "Low-confidence review threshold", 0.0, 1.0, 0.6, 0.05, key="pf_low_confidence_threshold",
        )
        submitted = st.form_submit_button("Classify dataset", icon=":material/query_stats:", type="primary")

    if submitted:
        if not model_path:
            st.error("A classification model is required.")
            return
        args.model_path = Path(model_path)
        args.low_confidence_threshold = low_confidence_threshold
        run_with_log("Classifying...", predict_run_batch, args)

        family, run_id = run_id_from_model_path(args.model_path)
        eval_tag = build_eval_tag(args.dataset.name, args.devices, args.landmarks_tps, args.run_label)
        out_dir = result_path(family, run_id, "predict", eval_tag)
        metrics = read_metrics(out_dir)
        predictions = pd.read_csv(out_dir / "predictions.csv")

        st.success(f"Classified with {model_display_name(args.model_path)}")
        if "accuracy_top1" in metrics:
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Top-1 accuracy", f"{metrics['accuracy_top1']:.1%}")
            col_b.metric("Top-3 accuracy", f"{metrics['accuracy_top3']:.1%}")
            col_c.metric("Specimens", metrics["n"])
        else:
            st.caption(f"{len(predictions)} specimen(s) classified (no known truth to score against).")
        st.dataframe(predictions, hide_index=True)
        st.caption(f"predictions.csv -> {out_dir / 'predictions.csv'}")


def step_build_or_predict() -> None:
    st.header(STEPS[6])
    if st.session_state.goal == "train":
        step_train()
    else:
        step_predict()


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
elif step == 7:
    step_build_or_predict()
