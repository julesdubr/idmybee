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

Raw/messy data (several naming conventions, identity conflicts) stays
CLI-only (tools.ingestion.prepare_dataset/ingest_raw/export_clean_dataset)
-- this app's own "prepare dataset" step covers the common case (an
already reasonably clean per-photo CSV, or an existing manifest.csv), same
scope boundary as train/anova staying CLI-only for classifiers (see
RESUME.md "Detail outil 1").
"""
from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from classifiers.predict import run_batch as predict_run_batch
from classifiers.train import main as train_main
from tools.ingestion import build_manifest
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

LOGO_PATH = Path("logo/logo3fb2_orig.png")
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


def discover(pattern: str) -> list[str]:
    return sorted(str(p) for p in Path(".").glob(pattern))


def restart() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


def advance(step: int) -> None:
    st.session_state.step = step
    st.rerun()


def run_with_log(label: str, fn, *args, **kwargs):
    """Runs a pipeline stage inside a spinner, capturing everything it
    prints (every stage's own main() reports progress via print(), see
    CONVENTIONS.md "Logging") into a collapsed expander instead of a
    terminal this Streamlit process doesn't have. logger.info/warning
    messages (internal status, not user-facing) still go to the terminal
    running `streamlit run`, same as any other script -- only print()
    output is captured here."""
    buffer = io.StringIO()
    with st.spinner(label):
        with contextlib.redirect_stdout(buffer):
            result = fn(*args, **kwargs)
    with st.expander("Run log", icon=":material/terminal:"):
        st.code(buffer.getvalue() or "(nothing printed)", language=None)
    return result


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
        "Dataset source", options=["existing", "build"],
        format_func=lambda m: "Use an existing dataset root" if m == "existing" else "Build a manifest from a per-photo CSV",
        default="existing", required=True, key="source_mode_control",
    )

    if source_mode == "existing":
        dataset_root = st.text_input(
            "Dataset root", value=st.session_state.dataset_root, placeholder="data/Bombus/collection",
            help="Must already contain manifest.csv and biological_data.csv (see tools.ingestion.build_manifest, "
                 "or tools.ingestion.prepare_dataset for raw/messy data -- CLI only).",
            key="dataset_root_input",
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
        with st.form("build_manifest_form", border=True):
            dataset_csv = st.text_input("Per-photo CSV", placeholder="data/clean/collection/dataset.csv", key="bm_dataset_csv")
            path_column = st.text_input("Path column", value="path", key="bm_path_column")
            base_dir = st.text_input("Base dir (optional, resolves relative paths)", value="", key="bm_base_dir")
            default_device_type = st.text_input("Default device type (for rows missing it)", value="S", key="bm_default_device_type")
            output_dir = st.text_input("Output dir", placeholder="data/clean/collection", key="bm_output_dir")
            submitted = st.form_submit_button("Build manifest", icon=":material/build:")
        if submitted:
            if not (dataset_csv and output_dir):
                st.error("Per-photo CSV and output dir are required.")
            else:
                argv = [
                    dataset_csv, "--output-dir", output_dir,
                    "--path-column", path_column, "--default-device-type", default_device_type,
                ]
                if base_dir:
                    argv += ["--base-dir", base_dir]
                run_with_log("Building manifest...", build_manifest.main, argv)
                if (Path(output_dir) / "manifest.csv").exists():
                    st.session_state.dataset_root = output_dir
                    st.success(f"manifest.csv/biological_data.csv written to {output_dir}.")
                else:
                    st.error(
                        "Not every row validated -- see manifest_raw.csv below "
                        "(status/status_reason explains each row)."
                    )
                    raw_path = Path(output_dir) / "manifest_raw.csv"
                    if raw_path.exists():
                        st.dataframe(pd.read_csv(raw_path), hide_index=True)
                    inconsistencies = Path(output_dir) / "reports" / "biological_inconsistencies.csv"
                    if inconsistencies.exists():
                        st.caption("Biological inconsistencies:")
                        st.dataframe(pd.read_csv(inconsistencies), hide_index=True)

    dataset_root = st.session_state.dataset_root
    dataset_ready = bool(dataset_root) and (Path(dataset_root) / "manifest.csv").exists()
    if not dataset_ready:
        return

    st.subheader("Pipeline parameters")
    unet_choices = discover("data/models/unet_landmarks/*/weights.pt")
    with st.form("landmarking_form", border=True):
        mode = st.segmented_control(
            "Detection backend", options=["light", "heavy"], default="light", required=True, key="lf_mode",
        )
        detector_model = st.text_input("Wing detector weights", value=str(DEFAULT_DETECTOR_MODEL), key="lf_detector_model")
        heavy_ref = st.text_input(
            "YOLOE references JSON (--mode heavy only)", value="", key="lf_heavy_ref",
        ) if mode == "heavy" else ""

        if unet_choices:
            unet_model = st.selectbox(
                "UNet landmark weights", unet_choices, format_func=lambda p: Path(p).parent.name, key="lf_unet_choice",
            )
        else:
            unet_model = st.text_input("UNet landmark weights path", key="lf_unet_text")
        n_landmarks = st.number_input("Number of landmarks", min_value=1, value=19, step=1, key="lf_n_landmarks")
        reference = st.text_input(
            "GPA reference shape (blank = default for the landmark count above)",
            value="", placeholder=str(default_gpa_reference(19)), key="lf_reference",
        )

        col_a, col_b = st.columns(2)
        imgsz = col_a.number_input("Detection image size", min_value=64, value=1024, step=32, key="lf_imgsz")
        conf = col_b.slider("Detection confidence threshold", 0.0, 1.0, 0.10, 0.01, key="lf_conf")
        padding = col_a.slider("Crop padding", 0.0, 0.5, 0.10, 0.01, key="lf_padding")
        device = col_b.selectbox("Device", ["cpu", "cuda"], index=0, key="lf_device")
        out_width = col_a.number_input("Crop width", min_value=32, value=512, step=32, key="lf_out_width")
        out_height = col_b.number_input("Crop height", min_value=32, value=256, step=32, key="lf_out_height")

        base_dir = st.text_input(
            "Base dir (optional, resolves manifest.csv's path if relative)", value="", key="lf_base_dir",
        )
        overwrite = st.checkbox("Overwrite crops/landmarks already logged (skip resuming a prior run)", key="lf_overwrite")
        retry_failed = st.checkbox("Retry photos previously logged FAILED", key="lf_retry_failed")

        submitted = st.form_submit_button("Continue to detection & crop", icon=":material/arrow_forward:", type="primary")

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

    view = _filtered_view(full_df, "crop")
    display_cols = ["photo_id", "inv_id", "auto_status", "reviewed_status", "error_reason"]
    edited = st.data_editor(
        view[display_cols], key="crop_editor", hide_index=True,
        column_config={"reviewed_status": st.column_config.SelectboxColumn(options=STATUSES, required=True)},
        disabled=[c for c in display_cols if c != "reviewed_status"],
    )
    full_df.loc[edited.index, "reviewed_status"] = edited["reviewed_status"]
    st.session_state.crop_review_df = full_df

    if len(view):
        preview_id = st.selectbox("Preview", view["photo_id"].tolist(), key="crop_preview_select")
        row = full_df.loc[full_df["photo_id"] == preview_id].iloc[0]
        output_path = row.get("output_path")
        if isinstance(output_path, str) and output_path and Path(output_path).exists():
            image = cv2.imread(output_path)
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=f"{preview_id} -- {row['reviewed_status']}")
        else:
            st.info(f"No crop image to show for {preview_id} (auto_status={row['auto_status']}: {row['error_reason']}).")

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

    view = _filtered_view(full_df, "landmark")
    display_cols = [
        "photo_id", "inv_id", "auto_status", "reviewed_status",
        "registration_cost", "n_outlier_landmarks", "error_reason",
    ]
    edited = st.data_editor(
        view[display_cols], key="landmark_editor", hide_index=True,
        column_config={
            "reviewed_status": st.column_config.SelectboxColumn(options=STATUSES, required=True),
            "registration_cost": st.column_config.NumberColumn(format="%.4f"),
        },
        disabled=[c for c in display_cols if c != "reviewed_status"],
    )
    full_df.loc[edited.index, "reviewed_status"] = edited["reviewed_status"]
    st.session_state.landmark_review_df = full_df

    if len(view):
        preview_id = st.selectbox("Preview", view["photo_id"].tolist(), key="landmark_preview_select")
        row = full_df.loc[full_df["photo_id"] == preview_id].iloc[0]
        landmarks_by_photo = load_numbered_landmarks_by_photo_id(args.dataset)
        specimen = landmarks_by_photo.get(preview_id)
        if specimen is not None:
            image_path = resolve_path(specimen.image_path)
            image = cv2.imread(str(image_path)) if image_path.exists() else None
            if image is not None:
                overlay = draw_landmarks(image, specimen.landmarks)
                caption = f"{preview_id} -- {row['reviewed_status']} (registration cost {row['registration_cost']:.4f})"
                st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption=caption)
            else:
                st.info(f"Crop image not found at {image_path}.")
        else:
            st.info(f"{preview_id} has no numbered landmarks to preview (auto_status={row['auto_status']}: {row['error_reason']}).")

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
