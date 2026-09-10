"""train_model.py
Streamlit model-building tool -- Tool 2 of 3 (see README.md "Scenario 1").
Two independent sections:

  - "Build a reference shape": interactive counterpart to
    landmarks.build_reference -- GPA consensus of a ground-truth TPS (with
    an optional --drop), written as a plain .tps (see core.tps_io), no CLI
    required.
  - "Train a model": fits a GPA -> PCA -> LDA model (classifiers.train) on
    a dataset root already prepared and exported by app/setup_dataset.py --
    this tool never runs detection/landmark placement itself, it just
    points classifiers.train at an already-exported dataset.

    streamlit run app/train_model.py

Reuses the same dataset-filter argument definitions as the CLI
(utils.cli.add_dataset_args, utils.landmarking_pipeline.dataset_filter_argv)
to build classifiers.train's argv from the widget values, same reasoning
as app/setup_dataset.py (see CONVENTIONS.md "Fonctions core reutilisables").
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import LOGO_PATH, path_picker, plot_reference_shape, run_with_log  # noqa: E402
from classifiers.train import main as train_main  # noqa: E402
from utils.cli import add_dataset_args, add_dataset_positional  # noqa: E402
from utils.landmarking_pipeline import dataset_filter_argv, resolve_export_dir  # noqa: E402
from core.run_io import model_display_name, read_metrics  # noqa: E402
from core.tps_io import ImageLandmarks, parse_tps, write_tps  # noqa: E402
from landmarks.build_reference import build_reference_shape  # noqa: E402

st.set_page_config(
    page_title="idmybee -- train a model",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else None,
    layout="wide",
)

col_logo, col_title = st.columns([1, 8])
if LOGO_PATH.exists():
    col_logo.image(str(LOGO_PATH), width=100)
with col_title:
    st.title("Train a model")
    st.caption("Build a GPA reference shape, and/or fit a GPA-PCA-LDA classifier on a prepared dataset.")


# ---------------------------------------------------------------------------
# Section 1 -- build a reference shape
# ---------------------------------------------------------------------------

def build_reference_section() -> None:
    st.header("Build a reference shape")
    st.caption(
        "GPA consensus of a ground-truth TPS (e.g. Tancrede's blueprint), frozen as a plain "
        ".tps -- see app/setup_dataset.py's 'Pipeline parameters' to pick one for landmarking."
    )

    with st.form("build_reference_form", border=True):
        ref_tps = path_picker("Ground-truth TPS (ordered landmarks)", mode="file", key="br_ref_tps")
        col_a, col_b = st.columns(2)
        n_landmarks = col_a.number_input(
            "Expected landmark count (0 = most frequent found)", min_value=0, value=0, step=1, key="br_n_landmarks",
        )
        drop = col_b.number_input(
            "Landmark index to drop (optional, e.g. absent from the detector's scheme)",
            min_value=-1, value=-1, step=1, key="br_drop",
            help="-1 = keep every landmark.",
        )
        out_path = path_picker(
            "Output .tps path", mode="file", key="br_out_path",
            help="e.g. references/shapes/<name>.tps",
        )
        submitted = st.form_submit_button("Build reference shape", icon=":material/build:", type="primary")

    if submitted:
        if not ref_tps or not out_path:
            st.error("A ground-truth TPS and an output path are both required.")
            return

        specimens, errors = parse_tps(ref_tps, strict=False)
        if not specimens:
            st.error(f"No valid specimen in {ref_tps} ({len(errors)} parsing error(s)).")
            return

        expected_lm = int(n_landmarks) if n_landmarks else int(np.bincount([s.n_points for s in specimens]).argmax())
        try:
            consensus = build_reference_shape(specimens, expected_lm)
        except SystemExit as exc:
            st.error(str(exc))
            return

        drop_idx = int(drop) if drop >= 0 else None
        zones = np.delete(consensus, drop_idx, axis=0) if drop_idx is not None else consensus

        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        write_tps(out, [ImageLandmarks(n_points=len(zones), landmarks=zones, image_path="", tps_id=1)])

        n_used = sum(1 for s in specimens if s.n_points == expected_lm)
        st.success(f"Reference shape built from {n_used} specimen(s) -> {out}")
        st.caption(f"{len(zones)} landmark(s)" + (f" (LM{drop_idx} dropped)" if drop_idx is not None else ""))
        st.pyplot(plot_reference_shape(zones))


# ---------------------------------------------------------------------------
# Section 2 -- train a model
# ---------------------------------------------------------------------------

def _dataset_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_dataset_positional(parser)
    add_dataset_args(parser)
    return parser


def train_section() -> None:
    st.header("Train a model")

    dataset_root = path_picker(
        "Dataset root (already prepared/exported by app/setup_dataset.py)",
        mode="dir", key="tr_dataset_root",
    )
    dataset_ready = bool(dataset_root) and (Path(dataset_root) / "manifest.csv").exists()
    if dataset_root and not dataset_ready:
        st.warning("manifest.csv not found at this path -- prepare it with app/setup_dataset.py first.")
    if dataset_root:
        export_dir = resolve_export_dir(argparse.Namespace(dataset=Path(dataset_root), export_dir=None))
        if not export_dir.exists():
            st.warning(f"{export_dir} not found -- run app/setup_dataset.py's export step on this dataset first.")

    if not dataset_ready:
        return

    with st.form("train_form", border=True):
        col_a, col_b = st.columns(2)
        devices = col_a.text_input("Devices (comma-separated, optional)", key="tr_devices")
        species = col_b.text_input("Species (comma-separated, optional)", key="tr_species")
        col_a, col_b = st.columns(2)
        castes = col_a.text_input("Castes (comma-separated, optional)", key="tr_castes")
        run_label = col_b.text_input("Run label (optional)", key="tr_run_label")
        include_outliers = st.checkbox("Include SUSPECT/FAILED photos", key="tr_include_outliers")
        landmarks_tps = path_picker("Landmarks .tps override (optional)", mode="file", key="tr_landmarks_tps")

        level = st.segmented_control(
            "Classify by", options=["species", "caste"], default="species", required=True, key="tr_level",
        )
        lda_components = st.number_input(
            "LDA components kept in the saved model", min_value=1, value=2, step=1, key="tr_lda_components",
        )
        model_name = st.text_input(
            "Model name", placeholder="e.g. Red-rumped bumblebee identifier",
            help="Shown by a model picker (this app or app/single_image.py/app/predict_dataset.py) "
                 "instead of an abstract id, and now also names the output folder under models/lda/ "
                 "(slugified; a name already in use gets an automatic _v2/_v3/... suffix). Defaults "
                 "to level_dataset_label[_devices][_source] if left blank.",
            key="tr_model_name",
        )
        submitted = st.form_submit_button("Build reference model", icon=":material/model_training:", type="primary")

    if submitted:
        parser_ns = _dataset_args_parser().parse_args([
            dataset_root,
            *(["--devices", *[d.strip() for d in devices.split(",") if d.strip()]] if devices else []),
            *(["--species", *[s.strip() for s in species.split(",") if s.strip()]] if species else []),
            *(["--castes", *[c.strip() for c in castes.split(",") if c.strip()]] if castes else []),
            *(["--include-outliers"] if include_outliers else []),
            *(["--tps", landmarks_tps] if landmarks_tps else []),
            *(["--run-label", run_label] if run_label else []),
        ])

        train_argv = [
            str(parser_ns.dataset), "--level", level, "--lda-components", str(int(lda_components)),
            *dataset_filter_argv(parser_ns),
        ]
        if model_name:
            train_argv += ["--model-name", model_name]
        out_dir = run_with_log("Fitting GPA -> PCA -> LDA...", train_main, train_argv)

        metrics = read_metrics(out_dir)
        st.success(f"Model built: {model_display_name(out_dir / 'model.joblib')}")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Top-1 accuracy (LOOCV)", f"{metrics['accuracy_top1']:.1%}")
        col_b.metric("Top-3 accuracy (LOOCV)", f"{metrics['accuracy_top3']:.1%}")
        col_c.metric("Specimens", metrics["n_specimens"])
        st.caption(f"model.joblib -> {out_dir / 'model.joblib'}")


build_reference_section()
st.divider()
train_section()
