"""predict_dataset.py
Streamlit prediction tool -- Tool 3 of 3 (see README.md "Scenario 1").
Classifies a dataset root already prepared and exported by
app/setup_dataset.py with an existing model (see app/train_model.py) --
this tool never runs detection/landmark placement itself, it just points
classifiers.predict at an already-exported dataset.

    streamlit run app/predict_dataset.py

Reuses the same dataset-filter argument definitions as the CLI
(utils.cli.add_dataset_args, utils.landmarking_pipeline.dataset_filter_argv)
to build classifiers.predict's argv from the widget values, same reasoning
as app/setup_dataset.py/app/train_model.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import LOGO_PATH, discover, path_picker, run_with_log  # noqa: E402
from classifiers.predict import run_batch as predict_run_batch  # noqa: E402
from utils.cli import add_dataset_args, add_dataset_positional  # noqa: E402
from utils.landmarking_pipeline import resolve_export_dir  # noqa: E402
from core.run_io import (  # noqa: E402
    build_eval_tag, model_display_name, read_metrics, result_path, run_id_from_model_path,
)

st.set_page_config(
    page_title="idmybee -- predict",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else None,
    layout="wide",
)

col_logo, col_title = st.columns([1, 8])
if LOGO_PATH.exists():
    col_logo.image(str(LOGO_PATH), width=100)
with col_title:
    st.title("Predict")
    st.caption("Classify a prepared dataset with an existing GPA-PCA-LDA model.")


def _dataset_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_dataset_positional(parser)
    add_dataset_args(parser)
    return parser


dataset_root = path_picker(
    "Dataset root (already prepared/exported by app/setup_dataset.py)",
    mode="dir", key="pr_dataset_root",
)
dataset_ready = bool(dataset_root) and (Path(dataset_root) / "manifest.csv").exists()
if dataset_root and not dataset_ready:
    st.warning("manifest.csv not found at this path -- prepare it with app/setup_dataset.py first.")
if dataset_root:
    export_dir = resolve_export_dir(argparse.Namespace(dataset=Path(dataset_root), export_dir=None))
    if not export_dir.exists():
        st.warning(f"{export_dir} not found -- run app/setup_dataset.py's export step on this dataset first.")

if dataset_ready:
    model_choices = discover("models/lda/*/train/model.joblib")
    with st.form("predict_form", border=True):
        if model_choices:
            model_path = st.selectbox(
                "Classification model", model_choices, format_func=model_display_name, key="pr_model_choice",
            )
        else:
            model_path = st.text_input("Classification model path", key="pr_model_text")

        col_a, col_b = st.columns(2)
        devices = col_a.text_input("Devices (comma-separated, optional)", key="pr_devices")
        species = col_b.text_input("Species (comma-separated, optional)", key="pr_species")
        col_a, col_b = st.columns(2)
        castes = col_a.text_input("Castes (comma-separated, optional)", key="pr_castes")
        run_label = col_b.text_input("Run label (optional)", key="pr_run_label")
        include_outliers = st.checkbox("Include SUSPECT/FAILED photos", key="pr_include_outliers")
        landmarks_tps = path_picker("Landmarks .tps override (optional)", mode="file", key="pr_landmarks_tps")

        low_confidence_threshold = st.slider(
            "Low-confidence review threshold", 0.0, 1.0, 0.6, 0.05, key="pr_low_confidence_threshold",
        )
        submitted = st.form_submit_button("Classify dataset", icon=":material/query_stats:", type="primary")

    if submitted:
        if not model_path:
            st.error("A classification model is required.")
        else:
            args = _dataset_args_parser().parse_args([
                dataset_root,
                *(["--devices", *[d.strip() for d in devices.split(",") if d.strip()]] if devices else []),
                *(["--species", *[s.strip() for s in species.split(",") if s.strip()]] if species else []),
                *(["--castes", *[c.strip() for c in castes.split(",") if c.strip()]] if castes else []),
                *(["--include-outliers"] if include_outliers else []),
                *(["--tps", landmarks_tps] if landmarks_tps else []),
                *(["--run-label", run_label] if run_label else []),
            ])
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
