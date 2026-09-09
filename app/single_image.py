"""single_image.py
Streamlit single-image / field mode: one photo in, in memory, no file
written anywhere -- detection -> crop -> UNet landmark placement ->
renumbering (utils.landmarking_pipeline.place_landmarks) -> LDA
classification (classifiers.predict.predict_specimens), displayed as a
ranked species/caste prediction with confidence and an annotated landmark
overlay. See TODO.md Phase 2/3 "Mode terrain / single" and PIPELINE.md
stage 8 ("single" -- "This is the function the future single-image UI tool
will call directly.").

Only the light (YOLO-OBB) detector backend is wired up here -- the heavy
(YOLOE) backend additionally needs a --heavy-ref JSON of reference
embeddings, not exposed in this first version, and isn't what the two
existing production models (data/models/yolon_obb/best.pt) were run with.

    streamlit run app/single_image.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from classifiers.predict import predict_specimens
from core.model_io import load_model as load_lda_model_impl
from core.run_io import model_display_name
from core.tps_io import ImageLandmarks
from extraction.light import detection as light_backend
from landmarks.build_reference import load_reference
from landmarks_trainer.model import load_weights as load_unet_weights
from utils.landmarking_pipeline import DEFAULT_DETECTOR_MODEL, default_gpa_reference, place_landmarks
from utils.tps_overlay import draw_landmarks

LOGO_PATH = Path("app/assets/idmb_logo.png")

st.set_page_config(
    page_title="idmybee -- bees prediction",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else None,
    layout="wide",
)

col_logo, col_title = st.columns([1, 8])
if LOGO_PATH.exists():
    col_logo.image(str(LOGO_PATH), width=100)
with col_title:
    st.title("Identifying bees by their wings")
    st.caption(
        "Geometric morphometric analysis of bee wings for identification using automatic"
        "wing detection and landmarks placement, before running GPA-PCA-LDA."
    )


def draw_obb(image: np.ndarray, box_xy: np.ndarray, color: tuple = (0, 255, 0)) -> np.ndarray:
    """Draws the detected wing's oriented bounding box (4 pixel corners, same
    space as `image`) as a closed quadrilateral on a copy of `image`."""
    annotated = image.copy()
    thickness = max(2, round(max(image.shape[:2]) / 400))
    points = box_xy.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(annotated, [points], isClosed=True, color=color, thickness=thickness)
    return annotated


@st.cache_resource(show_spinner="Loading wing detector...")
def load_detector(model_path: str):
    return light_backend.load_model(argparse.Namespace(model=model_path))


@st.cache_resource(show_spinner="Loading UNet landmark model...")
def load_unet(weights_path: str, device: str):
    return load_unet_weights(weights_path, device=device)


@st.cache_resource(show_spinner="Loading reference shape...")
def load_zones(reference_path: str):
    zones, _meta = load_reference(Path(reference_path))
    return zones


@st.cache_resource(show_spinner="Loading classification model...")
def load_lda_model(model_path: str):
    return load_lda_model_impl(model_path)


def discover(pattern: str) -> list[str]:
    return sorted(str(p) for p in Path(".").glob(pattern))


with st.sidebar:
    st.header("Models")

    lda_choices = discover("data/models/lda/*/train/model.joblib")
    if lda_choices:
        lda_model_path = st.selectbox(
            "Classification model", lda_choices, format_func=model_display_name,
            help="A model.joblib produced by classifiers/train.py or tools/pipeline/train_dataset.py "
                 "-- shown by its --model-name if it was given one, otherwise its run_id.",
        )
    else:
        lda_model_path = st.text_input("Classification model path")

    detector_model_path = st.text_input("Wing detector (YOLO-OBB) weights", value=str(DEFAULT_DETECTOR_MODEL))

    unet_choices = discover("data/models/unet_landmarks/*/weights.pt")
    if unet_choices:
        unet_model_path = st.selectbox(
            "UNet landmark weights", unet_choices, format_func=lambda p: Path(p).parent.name,
        )
    else:
        unet_model_path = st.text_input("UNet weights path")

    device = st.selectbox("Device", ["cpu", "cuda"], index=0)

    st.header("Pipeline parameters")
    n_landmarks = st.number_input("Number of landmarks", min_value=1, value=19, step=1)
    reference_path = st.text_input(
        "GPA reference shape", value=str(default_gpa_reference(n_landmarks)),
        key=f"reference_path_{n_landmarks}",
    )
    imgsz = st.number_input("Detection image size", min_value=64, value=1024, step=32)
    conf = st.slider("Detection confidence threshold", 0.0, 1.0, 0.10, 0.01)
    max_det = st.number_input("Max detections", min_value=1, value=10, step=1)
    padding = st.slider("Crop padding", 0.0, 0.5, 0.10, 0.01)
    out_width = st.number_input("Crop width", min_value=32, value=512, step=32)
    out_height = st.number_input("Crop height", min_value=32, value=256, step=32)

uploaded = st.file_uploader("Wing photo", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    file_bytes = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image_bgr is None:
        st.error("Could not decode this image (unsupported format or corrupted file).")
    else:
        col_original, col_result = st.columns(2)
        original_slot = col_original.empty()
        original_slot.image(
            cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption="Uploaded photo",
        )

        if st.button("Run pipeline", type="primary"):
            if not (lda_model_path and detector_model_path and unet_model_path):
                st.error("A classification model, a detector, and a UNet model are all required.")
            else:
                detector_ctx = load_detector(detector_model_path)
                unet_model = load_unet(unet_model_path, device)
                reference_zones = load_zones(reference_path)
                lda_model = load_lda_model(lda_model_path)

                detector_args = argparse.Namespace(
                    imgsz=int(imgsz), conf=float(conf), device=device, max_det=int(max_det),
                )

                with st.spinner("Running detection -> crop -> landmarks -> renumbering..."):
                    result = place_landmarks(
                        image_bgr,
                        detector_mode="light", detector_ctx=detector_ctx, detector_args=detector_args,
                        padding=float(padding), out_width=int(out_width), out_height=int(out_height),
                        unet_model=unet_model, unet_device=device, n_landmarks=int(n_landmarks),
                        reference_zones=reference_zones,
                    )

                if result.detection_box is not None:
                    original_slot.image(
                        cv2.cvtColor(draw_obb(image_bgr, result.detection_box), cv2.COLOR_BGR2RGB),
                        caption="Detected wing (OBB)",
                    )

                if result.status != "OK":
                    col_result.error(f"Failed at stage '{result.stage}': {result.error_reason}")
                    if result.crop_image is not None:
                        col_result.image(
                            cv2.cvtColor(result.crop_image, cv2.COLOR_BGR2RGB),
                            caption="Normalized crop (before failure)",
                        )
                else:
                    overlay = draw_landmarks(result.crop_image, result.landmarks)
                    col_result.image(
                        cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                        caption="Numbered landmarks (crop space)",
                    )
                    col_result.caption(f"Registration cost: {result.registration_score:.4f}")

                    specimen = ImageLandmarks(
                        n_points=int(n_landmarks), landmarks=result.landmarks,
                        image_path=uploaded.name, tps_id=0,
                    )
                    predictions = predict_specimens(lda_model, [specimen])
                    row = predictions.iloc[0]
                    level = lda_model.level

                    st.subheader(f"Predicted {level}")
                    st.caption(f"Model: {model_display_name(lda_model_path)}")
                    ranked = [
                        (row[f"predicted_{level}"], row["confidence"]),
                        (row["second_choice"], row["second_confidence"]),
                        (row["third_choice"], row["third_confidence"]),
                    ]
                    for rank, (name, confidence) in enumerate(ranked, start=1):
                        if name is None:
                            continue
                        st.metric(f"#{rank}: {name}", f"{confidence:.1%}")

                    procrustes_distance = row["procrustes_distance"]
                    if procrustes_distance > 0.3:  # heuristic threshold, no calibrated value yet
                        st.warning(
                            f"Procrustes distance to the model's reference shape is high "
                            f"({procrustes_distance:.4f}) -- possible atypical shape or landmark "
                            "placement issue; treat this prediction with caution."
                        )
                    else:
                        st.caption(f"Procrustes distance to reference: {procrustes_distance:.4f}")
