"""single_image.py
Streamlit single-image / field mode: one photo in, in memory, no file
written anywhere -- detection -> crop -> UNet landmark placement ->
renumbering (utils.landmarking_pipeline.place_landmarks) -> LDA
classification (classifiers.predict.predict_specimens), displayed as a
ranked species/caste prediction with confidence and an annotated landmark
overlay.

Only the light (YOLO-OBB) detector backend is wired up here -- the heavy
(YOLOE) backend additionally needs a --heavy-ref JSON of reference
embeddings, not exposed in this first version, and isn't what the two
existing production models (models/yolon_obb/best.pt) were run with.

    streamlit run app/single_image.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# app/ itself isn't an installed package -- see app/setup_dataset.py's
# identical sys.path assist for why this is needed for a sibling import.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import plot_reference_shape, unet_crop_size, unet_landmark_count  # noqa: E402
from classifiers.predict import predict_specimens
from core.model_io import load_model as load_lda_model_impl
from core.run_io import model_display_name
from core.tps_io import ImageLandmarks
from extraction.light import detection as light_backend
from landmarks.build_reference import load_reference
from landmarks_trainer.model import load_weights as load_unet_weights
from utils.landmarking_pipeline import DEFAULT_DETECTOR_MODEL, REFERENCE_SHAPES_DIR, place_landmarks
from utils.tps_overlay import draw_landmarks

LOGO_PATH = Path("app/assets/idmb_logo.png")

st.set_page_config(
    page_title="idmybee - identification bourdon",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else None,
    layout="wide",
)

col_logo, col_title = st.columns([1, 8])
if LOGO_PATH.exists():
    col_logo.image(str(LOGO_PATH), width=100)
with col_title:
    st.title("Identifier des bourdons par leurs ailes")
    st.caption(
        "Analyse de morphométrie géométrique des ailes de bourdons pour l'identification, avec "
        "détection automatique de l'aile et placement des landmarks, avant un GPA-PCA-LDA."
    )


def draw_obb(image: np.ndarray, box_xy: np.ndarray, color: tuple = (0, 255, 0)) -> np.ndarray:
    """Draws the detected wing's oriented bounding box (4 pixel corners, same
    space as `image`) as a closed quadrilateral on a copy of `image`."""
    annotated = image.copy()
    thickness = max(2, round(max(image.shape[:2]) / 400))
    points = box_xy.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(annotated, [points], isClosed=True, color=color, thickness=thickness)
    return annotated


@st.cache_resource(show_spinner="Chargement du détecteur d'aile...")
def load_detector(model_path: str):
    return light_backend.load_model(argparse.Namespace(model=model_path))


@st.cache_resource(show_spinner="Chargement du modèle UNet de landmarks...")
def load_unet(weights_path: str, device: str):
    return load_unet_weights(weights_path, device=device)


@st.cache_resource(show_spinner="Chargement de la forme de référence...")
def load_zones(reference_path: str):
    return load_reference(Path(reference_path))


@st.cache_resource(show_spinner="Chargement du modèle de classification...")
def load_lda_model(model_path: str):
    return load_lda_model_impl(model_path)


def discover(pattern: str) -> list[str]:
    return sorted(str(p) for p in Path(".").glob(pattern))


with st.sidebar:
    st.header("Modèles")
    st.caption("Dans l'ordre du pipeline : détection -> landmarks -> classification.")

    yolo_choices = discover("models/yolon_obb/*.pt")
    if yolo_choices:
        detector_model_path = st.selectbox(
            "Détecteur d'aile (YOLO-OBB)", yolo_choices, format_func=lambda p: Path(p).stem,
        )
    else:
        detector_model_path = st.text_input("Poids du détecteur d'aile (YOLO-OBB)", value=str(DEFAULT_DETECTOR_MODEL))

    unet_choices = discover("models/unet_landmarks/*/weights.pt")
    if unet_choices:
        unet_model_path = st.selectbox(
            "UNet (landmarks)", unet_choices, format_func=lambda p: Path(p).parent.name,
        )
    else:
        unet_model_path = st.text_input("Chemin des poids UNet")

    lda_choices = discover("models/lda/*/model.joblib")
    if lda_choices:
        lda_model_path = st.selectbox(
            "Classification (LDA)", lda_choices, format_func=model_display_name,
            help="Un model.joblib produit par classifiers/train.py ou tools/pipeline/train_dataset.py "
                 "-- affiché par son --model-name s'il en a un, sinon par son run_id.",
        )
    else:
        lda_model_path = st.text_input("Chemin du modèle de classification")

    device = st.selectbox("Calcul (CPU/GPU)", ["cpu", "cuda"], index=0)

    st.header("Paramètres du pipeline")

    # n_landmarks/crop_width/crop_height are properties of the selected UNet
    # checkpoint (see its train_config.json), not free parameters -- no
    # widget for them, unlike the detection/crop knobs below (see
    # app/setup_dataset.py's identical derivation for the file-based path).
    unet_n_landmarks = unet_landmark_count(unet_model_path) if unet_model_path else None
    checkpoint_crop_size = unet_crop_size(unet_model_path) if unet_model_path else None
    out_width, out_height = checkpoint_crop_size or (512, 256)
    if unet_n_landmarks:
        n_landmarks = unet_n_landmarks
        st.caption(
            f"Ce checkpoint UNet prédit {n_landmarks} landmark(s) à une résolution d'entrée de "
            f"{out_width}x{out_height} (depuis son train_config.json)."
        )
    else:
        n_landmarks = st.number_input(
            "Nombre de landmarks", min_value=1, value=19, step=1,
            help="Aucun n_landmarks trouvé dans le train_config.json de ce checkpoint -- à définir "
                 "manuellement (19 = patron complet de Tancrède, 18 pour un ancien modèle UNet).",
        )

    reference_choices = discover(f"{REFERENCE_SHAPES_DIR}/*.tps")
    if reference_choices:
        reference_path = st.selectbox(
            "Forme de référence GPA", reference_choices, format_func=lambda p: Path(p).stem,
            help="Un .tps simple, un seul bloc spécimen (voir landmarks.build_reference).",
        )
    else:
        reference_path = st.text_input("Chemin de la forme de référence GPA")

    landmark_mismatch = False
    if reference_path and Path(reference_path).exists():
        zones = load_zones(reference_path)
        with st.expander("Aperçu de la forme de référence"):
            st.caption(f"{len(zones)} landmark(s)")
            st.pyplot(plot_reference_shape(zones))
        if len(zones) != int(n_landmarks):
            landmark_mismatch = True
            st.error(
                f"Nombre de landmarks incohérent : le modèle UNet prédit {int(n_landmarks)} landmark(s), "
                f"mais la forme de référence ({Path(reference_path).name}) en a {len(zones)}."
            )

    if lda_model_path and Path(lda_model_path).exists():
        lda_model_preview = load_lda_model(lda_model_path)
        if lda_model_preview.n_points != int(n_landmarks):
            landmark_mismatch = True
            st.error(
                f"Nombre de landmarks incohérent : le modèle UNet prédit {int(n_landmarks)} landmark(s), "
                f"mais le modèle de classification ({model_display_name(lda_model_path)}) en attend "
                f"{lda_model_preview.n_points}."
            )

    imgsz = st.number_input("Taille d'image pour la détection", min_value=64, value=1024, step=32)
    conf = st.slider("Seuil de confiance de détection", 0.0, 1.0, 0.10, 0.01)
    max_det = st.number_input("Détections maximum", min_value=1, value=10, step=1)
    padding = st.slider("Marge du recadrage", 0.0, 0.5, 0.10, 0.01)

uploaded = st.file_uploader("Photo d'aile", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    file_bytes = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image_bgr is None:
        st.error("Impossible de décoder cette image (format non supporté ou fichier corrompu).")
    else:
        col_original, col_result = st.columns(2)
        original_slot = col_original.empty()
        original_slot.image(
            cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption="Photo importée",
        )

        if st.button("Lancer le pipeline", type="primary"):
            if not (lda_model_path and detector_model_path and unet_model_path):
                st.error("Un modèle de classification, un détecteur et un modèle UNet sont tous requis.")
            elif landmark_mismatch:
                st.error(
                    "Nombre de landmarks incohérent entre le modèle UNet, la forme de référence et/ou le "
                    "modèle de classification -- voir le détail dans la barre latérale."
                )
            else:
                detector_ctx = load_detector(detector_model_path)
                unet_model = load_unet(unet_model_path, device)
                reference_zones = load_zones(reference_path)
                lda_model = load_lda_model(lda_model_path)

                detector_args = argparse.Namespace(
                    imgsz=int(imgsz), conf=float(conf), device=device, max_det=int(max_det),
                )

                with st.spinner("Détection -> recadrage -> landmarks -> renumérotation en cours..."):
                    result = place_landmarks(
                        image_bgr,
                        detector_mode="light", detector_ctx=detector_ctx, detector_args=detector_args,
                        padding=float(padding), out_width=int(out_width), out_height=int(out_height),
                        unet_model=unet_model, unet_device=device, n_landmarks=int(n_landmarks),
                        reference_zones=reference_zones,
                    )

                if result.detection_box is not None:
                    annotated_original = draw_obb(image_bgr, result.detection_box)
                    caption = "Aile détectée (OBB)"
                    if result.status == "OK" and result.original_landmarks is not None:
                        annotated_original = draw_landmarks(annotated_original, result.original_landmarks)
                        caption = "Aile détectée (OBB) + landmarks (image originale)"
                    original_slot.image(
                        cv2.cvtColor(annotated_original, cv2.COLOR_BGR2RGB), caption=caption,
                    )

                if result.status != "OK":
                    col_result.error(f"Échec à l'étape « {result.stage} » : {result.error_reason}")
                    if result.crop_image is not None:
                        col_result.image(
                            cv2.cvtColor(result.crop_image, cv2.COLOR_BGR2RGB),
                            caption="Recadrage normalisé (avant l'échec)",
                        )
                else:
                    overlay = draw_landmarks(result.crop_image, result.landmarks)
                    col_result.image(
                        cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                        caption="Landmarks numérotés (espace du recadrage)",
                    )
                    col_result.caption(f"Coût d'enregistrement : {result.registration_score:.4f}")

                    specimen = ImageLandmarks(
                        n_points=int(n_landmarks), landmarks=result.landmarks,
                        image_path=uploaded.name, tps_id=0,
                    )
                    predictions = predict_specimens(lda_model, [specimen])
                    row = predictions.iloc[0]
                    level = lda_model.level
                    level_label = {"species": "espèce", "caste": "caste"}.get(level, level)

                    st.subheader(f"{level_label.capitalize()} prédite")
                    st.caption(f"Modèle : {model_display_name(lda_model_path)}")
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
                            f"La distance de Procrustes à la forme de référence du modèle est élevée "
                            f"({procrustes_distance:.4f}) -- forme possiblement atypique ou problème de "
                            "placement des landmarks ; à considérer avec précaution."
                        )
                    else:
                        st.caption(f"Distance de Procrustes à la référence : {procrustes_distance:.4f}")
