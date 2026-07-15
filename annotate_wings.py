#!/usr/bin/env python3
"""
App Streamlit pour annoter les ailes de bourdons avec dessin interactif.

Utilisation:
    pip install streamlit streamlit-drawable-canvas opencv-python pillow numpy
    streamlit run annotate_wings_canvas.py
"""

import streamlit as st
import cv2
import json
import numpy as np
from pathlib import Path
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Configuration Streamlit
st.set_page_config(
    page_title="Annotation Ailes Bourdons",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🐝 Annotation des Ailes de Bourdons")

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def find_image_in_organized(image_name, root_dir="data/images/organized"):
    """Retrouve le chemin complet d'une image dans l'arborescence organized."""
    root_path = Path(root_dir)
    
    if not root_path.exists():
        return None
    
    for file_path in root_path.rglob(f"{image_name}*"):
        if file_path.is_file() and file_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            return str(file_path)
    
    return None


def load_image_cv2(image_path):
    """Charge une image avec OpenCV et retourne un PIL Image."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_img)


def extract_bbox_from_canvas(canvas_data, scale_x=1.0, scale_y=1.0):
    """Extrait les coordonnées de la bounding box du canvas."""
    if canvas_data is None or "objects" not in canvas_data:
        return None
    
    objects = canvas_data["objects"]
    if not objects or len(objects) == 0:
        return None
    
    # Récupérer le dernier rectangle dessiné
    rect = objects[-1]
    
    if rect["type"] != "rect":
        return None
    
    # streamlit-drawable-canvas applique parfois scaleX/scaleY sur l'objet
    # (notamment après un redimensionnement par les poignées) -> il faut les
    # prendre en compte, sinon la largeur/hauteur réelle est fausse.
    obj_scale_x = rect.get("scaleX", 1.0) or 1.0
    obj_scale_y = rect.get("scaleY", 1.0) or 1.0
    
    left = rect["left"]
    top = rect["top"]
    width = rect["width"] * obj_scale_x
    height = rect["height"] * obj_scale_y
    
    x1 = int(left * scale_x)
    y1 = int(top * scale_y)
    x2 = int((left + width) * scale_x)
    y2 = int((top + height) * scale_y)
    
    return [x1, y1, x2, y2]


def clamp(value, lo, hi):
    """Contraint value dans [lo, hi]."""
    return max(lo, min(hi, value))


# ============================================================================
# INITIALISATION SESSION STATE
# ============================================================================

if "annotations" not in st.session_state:
    st.session_state.annotations = {}
if "current_index" not in st.session_state:
    st.session_state.current_index = 0
if "image_list" not in st.session_state:
    st.session_state.image_list = []

# ============================================================================
# SIDEBAR: CHARGEMENT DE LA LISTE
# ============================================================================

with st.sidebar:
    st.header("📋 Configuration")
    
    st.subheader("1. Chemin du répertoire")
    root_dir = st.text_input(
        "Répertoire organized/",
        value="data/images/organized",
        help="Chemin racine contenant species/role/images.jpg"
    )
    
    st.subheader("2. Charger la liste de photos")
    
    upload_method = st.radio(
        "Méthode de chargement:",
        ["Copier-coller", "Télécharger fichier"]
    )
    
    if upload_method == "Copier-coller":
        images_text = st.text_area(
            "Noms de photos (un par ligne)",
            height=150,
            placeholder="photo1.jpg\nphoto2.jpg\nphoto3.jpg"
        )
        if images_text:
            st.session_state.image_list = [
                line.strip() for line in images_text.split("\n") if line.strip()
            ]
    
    else:  # Télécharger fichier
        uploaded_file = st.file_uploader("Fichier texte ou JSON", type=["txt", "json"])
        if uploaded_file:
            content = uploaded_file.read().decode()
            if uploaded_file.name.endswith(".json"):
                data = json.loads(content)
                st.session_state.image_list = [
                    item.get("image", "") if isinstance(item, dict) else item
                    for item in (data if isinstance(data, list) else [])
                ]
            else:
                st.session_state.image_list = [
                    line.strip() for line in content.split("\n") if line.strip()
                ]
    
    if st.session_state.image_list:
        st.success(f"✅ {len(st.session_state.image_list)} image(s) chargée(s)")
    
    st.divider()
    st.subheader("📊 Progression")
    if st.session_state.image_list:
        progress = len([
            img for img in st.session_state.image_list
            if img in st.session_state.annotations and st.session_state.annotations[img]
        ])
        st.metric("Images annotées", f"{progress} / {len(st.session_state.image_list)}")
        if progress > 0:
            st.progress(progress / len(st.session_state.image_list))

# ============================================================================
# ZONE PRINCIPALE: ANNOTATION
# ============================================================================

if not st.session_state.image_list:
    st.info("👈 Charger une liste de photos dans le sidebar pour commencer.")
else:
    # Navigation
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col1:
        if st.button("⬅️ Précédent", width=150):
            st.session_state.current_index = max(0, st.session_state.current_index - 1)
            st.rerun()
    
    with col2:
        st.write(f"**Image {st.session_state.current_index + 1} / {len(st.session_state.image_list)}**")
    
    with col3:
        if st.button("Suivant ➡️", width=150):
            st.session_state.current_index = min(
                len(st.session_state.image_list) - 1,
                st.session_state.current_index + 1
            )
            st.rerun()
    
    # Charger l'image actuelle
    current_image_name = st.session_state.image_list[st.session_state.current_index]
    image_path = find_image_in_organized(current_image_name, root_dir)
    
    if image_path is None:
        st.error(f"❌ Image introuvable: {current_image_name}")
        st.info(f"Recherche dans: {root_dir}")
    else:
        # Charger avec OpenCV
        pil_image = load_image_cv2(image_path)
        
        if pil_image is None:
            st.error(f"❌ Impossible de charger l'image: {image_path}")
        else:
            w, h = pil_image.size
            
            # Info image
            st.subheader("🖼️ Image à annoter")
            st.caption(f"{Path(image_path).name} • {w}×{h}px • {Path(image_path).parent.parent.name}/{Path(image_path).parent.name}")
            
            # ----- Zoom -----
            zoom_pct = st.slider(
                "🔍 Zoom (%)", min_value=25, max_value=300, value=100, step=5,
                key=f"zoom_{current_image_name}_{st.session_state.current_index}",
                help="Change uniquement l'affichage. Les coordonnées enregistrées restent en pixels de l'image originale."
            )
            
            base_display_width = 800 if w > 800 else w
            display_width = max(50, int(base_display_width * (zoom_pct / 100)))
            ratio = display_width / w
            display_height = max(50, int(h * ratio))
            
            pil_image_resized = pil_image.resize((display_width, display_height), Image.Resampling.LANCZOS)
            
            # scale_x/scale_y convertissent les coordonnées canvas -> coordonnées image originale
            scale_x = w / display_width
            scale_y = h / display_height
            
            saved_bbox = st.session_state.annotations.get(current_image_name, None)
            
            st.caption("🖱️ Cliquez-glissez sur l'image pour tracer (ou retracer) la box autour de l'aile.")
            
            canvas_result = st_canvas(
                fill_color="rgba(0, 255, 0, 0.1)",
                stroke_width=2,
                stroke_color="rgb(0, 255, 0)",
                background_image=pil_image_resized,
                height=display_height,
                width=display_width,
                drawing_mode="rect",
                point_display_radius=0,
                update_streamlit=True,
                key=f"canvas_{current_image_name}_{st.session_state.current_index}_{zoom_pct}",
            )
            
            # Récupérer les coordonnées du dernier rectangle tracé sur CE canvas
            bbox_from_canvas = None
            if canvas_result.json_data is not None:
                objects = canvas_result.json_data.get("objects", [])
                if objects:
                    bbox_from_canvas = extract_bbox_from_canvas(
                        {"objects": objects[-1:]}, scale_x, scale_y
                    )
                    if bbox_from_canvas:
                        bbox_from_canvas = [
                            clamp(bbox_from_canvas[0], 0, w),
                            clamp(bbox_from_canvas[1], 0, h),
                            clamp(bbox_from_canvas[2], 0, w),
                            clamp(bbox_from_canvas[3], 0, h),
                        ]
            
            # ===== SECTION ANNOTATION =====
            st.divider()
            st.subheader("✏️ Coordonnées")
            
            if bbox_from_canvas:
                display_x1, display_y1, display_x2, display_y2 = bbox_from_canvas
            elif saved_bbox:
                display_x1, display_y1, display_x2, display_y2 = saved_bbox
            else:
                display_x1, display_y1, display_x2, display_y2 = 0, 0, w, h
            
            display_x1 = clamp(int(display_x1), 0, w)
            display_y1 = clamp(int(display_y1), 0, h)
            display_x2 = clamp(int(display_x2), 0, w)
            display_y2 = clamp(int(display_y2), 0, h)
            
            # Les number_input gardent leur dernière valeur tapée si la clé ne change pas
            # (Streamlit n'applique `value=` qu'à la création du widget). On inclut donc
            # les coordonnées elles-mêmes dans la clé pour forcer le rafraîchissement
            # dès que le canvas produit une nouvelle box.
            coords_sig = f"{display_x1}_{display_y1}_{display_x2}_{display_y2}"
            base_key = f"{current_image_name}_{st.session_state.current_index}_{coords_sig}"
            
            col_x1, col_y1, col_x2, col_y2 = st.columns(4)
            with col_x1:
                x1 = st.number_input("x1", min_value=0, max_value=w, value=display_x1, step=1,
                                      key=f"x1_{base_key}")
            with col_y1:
                y1 = st.number_input("y1", min_value=0, max_value=h, value=display_y1, step=1,
                                      key=f"y1_{base_key}")
            with col_x2:
                x2 = st.number_input("x2", min_value=0, max_value=w, value=display_x2, step=1,
                                      key=f"x2_{base_key}")
            with col_y2:
                y2 = st.number_input("y2", min_value=0, max_value=h, value=display_y2, step=1,
                                      key=f"y2_{base_key}")
            
            # Boutons d'action
            col_btn1, col_btn2, col_btn3 = st.columns(3)
            
            with col_btn1:
                if st.button("✅ Valider box", width=200, type="primary"):
                    st.session_state.annotations[current_image_name] = [x1, y1, x2, y2]
                    st.success("✓ Box validée!")
                    st.rerun()
            
            with col_btn2:
                if st.button("🗑️ Effacer", width=200):
                    if current_image_name in st.session_state.annotations:
                        del st.session_state.annotations[current_image_name]
                    st.rerun()
            
            with col_btn3:
                if saved_bbox:
                    st.info(f"📌 [{int(saved_bbox[0])}, {int(saved_bbox[1])}, {int(saved_bbox[2])}, {int(saved_bbox[3])}]")

# ============================================================================
# EXPORT
# ============================================================================

st.divider()
st.subheader("💾 Export")

export_col1, export_col2 = st.columns(2)

with export_col1:
    if st.button("📥 Générer JSON", width=200):
        if not st.session_state.annotations:
            st.warning("Aucune annotation à exporter.")
        else:
            # Construire le JSON
            export_data = []
            for image_name in st.session_state.image_list:
                if image_name in st.session_state.annotations:
                    image_path = find_image_in_organized(image_name, root_dir)
                    if image_path:
                        bbox = st.session_state.annotations[image_name]
                        export_data.append({
                            "image": image_path,
                            "box": bbox
                        })
            
            # Afficher aperçu
            with st.expander("👁️ Aperçu JSON", expanded=True):
                st.json(export_data[:3] if len(export_data) > 3 else export_data)
                if len(export_data) > 3:
                    st.caption(f"... et {len(export_data) - 3} autres")
            
            # Télécharger
            json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
            st.download_button(
                label="⬇️ Télécharger annotations.json",
                data=json_str,
                file_name="annotations.json",
                mime="application/json",
            )

with export_col2:
    if st.button("🔄 Réinitialiser tout", width=200):
        st.session_state.annotations = {}
        st.session_state.current_index = 0
        st.rerun()

# ============================================================================
# PIED DE PAGE
# ============================================================================

st.divider()
with st.expander("ℹ️ Aide"):
    st.markdown("""
    ### Mode d'emploi
    
    1. **Charger les photos** : Entrez les noms des images dans le sidebar
    2. **Dessiner** : Tracez un rectangle directement sur l'image en cliquant et glissant
    3. **Ajuster** : Les coordonnées se remplissent automatiquement, vous pouvez aussi les modifier manuellement
    4. **Valider** : Cliquez "Valider box"
    5. **Exporter** : Cliquez "Générer JSON" pour télécharger
    
    ### Format de sortie
    ```json
    [
        {
            "image": "data/images/organized/Bombus_sylvarum/worker/photo1.jpg",
            "box": [100, 150, 250, 300]
        }
    ]
    ```
    
    ### Coordonnées
    - **x1, y1** : coin supérieur gauche
    - **x2, y2** : coin inférieur droit
    """)