"""train_model.py
Streamlit model-building tool -- Tool 2 of 3 (see README.md "Scenario 1").

Fits a GPA -> PCA -> LDA model (classifiers.train) on landmarks + biological
data uploaded directly (drag-and-drop TPS + CSV, the export produced by
app/setup_dataset.py/tools.pipeline.export_final_landmarks) -- this tool
never runs detection/landmark placement itself, it just merges the
uploaded files into a throwaway dataset root (see
utils.uploaded_dataset.write_dataset_root) and points classifiers.train at
it.

    streamlit run app/train_model.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import LOGO_PATH, run_with_log, save_uploaded_files  # noqa: E402
from classifiers.train import main as train_main  # noqa: E402
from core.run_io import model_display_name, read_metrics, slugify  # noqa: E402
from utils.uploaded_dataset import (  # noqa: E402
    DatasetMergeError, join_specimens_to_bio, parse_uploaded_bio_csv, parse_uploaded_tps_files,
    write_dataset_root,
)

st.set_page_config(
    page_title="idmybee -- entraîner un modèle",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else None,
    layout="wide",
)

col_logo, col_title = st.columns([1, 8])
if LOGO_PATH.exists():
    col_logo.image(str(LOGO_PATH), width=100)
with col_title:
    st.title("Entraîner un modèle")
    st.caption("Entraîner un classifieur GPA-PCA-LDA sur des landmarks + données biologiques déposés.")


# Plain container, not st.form: st.file_uploader can't be nested inside a
# form together with more than one button anyway, and this stays
# consistent with the rest of the app's containers (see app/_common.py
# path_picker's own note).
with st.container(border=True):
    st.subheader("Jeu de données")
    tps_uploads = st.file_uploader(
        "Fichier(s) TPS de landmarks", type=["tps"], accept_multiple_files=True, key="tr_tps_upload",
        help="landmarks_numbered.tps (ou plusieurs, ex. un par source) -- exporté par "
             "app/setup_dataset.py / tools.pipeline.export_final_landmarks, ou landmarké à la main dans "
             "un autre outil (pas besoin de COMMENT= dans ce cas, voir le champ CSV ci-dessous).",
    )
    csv_uploads = st.file_uploader(
        "Fichier(s) CSV de données biologiques", type=["csv"], accept_multiple_files=True, key="tr_csv_upload",
        help="biological_data.csv -- doit avoir au moins les colonnes inv_id, species, caste. Si le(s) "
             "fichier(s) TPS ci-dessus n'ont pas été exportés par ce code (pas d'inv_id pour la jointure), "
             "ajoutez une colonne 'tps_id' avec une ligne par spécimen, dans le même ordre que le(s) "
             "fichier(s) TPS.",
    )

    st.subheader("Modèle")
    col_a, col_b = st.columns(2)
    level = col_a.segmented_control(
        "Classifier par", options=["species", "caste"], default="species", required=True, key="tr_level",
        format_func=lambda v: {"species": "espèce", "caste": "caste"}.get(v, v),
    )
    lda_components = col_b.number_input(
        "Composantes LDA conservées dans le modèle", min_value=1, value=2, step=1, key="tr_lda_components",
    )
    model_name = st.text_input(
        "Nom du modèle", placeholder="ex. Identifiant bourdon terricole",
        help="Affiché par un sélecteur de modèle (cette appli ou app/single_image.py/"
             "app/predict_dataset.py) plutôt qu'un id abstrait, et nomme le dossier de sortie sous "
             "models/lda/ (slugifié). Un tag '_<n>LM_<level>' (nombre de landmarks du TPS + "
             "espèce/caste) est toujours ajouté, et un nom déjà pris reçoit un suffixe automatique "
             "_v2/_v3/.... Par défaut, un nom dérivé du premier fichier TPS si laissé vide.",
        key="tr_model_name",
    )
    submitted = st.button("Construire le modèle", icon=":material/model_training:", type="primary", key="tr_submit")

if submitted:
    if not tps_uploads or not csv_uploads:
        st.error("Au moins un fichier TPS et un fichier CSV sont requis.")
        st.stop()

    with tempfile.TemporaryDirectory(prefix="idmybee_train_upload_") as tmp:
        tmp_path = Path(tmp)
        tps_paths = save_uploaded_files(tps_uploads, tmp_path / "uploaded_tps")
        csv_paths = save_uploaded_files(csv_uploads, tmp_path / "uploaded_csv")

        try:
            specimens = parse_uploaded_tps_files(tps_paths)
            bio_df = parse_uploaded_bio_csv(csv_paths)
            specimens = join_specimens_to_bio(specimens, bio_df)
        except DatasetMergeError as exc:
            st.error(str(exc))
            st.stop()

        dataset_label = slugify(Path(tps_uploads[0].name).stem)
        dataset_root = write_dataset_root(specimens, bio_df, tmp_path / dataset_label)

        train_argv = [str(dataset_root), "--level", level, "--lda-components", str(int(lda_components))]
        if model_name:
            train_argv += ["--model-name", model_name]
        result = run_with_log("Entraînement GPA -> PCA -> LDA...", train_main, train_argv)

        metrics = read_metrics(result.runs_dir)
        st.success(f"Modèle construit : {model_display_name(result.model_path)}")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Précision top-1 (LOOCV)", f"{metrics['accuracy_top1']:.1%}")
        col_b.metric("Précision top-3 (LOOCV)", f"{metrics['accuracy_top3']:.1%}")
        col_c.metric("Spécimens", metrics["n_specimens"])
        st.caption(f"model.joblib -> {result.model_path}")
        st.caption(f"Enregistrement de performance -> {result.runs_dir}")
