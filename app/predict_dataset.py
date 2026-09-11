"""predict_dataset.py
Streamlit prediction tool -- Tool 3 of 3 (see README.md "Scenario 1").
Classifies landmarks + biological data uploaded directly (drag-and-drop
TPS + CSV, same export format as app/train_model.py) with an existing
model (see app/train_model.py) -- this tool never runs detection/landmark
placement itself, it just merges the uploaded files into a throwaway
dataset root (see utils.uploaded_dataset.write_dataset_root) and points
classifiers.predict at it.

    streamlit run app/predict_dataset.py
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import LOGO_PATH, discover, path_picker, run_with_log, save_uploaded_files  # noqa: E402
from classifiers.predict import run_batch as predict_run_batch  # noqa: E402
from core.dataset_config import resolve_dataset_name  # noqa: E402
from core.model_io import load_model  # noqa: E402
from core.run_io import (  # noqa: E402
    RUNS_ROOT, build_eval_tag, model_display_name, read_metrics, result_path, run_id_from_model_path, slugify,
)
from utils.uploaded_dataset import (  # noqa: E402
    DatasetMergeError, join_specimens_to_bio, parse_uploaded_bio_csv, parse_uploaded_tps_files,
    write_dataset_root,
)

st.set_page_config(
    page_title="idmybee -- prédire",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else None,
    layout="wide",
)

col_logo, col_title = st.columns([1, 8])
if LOGO_PATH.exists():
    col_logo.image(str(LOGO_PATH), width=100)
with col_title:
    st.title("Prédire")
    st.caption("Classifier des landmarks + données biologiques déposés avec un modèle GPA-PCA-LDA existant.")


with st.container(border=True):
    st.subheader("Jeu de données")
    tps_uploads = st.file_uploader(
        "Fichier(s) TPS de landmarks", type=["tps"], accept_multiple_files=True, key="pr_tps_upload",
        help="landmarks_numbered.tps (ou plusieurs) -- exporté par app/setup_dataset.py / "
             "tools.pipeline.export_final_landmarks, ou landmarké à la main dans un autre outil (pas "
             "besoin de COMMENT= dans ce cas, voir le champ CSV ci-dessous).",
    )
    csv_uploads = st.file_uploader(
        "Fichier(s) CSV de données biologiques", type=["csv"], accept_multiple_files=True, key="pr_csv_upload",
        help="biological_data.csv -- doit avoir au moins les colonnes inv_id, species, caste. Si le(s) "
             "fichier(s) TPS ci-dessus n'ont pas été exportés par ce code (pas d'inv_id pour la jointure), "
             "ajoutez une colonne 'tps_id' avec une ligne par spécimen, dans le même ordre que le(s) "
             "fichier(s) TPS.",
    )

    st.subheader("Modèle")
    model_choices = discover("models/lda/*/model.joblib")
    if model_choices:
        model_path = st.selectbox(
            "Modèle de classification", model_choices, format_func=model_display_name, key="pr_model_choice",
        )
    else:
        model_path = st.text_input("Chemin du modèle de classification", key="pr_model_text")

    low_confidence_threshold = st.slider(
        "Seuil de confiance faible à revérifier", 0.0, 1.0, 0.6, 0.05, key="pr_low_confidence_threshold",
    )
    save_path = path_picker(
        "Copier aussi predictions.csv vers (optionnel)", mode="save", key="pr_save_path",
        default_filename="predictions.csv",
        help="predictions.csv est toujours écrit sous runs/ -- ne renseigner ce champ que pour en "
             "copier aussi une version ailleurs (ex. pour partager le résultat).",
    )
    submitted = st.button("Classifier le jeu de données", icon=":material/query_stats:", type="primary", key="pr_submit")

if submitted:
    if not tps_uploads or not csv_uploads:
        st.error("Au moins un fichier TPS et un fichier CSV sont requis.")
    elif not model_path:
        st.error("Un modèle de classification est requis.")
    else:
        with tempfile.TemporaryDirectory(prefix="idmybee_predict_upload_") as tmp:
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

            model = load_model(Path(model_path))
            mismatched = [sp for sp in specimens if sp.n_points != model.n_points]
            if mismatched:
                other_counts = sorted({sp.n_points for sp in mismatched})
                if len(mismatched) == len(specimens):
                    st.error(
                        f"Nombre de landmarks incohérent : {model_display_name(model_path)!r} attend "
                        f"{model.n_points} landmark(s), mais aucun des {len(specimens)} spécimen(s) "
                        f"déposés n'en a autant (trouvé {other_counts}). Déposez un TPS avec le même "
                        "schéma de landmarks que celui sur lequel le modèle a été entraîné."
                    )
                    st.stop()
                st.warning(
                    f"{len(mismatched)} spécimen(s) sur {len(specimens)} ont un nombre de landmarks "
                    f"différent du modèle ({model.n_points}, trouvé {other_counts}) et seront ignorés."
                )

            dataset_label = slugify(Path(tps_uploads[0].name).stem)
            dataset_root = write_dataset_root(specimens, bio_df, tmp_path / dataset_label)

            args = argparse.Namespace(
                dataset=dataset_root, model_path=Path(model_path),
                devices=None, species=None, castes=None, exclude_outliers=True, non_strict=False,
                landmarks_tps=None, landmarks_status_csv=None, run_label=None,
                low_confidence_threshold=low_confidence_threshold,
            )
            run_with_log("Classification en cours...", predict_run_batch, args)

            family, run_id = run_id_from_model_path(args.model_path)
            eval_tag = build_eval_tag(
                resolve_dataset_name(dataset_root), args.devices, args.landmarks_tps, args.run_label,
            )
            out_dir = result_path(family, run_id, "predict", eval_tag, root=RUNS_ROOT)
            metrics = read_metrics(out_dir)
            predictions = pd.read_csv(out_dir / "predictions.csv")

            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(out_dir / "predictions.csv", save_path)

            st.success(f"Classifié avec {model_display_name(args.model_path)}")
            if "accuracy_top1" in metrics:
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Précision top-1", f"{metrics['accuracy_top1']:.1%}")
                col_b.metric("Précision top-3", f"{metrics['accuracy_top3']:.1%}")
                col_c.metric("Spécimens", metrics["n"])
            else:
                st.caption(f"{len(predictions)} spécimen(s) classifié(s) (pas de vérité connue pour scorer).")
            st.dataframe(predictions, hide_index=True)
            st.caption(f"predictions.csv -> {out_dir / 'predictions.csv'}")
            if save_path:
                st.caption(f"Copié aussi vers -> {save_path}")
