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
    LOGO_PATH, discover, file_picker, image_to_data_url, array_to_data_url, path_picker, plot_reference_shape,
    run_with_log,
)
from tools.ingestion import prepare_dataset  # noqa: E402
from manifest.origin_table import (  # noqa: E402
    assign_constant_origin, build_origin_table, write_origin_codes,
)
from utils.cli import add_dataset_args, add_dataset_positional, add_logging_args  # noqa: E402
from utils.landmarking_pipeline import (  # noqa: E402
    REFERENCE_SHAPES_DIR,
    add_landmarking_args,
    landmarks_dirname,
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
from core.pipeline_io import format_duration, resolve_path  # noqa: E402
from landmarks.build_reference import load_reference  # noqa: E402

STATUSES = ["OK", "SUSPECT", "FAILED"]
STEPS = ["Configuration", "Détection & recadrage", "Validation du recadrage", "Placement des landmarks",
         "Validation des landmarks", "Export"]

st.set_page_config(
    page_title="idmybee -- préparer un jeu de données",
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
        st.title("Préparer un jeu de données")
        st.caption(
            "Préparer un jeu de données et le valider -- détection, placement des landmarks, "
            "validation humaine, export."
        )
    with st.sidebar:
        st.progress(st.session_state.step / len(STEPS), text=f"Étape {st.session_state.step} sur {len(STEPS)}")
        for i, name in enumerate(STEPS, start=1):
            if i < st.session_state.step:
                st.caption(f":material/check: ~~{name}~~")
            elif i == st.session_state.step:
                st.markdown(f"**:material/arrow_right: {name}**")
            else:
                st.caption(name)
        st.divider()
        st.button("Recommencer", icon=":material/restart_alt:", on_click=restart)


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
    source["origin_codes"] = file_picker(
        "CSV des codes d'origine (optionnel -- remplace le tableau automatique ci-dessous)",
        key=f"src_{sid}_origin_codes", type=["csv"],
    )
    source["origin_column"] = st.text_input(
        "Colonne d'origine dans le CSV d'identification (optionnel, défaut : collection_origin)",
        key=f"src_{sid}_origin_column",
    )
    if source["origin_codes"]:
        return

    source["_auto_origin"] = st.checkbox(
        "Construire les codes inv_name automatiquement pour cette source", key=f"src_{sid}_auto_origin",
        help="Affiche un tableau éditable pré-rempli avec les valeurs d'origine trouvées dans le CSV "
             "d'identification (détection automatique) -- ou, s'il n'a pas de colonne d'origine, un "
             "seul inv_name pour toute la source. Une ligne laissée vide dans le tableau garde sa "
             "valeur d'origine comme inv_name, inchangée.",
    )
    if not source["_auto_origin"]:
        return
    identification_csv = source.get("identification_csv")
    if not identification_csv or not Path(identification_csv).exists():
        st.caption("Choisissez d'abord un CSV d'identification ci-dessus.")
        return

    origin_column = source["origin_column"] or "collection_origin"
    try:
        identification_df = pd.read_csv(identification_csv)
    except (OSError, pd.errors.ParserError) as exc:
        st.error(f"Impossible de lire {identification_csv} : {exc}")
        return

    if origin_column in identification_df.columns:
        table = build_origin_table(identification_df, origin_column)
        if table.empty:
            st.warning(f"Aucune valeur trouvée dans la colonne {origin_column!r}.")
            return
        st.caption(f"inv_name pour chaque valeur de {origin_column!r} trouvée -- laisser une ligne vide pour la garder telle quelle :")
        source["_origin_table"] = st.data_editor(
            table, key=f"src_{sid}_origin_table", hide_index=True, num_rows="fixed",
        )
    else:
        st.caption(f"Aucune colonne {origin_column!r} trouvée dans le CSV d'identification.")
        source["_constant_inv_name"] = st.text_input(
            "inv_name à utiliser pour tous les spécimens de cette source", key=f"src_{sid}_constant_inv_name",
        )


def _render_source(index: int, source: dict, n_sources: int) -> None:
    sid = source["id"]
    with st.container(border=True):
        col_title, col_remove = st.columns([5, 1])
        col_title.markdown(f"**Source {index + 1}**")
        if n_sources > 1 and col_remove.button(":material/delete:", key=f"src_{sid}_remove", help="Supprimer cette source"):
            st.session_state.prep_sources.remove(source)
            st.rerun()

        source["images_dir"] = path_picker(
            "Dossier des photos brutes", mode="dir", key=f"src_{sid}_images_dir",
            help="Scanné pour trouver des photos ; chaque nom de fichier est analysé pour en tirer un "
                 "id original/appareil/numéro de prise.",
        )
        source["identification_csv"] = file_picker(
            "CSV d'identification (données biologiques)", key=f"src_{sid}_id_csv", type=["csv"],
        )
        col_a, col_b = st.columns(2)
        source["source_type"] = col_a.text_input("Type de source", key=f"src_{sid}_source_type", placeholder="collection")
        source["key_column"] = col_b.text_input(
            "Colonne clé", key=f"src_{sid}_key_column", placeholder="inv_id",
            help="Colonne du CSV d'identification correspondant à l'id original analysé de chaque photo.",
        )
        source["output_dir"] = path_picker(
            "Dossier de sortie (dataset.csv + manifest.csv propres à cette source)", mode="dir", key=f"src_{sid}_output_dir",
        )
        with st.expander("Codes d'origine (inv_name)", icon=":material/tag:"):
            _render_origin_codes(sid, source)
        with st.expander("Options avancées"):
            source["photographer_subfolder"] = st.checkbox(
                "Le premier sous-dossier est le photographe (style terrain)", key=f"src_{sid}_photog",
            )
            col_a, col_b = st.columns(2)
            source["device_column"] = col_a.text_input("Colonne appareil photo (optionnel)", key=f"src_{sid}_device_column")
            source["device_name_column"] = col_b.text_input(
                "Colonne nom de l'appareil photo (optionnel, défaut : device)", key=f"src_{sid}_device_name_column",
            )
            source["compare_columns"] = st.text_input(
                "Colonnes à comparer (séparées par des virgules, optionnel, défaut : genus,species,caste,dd,mm,yyyy)",
                key=f"src_{sid}_compare_columns",
            )
            source["image_group_by"] = st.text_input(
                "Grouper les images copiées par colonnes (séparées par des virgules, optionnel)", key=f"src_{sid}_image_group_by",
            )
            source["default_device_type"] = st.text_input("Type d'appareil par défaut", value="S", key=f"src_{sid}_ddt")
            source["no_copy_images"] = st.checkbox("Ne pas copier les fichiers image (CSV/rapports seulement)", key=f"src_{sid}_no_copy")
            source["keep_raw_manifest"] = st.checkbox(
                "Conserver le rapport de scan brut après la préparation (écrit dans le dossier des "
                "photos sous .idmybee_ingest/ -- aussi réutilisé plutôt que rescanné au prochain run)",
                key=f"src_{sid}_keep_raw",
            )


def _resolve_source_for_submit(source: dict) -> dict:
    """Returns a copy of `source` with origin_codes/origin_column/
    identification_csv finalized for tools.ingestion.prepare_dataset --
    writes the auto-built origin table (and, for the no-origin-column
    case, an adjusted identification CSV with a synthetic constant column)
    into this source's own output_dir. Raises ValueError if the user
    opted into auto-building but no table/inv_name was ever rendered
    (identification CSV missing/unreadable, or no origin values found --
    a table with blank rows is fine, see manifest.origin_table.
    resolve_origin_codes). See _render_origin_codes()/manifest.origin_table."""
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
        if table is None or table.empty:
            raise ValueError(f"Source {label!r} : aucun tableau de codes d'origine à préparer.")
        codes_path = output_dir / "collection_origin_codes.csv"
        write_origin_codes(table, codes_path)
        resolved["origin_codes"] = str(codes_path)
        resolved["origin_column"] = origin_column
    else:
        inv_name = (resolved.get("_constant_inv_name") or "").strip()
        if not inv_name:
            raise ValueError(f"Source {label!r} : saisissez un inv_name avant de préparer.")
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
    with st.expander("Format attendu", icon=":material/info:"):
        st.markdown(
            "Chaque source est un dossier de photos brutes (forme du nom de fichier auto-détectée, "
            "aucune convention à choisir -- le même libellé brut peut être réutilisé entre spécimens) "
            "plus un CSV d'identification avec les données biologiques (espèce, caste, ...) et une "
            "colonne correspondant à l'id analysé de chaque photo. Les conflits d'identité sont "
            "résolus automatiquement en un jeu de données propre et canonique -- aucun CSV par photo "
            "à préparer à la main.\n"
            "- Ajoutez plusieurs sources (ex. collection + terrain) pour les combiner en une seule "
            "racine de jeu de données.\n"
            "- inv_id_mapping.csv est écrit dans le dossier de sortie ci-dessous (partagé entre "
            "sources) -- aucun fichier de mapping séparé à indiquer, sauf pour en partager un "
            "délibérément entre plusieurs racines de jeux de données."
        )

    col_a, col_b = st.columns(2)
    with col_a:
        dataset_name = st.text_input(
            "Nom du jeu de données", key="prep_dataset_name", placeholder="Bombus/collection",
            help="Nomme le dossier propre à ce jeu de données -- manifest.csv/biological_data.csv "
                 "finaux atterrissent dans <dossier de sortie>/<nom du jeu de données>.",
        )
    with col_b:
        output_base_dir = path_picker("Dossier de sortie", mode="dir", value="data/", key="prep_output_base_dir")
    output_dir = str(Path(output_base_dir) / dataset_name) if output_base_dir and dataset_name else ""
    mapping_file = file_picker(
        "Fichier de mapping d'identité (optionnel -- défaut : <dossier de sortie>/inv_id_mapping.csv)",
        key="prep_mapping_file", type=["csv"],
    )

    sources: list[dict] = st.session_state.prep_sources
    st.caption("Sources")
    for i, source in enumerate(sources):
        _render_source(i, source, len(sources))

    if st.button(":material/add: Ajouter une source", key="prep_add_source"):
        sources.append({"id": st.session_state.prep_next_id})
        st.session_state.prep_next_id += 1
        st.rerun()

    st.divider()
    if st.button("Préparer le jeu de données", icon=":material/build:", type="primary", key="prep_submit"):
        errors = []
        if not dataset_name:
            errors.append("Le nom du jeu de données est requis.")
        if not output_base_dir:
            errors.append("Le dossier de sortie est requis.")
        required = ["images_dir", "identification_csv", "source_type", "key_column", "output_dir"]
        for i, source in enumerate(sources):
            if any(not source.get(field) for field in required):
                errors.append(f"Source {i + 1} : {', '.join(required)} sont tous requis.")

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

        result = run_with_log("Préparation du jeu de données...", prepare_dataset.run, config)

        if (Path(output_dir) / "manifest.csv").exists():
            st.session_state.dataset_root = output_dir
            n_photos = len(pd.read_csv(Path(output_dir) / "manifest.csv"))
            st.success(f"Jeu de données préparé -- {n_photos} photo(s) -> {output_dir}")
        if result["failed_sources"]:
            st.warning(f"{len(result['failed_sources'])} source(s) en échec : {result['failed_sources']}")
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


def _unet_train_config(weights_path: str) -> dict:
    """Sibling train_config.json for a UNet checkpoint (see
    landmarks_trainer/train.py) -- {} if that file is missing or unreadable
    (e.g. a migrated legacy checkpoint), letting callers fall back to
    manual entry/defaults."""
    config_path = Path(weights_path).parent / "train_config.json"
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _unet_landmark_count(weights_path: str) -> int | None:
    """n_landmarks this UNet checkpoint was trained for -- None if its
    train_config.json doesn't record it, letting the caller fall back to
    manual entry."""
    return _unet_train_config(weights_path).get("n_landmarks")


def _unet_crop_size(weights_path: str) -> tuple[int, int] | None:
    """(crop_width, crop_height) this UNet checkpoint expects as input,
    read from its train_config.json -- None if missing, letting the caller
    fall back to the pipeline's own defaults. Never user-editable in this
    UI: a checkpoint predicts landmarks for the resolution it was trained
    on, not an arbitrary one picked afterwards."""
    config = _unet_train_config(weights_path)
    width, height = config.get("img_width"), config.get("img_height")
    return (int(width), int(height)) if width and height else None


def step_setup() -> None:
    st.header(STEPS[0])

    st.subheader("Jeu de données")
    source_mode = st.segmented_control(
        "Source du jeu de données", options=["existing", "prepare"],
        format_func=lambda m: "Utiliser un jeu de données existant" if m == "existing" else "Préparer un nouveau jeu de données",
        default="existing", required=True, key="source_mode_control",
    )

    if source_mode == "existing":
        dataset_root = path_picker(
            "Racine du jeu de données", mode="dir", value=st.session_state.dataset_root, key="dataset_root",
            help="Doit déjà contenir manifest.csv et biological_data.csv.",
        )
        st.session_state.dataset_root = dataset_root
        if dataset_root:
            root = Path(dataset_root)
            if (root / "manifest.csv").exists() and (root / "biological_data.csv").exists():
                n_photos = len(pd.read_csv(root / "manifest.csv"))
                n_specimens = len(pd.read_csv(root / "biological_data.csv"))
                st.success(f"{n_photos} photo(s), {n_specimens} spécimen(s) trouvé(s) à {dataset_root}.")
            else:
                st.warning("manifest.csv/biological_data.csv introuvables à cet emplacement pour l'instant.")
    else:
        _prepare_dataset_wizard()

    dataset_root = st.session_state.dataset_root
    dataset_ready = bool(dataset_root) and (Path(dataset_root) / "manifest.csv").exists()
    if not dataset_ready:
        return

    st.divider()
    st.subheader("Paramètres du pipeline")
    unet_choices = discover("models/unet_landmarks/*/weights.pt")
    yolo_choices = discover("models/yolon_obb/*.pt")
    with st.container(border=True):
        st.markdown("**Moteur de détection**")
        mode = st.segmented_control(
            "Moteur de détection", options=["light", "heavy"], default="light", required=True, key="lf_mode",
            label_visibility="collapsed",
        )
        if mode == "light":
            if yolo_choices:
                detector_model = st.selectbox(
                    "Poids du détecteur d'aile (YOLO-OBB)", yolo_choices,
                    format_func=lambda p: Path(p).stem, key="lf_detector_choice",
                )
            else:
                detector_model = file_picker(
                    "Poids du détecteur d'aile (YOLO-OBB)", key="lf_detector_upload", type=["pt"],
                )
            heavy_ref = ""
        else:
            detector_model = ""
            heavy_ref = file_picker(
                "JSON de références YOLOE", key="lf_heavy_ref", type=["json"],
                help="Aucun poids de détecteur requis en mode heavy -- le moteur YOLOE utilise ses "
                     "propres poids embarqués. Forme JSON attendue : "
                     '{"base_root": {"darwin": "/path", "win32": "C:/path", ...}, "references": '
                     '[{"image": "relative/path.jpg", "boxes": [[x1, y1, x2, y2]]}, ...]}. '
                     "`base_root` résout chaque image de référence selon l'OS (clé sys.platform) ; "
                     "chaque référence liste une ou plusieurs boîtes pixel [x1, y1, x2, y2] axe-aligné "
                     "autour de l'aile dans cette image.",
            )

        st.divider()

        st.markdown("**Modèle UNet de landmarks**")
        if unet_choices:
            unet_model = st.selectbox(
                "Poids UNet (landmarks)", unet_choices, format_func=lambda p: Path(p).parent.name, key="lf_unet_choice",
            )
        else:
            unet_model = file_picker("Poids UNet (landmarks)", key="lf_unet_upload", type=["pt"])

        unet_n_landmarks = _unet_landmark_count(unet_model) if unet_model else None
        unet_crop_size = _unet_crop_size(unet_model) if unet_model else None
        out_width, out_height = unet_crop_size or (512, 256)
        if unet_n_landmarks:
            n_landmarks = unet_n_landmarks
        else:
            n_landmarks = st.number_input(
                "Nombre de landmarks", min_value=1, value=19, step=1, key="lf_n_landmarks",
                help="Aucun n_landmarks trouvé dans le train_config.json de ce checkpoint -- à définir "
                     "manuellement (19 = patron complet de Tancrède, 18 pour un ancien modèle UNet).",
            )
        if unet_n_landmarks or unet_crop_size:
            st.caption(
                f"Ce checkpoint UNet prédit {n_landmarks} landmark(s) à une résolution d'entrée de "
                f"{out_width}x{out_height} (depuis son train_config.json)."
            )

        st.divider()

        st.markdown("**Forme de référence GPA**")
        reference_choices = discover(f"{REFERENCE_SHAPES_DIR}/*.tps")
        if reference_choices:
            reference = st.selectbox(
                "Forme de référence GPA", reference_choices, format_func=lambda p: Path(p).stem, key="lf_reference_choice",
                help="Un .tps simple, un seul bloc spécimen -- voir 'Build a reference shape' dans "
                     "app/train_model.py pour en construire une nouvelle.",
            )
        else:
            reference = file_picker("Forme de référence GPA", key="lf_reference_upload", type=["tps"])
        if reference and Path(reference).exists():
            with st.expander("Aperçu de la forme de référence"):
                try:
                    zones = load_reference(Path(reference))
                    st.caption(f"{len(zones)} landmark(s)")
                    st.pyplot(plot_reference_shape(zones))
                except SystemExit as exc:
                    st.error(str(exc))

        st.divider()

        st.markdown("**Paramètres de détection & recadrage**")
        col_a, col_b = st.columns(2)
        imgsz = col_a.number_input("Taille d'image pour la détection", min_value=64, value=1024, step=32, key="lf_imgsz")
        device = col_b.selectbox("Calcul (CPU/GPU)", ["cpu", "cuda"], index=0, key="lf_device")
        conf = col_a.slider("Seuil de confiance de détection", 0.0, 1.0, 0.10, 0.01, key="lf_conf")
        padding = col_b.slider("Marge du recadrage", 0.0, 0.5, 0.10, 0.01, key="lf_padding")

        st.divider()

        st.markdown("**Options d'exécution**")
        landmarks_tag = st.text_input(
            "Tag des landmarks (optionnel)", key="lf_landmarks_tag",
            placeholder=f"ex. {int(n_landmarks)}lm",
            help="Laisser vide pour écrire dans <dataset>/landmarks/ (défaut, comme avant). À définir "
                 "en relançant le placement des landmarks sur ce même dataset déjà recadré avec un "
                 "autre modèle UNet/nombre de landmarks (ex. 18lm après 19lm) -- écrit alors dans "
                 "<dataset>/landmarks_<tag>/ à la place, pour ne pas écraser les fichiers du run "
                 "précédent. Pointer --tps (ou le dépôt TPS) de app/train_model.py/app/predict_dataset.py "
                 "vers le landmarks_numbered.tps du run voulu ensuite.",
        )
        overwrite = st.checkbox("Écraser les recadrages/landmarks déjà enregistrés (ne pas reprendre un run précédent)", key="lf_overwrite")
        retry_failed = st.checkbox("Réessayer les photos marquées FAILED", key="lf_retry_failed")

        submitted = st.button("Passer à la détection et au recadrage", icon=":material/arrow_forward:", type="primary", key="lf_submit")

    if submitted:
        if mode == "heavy" and not heavy_ref:
            st.error("--mode heavy nécessite un JSON de références YOLOE.")
            return
        if not unet_model:
            st.error("Un chemin de poids UNet (landmarks) est requis.")
            return
        if not reference:
            st.error("Une forme de référence GPA est requise.")
            return
        try:
            ref_zones = load_reference(Path(reference))
        except SystemExit as exc:
            st.error(str(exc))
            return
        if len(ref_zones) != int(n_landmarks):
            st.error(
                f"Nombre de landmarks incohérent : le modèle UNet prédit {int(n_landmarks)} landmark(s), "
                f"mais la forme de référence ({Path(reference).name}) en a {len(ref_zones)}. La "
                "renumérotation échouerait pour chaque spécimen -- choisissez une référence construite "
                "pour ce modèle (voir landmarks.build_reference, --drop/--n-landmarks) ou un checkpoint "
                "UNet correspondant."
            )
            return
        argv = [
            dataset_root, "--mode", mode,
            "--unet-model", unet_model, "--n-landmarks", str(int(n_landmarks)),
            "--reference", reference,
            "--imgsz", str(int(imgsz)), "--conf", str(conf),
            "--padding", str(padding), "--out-width", str(int(out_width)), "--out-height", str(int(out_height)),
            "--device", device,
        ]
        if detector_model:
            argv += ["--detector-model", detector_model]
        if heavy_ref:
            argv += ["--heavy-ref", heavy_ref]
        if landmarks_tag:
            argv += ["--landmarks-tag", landmarks_tag]
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
    st.caption(f"Jeu de données : {args.dataset}")

    slug = run_label.replace(" ", "_")
    col_run, col_skip = st.columns(2)
    if col_run.button(f"Lancer {run_label}", icon=":material/play_arrow:", type="primary", key=f"run_{slug}"):
        run_with_log(f"{run_label.capitalize()} en cours...", fn, args)
        st.session_state.step = next_step
        st.rerun()
    if col_skip.button("Passer -- déjà fait", icon=":material/skip_next:", help=skip_hint, key=f"skip_{slug}"):
        advance(next_step)


# ---------------------------------------------------------------------------
# Steps 3 / 5 -- validation review (shared rendering, different data source)
# ---------------------------------------------------------------------------

def _filtered_view(df: pd.DataFrame, state_key: str) -> pd.DataFrame:
    col_filter, col_search = st.columns([1, 2])
    status_filter = col_filter.segmented_control(
        "Filtrer par statut", options=["All"] + STATUSES,
        format_func=lambda s: "Tous" if s == "All" else s,
        default="All", key=f"{state_key}_filter", required=True,
    )
    search = col_search.text_input(
        "Rechercher photo_id / inv_id", key=f"{state_key}_search", placeholder="Rechercher photo_id / inv_id",
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
        "Taille des vignettes", options=["medium", "large"],
        format_func=lambda s: "Moyenne" if s == "medium" else "Grande",
        default="medium", required=True, key="thumb_size",
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
            "thumbnail": st.column_config.ImageColumn("Recadrage", width=thumb_width),
            "reviewed_status": st.column_config.SelectboxColumn(options=STATUSES, required=True),
        },
        disabled=[c for c in display_cols if c != "reviewed_status"],
    )
    full_df.loc[edited.index, "reviewed_status"] = edited["reviewed_status"]
    st.session_state.crop_review_df = full_df

    if st.button("Enregistrer la validation et continuer", icon=":material/arrow_forward:", type="primary"):
        crops_reviewed_path, _audit = write_crop_review(args.dataset, args.mode, full_df)
        args.crops_csv = str(crops_reviewed_path)
        n_changed = int((full_df["auto_status"] != full_df["reviewed_status"]).sum())
        st.toast(f"{n_changed} correction(s) enregistrée(s) -> {crops_reviewed_path}", icon=":material/check:")
        advance(4)


def step_landmark_review() -> None:
    st.header(STEPS[4])
    args = st.session_state.args
    landmarks_dir = landmarks_dirname(args)

    if st.session_state.landmark_review_df is None:
        try:
            st.session_state.landmark_review_df = build_landmark_review_df(args.dataset, landmarks_dir)
        except FileNotFoundError as exc:
            st.error(str(exc))
            return

    full_df = st.session_state.landmark_review_df
    counts = full_df["reviewed_status"].value_counts()
    st.caption(" | ".join(f"{s}: {counts.get(s, 0)}" for s in STATUSES))

    thumb_width = _thumbnail_width()
    view = _filtered_view(full_df, "landmark")

    landmarks_by_photo = load_numbered_landmarks_by_photo_id(args.dataset, landmarks_dir=landmarks_dir)

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

    if st.button("Enregistrer la validation et continuer", icon=":material/arrow_forward:", type="primary"):
        landmarks_reviewed_path, _audit = write_landmarks_review(args.dataset, full_df, landmarks_dir)
        args.landmarks_status_csv = str(landmarks_reviewed_path)
        n_changed = int((full_df["auto_status"] != full_df["reviewed_status"]).sum())
        st.toast(f"{n_changed} correction(s) enregistrée(s) -> {landmarks_reviewed_path}", icon=":material/check:")
        advance(6)


# ---------------------------------------------------------------------------
# Step 6 -- export (terminal: no further step, dataset is ready)
# ---------------------------------------------------------------------------

def _render_pipeline_stats(dataset: str) -> None:
    """Per (step, approach) summary from <dataset>/pipeline_stats.csv (see
    core.pipeline_io.update_pipeline_stats) -- lets a user sanity-check
    OK/SUSPECT/SKIPPED/FAILED counts and per-image timing before spending
    the export step on a run they may want to redo instead."""
    stats_path = Path(dataset) / "pipeline_stats.csv"
    if not stats_path.exists():
        return
    stats_df = pd.read_csv(stats_path)
    if stats_df.empty:
        return
    stats_df["mean_time"] = stats_df["mean_time_s"].apply(format_duration)
    stats_df["total_time"] = stats_df["total_time_s"].apply(format_duration)
    display_cols = ["step", "approach", "total", "ok", "suspect", "skipped", "failed", "mean_time", "total_time"]
    st.caption("Résumé du pipeline (par étape/méthode) :")
    st.dataframe(stats_df[display_cols], hide_index=True)
    st.divider()


def step_export() -> None:
    st.header(STEPS[5])
    args = st.session_state.args
    st.caption(f"Jeu de données : {args.dataset}")

    _render_pipeline_stats(args.dataset)

    col_run, col_skip = st.columns(2)
    if col_run.button("Lancer l'export", icon=":material/play_arrow:", type="primary", key="run_export"):
        st.session_state.export_dir = run_with_log("Export du paquet de landmarks...", run_export, args)
    if col_skip.button(
        "Passer -- déjà fait", icon=":material/skip_next:",
        help="Si <dataset>/export/ a déjà été écrit.", key="skip_export",
    ):
        st.session_state.export_dir = resolve_export_dir(args)

    if st.session_state.export_dir:
        st.success(f"Jeu de données prêt -> {args.dataset}")
        st.caption(f"Paquet de landmarks -> {st.session_state.export_dir}")
        st.info(
            "Ensuite : `streamlit run app/train_model.py` pour entraîner un modèle sur ce jeu de "
            "données, ou `streamlit run app/predict_dataset.py` pour le classifier avec un modèle "
            "existant."
        )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

if st.session_state.args is not None:
    st.sidebar.caption(f"Jeu de données : {st.session_state.args.dataset}")

step = st.session_state.step
if step == 1:
    step_setup()
elif step == 2:
    step_run(STEPS[1], "détection et recadrage", run_detection_and_crop, next_step=3,
             skip_hint="Si extraction/<mode>/crops.csv a déjà été écrit pour ce jeu de données.")
elif step == 3:
    step_crop_review()
elif step == 4:
    _landmarks_dir = landmarks_dirname(st.session_state.args)
    step_run(STEPS[3], "placement des landmarks", run_landmark_placement, next_step=5,
             skip_hint=f"Si {_landmarks_dir}/landmarks_numbered.csv a déjà été écrit pour ce jeu de données.")
elif step == 5:
    step_landmark_review()
elif step == 6:
    step_export()
