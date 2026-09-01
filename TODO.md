# TODO -- refactor idmybee

Coché = fait. Voir `RESUME.md` pour le contexte, `CONVENTIONS.md` pour le
style (validé, y compris l'anglais partout). Organisé par phase ; ne pas
commencer une phase avant que la précédente soit close.

## Phase 0 -- groundwork (bien avancée)

- [x] Diagnostic du code existant
- [x] `CONVENTIONS.md` -- validé (docstrings, nommage CLI, logging,
      imports, tests, **langue : anglais partout**)
- [x] `pyproject.toml` + suppression des `sys.path.insert`
- [x] Socle de tests sur les fonctions pures (gpa, alignment, outliers,
      tps_io, run_io, pipeline_io, hungarian_umeyama) -- 60 tests
- [x] `utils/cli.py` étendu : `add_logging_args`/`log_level_from_args`,
      `add_dataset_positional`
- [x] Fix du desync `parse_tps`
- [x] Suppression de `tools/find_missing_landmark.py`
- [x] Unification de `should_skip` (`utils/pipeline_io.py`, avec
      `output_exists` paramétrable)
- [x] `requirements.txt` réduit à un pointeur vers `pyproject.toml`
- [x] Anglais + logging + `dataset` positionnel + `argv` sur tout le
      pipeline "outil 1" : `extraction/detect_wing.py`,
      `extraction/normalize_crop.py`, `extraction/extraction_io.py`,
      `landmarks/predict.py`, `landmarks/renumber.py`,
      `landmarks/build_reference.py`, `landmarks/methods/base.py`,
      `landmarks/methods/hungarian_umeyama.py`
- [x] Anglais + logging sur le cœur partagé : `utils/tps_io.py`,
      `gpa.py`, `alignment.py`, `outliers.py`, `pipeline_io.py`,
      `run_io.py`, `model_io.py`, `cli.py`

Reste en Phase 0 (anglais + logging, fichiers non touchés cette session) :

- [ ] `utils/dataset.py`, `predictions.py`, `tps_overlay.py`,
      `repair_images.py`
- [ ] `manifest/build_dataset.py`
- [ ] `classifiers/train.py`, `classifiers/predict.py`
- [ ] `analysis/variance.py`, `variance_report.py`,
      `classification_report.py`, `compare_runs.py`
      (+ raccorder `--verbose`/`--quiet` via `utils/cli.py` : ces scripts
      appellent déjà `setup_console_logging()` sans flag)
- [ ] `landmarks_trainer/export_dataset.py`, `reproject_reference.py`
      (le reste de `landmarks_trainer/` est déjà en anglais)
- [ ] `tools/clean_tps.py`, `convert_heic_to_jpeg.py`,
      `drop_landmark_from_tps.py`, `flatten_image_dirs.py`, `verify_tps.py`
- [ ] `extraction/heavy/`, `extraction/light/` (backends de détection,
      pas inspectés cette session)
- [ ] Auditer `extraction/heavy/vpe.py` (repéré en français, contenu non
      vérifié)

## Phase 1 -- cœur partagé (`core/`)

- [ ] Renommer `utils/` -> `core/` (ou nom retenu), déplacer
      `tps_io.py`, `gpa.py`, `alignment.py`, `outliers.py`,
      `model_io.py`
- [ ] Décider du sort de `utils/dataset.py`, `predictions.py`,
      `pipeline_io.py`, `run_io.py`, `cli.py`, `repair_images.py`,
      `tps_overlay.py` -- lesquels sont vraiment "core" (géométrie/IO pure)
      vs spécifiques à un des 3 outils
- [ ] Mettre à jour tous les imports (`from utils.xxx` -> `from core.xxx`)

## Phase 2 -- outil 1 : landmarking

- [ ] Orchestrateur bout-en-bout : dossier d'images -> détection ->
      normalisation -> prédiction UNet -> renumérotation -> TPS + log
      (remplace le `extract_wings.py` jamais écrit -- la plomberie
      individuelle de chaque étape est prête depuis la Phase 0)
- [ ] Sortie configurable : crops sauvegardés (o/n), TPS espace original /
      crop / les deux (généraliser la transformation inverse déjà présente
      dans `reproject_reference.py`, écrite pour Tancrède spécifiquement)
- [ ] Référence de renumérotation (blueprint Tancrède) embarquée comme
      asset par défaut, pas à fournir par l'utilisateur
- [ ] UI Streamlit ou Gradio (à choisir) -- expose les fonctions "un item"
      déjà découplées (`detect_one_image`, `normalize_one`,
      `predict_landmarks`, `numerate_one`)

## Phase 3 -- outil 2 : classification / analyse

- [ ] CLI unique train/predict/anova depuis un couple TPS/CSV, join par
      clé avec repli par ordre (voir RESUME.md)
- [ ] Valider que `classifiers/`, `analysis/` n'ont plus besoin de
      `manifest/specimens.csv` complets pour un usage "TPS/CSV autonome"
      (garder le chemin manifest complet pour l'usage interne au labo)

## Phase 4 -- outil 3 : trainers

- [ ] Harmoniser les 3 conventions de sortie (`run_io.py` /
      `landmarks_trainer/checkpoint.py` / natif Ultralytics) en une seule
- [ ] Auditer `obb_trainer/` pour la convention de stockage (déjà en
      anglais, convention de sortie non encore harmonisée)

## Non planifié / à discuter

- Tests de non-régression avec données réelles (nécessite le micro-dataset
  -- redemander à Jules le moment venu, pas avant)
- Suppression définitive du contenu de `tools/` une fois ses scripts
  utiles migrés dans les 3 outils (ou parts conservées comme diagnostics
  ponctuels documentés)
