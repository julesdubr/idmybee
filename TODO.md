# TODO -- refactor idmybee

Coché = fait. Voir `RESUME.md` pour le contexte, `CONVENTIONS.md` pour le
style (validé, y compris l'anglais partout). Organisé par phase ; ne pas
commencer une phase avant que la précédente soit close.

## Phase 0 -- groundwork

- [x] Diagnostic du code existant
- [x] `CONVENTIONS.md` -- validé
- [x] `pyproject.toml` + suppression des `sys.path.insert`
- [x] Socle de tests sur les fonctions pures -- 60 tests
- [x] `utils/cli.py` étendu
- [x] Fix du desync `parse_tps`
- [x] Suppression de `tools/find_missing_landmark.py`
- [x] Unification de `should_skip`
- [x] `requirements.txt` réduit à un pointeur vers `pyproject.toml`
- [x] Anglais + logging + `dataset` positionnel + `argv` sur le pipeline
      "outil 1" (extraction/landmarks -- liste complète dans la version
      précédente de ce fichier)
- [x] Anglais + logging sur le cœur partagé (`utils/tps_io.py`, `gpa.py`,
      `alignment.py`, `outliers.py`, `pipeline_io.py`, `run_io.py`,
      `model_io.py`, `cli.py`)
- [x] `utils/dataset.py`, `predictions.py`, `tps_overlay.py`,
      `repair_images.py`
- [x] `manifest/build_dataset.py`
- [x] `classifiers/train.py`, `classifiers/predict.py`
- [x] `analysis/variance.py`, `variance_report.py`,
      `classification_report.py`, `compare_runs.py` -- `--verbose`/`--quiet`
      raccordé via `utils/cli.py` sur les 4
- [x] `landmarks_trainer/export_dataset.py`, `reproject_reference.py`
- [x] `tools/clean_tps.py`, `convert_heic_to_jpeg.py`,
      `drop_landmark_from_tps.py`, `flatten_image_dirs.py`, `verify_tps.py`
- [x] `extraction/heavy/`, `extraction/light/` -- traduits, `error_reason`
      passés en anglais
- [x] Auditer `extraction/heavy/vpe.py` -- déjà conforme, 1 chaîne corrigée

Reste en Phase 0 -- pas fait, décision prise (voir RESUME.md
"Questions ouvertes") :

- [ ] Aligner `landmarks_trainer/` (train.py, evaluate.py, checkpoint.py,
      augment.py, heatmap.py, model.py, dataset.py, constants.py) sur les
      conventions CLI/logging (`main(argv=None)`, `logger`,
      `add_logging_args`) -- déjà en anglais mais pas conformes
      structurellement. `export_dataset.py`/`reproject_reference.py`,
      traités cette session, le sont déjà, ce qui crée une incohérence
      locale entre-temps. **Décidé (session Phase 1, 2 sept. 2026) :
      reporté à la Phase 4** -- priorité basse confirmée par Jules,
      `landmarks_trainer/` sera de toute façon retouché à ce moment-là
      pour l'harmonisation des sorties de run.

## Phase 1 -- cœur partagé (`core/`)

- [x] Créer `src/core/` (nouveau package -- `utils/` continue d'exister en
      parallèle pour les 7 fichiers non tranchés, voir item suivant),
      déplacer `tps_io.py`, `gpa.py`, `alignment.py`, `outliers.py`,
      `model_io.py`. Imports croisés entre ces 5 fichiers
      mis à jour (`gpa.py` -> `core.alignment`, `outliers.py` ->
      `core.gpa`/`core.tps_io`). Mentions en docstring (pas seulement les
      `import`) corrigées aussi (ex. `alignment.py` référençait
      `utils/gpa.py`).
- [ ] Décider du sort de `utils/dataset.py`, `predictions.py`,
      `pipeline_io.py`, `run_io.py`, `cli.py`, `repair_images.py`,
      `tps_overlay.py` -- **toujours ouvert**, non traité cette session
      (voir RESUME.md "Où on en est" / "Questions ouvertes" pour le détail).
      Ces 7 fichiers restent dans `utils/` pour l'instant.
- [x] Mettre à jour tous les imports (`from utils.xxx` -> `from core.xxx`)
      -- fait, mais seulement pour les 5 modules déplacés ci-dessus. Les
      imports vers les 7 fichiers restés dans `utils/` (`utils.dataset`,
      `utils.predictions`, `utils.pipeline_io`, `utils.run_io`,
      `utils.cli`, `utils.repair_images`, `utils.tps_overlay`) sont
      inchangés -- normal, ces fichiers n'ont pas bougé. Fichiers
      concernés par le renommage effectif des imports (grep sur
      `utils.tps_io|gpa|alignment|outliers|model_io`) : `classifiers/
      predict.py`, `train.py`, `landmarks/renumber.py`, `landmarks/
      methods/hungarian_umeyama.py`, `landmarks/build_reference.py`,
      `landmarks/predict.py`, `tools/drop_landmark_from_tps.py`,
      `verify_tps.py`, `clean_tps.py`, `landmarks_trainer/
      reproject_reference.py`, `export_dataset.py`, `analysis/
      classification_report.py`, `variance_report.py`, `utils/
      tps_overlay.py`, `dataset.py`, `predictions.py`, + les 5 fichiers
      `tests/test_*.py` correspondants (`test_tps_io.py`, `test_gpa.py`,
      `test_alignment.py`, `test_outliers.py`). 60 tests toujours verts,
      compilation OK sur tout `src/`+`tests/`.

## Phase 2 -- outil 1 : landmarking

Conception validée session du 2 sept. 2026 -- détail complet dans
`RESUME.md` "Détails outil 1". Résumé actionnable ci-dessous.

- [x] Framework UI : **Streamlit** (raisons dans `RESUME.md`)
- [ ] Fonction core en mémoire `place_landmarks(image) -> Landmarks`
      (pure Python, pas de lecture/écriture disque) -- prérequis du mode
      terrain, voir `CONVENTIONS.md` "Fonctions core réutilisables"
- [ ] Fonction core `draw_landmarks_overlay(image, landmarks) -> Image`
      (overlay annoté partagé CLI/UI)
- [ ] **Mode dataset** -- orchestrateur bout-en-bout : dossier d'images ->
      détection -> normalisation -> prédiction UNet -> renumérotation ->
      TPS + log d'échecs ; + CSV de données biologiques si un
      manifest/specimens.csv est fourni
- [ ] Sortie configurable (mode dataset) : crops sauvegardés (o/n), TPS
      espace original / crop / les deux
- [ ] Référence de renumérotation (blueprint Tancrède) embarquée comme
      asset par défaut
- [ ] **Mode terrain / single** -- une photo ou un petit ensemble ad hoc
      (sans manifest) -> aucun fichier écrit ; stub temporaire = affiche
      les coordonnées TPS calculées (zone de code copiable en UI, stdout
      en CLI) en attendant la Phase 3 (voir item correspondant Phase 3)
- [ ] Statut auto `FAILED`/`SUSPECT`/`OK` calculé pendant le run, colonne
      `status`, identique CLI et UI (détail des critères dans
      `RESUME.md`)
- [ ] UI -- étape de validation (mode dataset) : galerie annotée
      (landmarks numérotés) + statut éditable OK/SUSPECT/FAILED,
      navigable, export final
- [ ] CLI -- parité fonctionnelle de la validation : `--export-review`
      (overlays + `review.csv`) et une commande de réconciliation (nom
      provisoire `reconcile-review`) qui réapplique un `review.csv` édité
      à la main au TPS/CSV final
- [x] Décider (question ouverte, voir `RESUME.md`) : le mode terrain
      garde-t-il un aperçu overlay sans statut persistant ? **Décidé
      (2 sept. 2026, Phase 1) : oui**, aperçu overlay conservé en mode
      terrain.
- [x] Décider (question ouverte) : app Streamlit unique (outil 1 + outil
      2) ou deux apps séparées ? **Décidé (2 sept. 2026, Phase 1) : une
      seule app.** Focus immédiat sur le landmarking ; la classification
      (outil 2) s'ajoute plus tard dans la même app, de façon
      incrémentale.
- [x] Mode dataset / mode terrain : **deux scripts distincts** (pas un
      seul script avec flag de mode) -- décidé (2 sept. 2026, Phase 1),
      confirme l'option déjà provisoirement retenue dans `RESUME.md`.
- [x] Nom de la commande de réconciliation : **`reconcile-review`
      confirmé comme nom définitif** (n'est plus provisoire) -- décidé
      (2 sept. 2026, Phase 1).

## Phase 3 -- outil 2 : classification / analyse

- [ ] CLI unique train/predict/anova depuis un couple TPS/CSV, join par
      clé avec repli par ordre
- [ ] Valider l'usage "TPS/CSV autonome" sans manifest complet
- [ ] `train.py` et l'ANOVA/variance restent **CLI uniquement** (réservé
      aux devs) -- décision confirmée session du 2 sept. 2026
- [ ] UI Streamlit (batch + single) pour **predict** uniquement :
      sélection d'un modèle LDA pré-fitté parmi les runs publiés (liste
      curatée depuis `data/models/lda/<run_id>/train/`)
- [ ] Brancher ce predict dans le **mode terrain de l'outil 1** (remplace
      le stub TPS-only du mode single/terrain, voir Phase 2) -- retourne
      un top-N espèce/caste + % de confiance, appel direct fonction à
      fonction, pas de TPS intermédiaire sur disque

## Phase 4 -- outil 3 : trainers

- [ ] Harmoniser les 3 conventions de sortie
      (`run_io.py` / `landmarks_trainer/checkpoint.py` / natif Ultralytics)
- [ ] Auditer `obb_trainer/` pour la convention de stockage
- [ ] (Si reporté depuis Phase 0) Aligner `landmarks_trainer/` sur les
      conventions CLI/logging

## Non planifié / à discuter

- Tests de non-régression avec données réelles (redemander le
  micro-dataset le moment venu)
- Suppression définitive du contenu de `tools/` une fois ses scripts
  utiles migrés dans les 3 outils
- Fixer l'annotation de type incorrecte sur `load_roots_config` dans
  `manifest/build_dataset.py` (`-> list[dict]` au lieu de `-> dict`,
  cosmétique, aucun effet à l'exécution)
