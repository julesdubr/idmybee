# RESUME -- état du refactor idmybee

Document vivant, à tenir à jour à chaque session. But : permettre de
reprendre le travail dans une nouvelle discussion sans tout réexpliquer.
Voir aussi `CONVENTIONS.md` (style, validé) et `TODO.md` (tâches).

## Contexte

`idmybee` : pipeline de morphométrie géométrique sur ailes de bourdons
(espèce + caste). Chaîne actuelle : détection de l'aile -> normalisation ->
landmarks UNet -> renumérotation -> GPA/PCA/LDA -> analyse de variance.
Toutes les briques existent et fonctionnent ; le code a été écrit bloc par
bloc sur plusieurs sessions, d'où des incohérences de convention (CLI,
logging, docstrings, langue, sorties de run) sans duplication de logique
majeure.

## Décisions d'architecture (validées)

Découpage en 3 outils + un cœur partagé :

1. **Landmarking** (CLI + UI Streamlit/Gradio, lequel des deux reste
   ouvert) -- dossier d'images en entrée -> TPS + log d'échecs en sortie.
   Pas de manifest/specimens.csv requis. Sortie configurable : crops
   sauvegardés ou non, TPS espace original / crop / les deux.
2. **Classification / analyse** (CLI uniquement) -- couple TPS + CSV
   (ordonnés pareil, sans les images en erreur) en entrée. Train, predict,
   ET l'ANOVA/variance. **LDA train/predict reste ici** (rapide, pas de
   GPU, c'est la brique consommée par predict/anova -- pas rangé dans
   "trainers").
3. **Trainers** (CLI uniquement, pas de UI prévue) -- UNet landmarks,
   YOLO-OBB. Entraînements lourds, GPU, produisent les poids consommés par
   l'outil 1. Sorties encore à harmoniser (3 conventions différentes
   aujourd'hui, voir diagnostic).
4. **Core** (pas un "outil", une dépendance partagée) -- TPS I/O, GPA,
   alignement (Kabsch-Umeyama), détection d'outliers. Utilisé à la fois par
   1 (renumérotation) et 2 (classification). Aujourd'hui c'est `utils/`,
   renommage en `core/` prévu au moment du découpage (pas fait maintenant
   pour éviter de bouger deux fois).

Le join TPS/CSV de l'outil 2 se fait par clé (`image_id`/`specimen_id` via
`COMMENT=`) quand disponible, avec repli sur l'ordre des lignes seulement
pour un TPS tiers sans clé -- pas un remplacement pur du join par ordre à
la Adrien-R.

**Langue : tout en anglais** (code, docstrings, logs, CLI, valeurs de champ
type `error_reason`). Voir `CONVENTIONS.md`. Les CSV déjà produits en
français ne sont pas migrés rétroactivement.

## Où on en est

**Phase actuelle : groundwork pré-découpage, bien avancée.** Le découpage
en 3 lui-même n'a pas commencé. Voir `TODO.md` pour le détail précis.

Fait :
- Diagnostic complet du code existant.
- `CONVENTIONS.md` validé (docstrings, nommage CLI, logging, imports,
  tests, langue).
- `pyproject.toml` : `pip install -e .`, plus de `sys.path.insert` (18
  fichiers nettoyés, imports "nus" requalifiés).
- `utils/cli.py` étendu : `add_logging_args`/`log_level_from_args`
  (`--verbose`/`--quiet`, partagé par tous les scripts) et
  `add_dataset_positional`.
- **Pipeline "outil 1" (landmarking) entièrement retravaillé** : anglais +
  logging + `dataset` positionnel + `--verbose`/`--quiet` + `main(argv=None)`
  sur `extraction/detect_wing.py`, `extraction/normalize_crop.py`,
  `extraction/extraction_io.py`, `landmarks/predict.py`,
  `landmarks/renumber.py`, `landmarks/build_reference.py`,
  `landmarks/methods/base.py`, `landmarks/methods/hungarian_umeyama.py`.
- **Cœur partagé (`utils/`) partiellement retravaillé** (anglais + logging) :
  `tps_io.py`, `gpa.py`, `alignment.py`, `outliers.py`, `pipeline_io.py`,
  `run_io.py`, `model_io.py`, `cli.py`. Restent en français :
  `dataset.py`, `predictions.py`, `tps_overlay.py`, `repair_images.py`.
- `parse_tps(strict=False)` : bug de désynchronisation corrigé (un bloc
  tronqué ne fait plus perdre le bloc suivant, pourtant valide) -- voir
  `tests/test_tps_io.py::test_parse_tps_non_strict_recovers_block_after_truncated_one`.
- `tools/find_missing_landmark.py` supprimé (cassé, référençait un module
  `numbering` qui n'existe plus).
- `should_skip` unifié : une seule implémentation dans `utils/pipeline_io.py`
  (généralisée avec un paramètre `output_exists` -- le sens de "la sortie
  existe encore" diffère selon l'étape : fichier sur disque vs entrée dans
  un TPS rechargé). `manifest/io.py` la réexporte ; `landmarks/predict.py`
  l'utilise avec `output_exists=(tps_id in working_tps)`, ce qu'elle ne
  faisait pas avant (elle ne vérifiait rien, contrairement à
  `manifest/io.py` -- qui elle n'était appelée nulle part : code mort).
- `requirements.txt` réduit à un pointeur vers `pyproject.toml`.
- Socle de tests (`tests/`, pytest) : 60 tests, tous passent, aucun
  n'a besoin du dataset réel. Ajout depuis la dernière session :
  `tests/test_hungarian_umeyama.py` (recouvrement de permutation sous
  rotation/échelle/miroir).

Pas fait (reporté, voir `TODO.md` pour le détail précis) :
- Anglais + logging sur `classifiers/`, `analysis/`, `manifest/build_dataset.py`,
  le reste de `utils/` (dataset.py, predictions.py, tps_overlay.py,
  repair_images.py), `landmarks_trainer/export_dataset.py` +
  `reproject_reference.py`, `tools/*.py` (5 scripts restants),
  `extraction/heavy/`, `extraction/light/`.
- `utils/cli.py` : pas encore utilisé par `classifiers/`/`analysis/` pour
  le `--verbose`/`--quiet` (ils utilisaient déjà `setup_console_logging()`
  sans flag -- à raccorder).
- Le découpage en 3 lui-même (déplacement de fichiers, `core/`,
  `landmarking/`, `classification/`, `training/`).
- UI Streamlit/Gradio de l'outil 1 (pas commencée -- lequel des deux reste
  à choisir).

## Bugs / incohérences trouvés en cours de route

- ~~`tools/find_missing_landmark.py` cassé~~ -- supprimé.
- ~~`should_skip` divergent~~ -- unifié.
- ~~`parse_tps` desync~~ -- corrigé.
- `extraction/detect_wing.py` référence toujours un `extract_wings.py` qui
  n'existe pas (orchestrateur bout-en-bout jamais écrit -- c'est
  précisément ce que l'outil 1 doit devenir).
- `extraction/heavy/`, `extraction/light/` (backends de détection appelés
  dynamiquement par `detect_wing.py`) n'ont pas été inspectés en détail --
  probablement encore en français, pas vérifiés par cette session.

## Questions ouvertes

- Streamlit vs Gradio pour l'outil 1 -- lequel, ou on regarde les deux
  avant de choisir ?
- Priorité de la suite : finir la traduction anglais/logging du reste du
  code (classifiers/analysis/manifest/trainers/tools), ou attaquer le
  découpage en 3 avec ce qui est déjà en anglais ?
