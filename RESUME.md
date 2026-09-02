# RESUME -- état du refactor idmybee
 
Document vivant, à tenir à jour à chaque session. But : permettre de
reprendre le travail dans une nouvelle discussion sans tout réexpliquer.
Voir aussi `CONVENTIONS.md` (style, validé) et `TODO.md` (tâches).

## Conventions de session (méta)

*(Ajouté session du 2 sept. 2026.)*

- Ces trois documents (`RESUME.md`, `TODO.md`, `CONVENTIONS.md`) sont
  amendés de façon **additive** : on ajoute/fusionne/corrige avec une note
  datée, on ne supprime jamais une information existante ni ne la résume
  silencieusement en renvoyant à "une version antérieure". Le travail se
  fait à travers de nombreuses conversations séparées (limite de contexte
  par session) et ces documents doivent rester complets à chaque reprise.
- Les modifications de code sont livrées sous forme de **patchs git**
  (diff/patch), pas de fichiers collés intégralement, pour rester
  applicables et traçables dans le dépôt.

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
 
1. **Landmarking** (CLI + UI). *Mis à jour session du 2 sept. 2026 :
   framework UI tranché (Streamlit) et le "mode batch/single" initialement
   envisagé s'est précisé en **deux usages distincts** -- détail complet
   dans "Détails outil 1" plus bas :
   - **Mode dataset** : dossier d'images en entrée -> TPS + log d'échecs
     en sortie, toujours ; CSV de données biologiques en plus si un
     manifest/specimens.csv est fourni (aligné, hors images en erreur),
     sinon TPS+log seul. Sortie configurable : crops sauvegardés ou non,
     TPS espace original / crop / les deux.
   - **Mode terrain / single** (nouveau) : une photo ou un petit ensemble
     ad hoc -> pas de fichier écrit sur disque, juste un résultat affiché
     (stub TPS en attendant la classification, puis top-N espèce/caste +
     confiance une fois l'outil 2 branché).
2. **Classification / analyse**. *Mis à jour session du 2 sept. 2026 :*
   train et ANOVA/variance restent **CLI uniquement** (réservé aux devs),
   mais **une UI (batch + single) est prévue pour predict** (choix d'un
   modèle LDA pré-fitté), consommée aussi bien directement que par le mode
   terrain de l'outil 1 -- voir "Détails outil 1" et Phase 3 dans
   `TODO.md`. Entrée : couple TPS + CSV (ordonnés pareil, sans les images
   en erreur). Train, predict, ET l'ANOVA/variance. **LDA train/predict
   reste ici** (rapide, pas de GPU, c'est la brique consommée par
   predict/anova -- pas rangé dans "trainers").
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

## Détails outil 1 -- conception validée (session du 2 sept. 2026)

Conception uniquement, aucune implémentation commencée (Phase 1 -- core --
pas encore lancée). Détail complet à conserver pour la reprise en Phase 2.

**Framework UI : Streamlit** (tranché cette session). Raisons : l'app va
héberger dans la durée deux outils (landmarking puis classification) qui
partagent des composants -- le multipage natif de Streamlit et
`st.session_state` collent mieux à ça qu'assembler plusieurs apps Gradio
séparées ; l'étape de validation a besoin d'un état navigable/éditable
persistant (parcourir/filtrer par statut, éditer un item, garder les
modifs entre reruns) qui est le cas d'usage central de `st.session_state` ;
les nombreux paramètres de pipeline (seuils, chemins de poids, booléens) se
mappent naturellement sur des widgets de sidebar, même logique que
l'`argparse` déjà en place -> bonne parité CLI/UI. Contre-argument honnête :
Gradio a un `gr.Gallery` (grille + légendes + sélection) pensé pour la
galerie de validation spécifiquement, ce qui coûterait moins de code sur
cette seule partie -- mais l'avantage se renverse en regardant l'app dans
son ensemble (multipage, état, formulaires). Question encore ouverte : une
app unique (outil 1 + outil 2) ou deux apps séparées -- voir "Questions
ouvertes".

**Deux usages du même outil, pas juste un mode batch/single :**

- **Mode dataset** -- construction/extension de dataset annoté. Dossier
  d'images (+ manifest optionnel) -> TPS + log d'échecs, + CSV bio si
  manifest fourni. Sortie configurable (crops sauvegardés ou non, TPS
  espace original/crop/les deux). Passe par l'étape de validation
  complète (galerie).
- **Mode terrain / single** -- prédiction rapide, pas de construction de
  dataset. Une photo, ou un petit ensemble ad hoc de photos (pas besoin de
  manifest ni de structure "dataset") -> **aucun fichier TPS/CSV écrit sur
  disque** (jugé sans intérêt pour une ligne unique). Tant que la
  classification (outil 2, Phase 3) n'est pas branchée : affiche
  uniquement les coordonnées TPS calculées -- zone de code copiable en UI,
  impression sur stdout en CLI -- en guise de stub temporaire. Une fois
  Phase 3 disponible : remplace le stub par un top-N espèce/caste + % de
  confiance (appel direct à `predict.py`/sa fonction Python, pas de TPS
  intermédiaire sur disque).

**Fonctions core en mémoire (nouvelle convention, voir aussi
`CONVENTIONS.md`)** : la pose de landmarks doit exister comme fonction
Python pure (image -> landmarks), pas seulement via CLI/fichiers, pour être
appelée directement par le mode terrain sans aller-retour disque. Idem pour
l'overlay annoté (`draw_landmarks_overlay(image, landmarks) -> Image`),
partagé tel quel entre CLI et UI.

**Étape de validation** (mode dataset ; extension possible au mode terrain
en aperçu seul, voir "Questions ouvertes") :

- Statut auto calculé pendant le run, **identique CLI et UI** (pas un
  artefact d'interface), écrit en colonne `status` : `FAILED` (détection,
  crop ou pose échoués), `SUSPECT` (ex. flag outlier MAD, erreur de
  reprojection GPA élevée, ambiguïté zones 0/1/13), `OK` sinon.
- UI : galerie (mode dataset) ou vue unique, image annotée avec landmarks
  numérotés + statut éditable (OK/SUSPECT/FAILED), navigable, export final
  du TPS/CSV avec les statuts corrigés.
- CLI (parité fonctionnelle, pas la même UX -- le geste humain de
  "parcourir et cliquer" reste par nature UI-only, c'est la mécanique qui
  est partagée) : un mode `--export-review` écrit les overlays annotés sur
  disque + un `review.csv` (`image_id`, statut auto, chemin de l'overlay) ;
  une commande de réconciliation (nom provisoire `reconcile-review`)
  relit un `review.csv` édité à la main (statuts corrigés dans un tableur
  ou éditeur de texte) et l'applique au TPS/CSV final.

**Paramètres pipeline** (détection OBB, cropping, pose UNet,
renumérotation) : configurables avec défauts, référence de renumérotation
(blueprint Tancrède) embarquée comme asset par défaut. Détail
d'implémentation à trancher en Phase 2 : un schéma de déclaration unique
(nom/type/défaut/aide) dont dérivent à la fois le flag `argparse` et le
widget Streamlit correspondant, pour éviter de dupliquer chaque paramètre.

## Où on en est

**Phase actuelle : groundwork pré-découpage, terminé.** Le découpage
en 3 (phase 1) lui-même n'a pas commencé. Voir `TODO.md` pour le détail précis.
 
Fait :
- Conception détaillée de l'outil 1 (landmarking, modes dataset/terrain,
  validation, framework UI) validée cette session (2 sept. 2026) -- voir
  "Détails outil 1" plus haut. Conception uniquement, pas de code.
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
- Anglais + logging + conventions CLI (`main(argv=None)`, positional
  `dataset` où pertinent, `--verbose`/`--quiet` via `utils/cli.py`) sur :
  `utils/dataset.py`, `predictions.py`, `tps_overlay.py`, `repair_images.py`,
  `manifest/build_dataset.py`, `classifiers/train.py`, `predict.py`,
  `analysis/variance.py`, `variance_report.py`, `classification_report.py`,
  `compare_runs.py`, `landmarks_trainer/export_dataset.py`,
  `reproject_reference.py`, `tools/clean_tps.py`,
  `convert_heic_to_jpeg.py`, `drop_landmark_from_tps.py`,
  `flatten_image_dirs.py`, `verify_tps.py`, `extraction/heavy/detection.py`,
  `extraction/light/detection.py`.
- `extraction/heavy/vpe.py` audité : déjà conforme (anglais, docstrings
  courtes), une seule chaîne française corrigée (`raise ValueError`).
- `tools/find_missing_landmark.py` supprimé -- il traînait encore dans ce
  zip alors que ce document indiquait déjà sa suppression (référençait un
  module `numbering` qui n'existe plus). Rien de neuf, juste réappliqué.
- `error_reason` : valeurs françaises de `extraction/heavy/detection.py` et
  `extraction/light/detection.py` traduites (`aucune_detection` ->
  `no_detection`, `pas_de_masque` -> `no_mask`, `obb_degenere` ->
  `degenerate_obb`, `aspect_ratio_insuffisant` -> `insufficient_aspect_ratio`)
  -- pas d'autre code ne dépendait de la valeur exacte (vérifié par grep),
  et par convention seules les nouvelles lignes écrites changent de libellé.
- 60 tests toujours verts après coup (aucun test n'a eu besoin d'être
  modifié -- la traduction n'a touché aucune signature consommée par les
  tests existants).
- Deux bugs latents corrigés au passage (détail dans "Bugs trouvés") :
  `tools/clean_tps.py` (désync avec la signature actuelle de `parse_tps`)
  et `analysis/compare_runs.py` (`--family` jamais branché, import mort).

## Non traité -- décision à prendre avant de continuer

- Le reste de `landmarks_trainer/` (`train.py`, `evaluate.py`,
  `checkpoint.py`, `augment.py`, `heatmap.py`, `model.py`, `dataset.py`,
  `constants.py`) est en anglais mais **ne suit pas** les conventions CLI/
  logging validées (`main(argv=None)`, `logger`, `add_logging_args`) --
  `train.py` vérifié explicitement, les autres probablement pareil mais pas
  vérifiés un par un. Ni `TODO.md` ni la session précédente ne mentionnaient
  ce point -- `RESUME.md` disait juste "déjà en anglais", ce qui est vrai
  mais incomplet vis-à-vis de `CONVENTIONS.md`. `export_dataset.py` et
  `reproject_reference.py`, que j'ai touchés cette session, suivent
  maintenant pleinement les conventions -- ce qui crée une incohérence
  temporaire à l'intérieur même de `landmarks_trainer/`.

## Bugs / incohérences trouvés en cours de route

- ~~`tools/find_missing_landmark.py` cassé~~ -- supprimé (déjà noté avant,
  réappliqué).
- ~~`should_skip` divergent~~ -- unifié (session précédente).
- ~~`parse_tps` desync~~ -- corrigé (session précédente).
- **`tools/clean_tps.py` : bug réel corrigé.** Appelait
  `specimens = parse_tps(args.tps)` sans déballer le tuple
  `(specimens, errors)` -- cassé contre la signature actuelle de
  `parse_tps`. Corrigé en traduisant.
- **`analysis/compare_runs.py` : `FAMILY_LDA` importé mais jamais
  utilisé, `main()` codait `"lda"` en dur.** Ajouté un vrai flag
  `--family` qui utilise la constante. Comportement par défaut inchangé.
- **`manifest/build_dataset.py` : annotation de type incorrecte sur
  `load_roots_config`** (`-> list[dict]` alors que la fonction retourne le
  dict JSON tel quel, utilisé comme `roots["base_root"]`/`roots["splits"]`
  juste après). Pas corrigée -- juste une annotation, aucun effet à
  l'exécution, je ne voulais pas faire un changement non demandé sur un
  fichier déjà gros à traduire. Signalée ici seulement.
- **`tools/convert_landmarks.py` : le CSV de sortie annoncé
  (`biological_data_<split>.csv`) n'est en réalité jamais écrit** --
  `bio_output.to_csv(...)` est en commentaire (lignes ~669-673) alors que
  le résumé affiché à la fin du script prétend l'avoir écrit. Le script
  duplique aussi, à la main et de façon fragile (regex sur le texte brut
  du TPS), une bonne partie de ce que `utils.tps_io`/`utils.dataset` font
  déjà correctement. Voir "Non traité" ci-dessus -- pas touché en attendant
  ta décision.
- `extraction/heavy/`, `extraction/light/` : inspectés cette session,
  RAS niveau logique -- juste la langue + les valeurs `error_reason` à
  traduire (fait).

## Questions ouvertes

- ~~Streamlit vs Gradio pour l'outil 1~~ -- résolu (session du 2 sept.
  2026) : Streamlit. Voir "Détails outil 1" pour les raisons.
- **Nouvelle question : aligner `landmarks_trainer/train.py` (et le reste
  du dossier) sur les conventions CLI/logging maintenant, ou reporter à la
  Phase 4 (harmonisation des sorties de run) puisque ce dossier va de
  toute façon être retouché à ce moment-là ?**
- **(2 sept. 2026) App Streamlit unique** (page landmarking + page
  classification, point d'entrée unique pour les utilisateurs
  non-techniques) **ou deux apps séparées ?**
- **(2 sept. 2026) Le mode terrain garde-t-il un aperçu overlay** (image
  annotée, sans statut persistant/éditable) **pour vérification rapide
  avant lecture de la prédiction** -- y compris pour un ensemble ad hoc de
  plusieurs photos, pas seulement une seule ?
- **(2 sept. 2026) Nommage/structure exacte** du mode dataset vs mode
  terrain -- deux scripts distincts partageant le core, ou un seul script
  avec un flag de mode ? Provisoire, pas figé.
- **(2 sept. 2026) Nom définitif de la commande CLI de réconciliation** de
  la validation (`reconcile-review` utilisé comme nom provisoire dans ce
  document).
