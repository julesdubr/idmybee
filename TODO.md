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
- [x] Décider du sort de `utils/dataset.py`, `predictions.py`,
      `pipeline_io.py`, `run_io.py`, `cli.py`, `repair_images.py`,
      `tps_overlay.py` -- resté ouvert plusieurs sessions (voir RESUME.md
      "Où on en est" / "Questions ouvertes" pour le détail historique).
      **Tranché (session 8 sept. 2026)** : `dataset.py`/`predictions.py`/
      `pipeline_io.py`/`run_io.py` -> `core/` ; `repair_images.py` ->
      `tools/maintenance/` ; `cli.py`/`tps_overlay.py` restent dans
      `utils/`. Voir `CONVENTIONS.md` "Clôture de la question ouverte
      `utils/` -> `core/`" pour le raisonnement complet.
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

## Hors plan -- nettoyage identification / dataset propre (session 4 sept. 2026)

Besoin ad hoc de Jules (les CSV d'identification bruts avaient des
conflits de num_inventaire entre collections + désync avec des images
renommées) -- prérequis pratique avant Phase 2 (mode dataset), pas une
phase numérotée à part entière. Détail complet des décisions et des
résultats du test dans `RESUME.md` "Où on en est".

- [x] Diagnostic sur les CSV réels (`IDMB_Bombus_collect.csv`/`terrain.csv`
      + `manifest.csv` fournis par Jules) : 3 collisions d'identité réelles
      (`CD388229c`, `CD388821d`, `CD388861` -- même étiquette utilisée pour
      deux spécimens biologiquement différents), 5 doublons de ligne
      inoffensifs, 4 (puis 5 avec un cas terrain) images sans ligne
      d'identification, 12 lignes terrain sans image, 364 lignes
      collection sans image (CSV avec plus de spécimens que d'images
      réellement digitalisées -- normal, pas un bug)
- [x] Convention `inv_id`/`photo_id` définie -- voir `CONVENTIONS.md`
      "Identification des spécimens"
- [x] `data/identification/collection_origin_codes.csv` créé (11
      `collection_origin` -> code court `NOM_INVENTAIRE`) -- **à relire par
      Jules**, en particulier si `Arthropologia` et `Arthropologia Lyon`
      doivent rester deux inventaires séparés ou non
- [x] `src/manifest/identification.py` -- fonctions pures (restriction aux
      images présentes, résolution des conflits/doublons, dérivation
      `NOM_INVENTAIRE`, mapping gelé, table specimen, `photo_id`) --
      testé bout en bout sur les 3 fichiers réels fournis par Jules dans
      cet environnement (résultats détaillés dans `RESUME.md`)
- [x] `src/tools/export_clean_dataset.py` -- CLI orchestrateur (lecture
      seule sur les sources, copie renommée des images vers
      `--output-dir`, jamais de modification sur place) -- testé avec
      `--no-copy-images` (bout en bout, résultats vérifiés) et avec copie
      réelle activée (échec propre et tracé faute de disque externe monté
      dans cet environnement, voir `reports/copy_failures.csv`)
- [ ] **À faire par Jules** : premier run réel sur son disque (avec les
      images) pour générer le `inv_id_mapping.csv` définitif et le
      dataset propre à renvoyer à Adrien -- non testé avec les vraies
      images depuis cet environnement
- [ ] Une fois le dataset propre validé : le brancher comme nouvelle
      entrée de `tools/convert_landmarks.py --identification-csv` (remplace
      les CSV bruts) -- pourrait aussi permettre de simplifier/supprimer le
      dédoublonnage "1ère ligne gardée" déjà présent dans
      `convert_landmarks.py`, puisque le CSV propre n'aura plus de clé
      dupliquée

### Suite -- passage à "un run = une source" + manifest pipeline (session 5 sept. 2026)

Reprise du point ci-dessus suite aux retours de Jules : `identification.py`/
`export_clean_dataset.py` réécrits (un run = une source, plus de fusion
collection/terrain interne -- voir `CONVENTIONS.md` "Export propre par
source, un run = une source"), et `manifest/build_dataset.py` +
`tools/ingest_raw.py` livrés (rôle détaillé dans `CONVENTIONS.md`
"Statuts harmonisés..."). Testé bout en bout dans cet environnement avec
des fixtures synthétiques (collection + terrain -> export propre ->
manifest pipeline combiné) -- toujours pas testé avec les vraies données
de Jules.

- [x] `identification.py`/`export_clean_dataset.py` réécrits pour "un run
      = une source" -- `biological_data_all.csv` (tous les spécimens de la
      source) + `biological_data.csv` (avec image), `--image-group-by`
      générique pour le rangement des images copiées,
      `assign_nom_inventaire`/`parse_nom_inventaire_embedded` pilotés par
      la présence de `--origin-codes` plutôt que par le nom de la source
- [x] Conflits d'identité étendus aux spécimens sans image (résolution
      lancée sur la CSV d'identification complète, plus seulement sur les
      lignes ayant une image) -- couvre le cas remonté par Jules (même
      `num_inv`, deux `collection_origin` différents, un des deux côtés
      sans image)
- [x] Bug trouvé et corrigé : quand `--key-column` s'appelle littéralement
      `inv_id` (cas terrain), `build_specimen_table` supprimait la colonne
      brute au lieu de la renommer -- ne laissait que `nom_inventaire`
      (juste le préfixe de campagne) en guise d'identifiant. Corrigé
      (`<key_column>_raw` conservé), `inv_id` toujours en première colonne
      désormais
- [x] Bug trouvé et corrigé (fixture terrain sans aucun `device_type`
      renseigné) : colonne lue en `float64` par pandas, plantait sur
      l'assignation du `--default-device-type` -- cast explicite en
      `object` avant.
- [x] `tools/ingest_raw.py` créé (déplacé/simplifié depuis l'ancien
      `manifest/build_dataset.py` -- scan brut multi-convention seul,
      `specimens.csv`/jointure identification retirés, ce rôle est
      maintenant entièrement couvert par `export_clean_dataset.py`)
- ~~`manifest/build_dataset.py` réécrit pour opérer sur un ou plusieurs
      datasets déjà propres (...) -- config JSON multi-racines pour
      combiner collection + terrain en un seul manifest pipeline~~ --
      **remplacé (session du 6 sept. 2026) : `manifest/build_dataset.py`
      supprimé**, `tools/export_clean_dataset.py` produit déjà
      directement le manifest pipeline. Voir section "Suite -- manifest
      indépendant, `build_dataset` supprimé" plus bas.
- ~~**Question ouverte, pas tranchée unilatéralement** : `manifest/
      build_dataset.py` est `main(argv)` + lancement seul (...) -- à
      confirmer avec Jules ou déplacer.~~ -- **sans objet (session du
      6 sept. 2026)** : le fichier est supprimé, la question de son
      emplacement ne se pose plus.
- [ ] **À faire par Jules** : premier run réel sur son disque (ingest_raw
      -> export_clean_dataset [-> combine_manifests si plusieurs sources],
      sur les vraies données) -- seulement testé avec des fixtures
      synthétiques dans cet environnement jusqu'ici
- ~~Reste ouvert : les colonnes des `identification_csv` de plusieurs
      racines combinées dans `manifest/build_dataset.py` sont fusionnées
      par simple concat/outer-join (...)~~ -- **sans objet (session du
      6 sept. 2026)** : `manifest/build_dataset.py` supprimé,
      `tools/combine_manifests.py` fait le même concat/outer-join mais au
      niveau des `biological_data.csv` déjà propres de chaque source (pas
      de CSV bruts différents à réconcilier) -- question implicitement
      tranchée par la suppression du problème sous-jacent.
- [ ] Toujours reporté (accord du 5 sept. 2026) : shape/TPS de référence
      pour la renumérotation -- pas abordé cette session non plus, à
      traiter avec la conception concrète de l'outil 1 (Phase 2)

### Suite -- manifest indépendant, `build_dataset` supprimé (session 6 sept. 2026)

Retour de Jules après son premier run réel sur le terrain (bug `inv_id`
en dernière colonne, cf. entrée précédente) + reformulation du besoin :
dataset propre indépendant du brut (pas de `raw_path` dans le manifest),
manifests trop chargés (`naming`/`source_root` redondants), et
questionnement légitime sur l'utilité d'un `build_dataset` séparé une fois
qu'`export_clean_dataset.py` produit déjà un manifest. Détail complet des
nouveaux schémas dans `CONVENTIONS.md` "Manifest indépendant du brut,
`build_dataset` supprimé".

- [x] `manifest/build_dataset.py` supprimé -- son rôle (rescanner un
      dossier propre) est redondant avec la sortie déjà produite par
      `export_clean_dataset.py`
- [x] `tools/combine_manifests.py` créé : concatène les `manifest.csv`/
      `biological_data.csv` déjà propres de plusieurs runs
      `export_clean_dataset.py` -- aucun rescan disque, aucun hash, aucun
      parsing de nom de fichier. Détecte et lève une erreur explicite sur
      collision de `photo_id` entre sources combinées.
- [x] `manifest.csv` (renommé depuis `manifest_clean.csv`) rendu
      indépendant du brut : `raw_path` retiré, `content_hash`/
      `file_size_bytes` recalculés sur la copie fraîchement écrite (plus
      hérités du scan brut) -- sauf `--no-copy-images`, qui retombe sur
      les valeurs du scan brut et statut `SKIPPED` (premier usage réel de
      ce statut)
- [x] Colonnes retirées du manifest final : `specimen_id`/`shot_index`
      (redondants avec `inv_id`/`photo_id`), les colonnes bio jointes
      temporairement pour `--image-group-by` (genus/species/caste...) ne
      sont plus persistées -- uniquement `inv_id` comme clé de jointure
      vers `biological_data.csv`
- [x] `tools/ingest_raw.py` : `image_id` (doublon de `content_hash`),
      `source_root` et `naming` retirés du `manifest.csv` brut ;
      `config/roots.json` : clé `"splits"` -> `"roots"`, champ par racine
      `"split"` -> `"source_type"` (connotation train/test retirée)
- [x] `tools/export_clean_dataset.py` filtre maintenant son manifest
      d'entrée sur `--source-type` si une colonne `source_type` y est déjà
      présente -- un seul run `ingest_raw` peut scanner toutes les racines
      d'un coup, chaque run `export_clean_dataset` prend sa part sans
      pré-découpage manuel du manifest par Jules
- [x] Testé bout en bout dans cet environnement avec les fixtures
      existantes (collection + terrain) : `ingest_raw` (nouveau schéma) ->
      `export_clean_dataset` (collection et terrain séparément) ->
      `combine_manifests` -- toujours pas testé avec les vraies données de
      Jules
- [ ] **À vérifier par Jules** : `config/roots.json` existant(s) à migrer
      vers le nouveau schéma (`"splits"` -> `"roots"`, `"split"` ->
      `"source_type"` par racine) avant le prochain `ingest_raw`

### Suite -- conflits dédupliqués par identité, `collector` -> `photographer` (session 6 sept. 2026)

- [x] `resolve_identification` : le rapport de conflits ne renvoie plus
      qu'une ligne par variante d'identité distincte au sein d'un
      `specimen_key` (au lieu d'une ligne par ligne brute/device) --
      confirmé par Jules qu'un désaccord entre devices d'un même spécimen
      n'a pas de sens, seul un désaccord entre spécimens différents en a
      un. Testé avec une fixture 2 spécimens x 2 devices (4 lignes brutes)
      -- rapport correctement réduit à 2 lignes.
- [x] `collector`/`collector_subfolder` renommés `photographer`/
      `photographer_subfolder` (`tools/ingest_raw.py`, `config/roots.json`,
      `--image-group-by`, `manifest.csv`) -- terminologie terrain, pas
      collection de musée
- [ ] **À vérifier par Jules** : `roots.json` existant(s), deuxième
      renommage à faire en plus de `"splits"`->`"roots"` :
      `"collector_subfolder"` -> `"photographer_subfolder"`

## Phase 2 -- outil 1 : landmarking

Conception validée session du 2 sept. 2026 -- détail complet dans
`RESUME.md` "Détails outil 1". Résumé actionnable ci-dessous.

- [x] Framework UI : **Streamlit** (raisons dans `RESUME.md`)
- [x] Fonction core en mémoire `place_landmarks(image) -> Landmarks`
      (pure Python, pas de lecture/écriture disque) -- prérequis du mode
      terrain, voir `CONVENTIONS.md` "Fonctions core réutilisables".
      **Fait (session 7 sept. 2026, suite 8)** : `utils/landmarking_pipeline.py::place_landmarks`,
      détail dans `RESUME.md`.
- [x] Fonction core `draw_landmarks_overlay(image, landmarks) -> Image`
      (overlay annoté partagé CLI/UI). **Fait (session 7 sept. 2026, suite
      8)** : déjà couvert par `utils/tps_overlay.py::draw_landmarks`
      (existait déjà, pure, sans I/O) -- réutilisé tel quel plutôt que
      dupliqué.
- [x] **Mode dataset** -- orchestrateur bout-en-bout : dossier d'images ->
      détection -> normalisation -> prédiction UNet -> renumérotation ->
      TPS + log d'échecs ; + CSV de données biologiques si un
      manifest/specimens.csv est fourni. **Fait (session 7 sept. 2026,
      suite 5/6)** en CLI : `tools/train_dataset.py`/`tools/predict_dataset.py`
      (voir `PIPELINE.md` "Orchestrator scripts"). **Étendu (session
      8 sept. 2026)** : accessible aussi depuis l'UI, `app/build_dataset.py`
      pilote exactement les mêmes fonctions (`utils/landmarking_pipeline.py`),
      avec en plus les deux étapes de validation ci-dessous.
- [ ] Sortie configurable (mode dataset) : crops sauvegardés (o/n), TPS
      espace original / crop / les deux
- [ ] Référence de renumérotation (blueprint Tancrède) embarquée comme
      asset par défaut
- [x] **Mode terrain / single** -- une photo ou un petit ensemble ad hoc
      (sans manifest) -> aucun fichier écrit ; stub temporaire = affiche
      les coordonnées TPS calculées (zone de code copiable en UI, stdout
      en CLI) en attendant la Phase 3 (voir item correspondant Phase 3).
      **Fait (session 7 sept. 2026, suite 8), stub sauté** : la Phase 3
      (classification) a été branchée directement plutôt que de passer par
      un stub TPS-only, voir `app/single_image.py` et l'item Phase 3
      correspondant.
- [x] Statut auto `FAILED`/`SUSPECT`/`OK` calculé pendant le run, colonne
      `status`, identique CLI et UI (détail des critères dans
      `RESUME.md`). Déjà vrai côté fichiers depuis plusieurs sessions
      (`crops.csv`, `landmarks_numbered.csv`) ; **le "identique CLI et UI"
      est maintenant vérifié pour de vrai (session 8 sept. 2026)** --
      `utils/review.py` lit ces mêmes statuts pour construire la review,
      qu'elle soit affichée par `app/build_dataset.py` ou par
      `tools/pipeline/export_review.py`.
- [x] UI -- étape de validation (mode dataset) : galerie annotée
      (landmarks numérotés) + statut éditable OK/SUSPECT/FAILED,
      navigable, export final. **Fait (session 8 sept. 2026)** :
      `app/build_dataset.py`, deux étapes de validation (recadrage puis
      landmarks) -- tableau filtrable/éditable (`st.data_editor`,
      colonne `reviewed_status`) + aperçu (crop ou overlay de landmarks
      numérotés) pour la ligne sélectionnée. "Galerie" au sens strict
      (une vignette par photo) pas retenue au profit d'un tableau +
      aperçu à la demande -- ne passe pas à l'échelle pour un dataset de
      plusieurs milliers de photos (la collection en a ~2600), voir
      `RESUME.md` pour le détail du choix.
- [x] CLI -- parité fonctionnelle de la validation : `--export-review`
      (overlays + `review.csv`) et une commande de réconciliation (nom
      provisoire `reconcile-review`) qui réapplique un `review.csv` édité
      à la main au TPS/CSV final. **Fait (session 8 sept. 2026)** :
      `tools/pipeline/export_review.py` (écrit les CSV de review + overlays
      optionnels) et `tools/pipeline/reconcile_review.py` (réapplique),
      tous deux au-dessus de `utils/review.py` -- même implémentation que
      l'UI, voir `CONVENTIONS.md` "Fonctions core réutilisables". Nom
      final `reconcile_review.py`/`export_review.py` (pas de tiret, pour
      rester cohérent avec le nommage `python -m tools.pipeline.xxx` du
      reste du paquet -- `reconcile-review` était un nom de COMMANDE
      envisagé à l'époque d'une éventuelle CLI unifiée avec sous-commandes,
      jamais construite).
- [x] Décider (question ouverte, voir `RESUME.md`) : le mode terrain
      garde-t-il un aperçu overlay sans statut persistant ? **Décidé
      (2 sept. 2026, Phase 1) : oui**, aperçu overlay conservé en mode
      terrain.
- [x] Décider (question ouverte) : app Streamlit unique (outil 1 + outil
      2) ou deux apps séparées ? **Décidé (2 sept. 2026, Phase 1) : une
      seule app.** Focus immédiat sur le landmarking ; la classification
      (outil 2) s'ajoute plus tard dans la même app, de façon
      incrémentale.
      **Révisé (session 8 sept. 2026)** : finalement DEUX apps distinctes
      -- `app/single_image.py` (terrain, une photo, déjà là depuis la
      session 7) et `app/build_dataset.py` (nouveau, dataset -- préparer,
      valider, entraîner ou prédire). Les deux usages se sont avérés assez
      différents dans leur flux (une photo + résultat immédiat, vs. un
      assistant à plusieurs étapes avec deux points de validation) pour
      qu'une seule app avec un mode caché aurait ajouté de la complexité
      sans bénéfice réel ; ils partagent déjà le code qui compte
      (`utils/landmarking_pipeline.py`, `classifiers/predict.py`,
      `core/run_io.py::model_display_name`), donc rien n'est dupliqué.
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
- [ ] L'ANOVA/variance reste **CLI uniquement** (réservé aux devs) --
      décision confirmée session du 2 sept. 2026, toujours valable pour
      `analysis/variance_report.py`.
      **`train.py`, en revanche, révisé (session 8 sept. 2026)** : demande
      explicite d'une UI couvrant "préparer un jeu de données, construire
      un modèle de référence, et prédire dessus" (scénario 1) -- construire
      un modèle est maintenant possible depuis `app/build_dataset.py`
      (nom du modèle, niveau espèce/caste), en plus de la ligne de
      commande. Reste réservé aux devs : l'ANOVA/variance (ci-dessus) et
      les options fines de `classifiers/train.py` non exposées dans l'UI
      (`--lda-components` l'est, mais pas par ex. `--devices`/`--species`/
      `--castes`/`--tps` -- l'UI couvre le cas courant, la CLI reste l'outil
      complet).
- [x] UI Streamlit (batch + single) pour **predict** : sélection d'un
      modèle LDA pré-fitté parmi les runs publiés (liste curatée depuis
      `data/models/lda/<run_id>/train/`). **Single fait (session 7 sept.
      2026, suite 8)** : `app/single_image.py`. **Batch fait (session
      8 sept. 2026)** : `app/build_dataset.py`, objectif "Predict with an
      existing model" -- sélection du modèle par nom (voir `--model-name`
      dans `CONVENTIONS.md`), résultats affichés en tableau +
      top-1/top-3 si vérité connue.
- [x] Brancher ce predict dans le **mode terrain de l'outil 1** (remplace
      le stub TPS-only du mode single/terrain, voir Phase 2) -- retourne
      un top-N espèce/caste + % de confiance, appel direct fonction à
      fonction, pas de TPS intermédiaire sur disque. **Fait (session
      7 sept. 2026, suite 8)** : `app/single_image.py` construit un
      `ImageLandmarks` en mémoire depuis `place_landmarks()` et appelle
      `classifiers.predict.predict_specimens` directement -- aucun TPS
      écrit. Seul le **mode single** de cet item est fait ; l'UI Streamlit
      **batch** pour predict (item précédent) reste à faire séparément.

## Phase 4 -- outil 3 : trainers

- [ ] Harmoniser les 3 conventions de sortie
      (`run_io.py` / `landmarks_trainer/checkpoint.py` / natif Ultralytics)
- [ ] Auditer `obb_trainer/` pour la convention de stockage
- [ ] (Si reporté depuis Phase 0) Aligner `landmarks_trainer/` sur les
      conventions CLI/logging
- [ ] **Nouveau (session du 7 sept. 2026)** : `obb_trainer/prepare_dataset.py`/
      `validate_crops.py` référencent encore un schéma `crops.csv`/
      `detection.csv` (`image_id`/`specimen_id`) qui ne correspond déjà
      plus aux colonnes actuelles d'`extraction/extraction_io.py` --
      semblent déconnectés du pipeline actuel indépendamment du
      renommage `photo_id`/`inv_id` fait cette session (non touchés,
      hors périmètre de ce refactor). À auditer en même temps que le
      point ci-dessus.

## Hors plan -- unification Pipeline A / Pipeline B sur `photo_id`/`inv_id` (session 7 sept. 2026)

Détail complet des décisions dans `RESUME.md` "Où en est-on" (session du
7 sept., "suite 3") et `PIPELINE.md` (nouveau, guide stage par stage).
Demande de Jules : combler l'écart entre le nouveau pipeline dataset
propre (`manifest/identification.py`/`tools/ingest_raw.py`/
`tools/export_clean_dataset.py`) et l'ancien pipeline extraction/
landmarks/classification, qui attendait un schéma différent
(`data/Bombus/`, `image_id`/`specimen_id`/`split` en colonne) -- pas une
couche d'adaptation, les fichiers eux-mêmes modifiés pour utiliser
`photo_id`/`inv_id` de bout en bout.

- [x] `core/tps_io.py` -- `ImageLandmarks.photo_id`/`.inv_id`,
      `assign_sequential_ids` remplace `image_id_to_sid`/`sid_to_image_id`
      (COMMENT= devient le seul mécanisme de jointure, plus de repli par
      hash)
- [x] `extraction/extraction_io.py`, `detect_wing.py`, `normalize_crop.py`
      -- schéma `photo_id`/`inv_id`/`path`, `--splits-csv` (nouveau,
      partagé), `build_output_path` simplifié
- [x] `landmarks/predict.py` -- fix resume/checkpoint (indexé par
      `photo_id`, plus par un `tps_id` recalculé par hash)
- [x] `landmarks/renumber.py` -- `--specimens` -> `--biological-data`,
      lit `biological_data.csv` (plus de colonne `is_labeled`)
- [x] `utils/dataset.py` -- réécriture du contrat de jointure ; bug de
      collision de nom trouvé et corrigé (`device` réutilisé pour
      *device_type*, collision avec la vraie colonne `device` ajoutée
      plus tôt dans la session -- voir `RESUME.md`)
- [x] `utils/cli.py` (`--splits-csv`), `utils/predictions.py`,
      `analysis/variance_report.py` (`LEVEL_CHOICES`)
- [x] `tools/convert_landmarks.py` supprimé, remplacé par
      `tools/export_final_landmarks.py` -- fusion `--identification-csv`
      obsolète supprimée, colonnes `species`/`caste` (commentées dans
      l'ancien fichier) réinstaurées
- [x] `classifiers/train.py`/`predict.py` audités -- confirmé aucun
      changement nécessaire (dépendance au schéma déjà isolée dans
      `utils/dataset.py`)
- [x] Fichiers hors plan initial corrigés en cours de route (auraient
      cassé avec `AttributeError`) : `tools/verify_tps.py`,
      `utils/tps_overlay.py`, `landmarks_trainer/export_dataset.py`
- [x] `PIPELINE.md` créé -- guide complet stage par stage
- [x] `tests/test_dataset.py` créé (6 tests), `tests/test_tps_io.py`/
      `test_pipeline_io.py` complétés. 77 tests verts, `py_compile` sur
      tout `src/` OK, smoke-test fonctionnel `extraction_io`/
      `normalize_crop` sur manifest.csv synthétique, `--help` vérifié
      sans crash sur les 8 scripts CLI modifiés
- [ ] **À faire par Jules** : premier run réel de bout en bout
      (vraies images + modèle UNet réel) sur un dataset issu
      d'`export_clean_dataset.py` -- seulement testé via tests
      synthétiques + smoke tests dans cet environnement, aucun accès à
      un modèle UNet entraîné ni à de vraies images ici
- [ ] `data/Bombus/` (dataset historique, ancien schéma) : incompatible
      avec le code actuel -- coupure nette assumée (voir "Migration/
      compatibility note" dans le plan de session), à régénérer depuis
      `data/clean/` ou laisser de côté comme référence historique

## `original_id` conditionné à `--origin-codes` + crash réel confirmé (session 7 sept. 2026, suite 4)

Détail complet dans `RESUME.md` "Où en est-on" (session du 7 sept.,
"suite 4"). Jules a lancé la vraie commande sur la collection et pris un
`KeyError: 'original_id'` -- confirme que la migration "À faire par Jules"
listée plus haut (renommer `specimen_id` -> `original_id` dans les
`manifest.csv` déjà produits par `tools/ingest_raw.py`) n'est pas encore
faite ; ce n'est pas un bug de code (vérifié en testant sur une copie avec
l'en-tête renommé, ça tourne).

- [x] `attach_canonical_inv_id`/`build_specimen_table` : nouveau paramètre
      `keep_original_id`, câblé dans `tools/export_clean_dataset.py`
      (`origin_codes is not None`) -- `original_id`/`inv_id_raw` n'apparaît
      plus dans `biological_data*.csv`/les rapports quand
      `--origin-codes` est omis (là, `inv_id == original_id` par
      construction, colonne dupliquée)
- [x] Print de résumé (`Identity conflicts: ... specimen(s)`) corrigé pour
      ne plus dépendre de `conflicts['original_id']`, absente dans ce cas
- [x] Docstrings de `manifest/identification.py`/
      `tools/export_clean_dataset.py` nettoyées suivant `CONVENTIONS.md`
      (exemples d'usage à jour, rationale élaguée)
- [x] Testé sur les deux sources réelles (collection + terrain, copie
      scratch avec en-tête renommé) : 79 tests verts, `py_compile` OK
- [ ] **À faire par Jules (toujours en attente)** : re-lancer
      `tools/ingest_raw.py` (ou renommer l'en-tête à la main) sur
      `data/bombus_collection_raw/manifest.csv` et
      `data/bombus_terrain_raw/manifest.csv` -- bloque toute commande
      réelle tant que ce n'est pas fait

## Suppression de `--split`/`--splits-csv` + orchestrateurs collection/terrain (session 7 sept. 2026, suite 5)

Trois demandes de Jules, détail complet des décisions dans `RESUME.md`
"Où en est-on" (session du 7 sept., "suite 5") et `PIPELINE.md` (mis à
jour) : le concept train/test `--split`/`--splits-csv` n'est plus
nécessaire (les jeux de données sont maintenant séparés par racine --
collection pour entraîner, terrain pour prédire -- plus par split d'un même
jeu de données) ; un script bout-en-bout pour construire le modèle de
référence depuis `data/Bombus/collection` ; un script bout-en-bout pour
classifier `data/Bombus/terrain` avec ce modèle.

- [x] `--split`/`--splits-csv` retirés de tout le pipeline : `utils/cli.py`
      (`add_dataset_args`/`dataset_kwargs`, `resolve_split` supprimée),
      `utils/pipeline_io.py` (`load_splits` supprimée), `utils/dataset.py`
      (`load_dataset` -- plus de colonne `split` dans `meta_df`),
      `extraction/detect_wing.py`, `extraction/extraction_io.py`
      (`select_images`), `landmarks/predict.py` (`load_target_crops`),
      `tools/export_final_landmarks.py` (renommage des sorties sans
      suffixe split : `landmarks_<n>lm_crop.tps`/`_original.tps`,
      `biological_data.csv`, `failed.csv` -- un run = un dataset root
      entier, plus de découpage), `classifiers/train.py`,
      `classifiers/predict.py`, `analysis/variance_report.py`,
      `analysis/classification_report.py`. **`obb_trainer/`/
      `landmarks_trainer/dataset.py` non touchés** : leur `--split`
      train/val/test est un concept ML différent (découpage d'entraînement
      du détecteur/UNet), sans rapport avec le split biologique du
      pipeline -- déjà hors périmètre du refactor `photo_id`/`inv_id`
      (voir Phase 4).
- [x] **`utils/run_io.py` : `split` remplacé par `dataset_label`** dans
      `build_run_id`/`build_eval_tag`/`build_variance_id` (le nom de la
      racine du dataset, ex. `Path(dataset).name` -> `"collection"`/
      `"terrain"`, remplace le rôle que jouait `split` pour distinguer les
      runs -- ex. `run_id="species_collection"` au lieu de
      `"species_train"`, `eval_tag="terrain"` au lieu de `"test"`).
- [x] **`core/model_io.py` : `TrainedModel.split` remplacé par
      `TrainedModel.dataset_label`** (même rôle informatif, ex. dans
      `classifiers/predict.py::_print_model_info`). **Casse les
      `model.joblib` déjà entraînés** (l'ancien champ `split` n'existe
      plus sur la classe) -- pas un bug, un cutover net comme les
      précédents de ce refactor ; retrainer avec
      `tools/build_collection_reference.py`.
- [x] Tests mis à jour : `tests/test_pipeline_io.py` (tests `load_splits`
      supprimés), `tests/test_dataset.py` (tests split supprimés),
      `tests/test_run_io.py` (signatures `dataset_label` à la place de
      `split`). 73 tests verts (hors `test_outliers.py`, qui échoue à la
      collecte dans cet environnement Python 3.9 -- syntaxe `int | None`
      sans `from __future__ import annotations`, limitation d'environnement
      préexistante et sans rapport avec cette session, le projet exige
      Python >=3.11).
- [x] **`tools/build_collection_reference.py` créé** : orchestrateur
      bout-en-bout (détection -> crop -> landmarks -> renumérotation ->
      `classifiers.train`) sur `data/Bombus/collection` par défaut, produit
      le modèle de référence (`model.joblib`). Appelle le `main(argv)` de
      chaque étage en process (pas de sous-processus, pas de logique
      dupliquée). `--reference` par défaut déduit de `--n-landmarks`
      (`data/references/shapes/reference_shape_<n>.npz`, déjà gelé --
      `landmarks/build_reference.py` n'a pas besoin d'être relancé).
      `--detector-model` par défaut `data/models/yolon_obb/best.pt`.
- [x] **`tools/predict_terrain.py` créé** : même chaîne (détection -> crop
      -> landmarks -> renumérotation) sur `data/Bombus/terrain` par défaut,
      puis `classifiers.predict batch` avec un `--model` déjà entraîné
      (typiquement celui produit par `build_collection_reference.py`).
- [x] `PIPELINE.md` mis à jour : section `splits.csv` (ancien stage 6)
      supprimée, stages renumérotés, section "Orchestrator scripts"
      ajoutée, "Known gaps" mis à jour (note sur la casse des anciens
      `model.joblib`, `data/Bombus/collection`/`terrain` comme racines
      actuelles).
- [x] ~~**À faire par Jules** : premier run réel...~~ -- **fait (suite 6)** :
      run réel réussi sur `data/Bombus/collection` puis `data/Bombus/terrain`,
      voir section suivante.
- [ ] Anciens runs `data/models/lda/species_train_*` (schéma `split`
      pré-refactor, sur l'ancien `data/Bombus/` maintenant remplacé) :
      obsolètes, à laisser de côté ou supprimer une fois le nouveau modèle
      de référence validé -- non touchés cette session.

## Renommage générique + factorisation `utils/landmarking_pipeline.py` (session 7 sept. 2026, suite 6)

Retour de Jules sur les deux orchestrateurs de la suite 5 : plus de noms
spécifiques à "collection"/"terrain", entrées/sorties trop éparpillées,
sorties d'`export_final_landmarks.py` à organiser dans le script lui-même.
Détail complet dans `RESUME.md` "Où en est-on" (session du 7 sept.,
"suite 6") et `PIPELINE.md` ("Orchestrator scripts").

- [x] **Renommage, comportement générique** : `tools/build_collection_reference.py`
      -> `tools/train_dataset.py`, `tools/predict_terrain.py` ->
      `tools/predict_dataset.py`. Le positionnel `dataset` n'a plus de
      valeur par défaut (avant : `data/Bombus/collection`/`data/Bombus/terrain`
      en dur) -- fonctionne sur n'importe quel dataset propre
      (`manifest.csv` + `biological_data.csv` + images).
- [x] **`utils/landmarking_pipeline.py` créé** : factorise la chaîne des
      4 étapes (détection -> crop -> landmarks -> renumérotation),
      auparavant dupliquée dans les deux scripts -- `add_landmarking_args`
      (tous les flags des étapes 2-6 déclarés une seule fois),
      `run_landmarking` (étapes 2-5), `run_export` (étape 6).
- [x] **`tools/export_final_landmarks.py` : `--output-dir` optionnel**,
      défaut `<dataset>/export/` via `utils.pipeline_io.dataset_export_dir`
      -- fini les chemins `out/collection/19` inventés à la main à chaque
      run ; `tools/train_dataset.py`/`tools/predict_dataset.py` appellent
      maintenant systématiquement cette étape d'export en plus de
      l'entraînement/la classification.
- [x] **Premier run réel de bout en bout, vraies données** :
      `tools/train_dataset.py data/Bombus/collection` (2600 spécimens, 19
      landmarks) -> LOOCV top-1 = 94.77 %, top-3 = 98.73 % ; puis
      `tools/predict_dataset.py data/Bombus/terrain` avec ce modèle -> 265
      spécimens évalués, top-1 = 82.26 %, top-3 = 96.98 %. Confirme
      `dataset_label`/`run_id`/`eval_tag` (suite 5) en conditions réelles
      (`run_id="species_collection"`, `eval_tag="terrain"`).
- [x] `PIPELINE.md` mis à jour (nouveaux noms, section "Orchestrator
      scripts" enrichie, résultats du run réel, stage 6 documente le
      nouveau défaut `<dataset>/export/`).
- [x] Testé : 73 tests toujours verts, `py_compile` OK sur tout `src/`.

## Manifest découplé du nettoyage, `build_manifest.py` créé, symlinks supprimés (session 7 sept. 2026, suite 7)

Retour de Jules : l'étape "pré-pipeline" (copier les images localement puis
les symlinker vers un disque externe, `tools/link_external_data.py`) jugée
confuse, et `tools/export_clean_dataset.py` jugé faire deux métiers à la
fois (nettoyage d'identité + construction du manifest). Détail complet des
décisions dans `RESUME.md` "Où on en est" (session du 7 sept., "suite 7")
et `CONVENTIONS.md` "Manifest découplé du nettoyage, `build_manifest.py`
créé".

- [x] `src/manifest/build.py` créé -- validation structurelle pure
      (colonnes obligatoires, vérification/hash des fichiers image,
      `photo_id` dupliqué, cohérence des colonnes biologiques par
      `inv_id`, dérivation `photo_id`/`photo_index` séquentielle quand
      absents) sur un CSV par photo générique, jamais lié à une convention
      de nommage particulière -- volontairement sans résolution de conflit
      d'identité (ça reste le rôle d'`export_clean_dataset.py`).
- [x] `src/tools/build_manifest.py` créé -- CLI toujours lancée, seul point
      d'entrée réel vers la suite du pipeline. Produit `manifest.csv`/
      `biological_data.csv` si tout valide OK, sinon `manifest_raw.csv`
      seul (+ `reports/biological_inconsistencies.csv` si le problème est
      une incohérence biologique) -- jamais de crash, diagnostic
      inspectable.
- [x] `tools/export_clean_dataset.py` : ne produit plus `manifest.csv`/
      `biological_data.csv` -- sortie renommée `dataset.csv` (une ligne par
      photo copiée avec succès, colonnes biologiques déjà fusionnées),
      compatible tel quel avec `build_manifest.py`. `biological_data_all.csv`
      inchangé.
- [x] `src/tools/combine_manifests.py` créé -- **corrige une dérive
      documentation/code** : ce fichier était marqué `[x]` fait depuis la
      suite 6 (et avant) mais n'existait jamais réellement dans le dépôt.
      Concat pur de `manifest.csv`/`biological_data.csv` déjà produits par
      `build_manifest.py`, erreur explicite sur collision `photo_id`/
      `inv_id` entre racines combinées.
- [x] `tools/link_external_data.py` et `config/derived_root.json`
      supprimés -- plus de raison d'être : `build_manifest.py` ne copie
      jamais de fichier, `export_clean_dataset.py --output-dir` écrit déjà
      directement sur un volume externe sans copie locale intermédiaire.
- [x] `PIPELINE.md` : section "External storage" remplacée par "Dataset
      roots can live anywhere on disk", stage 1 réécrit en 1a (nettoyage,
      optionnel) / 1b (`build_manifest.py`, toujours lancé), diagramme de
      flux mis à jour.
- [x] `tests/test_manifest_build.py` créé -- couvre les fonctions pures de
      `manifest/build.py`.
- [ ] **À faire par Jules** : premier run réel de bout en bout sur les
      vraies données (`export_clean_dataset.py` -> `build_manifest.py`,
      collection + terrain) -- seulement vérifié par tests synthétiques
      dans cet environnement.

## Outil de classification à une image (UI) -- fait (session 7 sept. 2026, suite 8)

Demande explicite de Jules (suite 5) : garder en tête l'objectif d'un petit
outil UI pour essayer de classifier depuis une image unique. Déjà décrit en
détail dans Phase 2/Phase 3 ci-dessus (mode terrain/single de l'outil 1,
branché sur `classifiers/predict.py single` -- voir `RESUME.md` "Détails
outil 1") : rien de nouveau à concevoir, ce fil est confirmé toujours
pertinent maintenant que `tools/train_dataset.py` donne un chemin concret,
validé sur données réelles (suite 6), pour obtenir le `model.joblib` que
cet outil UI consommera.

**Fait** : `app/single_image.py` (`streamlit run app/single_image.py`) --
voir la section détaillée dans `RESUME.md` ("Fait (session, 7 sept. 2026,
suite 8)").

## Interface Scénario 1, orchestrateur étape 0, review de validation, `core`/`tools` réorganisés, `--model-name` (session 8 sept. 2026)

Demande de Jules : un orchestrateur CLI pour l'étape 0 (ingestion brute),
en profitant pour clarifier `tools/`/`utils/` ; une interface pour le
scénario 1 (préparer un jeu de données, construire un modèle de référence,
prédire dessus), indépendante de `single_image.py`, avec deux étapes de
validation (recadrage, landmarks) à statut éditable ; pouvoir nommer
clairement le modèle produit ; mettre à jour les `.md`. Détail complet de
chaque décision dans les sections déjà référencées ci-dessus/dans
`CONVENTIONS.md`/`PIPELINE.md` -- résumé actionnable ici :

- [x] **Clôture de la question ouverte `utils/` -> `core/`** (Phase 1) et
      **découpage de `tools/`** en `ingestion/`/`pipeline/`/`maintenance/`
      -- voir `CONVENTIONS.md` "Clôture de la question ouverte...".
- [x] **`tools/ingestion/prepare_dataset.py`** : orchestrateur étape 0,
      config JSON, chaîne `ingest_raw` (optionnel) -> `export_clean_dataset`
      (par source "raw") -> `build_manifest` (toujours) ->
      `combine_manifests` -- voir `PIPELINE.md` "Stage 0 orchestrator".
- [x] **`utils/review.py`** + `tools/pipeline/export_review.py`/
      `reconcile_review.py` (parité CLI) : validation recadrage/landmarks,
      statut éditable OK/SUSPECT/FAILED, réconciliation transparente pour
      la suite du pipeline (`--crops-csv`, `--landmarks-status-csv`) --
      voir `PIPELINE.md` "Validation review" et `CONVENTIONS.md`.
- [x] **`app/build_dataset.py`** : assistant Streamlit en 7 étapes
      couvrant le scénario 1 de bout en bout (choix objectif -> dataset ->
      paramètres -> détection/crop -> **validation recadrage** ->
      landmarks -> **validation landmarks** -> export -> construction du
      modèle ou prédiction), indépendant de `single_image.py` -- voir
      `README.md` "Scénario 1" et `RESUME.md` pour le détail de conception
      (dont le choix `st.data_editor` + aperçu plutôt qu'une vraie galerie
      d'images, pour rester utilisable sur ~2600 photos).
- [x] **`--model-name`** (`classifiers/train.py`, relayé par
      `tools/pipeline/train_dataset.py` et `app/build_dataset.py`) :
      nom lisible optionnel stocké dans le modèle (`TrainedModel.model_name`)
      et ses métriques, affiché par un sélecteur de modèle
      (`core.run_io.model_display_name`) à la place du `run_id` technique
      -- répond directement à "les .joblib ont des noms abstraits" (ex.
      `species_collection` ne dit pas "identification bourdons à
      abdomen rouge"). Purement descriptif, n'affecte jamais le chemin.
- [x] `.md` mis à jour : `README.md` (scénario 1 enrichi, section 6, point
      d'attention sur les seuils de validation), `PIPELINE.md` (nouvelles
      sections "Stage 0 orchestrator"/"Validation review"/"Model naming",
      "Orchestrator scripts" et "Known gaps" mis à jour), `CONVENTIONS.md`
      (clôture de la question `core/`, conventions review/`--model-name`),
      `TODO.md`/`RESUME.md` (ce document).
- Testé : 109 tests verts (95 existants + 7 `tests/test_review.py` + 5
  `tests/test_prepare_dataset.py` -- le décompte grimpe encore avec les 2
  tests de régression ajoutés après le bug ci-dessous), `py_compile` sur
  tout `src/`+`app/`, `--help` sans crash sur chaque script CLI déplacé ou
  créé, `app/build_dataset.py` exercé via `streamlit.testing.v1.AppTest`
  (rendu de chaque étape + interactions clé : construction du manifest,
  soumission des paramètres pipeline, review recadrage avec écriture du
  fichier réconcilié, review landmarks avec aperçu overlay réel, export
  réel sur un jeu de données synthétique).
- **Bug réel trouvé et corrigé pendant les tests** (pas seulement un test
  qui aurait pu être vert par chance) : `utils/review.py::write_crop_review`/
  `write_landmarks_review` plantaient (`pandas.errors.LossySetitemError`)
  dès que TOUTES les valeurs de `error_reason` du fichier d'origine
  étaient vides -- pandas lit alors la colonne en `float64` (NaN) plutôt
  qu'en texte, et y assigner une chaîne lève une exception. Cas réel très
  probable (un dataset sans aucun échec avant la review). Corrigé par un
  cast explicite en `object` avant assignation ; test de régression ajouté
  pour les deux fonctions (`tests/test_review.py`).
- **À faire par Jules** : premier run réel de `app/build_dataset.py` sur
  un vrai dataset (collection ou terrain) avec les vrais poids UNet/YOLO-OBB
  -- testé dans cet environnement via `AppTest` sur un jeu synthétique
  (2 spécimens, 3 landmarks) pour valider l'orchestration et la review,
  pas encore au clic dans un vrai navigateur sur un jeu réel de plusieurs
  milliers de photos (où la pagination/le filtre du tableau de review
  n'ont encore jamais été éprouvés en pratique).

## Non planifié / à discuter

- ~~`src/manifest/io.py` : confirmé mort (...). À supprimer à l'occasion~~
  -- **fait (session 8 sept. 2026)**, supprimé au passage du tri
  `core/`/`utils/`/`tools/` (voir `CONVENTIONS.md`).
- Tests de non-régression avec données réelles (redemander le
  micro-dataset le moment venu)
- Suppression définitive du contenu de `tools/` une fois ses scripts
  utiles migrés dans les 3 outils
- Fixer l'annotation de type incorrecte sur `load_roots_config` dans
  `manifest/build_dataset.py` (`-> list[dict]` au lieu de `-> dict`,
  cosmétique, aucun effet à l'exécution)
- ~~Template de normalisation des colonnes des CSV d'identification~~ --
  **traité (même session, 4 sept. 2026)**, voir la nouvelle section
  "Hors plan -- nettoyage identification / dataset propre" plus haut. Pas
  un schéma de colonnes commun forcé (l'idée initiale ci-dessous) mais une
  granularité specimen propre avec `inv_id` stable + colonnes spécifiques
  à chaque source laissées telles quelles (voir `CONVENTIONS.md`
  "Identification des spécimens"). Ancienne note conservée pour mémoire :
  *"CSV d'identification (`IDMB_Bombus_collection.csv` vs
  `IDMB_Bombus_terrain.csv` : schémas incompatibles aujourd'hui à part
  `num_inv`/`Numéro d'inventaire` + espèce/caste) -- évoqué par Jules et
  Adrien (session 4 sept. 2026), pas à faire maintenant. En attendant :
  `tools/convert_landmarks.py --identification-csv` passe les colonnes du
  CSV donné telles quelles (sans renommage), et refuse `--split all` (les
  deux fichiers ne peuvent pas être fusionnés dans un même export tant que
  le template n'existe pas)."* Le refus de `--split all` dans
  `convert_landmarks.py` reste valable en l'état -- pas retouché cette
  session, `convert_landmarks.py` prend encore les CSV bruts en entrée
  jusqu'à ce que le nouveau CSV propre soit branché (voir item
  correspondant dans la nouvelle section).
- `extraction/normalize_crop.py` (et potentiellement `detect_wing.py`)
  devrait persister les paramètres réellement utilisés à l'exécution
  (`--padding`, `--out-width`, `--out-height`, `--mode`) -- ex. colonnes
  ajoutées à `crops.csv`, ou un `params.json` à côté -- plutôt que de
  compter sur le fait qu'un script en aval (comme `tools/
  convert_landmarks.py` pour la réprojection crop -> original) les
  redonne en argument par convention et espère qu'ils correspondent à ce
  qui a servi à produire les crops. Repéré (session 4 sept. 2026) en
  repassant sur les valeurs par défaut de `convert_landmarks.py`
  (0.10/512/256) sans moyen de les vérifier automatiquement contre le run
  réel -- source d'erreur silencieuse (géométrie fausse sans exception).