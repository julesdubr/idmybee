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

*Voir `PIPELINE.md` (créé session du 7 sept. 2026) pour le détail exact
stage par stage -- entrées/sorties/CLI/statuts -- du pipeline unifié
(dataset propre -> extraction -> landmarks -> classification), maintenant
que Pipeline A et Pipeline B partagent le même schéma `photo_id`/`inv_id`.*
 
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

*Révisé session du 8 sept. 2026* : deux points ci-dessus ont finalement
changé à l'usage (détail complet dans "Fait (session, 8 sept. 2026)"
plus bas) -- (1) l'outil 1 (landmarking) et l'outil 2 (classification)
vivent dans **deux apps Streamlit séparées** (`app/single_image.py`,
`app/build_dataset.py`), pas une seule comme envisagé au point 1 ; (2)
**construire un modèle de référence (train) est maintenant aussi possible
depuis l'UI** (`app/build_dataset.py`), pas seulement en CLI comme prévu
au point 2 -- l'ANOVA/variance, elle, reste CLI uniquement comme prévu.
Le point 4 (`core/`) est acté : voir `CONVENTIONS.md` "Clôture de la
question ouverte `utils/` -> `core/`".

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

Fait (session, 8 sept. 2026) -- demande en plusieurs volets : un
orchestrateur CLI pour l'étape 0 (ingestion brute), en profitant pour
clarifier `tools/` (découpage possible en sous-modules) et `utils/`
(question ouverte depuis la Phase 1) ; une interface pour le scénario 1
(préparer un jeu de données, construire un modèle de référence, prédire
dessus), indépendante de `single_image.py`, avec deux étapes de validation
(recadrage, placement des landmarks) à statut éditable OK/SUSPECT/FAILED ;
pouvoir nommer clairement le modèle produit (les `.joblib` actuels ont des
noms abstraits, ex. `species_collection` ne dit pas "identification
bourdons à abdomen rouge") ; liberté de simplifier le code CLI existant si
ça facilite l'intégration UI ; mettre à jour les `.md`.

**`core`/`utils`/`tools` réorganisés (clôture de la question ouverte
Phase 1).** `utils/dataset.py`, `predictions.py`, `pipeline_io.py`,
`run_io.py` déplacés vers `core/` (aucune dépendance `argparse`/CLI,
partagés largement -- même critère que les 5 fichiers `core/` déjà là,
même si ce ne sont pas de la géométrie au sens strict) ; `core/` compte
maintenant 9 fichiers. `utils/repair_images.py` déplacé vers
`tools/maintenance/` (CLI autonome, jamais importé ailleurs -- répondait
déjà au critère `tools/`, resté dans `utils/` par oubli). `utils/cli.py`/
`tps_overlay.py` restent dans `utils/` (respectivement lié à `argparse`,
et double usage bibliothèque+CLI). `src/manifest/io.py` (confirmé mort
depuis la session du 7 sept., suite 7) supprimé au passage. `tools/`
découpé en trois sous-paquets par responsabilité -- `ingestion/` (brut ->
manifest propre : `ingest_raw.py`, `export_clean_dataset.py`,
`build_manifest.py`, `combine_manifests.py`, `prepare_dataset.py` --
nouveau, voir plus bas), `pipeline/` (orchestrateurs dataset-agnostiques +
outils de validation : `export_final_landmarks.py`, `export_review.py`/
`reconcile_review.py` -- nouveaux, `train_dataset.py`, `predict_dataset.py`),
`maintenance/` (scripts ponctuels sans rapport avec le pipeline courant).
Chaque commande `python -m tools.X` change en conséquence (ex.
`tools.build_manifest` -> `tools.ingestion.build_manifest`) -- tous les
`.md` ont été mis à jour en conséquence pour les commandes CLI qu'ils
documentent, MAIS PAS pour les mentions purement historiques dans
`RESUME.md`/`TODO.md`/`CONVENTIONS.md` (les anciens chemins plats y sont
volontairement laissés tels quels : ils étaient exacts au moment décrit).
Détail complet du raisonnement dans `CONVENTIONS.md` "Clôture de la
question ouverte `utils/` -> `core/`".

**`tools/ingestion/prepare_dataset.py` créé** : orchestrateur étape 0,
piloté par un fichier de config JSON (un bloc par source, puisque l'étape
0 a vraiment besoin d'options par source -- CSV d'identification, colonnes
-- pas réductible à une poignée de flags communs). Chaîne, en process
(comme les orchestrateurs existants -- `main(argv)` de chaque étage, pas
de sous-processus) : `ingest_raw` (une fois, optionnel) -> par source de
type `"raw"` : `export_clean_dataset` -> par source, toujours :
`build_manifest` -> une fois, sur N sources (N=1 marche aussi) :
`combine_manifests`. Une source de type `"compliant"` (déjà un CSV par
photo conforme) saute directement à `build_manifest`. Une source dont
`build_manifest` ne produit que `manifest_raw.csv` (pas `manifest.csv`)
est rapportée et exclue de la combinaison finale plutôt que de faire
échouer tout le run pour une seule source en défaut. `--skip-ingest`
réutilise un manifest brut déjà scanné (le scan complet est lent). Testé
avec 5 tests synthétiques (`tests/test_prepare_dataset.py`, les 4 scripts
enchaînés sont monkeypatchés -- chacun déjà couvert séparément par
ailleurs) : source "compliant" seule, source "raw" (export puis build),
absence de `mapping_file` sur une source "raw" (erreur explicite), source
en échec exclue de la combinaison, `--skip-ingest`.

**`utils/review.py` créé** : cœur partagé de la validation manuelle
(recadrage, placement des landmarks), utilisé identiquement par
`app/build_dataset.py` et par le nouveau couple CLI
`tools/pipeline/export_review.py`/`reconcile_review.py` -- voir
`CONVENTIONS.md` "Fonctions core réutilisables" et "Validation review :
format des CSV". Principe : `build_crop_review_df`/`build_landmark_review_df`
lisent le statut auto déjà écrit par le pipeline (`crops.csv`,
`landmarks_numbered.csv`) dans un DataFrame `photo_id, ..., auto_status,
reviewed_status` (les deux colonnes distinctes, jamais fusionnées -- sinon
on perd la trace de ce qui a été corrigé à la main) ; `write_crop_review`/
`write_landmarks_review` persistent la review éditée en DEUX choses :
un fichier réconcilié, même schéma que l'original
(`crops_reviewed.csv`/`landmarks_reviewed.csv`), qui se branche
directement sur un mécanisme déjà existant (`landmarks.predict
--crops-csv` -- nouveau flag, cf. plus bas ; `--landmarks-status-csv`, déjà
là) sans qu'aucun code consommateur n'ait besoin de savoir qu'une review a
eu lieu ; et un fichier d'audit minimal
(`<dataset>/review/<nom>_review.csv`, `photo_id, auto_status,
reviewed_status`) pensé pour être ouvert/édité dans un tableur. Jamais de
mutation des fichiers auto originaux (gardés intacts pour l'audit).

**`landmarks/predict.py` : `--crops-csv` ajouté** (défaut : chemin
inchangé `extraction/<mode>/crops.csv`) -- lit un fichier différent si
donné, pour permettre à une review de recadrage de faire sauter une photo
rejetée à la main avant de dépenser du calcul UNet dessus, sans muter
`crops.csv` lui-même.

**`utils/landmarking_pipeline.py::run_landmarking` scindé** en
`run_detection_and_crop` (étapes 1-2) + `run_landmark_placement` (étapes
3-4), `run_landmarking` devenant un simple appel des deux à la suite
(inchangé pour `tools/pipeline/train_dataset.py`/`predict_dataset.py`,
qui continuent à l'utiliser tel quel). Nécessaire pour qu'un appelant
comme `app/build_dataset.py` puisse s'arrêter entre les deux pour la
review du recadrage. `run_export` mis à jour pour transmettre les mêmes
filtres dataset (`--devices`/`--species`/`--castes`/`--include-outliers`/
`--tps`/`--landmarks-status-csv`, factorisés dans un nouveau
`dataset_filter_argv`) que `classifiers.train` -- **bug réel trouvé et
corrigé au passage** : avant cette session, `run_export` n'transmettait
PAS ces filtres, donc le paquet exporté (pour Adrien/R) pouvait ne pas
refléter les mêmes specimens/statuts que ceux effectivement utilisés pour
entraîner le modèle -- silencieux, jamais remarqué faute d'un cas d'usage
qui l'aurait révélé. `dataset_filter_argv` réutilisé aussi dans
`tools/pipeline/train_dataset.py`, qui dupliquait les 7 mêmes blocs `if`
à la main. `verbosity_flags` (défini seulement dans
`landmarking_pipeline.py`) remonté vers `utils/cli.py::verbosity_argv`,
seul point de définition maintenant.

**`app/build_dataset.py` créé** (`streamlit run app/build_dataset.py`) --
assistant en 7 étapes pour le scénario 1, indépendant de
`single_image.py` (fichier séparé, pas un mode caché dans le même) :
1. **Setup** : objectif (construire un modèle vs. prédire avec un modèle
   existant) ; jeu de données (racine déjà propre, ou construction du
   manifest depuis un CSV via `tools.ingestion.build_manifest` directement
   depuis l'UI) ; paramètres du pipeline de landmarking (mêmes flags que
   la CLI, formulaire). Construit l'objet `args` (un `argparse.Namespace`)
   en réutilisant le PARSER de la CLI (`add_dataset_positional`/
   `add_landmarking_args`/`add_dataset_args`) sur un argv construit depuis
   les widgets -- pas de Namespace reconstruit à la main champ par champ,
   une seule logique de parsing/validation/défauts pour la CLI et l'UI
   (voir `CONVENTIONS.md` "Fonctions core réutilisables").
2. **Detection & crop** : bouton "Run" (`run_detection_and_crop(args)`) ou
   "Skip -- already done" (reprendre un dataset déjà partiellement traité,
   par la CLI ou une session précédente de l'appli, sans tout relancer).
3. **Crop review** : `build_crop_review_df` -> tableau filtrable
   (statut, recherche `photo_id`/`inv_id`) et éditable (`st.data_editor`,
   colonne `reviewed_status`) + aperçu du crop de la ligne sélectionnée.
   "Enregistrer et continuer" -> `write_crop_review`, pose
   `args.crops_csv` sur le Namespace partagé (lu par l'étape suivante).
4. **Landmark placement** : même mécanique Run/Skip que l'étape 2, avec
   `run_landmark_placement(args)`.
5. **Landmark review** : même mécanique que l'étape 3, avec en plus
   l'aperçu de l'overlay de landmarks numérotés (`utils.tps_overlay.draw_landmarks`
   sur le crop chargé via le TPS numéroté) et le coût d'enregistrement.
   "Enregistrer et continuer" pose `args.landmarks_status_csv`.
6. **Export** : Run/Skip sur `run_export(args)` -- reflète maintenant la
   review landmarks (voir le bug corrigé ci-dessus).
7. **Construire le modèle ou prédire**, selon l'objectif choisi à
   l'étape 1 : formulaire nom du modèle (`--model-name`, voir plus bas) +
   niveau espèce/caste -> `classifiers.train`, résultats affichés
   (top-1/top-3 LOOCV, chemin du modèle) ; ou sélection d'un modèle
   existant (par nom si `--model-name` a été utilisé) + seuil de
   confiance basse -> `classifiers.predict.run_batch`, résultats affichés
   en tableau + top-1/top-3 si vérité connue.

**Choix "tableau + aperçu" plutôt qu'une vraie galerie d'images.** Le
brief évoquait une "galerie annotée" (vocabulaire déjà présent dans
`TODO.md` Phase 2) ; retenu à la place : un tableau (`st.data_editor`,
filtrable/triable, colonne de statut éditable directement) + un aperçu à
la demande (une image à la fois, sélectionnée dans une liste déroulante
filtrée) plutôt qu'une grille de vignettes. Raison : la collection compte
~2600 spécimens -- charger/afficher des milliers de vignettes ne passe pas
à l'échelle dans Streamlit (mémoire navigateur, temps de rendu), alors que
le tableau reste utilisable à cette taille (le composant sous-jacent,
glide-data-grid, est virtualisé) et couvre le besoin réel (voir/corriger
le statut) sans la partie "parcourir visuellement" que seule une vraie
galerie apporterait en plus. Pas testé au clic sur un dataset de cette
taille réelle depuis cet environnement (voir "À faire par Jules").

**`--model-name` ajouté** (`classifiers/train.py`, relayé par
`tools/pipeline/train_dataset.py --model-name` et le formulaire de l'étape
7 de l'UI) : nom lisible optionnel, stocké dans
`core.model_io.TrainedModel.model_name` et dans `metrics.json`. Purement
descriptif -- ne construit jamais `run_id`/le chemin de sortie (reste
dérivé de niveau/dataset/appareils/source de landmarks pour la
reproductibilité, voir `core/run_io.py::build_run_id`) ; à défaut, le
`run_id` technique reste utilisé comme avant. Nouvelle fonction
`core.run_io.model_display_name(model_path)` (lit `metrics.json`, ne
charge jamais le pickle du modèle juste pour un libellé) branchée dans les
deux sélecteurs de modèle (`app/single_image.py`, `app/build_dataset.py`)
et dans la bannière console de `classifiers/predict.py`.

**Bug réel trouvé et corrigé pendant les tests** (pas anticipé à la
conception) : `utils/review.py::write_crop_review`/`write_landmarks_review`
plantaient (`pandas.errors.LossySetitemError`) dès que TOUTES les valeurs
de `error_reason` du fichier d'origine étaient vides -- pandas lit alors
la colonne en `float64` (NaN) plutôt qu'en texte/objet, et y assigner une
chaîne lève une exception au lieu de l'accepter. Cas réel très probable
(un dataset qui n'a encore eu aucun échec avant la review). Repéré en
testant `app/build_dataset.py` via `streamlit.testing.v1.AppTest` sur un
jeu de données synthétique bout en bout (pas seulement des tests
unitaires sur `utils/review.py` isolément -- mon premier jeu de test
unitaire avait par coïncidence toujours au moins une ligne FAILED avec un
`error_reason` non vide, qui masquait le bug en forçant la colonne en
`object` dès la lecture). Corrigé par un cast explicite en `object` avant
assignation dans les deux fonctions ; deux tests de régression ajoutés
(`tests/test_review.py`).

**Testé** : 109 tests verts (95 existants + 7 `tests/test_review.py` + 5
`tests/test_prepare_dataset.py`, + 2 tests de régression ajoutés après le
bug ci-dessus), `py_compile` sur tout `src/`+`app/`, `--help` sans crash
sur chaque script CLI déplacé ou créé. `app/build_dataset.py` exercé avec
`streamlit.testing.v1.AppTest` (pas seulement un lancement headless +
`curl` comme pour `single_image.py` en session 7 -- cette fois
l'exécution réelle du script, étape par étape, avec de vraies
interactions simulées) : rendu de l'étape 1 (formulaire complet une fois
un dataset synthétique détecté), soumission des paramètres pipeline (avance
bien à l'étape 2), review recadrage (édition du tableau, aperçu image,
écriture réelle de `crops_reviewed.csv`), review landmarks (aperçu overlay
réel via un vrai crop + un vrai TPS numéroté synthétiques, écriture de
`landmarks_reviewed.csv`), export réel (`run_export` exécuté pour de vrai
sur le jeu synthétique, produit bien `export/landmarks_3lm_crop.tps`).
Limitation notée en cours de route : `streamlit.testing.v1.AppTest`
reconstruit son propre arbre de widgets à chaque `.run()` et ne gère pas
bien un script qui change de branche conditionnelle (`if
st.session_state.step == ...`) entre deux interactions successives DANS
LA MÊME session `AppTest` (`KeyError` sur un widget de l'étape
précédente qui ne se ré-affiche plus) -- confirmé être une limitation du
harnais de test lui-même (reproduit sur un script minimal de 15 lignes
suivant exactement le même motif, qui plante pareil) et non un bug de
l'appli réelle (chaque étape testée séparément, en pré-positionnant
`session_state`, fonctionne sans exception ; le serveur réel démarre sans
erreur, `curl` renvoie HTTP 200) -- contourné en testant chaque étape dans
sa propre instance `AppTest` plutôt qu'en enchaînant tous les clics dans
une seule session de test.
- **À faire par Jules** : premier run réel de `app/build_dataset.py` sur
  un vrai dataset (collection ou terrain) avec les vrais poids UNet/
  YOLO-OBB, au clic dans un vrai navigateur -- non testé au-delà du
  scénario synthétique dans cet environnement, en particulier le
  comportement du tableau de review sur les ~2600 photos réelles de la
  collection (voir "Choix tableau + aperçu" ci-dessus).

Fait (session, 7 sept. 2026, suite 8) -- Jules a demandé de passer à
l'implémentation du **mode single-image en Streamlit** (Phase 2/3 : "mode
terrain/single" de l'outil 1, branché directement sur la classification de
l'outil 2 -- voir "Détails outil 1" plus haut et la section "Outil de
classification à une image" du `TODO.md`). Avant d'écrire quoi que ce soit,
un agent Explore a vérifié le code réel de chaque étage
(`extraction/detect_wing.py`, `extraction/normalize_crop.py`,
`landmarks/predict.py`, `landmarks/renumber.py`, `classifiers/predict.py`) :
bonne nouvelle confirmée, **chaque étage exposait déjà une fonction pure,
sans I/O disque**, prévue explicitement pour cet usage (`detect_one_image`,
`normalize_one`/`compute_wing_transform`, `predict_landmarks`,
`numerate_one`, `predict_specimens`) -- plusieurs docstrings le disaient
même noir sur blanc (`landmarks/build_reference.py::load_reference` :
"Used by landmarks/renumber.py, in batch mode as well as single-image
mode." ; `PIPELINE.md` stage 8 sur `predict.py single` : "This is the
function the future single-image UI tool will call directly."). Il ne
manquait qu'une fonction qui les enchaîne en mémoire, et l'app elle-même.

- **`utils/landmarking_pipeline.py::place_landmarks()` créé** (+ dataclass
  `PlacementResult`) : le pendant en mémoire de `run_landmarking()` déjà
  présent dans ce fichier (utilisé par `tools/train_dataset.py`/
  `predict_dataset.py`), mais sur UNE image déjà chargée, sans jamais lire
  ni écrire de fichier -- appelle dans l'ordre `detect_one_image` ->
  `normalize_one` -> `predict_landmarks` -> `numerate_one`, les mêmes
  fonctions pures que le pipeline batch, donc une photo unique et une ligne
  de dataset traversent EXACTEMENT le même code géométrie/modèle. Un détail
  vérifié en lisant `landmarks/renumber.py` avant d'écrire quoi que ce
  soit : un `predict_landmarks` `SUSPECT` (sous-détection) n'est PAS
  court-circuité à part -- il retombe tel quel dans `numerate_one`, qui
  échoue proprement sur l'incompatibilité de nombre de points, exactement
  le comportement déjà documenté du pipeline batch ("every stage-4 SUSPECT
  ends up here"). `PlacementResult` n'a que deux statuts (`OK`/`FAILED`,
  pas de `SUSPECT`) : le seul signal `SUSPECT` existant est un diagnostic
  de population (comparaison à l'espèce, `core.outliers.flag_by_species`),
  sans objet pour un spécimen isolé non labellisé.
- **`draw_landmarks_overlay(image, landmarks) -> Image`** (item Phase 2
  du `TODO.md`) : pas recréé -- `utils/tps_overlay.py::draw_landmarks`
  faisait déjà exactement ça (pure, sans I/O, dessine des points numérotés
  sur une copie de l'image). Réutilisé tel quel dans l'app plutôt que
  dupliqué.
- **`app/single_image.py` créé** (`streamlit run app/single_image.py`) --
  une seule page : upload d'une photo -> bouton "Run pipeline" -> appelle
  `place_landmarks()` puis, si `OK`, construit un `core.tps_io.ImageLandmarks`
  en mémoire à partir du résultat et appelle directement
  `classifiers.predict.predict_specimens(model, [specimen])` -- aucun TPS
  intermédiaire écrit sur disque, conforme à la décision actée
  ("Détails outil 1" : "aucun fichier TPS/CSV écrit sur disque" en mode
  terrain). Affiche : la photo uploadée, l'overlay annoté (landmarks
  numérotés sur le crop, via `draw_landmarks`) avec le coût
  d'enregistrement (`registration_score`), puis le top-3 espèce/caste avec
  confiance (`st.metric`) et la distance de Procrustes à la référence du
  modèle (avertissement si > 0.3, seuil heuristique non calibré -- à
  affiner par Jules avec des cas réels). En cas d'échec à n'importe quel
  étage, affiche le stage et la raison exacte (même vocabulaire que le CLI)
  plutôt qu'un message générique.
  - Modèles mis en cache (`st.cache_resource`) : détecteur YOLO-OBB, poids
    UNet, shape de référence GPA, `model.joblib` LDA -- chacun chargé une
    seule fois par session, jamais rechargé à chaque clic.
  - Sélection des modèles dans la sidebar : `model.joblib` LDA et poids
    UNet **découverts automatiquement** (`glob` sur `data/models/lda/*/train/model.joblib`
    et `data/models/unet_landmarks/*/weights.pt`) et présentés par nom de
    run plutôt que par chemin brut -- reprend la "liste curatée" prévue
    dans "Détails outil 1"/Phase 3 du `TODO.md` pour le choix du modèle
    LDA, étendue au choix du modèle UNet par cohérence. Détecteur YOLO-OBB
    en simple chemin texte (un seul modèle existe aujourd'hui,
    `data/models/yolon_obb/best.pt`, déjà le défaut).
  - **Seul le mode détection `light` (YOLO-OBB) est câblé.** Le mode
    `heavy` (YOLOE) demande en plus un `--heavy-ref` JSON de références --
    non exposé dans cette première version, et ce n'est de toute façon pas
    le mode utilisé par les deux runs réels déjà validés (suite 6). Pas un
    oubli : `place_landmarks()` accepte `detector_mode` en paramètre (pas
    figé en dur), le mode heavy pourra être ajouté dans l'app plus tard
    sans toucher à la fonction elle-même.
- **`utils/landmarking_pipeline.py` : imports réorganisés** pour exposer
  aussi les fonctions pures des étages sous-jacents (`detect_one_image`,
  `normalize_one`, `predict_landmarks`, `numerate_one`), en plus de leurs
  `main` déjà importés pour `run_landmarking()`/`run_export()`.
- **`pyproject.toml`** : `streamlit` ajouté aux dépendances (absent
  jusqu'ici -- seul `app/annotate_wings.py`, un outil d'annotation manuelle
  antérieur à ce refactor et sans rapport avec cette session, en dépendait
  de façon informelle, via un commentaire `pip install` dans son
  docstring).
- **`tests/test_landmarking_pipeline.py` créé** : 5 tests synthétiques sur
  `place_landmarks()` (monkeypatch des 4 fonctions composées -- aucun vrai
  modèle/image nécessaire) : chemin `OK` bout en bout, échec à chacun des 4
  étages (détection/crop/landmarks/renumérotation), plus le cas
  `SUSPECT`-sous-détection décrit plus haut qui doit échouer à la
  renumérotation, pas être court-circuité avant.
- **Correction en cours de session : l'environnement Python du projet
  n'est PAS le `python3` système.** Après avoir écrit tout ce qui précède
  en supposant `torch`/`streamlit`/`ultralytics`/`pytest` absents de
  l'environnement (comme dans plusieurs sessions précédentes), Jules a
  signalé qu'il faut activer l'environnement mamba `idmybee`
  (`mamba activate idmybee`) -- il contient bien tout (`torch`,
  `streamlit`, `ultralytics`, `scikit-image`, `pytest`), et le paquet est
  déjà installé en mode éditable dedans. Une fois activé :
  - **95 tests verts** (90 existants + 5 nouveaux sur `place_landmarks`),
    `python -m pytest tests/` propre.
  - **Run réel de bout en bout, pas seulement des tests synthétiques** :
    `place_landmarks()` puis `predict_specimens()` exécutés directement
    (hors Streamlit) sur une vraie photo brute de la collection
    (`ARLY_0001_P_1.jpg`, montée sur le volume externe déjà branché dans
    cet environnement) avec les vrais poids (`yolon_obb/best.pt`, UNet
    `2026-08-29_131929`, `reference_shape_19.npz`, modèle
    `species_collection/train/model.joblib`) : détection -> crop ->
    landmarks -> renumérotation -> classification, statut `OK` de bout en
    bout, 19 landmarks, coût d'enregistrement bas (0.0002), distance de
    Procrustes basse (0.06) -- prédiction top-1 `lapidarius` (97.9 %) alors
    que l'étiquette du dossier est `rupestris` (2e choix, 2.1 %) : une
    vraie erreur du modèle sur ce spécimen (deux espèces à abdomen rouge
    visuellement proches, cohérent avec les 94.77 % de précision LOOCV déjà
    mesurés sur ce modèle -- pas un bug du nouveau code, le pipeline a
    fonctionné correctement, le modèle s'est juste trompé sur cet
    individu).
  - **`streamlit run app/single_image.py` démarre sans erreur** (lancé en
    headless, `curl` renvoie HTTP 200, aucune erreur dans les logs) --
    non testé au clic dans un vrai navigateur (pas d'accès UI depuis cet
    environnement), mais le serveur et tous les imports de la page sont
    confirmés sains.
- **À faire par Jules** : tester réellement l'app au clic dans un
  navigateur (upload, bouton "Run pipeline", lecture de l'overlay/du
  classement) -- le pipeline sous-jacent et le démarrage du serveur sont
  déjà vérifiés, seule l'interaction UI elle-même ne l'est pas depuis cet
  environnement. Calibrer le seuil d'avertissement sur la distance de
  Procrustes (0.3, posé arbitrairement) une fois des cas réels observés.

Fait (session, 7 sept. 2026, suite 7) -- Jules a demandé de revoir
l'architecture de construction du manifest et de supprimer les symlinks,
l'étape "pré-pipeline" étant jugée confuse. Sa reformulation, en substance :
`export_clean_dataset.py` ne devrait servir qu'à produire un dataset propre
(pas obligatoire -- à lancer seulement si le dataset donné n'est pas
utilisable, ou si Jules le choisit), pas à construire le manifest ; le
manifest devrait pouvoir être généré depuis n'importe quel dataset en
entrée (nettoyé ou non) tant qu'il respecte un standard simple (images +
CSV avec les données biologiques obligatoires par photo) ; en cas de
dataset non conforme, avertissement et/ou `manifest_raw` exploitable pour
nettoyage. Deux points d'architecture tranchés avec Jules avant
implémentation (via `AskUserQuestion`) :
1. Le CSV compatible est **une ligne par photo**, avec une colonne
   fichier/chemin explicite, des colonnes biologiques obligatoires
   (species, caste, ...) et des colonnes optionnelles de capture/photo
   (device, photographe, date...) -- fonctionne avec n'importe quelle
   convention de nommage, y compris un dataset tiers jamais passé par
   l'outillage de ce projet.
2. Le nouvel outil fait de la **validation structurelle uniquement**
   (fichiers présents/lisibles, `photo_id` unique, cohérence biologique
   par `inv_id`) -- jamais d'arbitrage d'identité (étiquette réutilisée
   pour deux spécimens différents) : ça reste le rôle
   d'`export_clean_dataset.py`, pour les données brutes réellement
   désordonnées.

Un agent Plan dédié a vérifié le contrat exact de chaque fichier concerné
(`export_clean_dataset.py`, `ingest_raw.py`, `link_external_data.py`,
`manifest/identification.py`, `utils/dataset.py`, `utils/pipeline_io.py`)
avant de proposer l'architecture, plutôt que de deviner les signatures --
conception ensuite vérifiée fichier par fichier avant implémentation.

- **Découverte en cours de route, sans rapport direct avec la demande de
  Jules :** `tools/combine_manifests.py` est marqué `[x]` fait dans
  `TODO.md`/`RESUME.md` depuis la suite 6 (et avant) mais n'existait **pas
  réellement** dans le dépôt -- dérive documentation/code, corrigée par la
  même occasion (voir plus bas).
- **`src/manifest/build.py` créé** (pure, sœur d'`identification.py`) :
  `validate_required_columns`/`biological_columns`/
  `resolve_and_verify_images` (hash/taille/statut recalculés depuis le
  fichier lui-même, jamais hérités d'un producteur en amont)/
  `flag_duplicate_photo_ids`/`check_biological_consistency` (rapporte les
  variantes en désaccord par `inv_id`, ne les arbitre jamais)/
  `assign_sequential_photo_ids` (même principe que
  `identification.assign_photo_ids`, généralisé pour trier sur la colonne
  chemin plutôt que sur `shot_index`/un fichier de mapping -- un CSV tiers
  n'a ni l'un ni l'autre ; un `photo_id` déjà présent n'est jamais
  recalculé, puisque le nom de fichier sur disque en dépend déjà)/
  `build_biological_data` (dédoublonnage par `inv_id` + pivot
  `n_photos_<device_type>`, version générique de
  `identification.build_specimen_table`)/`build_manifest_table`.
- **`src/tools/build_manifest.py` créé** : CLI toujours lancée, seul point
  d'entrée réel vers la suite du pipeline. Règle à un seul artefact (pas
  deux modes d'échec séparés, pour rester simple comme demandé par
  Jules) : si toutes les lignes valident `OK`, écrit `manifest.csv` +
  `biological_data.csv` ; sinon écrit `manifest_raw.csv` seul (même schéma,
  `status`/`status_reason` explicites) + `reports/biological_inconsistencies.csv`
  si le problème vient d'une incohérence biologique -- jamais de crash,
  code de sortie 0 dans les deux cas (diagnostic, pas une erreur fatale).
- **`tools/export_clean_dataset.py` : sortie renommée `dataset.csv`.** Ne
  produit plus `manifest.csv`/`biological_data.csv` -- `dataset.csv` est
  une ligne par photo copiée avec succès (colonnes biologiques déjà
  fusionnées dedans, dénormalisé), exactement le CSV compatible que
  `build_manifest.py` attend, sans étape intermédiaire. `content_hash`/
  `file_size_bytes`/`status`/`status_reason`/`source_type` ne sont
  délibérément plus repris dans `dataset.csv` -- `build_manifest.py`
  recalcule hash/taille/statut lui-même depuis `path`, une seule source de
  vérité. `biological_data_all.csv`, `device_types.csv`, tous les
  `reports/` restent inchangés.
- **`tools/combine_manifests.py` implémenté pour de vrai**, corrigeant la
  dérive doc/code trouvée ci-dessus -- concat pur de `manifest.csv`/
  `biological_data.csv` déjà produits par `build_manifest.py`, erreur
  explicite sur collision `photo_id`/`inv_id` entre racines combinées.
- **`tools/link_external_data.py` et `config/derived_root.json`
  supprimés.** Plus aucun scénario ne nécessite "copier localement puis
  déplacer+symlinker" : `build_manifest.py` ne copie jamais de fichier (son
  but est justement de référencer les images là où elles vivent déjà), et
  `export_clean_dataset.py --output-dir` peut déjà écrire directement sur
  un volume externe sans copie locale intermédiaire (aucun changement de
  code nécessaire pour ça).
- **`PIPELINE.md` mis à jour** : section "External storage" remplacée par
  "Dataset roots can live anywhere on disk" ; stage 1 réécrit en 1a
  (nettoyage via `ingest_raw.py`/`export_clean_dataset.py`, optionnel) et
  1b (`build_manifest.py`, toujours lancé) ; diagramme "Cross-stage data
  flow" mis à jour ; note sur `data/Bombus/collection`/`terrain` (Known
  gaps) mise à jour pour refléter la chaîne complète.
- **`CONVENTIONS.md`** : note datée ajoutée en tête de "Export propre par
  source" pointant vers la nouvelle section ; nouvelle section "Manifest
  découplé du nettoyage, `build_manifest.py` créé" ajoutée, expliquant
  explicitement pourquoi ce n'est **pas** la même erreur que la suppression
  de `manifest/build_dataset.py` (6 sept. 2026) : celui-ci ne fonctionnait
  que sur un dossier déjà nommé selon la convention canonique du projet ;
  `build_manifest.py` fonctionne sur n'importe quel dataset conforme, y
  compris un qui n'est jamais passé par l'outillage de ce projet, piloté
  par un CSV explicite plutôt que par une convention de nommage figée.
- **`tests/test_manifest_build.py` créé**, dans le style de
  `tests/test_identification.py` : validation des colonnes obligatoires,
  vérification des fichiers image, détection de `photo_id` dupliqué,
  détection d'incohérence biologique, dérivation séquentielle de
  `photo_id`, et la règle à un seul artefact (succès vs `manifest_raw`).
- **Non touché** (confirmé par grep) : `tools/ingest_raw.py`,
  `manifest/identification.py` (toujours utilisé seulement par
  `export_clean_dataset.py`), le contrat `utils.dataset.load_dataset()`,
  les stages 2-9 du pipeline, `tools/train_dataset.py`/`predict_dataset.py`,
  `utils/landmarking_pipeline.py`. `src/manifest/io.py` laissé tel quel --
  confirmé mort (aucune référence nulle part, schéma pré-refactor
  `images.csv`/`specimens.csv`/`crops.csv`), sans rapport avec ce
  changement, signalé comme piste de nettoyage séparée dans `TODO.md`.
- **À faire par Jules** : premier run réel de bout en bout sur les vraies
  données (`export_clean_dataset.py` -> `build_manifest.py`, collection et
  terrain séparément, puis éventuellement `combine_manifests.py`) --
  seulement vérifié par tests synthétiques dans cet environnement jusqu'ici.

Fait (session, 7 sept. 2026, suite 6) -- retour de Jules sur les deux
orchestrateurs de la suite 5 (`tools/build_collection_reference.py`,
`tools/predict_terrain.py`), trois demandes claires :
1. plus de scripts/noms spécifiques à "collection"/"terrain" -- pouvoir
   construire un modèle de référence depuis N'IMPORTE QUEL dataset ;
   revoir le nom de chaque fichier Python pour que ce soit clair.
2. clarifier les entrées/sorties, jugées éparpillées ("un peu le bazar").
3. un dossier qui organise clairement les sorties de
   `tools/export_final_landmarks.py`, à implémenter dans le script
   lui-même (pas laissé au choix ad hoc de `--output-dir` à chaque appel).

Traité par un renommage + une factorisation, pas juste un renommage de
façade :

- **`tools/build_collection_reference.py` -> `tools/train_dataset.py`,
  `tools/predict_terrain.py` -> `tools/predict_dataset.py`.** Le
  positionnel `dataset` n'a plus AUCUNE valeur par défaut (avant : repliait
  respectivement sur `data/Bombus/collection`/`data/Bombus/terrain`) --
  argument requis comme toute autre commande du pipeline (cf.
  `CONVENTIONS.md` "Nommage CLI"). Les deux docstrings énoncent
  explicitrement "Dataset-agnostic ... nothing here is specific to
  collection vs terrain vs a future source" pour que ça reste vrai dans le
  temps, pas juste au moment du renommage.
- **`utils/landmarking_pipeline.py` créé** -- la chaîne des 4 étapes
  (détection -> crop -> landmarks -> renumérotation), auparavant dupliquée
  presque à l'identique dans les deux scripts, factorisée une seule fois :
  `add_landmarking_args(parser)` (tous les flags des étapes 2 à 6 déclarés
  à un seul endroit), `run_landmarking(args)` (étapes 2-5), `run_export(args)`
  (étape 6, voir point suivant). Répond directement au point 2 : les
  entrées/sorties de chaque étape sont maintenant documentées une seule
  fois, au même endroit, plutôt que répétées (et risquant de diverger)
  dans chaque script appelant.
- **`tools/export_final_landmarks.py` : `--output-dir` devient optionnel**,
  défaut `<dataset>/export/` via une nouvelle fonction
  `utils.pipeline_io.dataset_export_dir(dataset)`. Choix : sous la racine
  du dataset plutôt qu'un chemin centralisé genre `data/exports/<label>/`,
  pour rester cohérent avec tout le reste (`extraction/`, `landmarks/`,
  `manifest.csv`, `biological_data.csv` vivent déjà tous sous
  `<dataset>/...`) -- un seul dossier à connaître par dataset, plus besoin
  d'inventer un chemin `out/collection/19` à la main à chaque run comme
  avant. `--output-dir` explicite reste possible (utile pour un export
  ponctuel vers un chemin précis, ex. pour Adrien). `tools/train_dataset.py`/
  `tools/predict_dataset.py` appellent maintenant aussi cette étape
  d'export (`run_export`) avant l'entraînement/la classification -- le
  paquet R-facing est donc systématiquement produit par les deux
  orchestrateurs, pas seulement le modèle/les prédictions.
- **Premier run réel de bout en bout, sur les vraies données** (levée
  d'une limitation signalée en suite 5 -- `skimage`/`ultralytics` absents
  de l'environnement utilisé pour écrire le code, mais l'exécution réelle
  a eu lieu dans un environnement qui les a) :
  - `tools/train_dataset.py data/Bombus/collection --unet-model
    data/models/unet_landmarks/2026-08-29_131929/weights.pt` : 2600
    spécimens, 19 landmarks, LOOCV top-1 = 94.77 %, top-3 = 98.73 % --
    modèle sauvé dans `data/models/lda/species_collection/train/model.joblib`
    (`run_id="species_collection"`, confirme le nouveau schéma
    `dataset_label` de la suite 5 en conditions réelles).
  - `tools/predict_dataset.py data/Bombus/terrain --model
    data/models/lda/species_collection/train/model.joblib --unet-model ...` :
    265 spécimens terrain évalués contre une vérité connue, top-1 = 82.26 %,
    top-3 = 96.98 % -- sorties dans
    `data/models/lda/species_collection/predict/terrain/` (`eval_tag="terrain"`,
    confirme aussi `build_eval_tag` en conditions réelles). Chute
    d'accuracy collection -> terrain (94.77 % -> 82.26 %) cohérente avec
    un usage terrain plus varié (éclairage, angle, usure des ailes) que la
    collection de musée sur laquelle le modèle a été entraîné -- pas un
    signal de bug, juste la réalité du champ.
- **`PIPELINE.md` mis à jour** : section "Orchestrator scripts" et "Known
  gaps" avec les nouveaux noms de fichiers, ajout des résultats du run réel
  ci-dessus, stage 6 (export) documente le nouveau défaut
  `<dataset>/export/`.
- **Testé** : 73 tests toujours verts (`test_outliers.py` toujours hors
  périmètre, limitation Python 3.9 de l'environnement d'écriture du code,
  sans rapport avec cette session), `py_compile` OK sur tout `src/`.
- **À faire par Jules** : rien de bloquant identifié -- le run réel a
  réussi du premier coup avec les noms/chemins par défaut actuels. Les
  anciens runs `data/models/lda/species_train_*lm` (pré-refactor) restent
  à nettoyer/ignorer si plus utiles (signalé en suite 5, toujours valable).

Fait (session, 7 sept. 2026, suite 5) -- trois demandes de Jules dans le
même message : retirer `--split`/`--splits-csv` du pipeline (plus besoin),
un script bout-en-bout construisant le modèle de référence depuis
`data/Bombus/collection`, un script bout-en-bout classifiant
`data/Bombus/terrain` avec ce modèle -- en gardant en tête l'objectif d'un
futur outil UI de classification à une image (déjà décrit en détail dans
"Détails outil 1" plus haut et Phase 3 du `TODO.md`, rien de nouveau à
concevoir cette session sur ce point, juste confirmé/gardé en tête).

- **Suppression de `--split`/`--splits-csv` pipeline-wide.** Constat qui a
  guidé le retrait : Jules a désormais deux racines de dataset propres et
  séparées (`data/Bombus/collection`, `data/Bombus/terrain`, sorties
  d'`export_clean_dataset.py` -- voir entrées précédentes), donc plus
  besoin de découper un seul dataset en train/test par un `splits.csv`
  séparé -- collection sert à entraîner, terrain à prédire, chacun étant
  chargé en entier. Retiré de `utils/cli.py` (`add_dataset_args`/
  `dataset_kwargs`, `resolve_split` supprimée), `utils/pipeline_io.py`
  (`load_splits` supprimée), `utils/dataset.py::load_dataset` (plus de
  colonne `split` dans `meta_df`, plus de filtre), `extraction/detect_wing.py`,
  `extraction/extraction_io.py::select_images`, `landmarks/predict.py::load_target_crops`,
  `classifiers/train.py`, `classifiers/predict.py`, `analysis/variance_report.py`,
  `analysis/classification_report.py`. **`obb_trainer/`/
  `landmarks_trainer/dataset.py` volontairement non touchés** : leur
  `--split train/val/test` est un concept ML différent (découpage
  d'entraînement du détecteur YOLO-OBB / de l'UNet), sans rapport avec le
  split biologique du pipeline de classification -- déjà hors périmètre
  (voir Phase 4 du `TODO.md`).
- **`utils/run_io.py` : `split` remplacé par `dataset_label`** dans
  `build_run_id`/`build_eval_tag`/`build_variance_id` -- `dataset_label`
  est le nom de la racine du dataset (`Path(dataset).name`, ex.
  `"collection"`/`"terrain"`), qui reprend exactement le rôle que jouait
  `split` pour distinguer les runs entre eux sans jamais désigner un
  sous-ensemble du même dataset. Exemple concret : un `train.py` sur
  `data/Bombus/collection` produit maintenant `run_id="species_collection"`
  (au lieu de `"species_train"`) ; un `predict.py batch` de ce modèle sur
  `data/Bombus/terrain` produit `eval_tag="terrain"` (au lieu de `"test"`),
  et le même modèle évalué sur un futur troisième dataset ne collision
  jamais avec cette évaluation puisque chaque `eval_tag` porte le nom de
  SON dataset.
- **`core/model_io.py` : `TrainedModel.split` remplacé par
  `TrainedModel.dataset_label`** (même rôle informatif -- affiché par
  `classifiers/predict.py::_print_model_info`). Effet de bord assumé,
  signalé explicitement (pas caché) : les `model.joblib` déjà entraînés
  avant ce changement ont un `__dict__` pickle contenant encore l'ancien
  champ `split`, mais la classe actuelle ne définit plus que
  `dataset_label` -- charger un vieux modèle et lire `model.dataset_label`
  lève une `AttributeError` (le champ n'a jamais existé dans ce pickle).
  Pas un bug à corriger : cohérent avec les autres cutovers nets déjà
  actés dans ce refactor (`data/Bombus/` ancien schéma, etc.) -- il suffit
  de retrainer avec le nouveau script (voir plus bas) pour obtenir un
  modèle au schéma actuel. Les runs `data/models/lda/species_train_*lm`
  existants (pré-refactor, entraînés sur l'ancien `data/Bombus/` remplacé
  depuis par `data/Bombus/collection`) sont donc déjà obsolètes
  indépendamment de ce changement de champ.
- **`tools/build_collection_reference.py` créé** : orchestrateur bout-en-
  bout appelant en process le `main(argv)` de chaque étage (pas de
  sous-processus, pas de logique dupliquée -- réutilise directement
  `extraction.detect_wing`/`extraction.normalize_crop`/`landmarks.predict`/
  `landmarks.renumber`/`classifiers.train`) : détection -> normalisation de
  crop -> pose de landmarks (UNet) -> renumérotation -> entraînement
  GPA-PCA-LDA, sur `data/Bombus/collection` par défaut (positionnel
  optionnel pour changer de racine). `--reference` par défaut déduit de
  `--n-landmarks` (`data/references/shapes/reference_shape_<n>.npz`, déjà
  gelé lors d'une session précédente -- `landmarks/build_reference.py` n'a
  pas besoin d'être relancé). `--detector-model` par défaut
  `data/models/yolon_obb/best.pt` (poids YOLO-OBB déjà entraînés,
  confirmés présents sur le disque de cet environnement). `--unet-model`
  reste obligatoire (pas de poids UNet par défaut choisi unilatéralement --
  deux runs existent, `2026-08-29_131929` et `legacy_baseline`, à Jules de
  choisir). Chacune des 3 premières étapes reste indépendamment reprenable
  (`--overwrite`/`--retry-failed` relayés là où l'étage cible les supporte
  réellement -- `detect_wing.py` n'a pas de mécanisme de reprise, seuls
  `normalize_crop.py`/`landmarks/predict.py` en ont).
- **`tools/predict_terrain.py` créé** : même chaîne des 4 premières étapes
  (détection -> crop -> landmarks -> renumérotation) sur
  `data/Bombus/terrain` par défaut, puis `classifiers.predict batch` avec
  un `--model` déjà entraîné (typiquement celui produit par
  `build_collection_reference.py`) -- appelle directement
  `classifiers.predict.run_batch(args)` (pas `main(argv)`, pour réutiliser
  tel quel le Namespace construit par `utils.cli.add_dataset_args`, dont
  les noms de destination correspondent exactement à ce que `run_batch`
  attend, plutôt que de re-sérialiser tous les flags en argv). `--unet-model`/
  `--n-landmarks`/`--reference` doivent être ceux utilisés pour entraîner
  `--model` (signalé dans le docstring -- un mauvais réglage donne des
  prédictions fausses sans erreur, `classifiers.predict` ne vérifie que le
  nombre de points, jamais l'ordre).
- **Vérifié dans cet environnement** : les deux nouveaux scripts compilent
  (`py_compile`), leur `parse_args()` produit un `Namespace` avec tous les
  champs attendus (testé en stubant `sys.modules['skimage']`/
  `sys.modules['ultralytics']`, absents de cet environnement -- pas de run
  réel possible ici faute de ces dépendances). Chaque nom de flag passé aux
  `main(argv)` des étages a été recoupé à la main contre le parser réel de
  chaque fichier (`detect_wing.py` n'a PAS de `--overwrite`/`--retry-failed`
  -- pas de mécanisme de reprise, contrairement à `normalize_crop.py`/
  `landmarks/predict.py` qui en ont tous les deux ; le `--model` du
  détecteur YOLO-OBB/YOLOE, ajouté par `backend.add_arguments()`, est
  distinct du `--model` UNet de `landmarks/predict.py` -- d'où
  `--detector-model`/`--unet-model` comme noms distincts côté
  orchestrateurs pour ne pas les confondre). 73 tests verts (hors
  `test_outliers.py`, limitation Python 3.9 de cet environnement, sans
  rapport avec cette session -- voir `TODO.md`).
- **À faire par Jules** : premier run réel des deux orchestrateurs (poids
  UNet/YOLO-OBB réels, vraies images) -- non exécutable dans cet
  environnement (dépendances `skimage`/`ultralytics` absentes). Décider
  quel run UNet utiliser (`data/models/unet_landmarks/2026-08-29_131929/weights.pt`
  vs `legacy_baseline/weights.pt`, ce dernier avec `--n-landmarks 18`).

Fait (session, 7 sept. 2026, suite 4) -- Jules a testé la commande réelle de
`tools/export_clean_dataset.py` sur la collection (avec `--key-column inv_id`,
la colonne brute de collection s'appelant maintenant `inv_id`, pas
`inv_name`/`original_id` comme dans l'ancien exemple d'usage) et est tombé sur
un `KeyError: 'original_id'` dans `restrict_to_present_images`. Confirmé par
inspection du fichier réel : c'est exactement la migration déjà signalée
"À faire par Jules" dans l'entrée précédente qui n'a pas encore été faite --
`data/bombus_collection_raw/manifest.csv` porte toujours l'en-tête pré-refactor
`specimen_id` au lieu de `original_id`. Pas un bug de code : reste à re-lancer
`tools/ingest_raw.py` (ou renommer l'en-tête à la main) pour ce fichier et son
équivalent terrain. Vérifié sur une copie scratch avec l'en-tête renommé : la
commande tourne bien de bout en bout une fois ça fait.

En creusant le retour de Jules, deux vraies demandes de fond :
- **`original_id` ne doit apparaître dans `biological_data*.csv` (et les
  rapports) que si `--origin-codes` est fourni.** Sans `--origin-codes`
  (numérotation "embedded", terrain-style), `extend_mapping` pose
  `inv_id = original_id` tel quel (voir son docstring) -- garder les deux
  colonnes est une pure duplication. Avec `--origin-codes` (collection-style),
  `inv_id` est réassigné (nom_inventaire + numéro séquentiel), donc distinct
  de la valeur brute -- `original_id` y reste une information réelle à
  garder. Nouveau paramètre `keep_original_id: bool` sur
  `attach_canonical_inv_id`/`build_specimen_table` (défaut `True`, rétro-
  compatible), câblé dans `tools/export_clean_dataset.py` via
  `keep_original_id = origin_codes is not None`, appliqué aux 4 sorties
  (`biological_data_all`, `biological_data`, `conflicts`,
  `excluded_no_image`). Effet de bord : le print de résumé
  `Identity conflicts: ... specimen(s)` lisait `conflicts['original_id']`,
  qui peut ne plus exister -- capturé plus tôt (`n_conflicted_specimens`,
  calculé sur `conflicted_keys` avant la transformation) pour ne pas
  dépendre de la colonne. Testé sur les deux vraies sources (collection avec
  `--origin-codes`, `inv_id_raw` présent avec la vraie étiquette de musée ;
  terrain sans `--origin-codes`, aucune colonne dupliquée).
- **Docstrings nettoyées** (`manifest/identification.py`,
  `tools/export_clean_dataset.py`) suivant le style de `CONVENTIONS.md` :
  exemples d'usage du module mis à jour pour matcher les vraies colonnes
  actuelles (`--key-column inv_id` pour les deux sources, plus
  `inv_name`/`original_id` obsolètes), rationale historique élaguée dans
  `resolve_identification`/le docstring de module, `--origin-codes` explicité
  dans l'aide CLI pour documenter la nouvelle règle `original_id`.

**Testé** : 79 tests verts (2 nouveaux : `attach_canonical_inv_id`/
`build_specimen_table` avec `keep_original_id=False`), `py_compile` OK,
export réel sur les deux sources (collection + terrain, via une copie
scratch du manifest brut avec l'en-tête renommé) vérifié colonne par
colonne.

Fait (session, 7 sept. 2026, suite 3) -- Jules a demandé de combler pour de
vrai l'écart entre le nouveau pipeline "dataset propre" (Pipeline A --
`manifest/identification.py`/`tools/ingest_raw.py`/`tools/export_clean_dataset.py`)
et l'ancien pipeline extraction/landmarks/classification (Pipeline B --
`extraction/*`, `landmarks/*`, `tools/convert_landmarks.py`, `utils/dataset.py`,
`classifiers/*`, `analysis/variance_report.py`), constaté en creusant la
demande de doc pipeline : Pipeline B attend un schéma différent
(`data/Bombus/manifest.csv` avec `image_id`/`raw_path`/`specimen_id`/`split`
en colonne, `data/Bombus/specimens.csv` avec `specimen_id`/`species`/`caste`/
`is_labeled`), jamais connecté au nouveau `data/clean/<source>/`. Demande
explicite : modifier les fichiers eux-mêmes (pas une couche d'adaptation),
`photo_id`/`inv_id` comme identifiants canoniques de bout en bout. Conception
validée via un agent Plan dédié (deux points d'architecture tranchés de façon
décisive, pas laissés ouverts) puis vérifiée fichier par fichier en lisant le
code réel avant de patcher (pas de suppositions) :

- **`ID=` du TPS** : `photo_id` est une string, pas un hash hex -- l'ancien
  `image_id_to_sid = int(image_id, 16)` est supprimé. `COMMENT=photo_id=...;
  inv_id=...` devient le SEUL mécanisme de jointure (déjà documenté comme
  "primaire", le hash n'était qu'un repli pour un TPS sans ce champ -- coupure
  nette avec `data/Bombus/`, ce repli n'a plus rien à couvrir). `ID=` devient
  un entier séquentiel, réassigné à chaque écriture
  (`core.tps_io.assign_sequential_ids`, trié par `photo_id`). Point délicat
  trouvé en lisant le code : `landmarks/predict.py` calculait `tps_id` AVANT
  toute résolution pour son mécanisme de reprise (checkpoint/resume) --
  corrigé en indexant `working_tps` par `photo_id` (déjà stable) au lieu de
  `tps_id`, qui n'est réassigné qu'au moment du checkpoint.
- **`split`** : absent du schéma Pipeline A (décision déjà actée -- "laisser
  le split aux trainers"). Nouveau `splits.csv` (`inv_id,split`), construit
  séparément par expérience, `--splits-csv` ajouté une seule fois à
  `utils.cli.add_dataset_args`/`dataset_kwargs`, chargé via un nouveau
  `utils.pipeline_io.load_splits()` réutilisé par `detect_wing.py`,
  `landmarks/predict.py` et `utils/dataset.py` -- un seul endroit qui définit
  "que veut dire --split train".
- **Collision de nom trouvée en croisant les deux parties de la session** :
  `utils/dataset.py::load_dataset()` réutilisait déjà le nom `device` pour
  dire *device_type* (`row["device"] = img_row["device_type"]`) -- collision
  avec la vraie colonne `device` (nom de caméra/téléphone) ajoutée à
  `manifest.csv` plus tôt dans la session. Corrigé : `device_type` et
  `device` attachés séparément et correctement nommés.

**Fichiers modifiés** (renommage `image_id`->`photo_id`, `specimen_id`->
`inv_id`, schéma `manifest.csv`/`biological_data.csv` de Pipeline A comme
entrée canonique) : `core/tps_io.py` (`ImageLandmarks.photo_id`/`.inv_id`,
`assign_sequential_ids` remplace `image_id_to_sid`/`sid_to_image_id`),
`extraction/extraction_io.py` (`DETECTION_FIELDS`/`CROP_FIELDS`,
`read_images_csv` attend `photo_id`/`inv_id`/`path`, `select_images` filtre
par `{inv_id: split}` au lieu d'une colonne `split` en ligne),
`extraction/detect_wing.py`, `extraction/normalize_crop.py`
(`build_output_path` simplifié : `{photo_id}.jpg`, plus de sous-dossier par
split ni de nom composé specimen/device/shot), `landmarks/predict.py`
(fix resume ci-dessus), `landmarks/renumber.py` (`--specimens` ->
`--biological-data`, lit `biological_data.csv`, "labeled" = `species` non
vide puisqu'il n'y a pas de colonne `is_labeled` dans ce schéma),
`utils/dataset.py` (réécriture complète du contrat de jointure -- voir
collision `device` ci-dessus), `utils/cli.py` (`--splits-csv`),
`utils/predictions.py`, `analysis/variance_report.py`
(`LEVEL_CHOICES`). **`tools/convert_landmarks.py` supprimé, remplacé par
`tools/export_final_landmarks.py`** : suppression complète de la fusion
`--identification-csv` (obsolète, `biological_data.csv` fournit déjà
genus/species/caste via `inv_id`) -- effet de bord positif, les colonnes
`species`/`caste` qui étaient commentées dans l'ancien fichier (pour éviter
une collision avec cette fusion) sont réinstaurées dans la sortie finale.
`classifiers/train.py`/`predict.py` **inchangés** (ne consomment que
`meta_df`/`specimens` déjà normalisés par `load_dataset()` -- confirme que
la dépendance au schéma était déjà bien isolée).

**Fichiers hors plan initial, corrigés en cours de route** (cassaient
autrement avec `AttributeError`/colonnes manquantes, trouvés par grep
systématique après le renommage principal) : `tools/verify_tps.py`,
`utils/tps_overlay.py` (`--image-id-col` -> `--photo-id-col`),
`landmarks_trainer/export_dataset.py` (consommateur direct de
`ImageLandmarks.image_id`/`.specimen_id`). **`obb_trainer/*.py` volontairement
non touchés** : ces scripts référencent déjà un schéma `crops.csv`/`detection.csv`
qui ne correspond pas aux colonnes actuelles de `extraction_io.py` (semblent
déjà déconnectés du pipeline actuel, indépendamment de cette session) --
hors périmètre, signalé dans `TODO.md`.

**`PIPELINE.md` créé** (nouveau fichier racine, référencé depuis ce
document) : guide complet stage par stage (entrées/sorties/CLI/statuts
exacts, vérifiés en lisant le code réel de chaque fichier), du dataset
propre jusqu'au modèle GPA-PCA-LDA fitté, incluant le nouveau `splits.csv`
et une section "Known gaps" explicite (pas d'orchestrateur Phase 2,
`data/Bombus/` incompatible -- coupure nette assumée, `obb_trainer/` hors
scope).

**Testé** : `tests/test_tps_io.py` (nouveau test `assign_sequential_ids`,
champs renommés), `tests/test_pipeline_io.py` (nouveau `load_splits`),
`tests/test_dataset.py` **créé** (6 tests synthétiques -- jointure
`photo_id`/`inv_id`, distinction `device_type`/`device`, `splits.csv`,
plusieurs photos par spécimen). 77 tests verts au total (69 + 8 nouveaux).
`python -m py_compile` sur tout `src/` : OK. Smoke-test fonctionnel direct
(sans modèle réel) de `extraction_io.select_images`/`normalize_crop.build_output_path`
sur un manifest.csv synthétique format Pipeline A : OK. `--help` vérifié
sans crash sur les 8 scripts CLI modifiés.

**Pas testé** (nécessite un vrai modèle UNet + de vraies images, hors de cet
environnement) : la chaîne complète `detect_wing.py` -> `normalize_crop.py`
-> `landmarks/predict.py` -> `renumber.py` -> `export_final_landmarks.py` ->
`train.py` sur un dataset réel issu d'`export_clean_dataset.py`.

**À faire par Jules** (migrations de fichiers déjà générés, valeurs
inchangées, juste des en-têtes de colonnes à renommer) :
- `data/identification/inv_id_mapping.csv` : colonne `specimen_key` ->
  `original_id` (voir entrée précédente).
- Tout `manifest.csv` déjà produit par `tools/ingest_raw.py` : colonne
  `specimen_id` -> `original_id`.
- **Premier run réel de bout en bout** sur un dataset `data/clean/<source>/`
  complet (avec de vraies images + un modèle UNet) pour valider Part B en
  conditions réelles -- seulement vérifié par tests synthétiques + smoke
  tests dans cet environnement.
- `data/Bombus/` : dataset historique, incompatible avec le code actuel --
  soit régénéré depuis `data/clean/` via les étapes 2-7 de `PIPELINE.md`,
  soit laissé de côté comme référence historique seule.

Fait (session, 7 sept. 2026, suite 2) -- retour de Jules sur les 3 points
ci-dessus après un run réel avec le code de la suite précédente, plus
demande explicite de combler l'écart entre le nouveau pipeline "dataset
propre" et l'ancien pipeline extraction/landmarks/classification :
- **`n_photos_P`/`n_photos_S` restaurés** (`build_specimen_table`) --
  retour au pivot par device, la somme unique de la session précédente est
  annulée (Jules veut le détail par device, pas un total).
- **Renommage `specimen_key`/`specimen_id` -> `original_id`** dans tout
  `manifest/identification.py` (y compris `MAPPING_COLUMNS`, donc le
  schéma d'`inv_id_mapping.csv` change -- colonne `specimen_key` ->
  `original_id`, migration triviale d'en-tête, valeurs inchangées) et
  `tools/ingest_raw.py` (`ImageRecord.specimen_id` -> `original_id`, donc
  **le `manifest.csv` brut produit par `ingest_raw.py` change de schéma
  aussi** -- migration nécessaire pour du `manifest.csv` déjà généré,
  simple renommage d'en-tête). But : Jules avait raison, `specimen_key`/
  `specimen_id` prêtaient à confusion avec le vrai `inv_id` canonique --
  `original_id` est maintenant le nom unique de bout en bout (CSV brut
  d'identification, manifest brut, rapports) pour "l'identifiant tel que
  trouvé dans la source, non garanti unique/stable".
- **Tentative initiale de dropper la colonne `key_column` redondante dans
  `build_specimen_table` (annulée après coup) :** parti du constat que
  `inv_name`/`original_id` étaient byte-identiques sur les 539 lignes
  réelles, j'ai d'abord ajouté un drop automatique de `key_column` --
  mais en retestant sur les vraies données j'ai découvert que **Jules
  avait entre-temps lui-même nettoyé `IDMB_Bombus_collect.csv`** :
  la colonne `inv_name` n'existe plus du tout dans le fichier brut,
  seule `original_id` subsiste comme clé. Avec `--key-column original_id`
  (désormais la bonne valeur), mon drop automatique supprimait alors la
  SEULE colonne d'id brut restante -- l'inverse de ce que Jules voulait
  ("garder original_id"). **Annulé** : `build_specimen_table` ne drop
  plus rien, `key_column` reste tel quel dans la sortie (renommé en
  `<key_column>_raw` seulement en cas de collision avec `inv_id`/
  `inv_num`, comme avant). Piège identifié en testant sur les vraies
  données à chaque étape plutôt qu'en supposant que le fichier brut était
  figé entre deux sessions.
- **Bug trouvé et corrigé en testant** : `attach_canonical_inv_id`
  utilisait en interne le nom littéral `"original_id"` comme colonne de
  jointure temporaire -- collision silencieuse avec la vraie colonne
  `original_id` du CSV brut collection (pandas suffixe en `_x`/`_y`,
  cassant le `.drop()` suivant). Corrigé : colonnes de jointure internes
  préfixées `_original_id_key`/`_clean_inv_id`, jamais un nom qui pourrait
  exister dans les données réelles.
- **`build_specimen_device_table` ajouté** (`manifest/identification.py`)
  -- une ligne par (`inv_id`, `device_type`) réellement utilisée par un
  spécimen, avec le nom de device précis. Câblé dans
  `tools/export_clean_dataset.py` : jointure sur `["inv_id",
  "device_type"]` dans `photos` juste après `assign_photo_ids`, colonne
  `device` ajoutée à `manifest.csv` (après `device_type`). Vérifié sur
  données réelles : `ARLY_SORO_0114` (photos P) -> `Canon EOS 70D - 40mm`,
  (photos S) -> `Smartphone Samsung s10`, correct par photo.
- **`--key-column` de la commande d'usage collection mis à jour** :
  `inv_name` -> `original_id` dans le docstring d'`export_clean_dataset.py`,
  pour refléter le CSV brut déjà nettoyé par Jules.
- `tests/test_identification.py` : 9 tests (ajout/adaptation --
  `build_specimen_device_table`, comportement "key_column conservé" au
  lieu de "dropped"). 69 tests verts au total (60 cœur + 9
  identification), aucune régression.
- **Testé sur les vraies données** (`data/identification/IDMB_Bombus_collect.csv`
  actuel + une copie scratch du `manifest.csv` brut avec l'en-tête migré
  `specimen_id`->`original_id`, supprimée après coup) : `biological_data.csv`
  a bien `original_id`/`n_photos_P`/`n_photos_S` ; `manifest.csv` a bien
  une colonne `device` correcte par photo.
- **À faire par Jules** : régénérer `data/bombus_collection_raw/manifest.csv`
  via `tools.ingest_raw` (nouveau schéma `original_id`), ou migrer l'en-tête
  à la main (`specimen_id` -> `original_id`, aucune valeur ne change) --
  sinon `export_clean_dataset.py` échoue sur ce fichier avec le nouveau
  code. Idem pour `data/identification/inv_id_mapping.csv` existant
  (colonne `specimen_key` -> `original_id`).

Fait (session, 7 sept. 2026, suite) -- Jules a nettoyé le placeholder
`inv_id`/`original_id` du CSV brut collection (n'a plus de suffixe device,
confirmant le bug corrigé plus haut) et a fait remonter un effet de bord
du fix précédent : `n_photos_P`/`n_photos_S` (colonnes pivotées par device
dans `biological_data.csv`) ne sont plus adaptées -- device n'est de toute
façon pas une propriété du spécimen (cf. "Identification des spécimens"),
et le fichier de base d'Adrien ne fait pas non plus cette distinction par
ligne. Décidé avec Jules :
- `build_specimen_table` (`manifest/identification.py`) ne pivote plus
  `n_photos` par device -- une seule colonne `n_photos` = somme sur tous
  les devices du spécimen (remplace `n_photos_P`/`n_photos_S`).
- Nouvelle fonction `build_device_type_table(identification_df,
  device_column, device_name_column)` : une ligne par paire distincte
  (`device_type`, nom de device) vue dans la source -- **pas forcément
  1:1** (vérifié sur les vraies données : `P` de la collection couvre 3
  appareils différents -- deux objectifs du même Canon EOS 70D + un Canon
  PowerShot G16 -- `S` un seul smartphone). Pas d'erreur sur cette
  ambiguïté (comportement réel des données, pas un bug) -- juste une ligne
  par paire. Écrite en sortie d'`export_clean_dataset.py` sous
  `device_types.csv` (à la racine de `--output-dir`, pas dans `reports/` --
  c'est une table de référence du dataset propre, pas un diagnostic
  d'anomalie), seulement si `--device-column` et `--device-name-column`
  (nouveau flag, défaut `device`) sont tous deux présents. Objectif :
  garder la correspondance `device_type` (code court, seul présent au
  niveau photo -- `manifest.csv`, futur CSV photo/TPS de
  `tools/convert_landmarks.py`) -> nom de device lisible, puisque
  `biological_data.csv` ne la porte plus par ligne.
- `tests/test_identification.py` mis à jour : le test de pivot devient
  `test_build_specimen_table_sums_n_photos_across_devices` (somme au lieu
  du pivot), + deux nouveaux tests pour `build_device_type_table`.
  66 tests verts au total (aucune régression).
- Vérifié sur les vraies données (même run que précédemment,
  `--no-copy-images`, scratch supprimé après) : `n_photos` = 5 pour
  `ARLY_SORO_0001` (2 P + 3 S, cohérent avec l'ancien pivot) ;
  `device_types.csv` liste bien les 4 paires réelles
  (`P`/`Canon EOS 70D - 40mm`, `P`/`Canon EOS 70D - 100mm macro`,
  `P`/`Canon PowerShot G16`, `S`/`Smartphone Samsung s10`). Vérifié aussi
  qu'une source sans `--device-column` (terrain) tourne toujours sans
  accroc (pas de `device_types.csv` écrit, comportement inchangé).

Fait (session, 7 sept. 2026) -- retour de Jules après relecture de
`reports/identification_conflicts.csv` (ouvert dans l'éditeur) : les deux
rapports `reports/excluded_identification_rows.csv` et
`reports/identification_conflicts.csv` affichaient encore une ligne par
device (P/S) pour ce qui est en réalité un seul événement au niveau
spécimen (une exclusion, un conflit d'identité) -- le device n'a pas de
sens à ce niveau (cf. `CONVENTIONS.md` "Identification des spécimens").
Root cause diagnostiquée (via un agent Explore puis lecture directe du
code) : les deux rapports sont construits à partir de
`identification_df`/`resolved_all` **avant** que `build_specimen_table`
ne tourne, donc avant que le `inv_id` canonique ne soit jamais substitué --
ils affichaient tel quel un **placeholder `inv_id` hérité du CSV brut,
suffixé par device** (ex. `CD388229c_P`), colonne legacy distincte de la
vraie clé (`inv_name` pour la collection). Ce n'était jusqu'ici géré que
dans `build_specimen_table`, jamais dans ces deux rapports :
- `excluded_identification_rows.csv` : littéralement une ligne par ligne
  brute post-`resolve_identification` (déduplication par
  `[specimen_key, device_column]`, donc jusqu'à une ligne par device par
  spécimen) -- vérifié sur les vraies données : 353 lignes brutes pour
  227 spécimens réellement exclus.
- `identification_conflicts.csv` : le nombre de lignes était déjà correct
  (une par variante d'identité distincte, fix de la session du 6 sept.),
  mais chaque variante affichait un `inv_id` legacy différent (suffixe
  device), ce qui donnait l'impression trompeuse d'un éclatement par
  device alors que le vrai problème était juste la colonne `inv_id`
  affichée.
- Fix : extraction d'une fonction réutilisable
  `attach_canonical_inv_id(df, mapping_df, key_column, source_type)`
  (`manifest/identification.py`) reprenant exactement la logique déjà
  présente dans `build_specimen_table` (drop des colonnes placeholder
  `inv_id`/`inv_num` héritées, renommage `<key_column>_raw` en cas de
  collision, `inv_id` toujours première colonne) -- `build_specimen_table`
  en devient un simple wrapper (comportement inchangé pour ses appelants
  existants). `tools/export_clean_dataset.py` applique maintenant cette
  fonction aux deux rapports une fois `mapping` finalisé (après
  `extend_mapping`, qui garantit un `inv_id` pour tout spécimen du run, y
  compris exclus/en conflit) ; `excluded_identification_rows.csv` est en
  plus dédupliqué à une ligne par `inv_id` (l'exclusion ne dépend jamais
  du device).
- `tests/test_identification.py` créé (n'existait pas encore malgré
  plusieurs sessions de bugs corrigés à la main sur ce module) : 4 tests
  synthétiques (`attach_canonical_inv_id` seul -- cas legacy et collision
  de nom de colonne --, `build_specimen_table` non régressé après le
  refactor, bout-en-bout sur une fixture 3 spécimens dont un exclu et un
  en conflit, tous deux avec des lignes P+S). 64 tests verts au total
  (60 existants + 4 nouveaux), aucune régression.
- Vérifié sur les vraies données (`data/bombus_collection_raw/manifest.csv`
  + `IDMB_Bombus_collect.csv`, `--no-copy-images`, mapping/output dans un
  répertoire scratch puis supprimé) : `excluded_identification_rows.csv`
  353 lignes brutes -> 227 (= nombre de spécimens réellement exclus, une
  ligne chacun) ; `identification_conflicts.csv` toujours 8 lignes (4
  conflits x 2 variantes, inchangé) mais `inv_id` maintenant canonique et
  identique entre les 2 variantes d'un même spécimen (ex. `ARLY_SORO_0114`
  pour les 2 lignes `CD388229c`, au lieu de `CD388229c_P` sur les deux).
- Pas retouché : `missing_biological_data.csv` (photos sans donnée
  d'identification) -- hors périmètre du retour de Jules, ce rapport n'a
  pas de notion d'`inv_id` à corriger (les photos concernées n'ont
  justement aucune ligne d'identification associée).

**Phase actuelle : Phase 1 (`core/`), démarrée.** Groundwork pré-découpage
(Phase 0) terminé pour tout ce qui était planifié ; `landmarks_trainer/`
volontairement reporté à la Phase 4 (voir "Questions ouvertes"). Voir
`TODO.md` pour le détail précis.

Fait (session, 6 sept. 2026) -- retour de Jules après son premier run réel
sur le terrain, reformulation de l'architecture d'export :
  décrit dans l'entrée précédente (colonne `nom_inventaire` au lieu de
  `inv_name`/id complet, `inv_id` en dernière colonne) -- déjà corrigé
  avant ce message, confirmé toujours correct.
- Trois demandes supplémentaires, toutes traitées :
  1. Le dataset propre exporté doit être **indépendant** du brut -- plus
     de `raw_path` dans le manifest de sortie.
  2. Les manifests trainaient des colonnes redondantes (`naming`,
     `source_root` -- redondants avec `raw_path`/la logique de parsing
     elle-même).
  3. Question de Jules, actée : si `export_clean_dataset.py` produit déjà
     un manifest utilisable, `manifest/build_dataset.py` (qui rescanne le
     dossier propre pour en reconstruire un) est redondant.
- **`manifest/build_dataset.py` supprimé.** Remplacé par
  `tools/combine_manifests.py`, un outil volontairement minimal : pour
  plusieurs sources (collection + terrain), il ne fait plus que concaténer
  les `manifest.csv`/`biological_data.csv` déjà propres de chaque run
  `export_clean_dataset` -- aucun rescan disque, hash ou parsing de nom de
  fichier (tout ça a déjà été fait une fois par source). Pour une seule
  source, plus rien à faire du tout après `export_clean_dataset.py`.
  Lève une erreur explicite sur collision de `photo_id` entre sources
  combinées (signe que deux sources partagent involontairement un
  `nom_inventaire`).
- `tools/export_clean_dataset.py` : sortie renommée `manifest_clean.csv`
  -> `manifest.csv`. `raw_path` retiré (plus jamais persisté) ;
  `content_hash`/`file_size_bytes` recalculés sur la copie fraîchement
  écrite dans `--output-dir`, pas hérités du scan brut -- le dataset
  propre se vérifie seul. Exception assumée : `--no-copy-images` (rien
  n'est copié) retombe sur les valeurs du scan brut par défaut, et
  `status="SKIPPED"` -- premier usage réel de ce statut, jusque-là
  seulement présent dans le vocabulaire commun sans être déclenché.
  `specimen_id`/`shot_index` retirés (redondants avec `inv_id`/`photo_id`) ;
  les colonnes biologiques jointes temporairement pour calculer le
  sous-dossier de `--image-group-by` (genus/species/caste...) ne sont plus
  persistées dans le manifest final -- seul `inv_id` y reste, à joindre à
  `biological_data.csv` pour le reste. Ajout : si le manifest brut fourni
  a déjà une colonne `source_type`, il est filtré sur `--source-type`
  automatiquement -- un seul `ingest_raw` peut scanner toutes les racines
  d'un coup, plus besoin de pré-découper le manifest par source à la main.
- `tools/ingest_raw.py` : `image_id` (doublon de `content_hash`),
  `source_root` et `naming` retirés de `manifest.csv`. **Changement de
  schéma `config/roots.json`** : clé `"splits"` -> `"roots"`, champ par
  racine `"split"` -> `"source_type"` -- purge de la dernière trace de la
  connotation train/test dans l'ingestion, cohérent avec la décision de
  laisser le split aux trainers (voir "Questions ouvertes"). **Les
  `roots.json` existants de Jules devront être migrés** avant le prochain
  `ingest_raw`.
- Testé bout en bout dans cet environnement (fixtures existantes,
  collection + terrain) : `ingest_raw` (nouveau schéma) ->
  `export_clean_dataset` (par source) -> `combine_manifests` -- toujours
  pas de test avec les vraies données de Jules au-delà de son retour sur
  le bug terrain.
- **Suite, même session** -- dernier retour de Jules sur
  `reports/identification_conflicts.csv` : le rapport de conflits
  affichait une ligne par ligne brute (donc par device), doublant/
  quadruplant inutilement une même désaccord d'identité quand le
  `specimen_key` en collision avait des lignes P+S pour chacun des deux
  vrais spécimens. Confirmé avec Jules qu'un conflit entre devices d'un
  même spécimen n'a pas de sens (ils s'accordent par construction) --
  corrigé pour ne garder qu'une ligne par variante d'identité distincte
  (`resolve_identification`, `manifest/identification.py`). Testé avec une
  fixture 2 spécimens x 2 devices (4 lignes brutes -> 2 lignes de rapport).
  `collector`/`collector_subfolder` renommés `photographer`/
  `photographer_subfolder` partout (terminologie terrain, pas héritage
  d'une collection de musée) -- **`roots.json` de Jules à migrer une
  deuxième fois** en plus du renommage `splits`->`roots`.

Fait (session, 5 sept. 2026) -- suite directe du nettoyage identification/
dataset du 4 sept., passage à l'architecture définitive :
- Jules a reformulé le besoin après la session du 4 sept. : pas de CSV
  global fusionnant les colonnes des deux sources, mais **un run
  `export_clean_dataset` par `source_type`** (collection, terrain, futurs
  autres datasets), partageant le même `--mapping-file`. `identification.py`
  et `export_clean_dataset.py` réécrits en conséquence -- `_process_source`
  (le branchement `if source_type == "collection"` de la session
  précédente) disparaît entièrement, `main()` ne traite plus qu'une seule
  source par invocation. Détail de l'architecture, des CLI et des deux
  sorties (`biological_data_all.csv`/`biological_data.csv`) dans
  `CONVENTIONS.md` "Export propre par source, un run = une source".
- Conflits d'identité étendus aux spécimens sans image : la résolution
  (`resolve_identification`) tourne maintenant sur la CSV d'identification
  complète, la restriction aux images présentes n'intervenant qu'après --
  réglait du même coup le point 3 de Jules (conflits invisibles quand un
  des deux côtés en collision n'a pas de photo) et son exemple concret
  (même `num_inv`, deux `collection_origin`, un des deux côtés non
  photographié).
- **Bug trouvé après un premier run réel de Jules sur le terrain** :
  quand `--key-column` vaut littéralement `inv_id` (le cas terrain, dont
  la CSV brute a déjà une colonne `inv_id`), `build_specimen_table`
  supprimait cette colonne au lieu de la renommer -- confondue avec le cas
  générique "colonne `inv_id` héritée à jeter". Résultat observé par
  Jules : `inv_id` en dernière colonne (place où `pandas.merge` l'avait
  mise) et `nom_inventaire` (le préfixe de campagne seul) comme seul
  identifiant restant, l'id complet ayant disparu. Corrigé : renommage en
  `<key_column>_raw` au lieu de suppression si collision avec `key_column`,
  `inv_id` toujours réordonné en première colonne du résultat. Même
  traitement appliqué à `assign_photo_ids` par cohérence.
- Deux bugs supplémentaires trouvés en testant moi-même avec des fixtures
  terrain synthétiques (device_type entièrement vide) : colonne lue en
  `float64` par pandas, plantait sur l'assignation de `--default-device-type`
  (`TypeError`) -- cast explicite en `object` ajouté ; et le manifest brut
  n'était pas filtré sur un éventuel statut `FAILED` avant traitement (une
  ligne au nom non reconnu se serait retrouvée classée à tort "image sans
  donnée biologique" plutôt que "échec de parsing du nom") -- filtre
  ajouté (rétrocompatible, no-op si la colonne `status` est absente).
- `tools/ingest_raw.py` créé : reprise du scan brut multi-convention de
  l'ancien `manifest/build_dataset.py` (naming parsers, `collector_subfolder`,
  dédoublonnage par hash), amputé de la jointure identification/
  `specimens.csv` -- ce rôle appartient maintenant entièrement à
  `export_clean_dataset.py`. Statuts alignés sur `utils/pipeline_io.py`
  (`OK`/`SUSPECT`/`FAILED`, `status_reason` explicite) au lieu de
  `parsed_ok`/`unparsed_name`/`unreadable`.
- `manifest/build_dataset.py` réécrit : opère sur un ou plusieurs datasets
  déjà propres (une seule convention de nommage `<inv_id>_<device_type>_<n>`,
  échoue explicitement -- `FAILED` -- sur tout ce qui ne la respecte pas),
  peut combiner plusieurs racines (ex. collection + terrain) via un config
  JSON `{path, identification_csv, source_type}` par racine pour produire
  un manifest pipeline unique. `collector`/`source_root`/`naming` supprimées
  (propres au scan brut, sans objet ici) ; `photo_id` (le nom de fichier
  canonique) comme identifiant, `content_hash` réservé au dédoublonnage.
  Laissé dans `manifest/` (pas déplacé vers `tools/`) par continuité avec
  son emplacement historique, bien qu'au sens strict de la règle
  `core/utils/tools` il soit `main(argv)` + lancement seul comme
  `ingest_raw.py` -- signalé comme question ouverte dans `TODO.md`.
- Testé bout en bout dans cet environnement avec des fixtures synthétiques
  (collection avec collision d'identité + spécimen sans photo ; terrain
  avec `collector` pour le rangement des images) : `ingest_raw` ->
  `export_clean_dataset` (collection et terrain séparément, même
  `--mapping-file`) -> `manifest/build_dataset` combinant les deux racines
  propres en un manifest pipeline unique. Pas encore testé avec les
  vraies données de Jules au-delà de son premier retour sur le bug terrain.
- Reste ouvert (voir `TODO.md`) : la fusion des colonnes d'identification
  de plusieurs racines dans `manifest/build_dataset.py` (concat/outer-join
  simple, comme pandas le fait nativement) n'a pas été explicitement
  validée par Jules à ce niveau -- il avait écarté ce comportement pour
  les sorties d'`export_clean_dataset.py`, mais le besoin diffère ici
  (dataset combiné pour l'entraînement/l'analyse downstream).

Fait (session, 4 sept. 2026) -- hors plan de phases, besoin ad hoc d'Adrien
(export TPS/CSV pour tester en R sur les landmarks posés automatiquement) :
- `tools/convert_landmarks.py` entièrement reconstruit sur `core.tps_io`/
  `utils.dataset.load_dataset()`/`utils.cli.add_dataset_args()` au lieu de
  reparser le TPS à la main (regex) et dupliquer la jointure manifest/
  specimens -- voir "Bugs / incohérences trouvées" plus bas, l'entrée
  correspondante est maintenant résolue.
- Nouvelle réprojection crop -> espace original : `read_image_size()` et
  `apply_wing_transform_to_points_inverse()` ajoutées dans
  `extraction/normalize_crop.py`, à côté de `compute_wing_transform`/
  `apply_wing_transform_to_points` dont elles sont l'inverse algébrique
  exact (même précaution de synchronisation avec `rotate_image()`/
  `crop_with_context()` que documentée là pour le sens forward).
- `tools/convert_landmarks.py` produit maintenant, pour un `--split`
  train/test/all donné : `landmarks_<split>_<n>lm_crop.tps` +
  `landmarks_<split>_<n>lm_original.tps` (IDs séquentiels 1..N, sans
  `COMMENT=`, même ordre entre les deux TPS), `biological_data_<split>.csv`
  (`id`, `specimen_id`, `species`, `caste`, `split`, `device`,
  `device_tag`, + toutes les colonnes de `--identification-csv` si donné),
  `failed_<split>.csv` (`image_id`, `specimen_id`, `stage`, `reason` --
  stages : `landmark_placement`, `outlier_registration`,
  `biological_metadata`, `reprojection`, `other`).
- Décisions prises sans re-solliciter Jules (délégué explicitement --
  "fais ce que tu penses être le plus pertinent, simple et pratique
  d'usage") :
  - Table d'échecs reconstruite dans le nouveau script à partir de
    `landmarks_numbered.csv`/`manifest.csv`/`specimens.csv`, sans toucher
    `utils.dataset.load_dataset()` (fonction partagée par `classifiers/
    train.py`/`predict.py`/`variance_report.py` -- changement jugé trop
    invasif pour ce besoin ponctuel). Portée volontairement limitée à
    `--split` seul : avec `--devices`/`--species`/`--castes` actifs, le
    residual "OK mais pas dans l'export" devient ambigu (filtre voulu vs.
    vraie donnée manquante) et n'est alors pas comptabilisé comme échec
    (juste dans le résumé imprimé).
  - Reprojection crop -> original placée dans `extraction/normalize_crop.py`
    (pas un fichier `utils/` séparé) pour rester à côté de la géométrie
    forward dont elle dépend directement.
  - `--split all` + `--identification-csv` lève une erreur explicite plutôt
    que de deviner : `IDMB_Bombus_collection.csv`/`IDMB_Bombus_terrain.csv`
    n'ont pas de schéma de colonnes commun (cf. template de normalisation
    évoqué par Jules en session, pas encore fait -- voir `TODO.md`).
  - Colonnes du CSV d'identification passées telles quelles (pas de
    renommage anticipé du futur template), sauf collision avec une colonne
    calculée par le script (ex. `caste` existe à la fois dans
    `IDMB_Bombus_collection.csv` en français et comme colonne calculée
    depuis `specimens.csv` en anglais) -- alors préfixée
    `identification_<nom>`.
  - Dédoublonnage de `IDMB_Bombus_collection.csv` sur sa clé (671/1437
    lignes dupliquées, une ligne par appareil/photo) : 1ʳᵉ ligne gardée,
    comme déjà fait pour `specimens.csv`.
  - Échec de réprojection (image brute illisible) : le specimen est exclu
    des TROIS sorties (pas seulement le TPS original), pour garder les IDs
    denses et les 3 fichiers strictement alignés -- tracé dans
    `failed_<split>.csv`, stage `reprojection`.
  - Garde-fou : si 5 images brutes consécutives sont illisibles, le script
    s'arrête avec un message explicite (drive externe probablement pas
    monté) plutôt que d'écrire 2500+ lignes d'échec une par une.
- Testé fonctionnellement dans le zip fourni (pas d'accès aux images
  brutes réelles depuis cet environnement) : `--split train`/`--tps
  landmarks_numbered_18.tps`/`--identification-csv IDMB_Bombus_collection.csv`
  et `--split test`/`landmarks_numbered.tps`/`IDMB_Bombus_terrain.csv`
  bout en bout (2571 et 243 specimens exportés respectivement) ; garde-fou
  `--split all` + `--identification-csv` ; collision de colonne `caste`
  détectée puis corrigée après un premier run bogué ; nettoyage des `nan`
  littéraux dans le CSV (device manquant) détecté puis corrigé de la même
  façon. Géométrie de réprojection crop->original validée séparément avec
  des images de test synthétiques (dimensions/plage de coordonnées
  cohérentes) -- pas de run complet avec le vrai drive, à valider par
  Jules.
- Patch livré en chat : `convert_landmarks_reprojection.patch` (diff sur
  `src/tools/convert_landmarks.py` + `src/extraction/normalize_crop.py`).
  À vérifier par Jules avant de lancer pour de vrai : `--padding`/
  `--out-width`/`--out-height` (défauts 0.10/512/256, doivent matcher ce
  qui a réellement été utilisé pour générer les crops).

Correctif (même session, après retour de Jules sur un run réel) :
- Bug trouvé : le TPS `..._original.tps` réutilisait `sp.image_path` (le
  chemin du **crop**) comme `IMAGE=`, au lieu du chemin de l'image brute
  -- copié-collé du TPS crop, pas remarqué en test faute d'images brutes
  réelles disponibles dans cet environnement pour le vérifier visuellement.
  Corrigé : `reproject_to_raw_space()` renvoie maintenant aussi le
  `raw_path` résolu (celui de `manifest.csv`), utilisé comme `IMAGE=` du
  TPS original ; celui du TPS crop reste `sp.image_path`. Revérifié :
  les deux chemins diffèrent bien et `raw_image_path == manifest["raw_path"]`.
- Patch incrémental livré (`convert_landmarks_original_space_path_fix.patch`,
  vérifié applicable avec `git apply` et `patch -p1` par-dessus le patch
  précédent déjà appliqué par Jules) -- **attention pour la suite** : tant
  que ce fichier n'est pas encore commité côté Jules, régénérer un diff
  complet (`git diff`) depuis cet environnement redonnerait un patch
  cumulatif non applicable en l'état ; il faut soit demander l'état actuel
  du fichier, soit continuer à raisonner en patchs incrémentaux comme
  celui-ci.
- Nouveau point à traiter (voir `TODO.md` "Non planifié") : la détection/
  normalisation devrait elle-même exposer les paramètres réellement
  utilisés (padding, out_width, out_height, mode) plutôt que de compter
  sur le fait que `convert_landmarks.py` les redonne en argument et
  espère qu'ils correspondent -- source d'erreur silencieuse identifiée
  par Jules en repassant sur `--padding`/`--out-width`/`--out-height`.

Fait (session, 4 sept. 2026, suite) -- nettoyage identification / dataset
propre, déclenché par Jules en repassant sur l'export TPS/CSV de la
session précédente (conflits de num_inventaire découverts en creusant
`--identification-csv`) :

- **Diagnostic sur données réelles.** Jules a fourni `IDMB_Bombus_collect.csv`
  (1437 lignes, déjà une passe de nettoyage de son côté), `IDMB_Bombus_terrain.csv`
  (104 lignes, terrain déjà propre) et `manifest.csv` (3069 images). Croisement
  fait dans cet environnement (pandas), pas de simulation :
  - 3 **collisions d'identité réelles** (pas de simples doublons) :
    `CD388229c`, `CD388821d`, `CD388861` -- même étiquette d'inventaire
    utilisée pour deux spécimens biologiquement différents (espèce/caste/
    origine différentes), 4 lignes brutes chacune (2 devices x 2 variantes),
    présentes dans le manifest (vraies photos, pas des lignes fantômes).
    Un 4ᵉ cas (`MET-2016.18309`, écart mineur sur `identification_year`
    seulement) existe dans le CSV mais n'a pas d'image dans le manifest
    actuel -- sans effet pour l'instant.
  - 5 groupes de doublons de ligne inoffensifs (`15300`, `N174`, `N816`,
    `FB.F19.0212`) -- même donnée biologique répétée, aucun conflit.
  - 5 images avec strictement aucune ligne d'identification (`2020.00471`,
    `ABAURA8672b`, `OPPO.7362`, `Polset_1` côté collection + `WB1_23` côté
    terrain -- ce dernier vient de 19 fichiers `terrain/adrien/WB1_23_136.jpg`
    .. `154.jpg` mal parsés par le manifest, numérotation sur 3 chiffres au
    lieu de 4 ; **Jules a renommé ces fichiers sur le disque** pour la
    prochaine ingestion, à revérifier au prochain rebuild du manifest).
  - 364 lignes collection + 12 lignes terrain sans image correspondante
    dans le manifest -- attendu (CSV avec plus de spécimens que
    d'images réellement digitalisées), pas un bug.
- **Décisions prises avec Jules cette session :**
  - `inv_id` (spécimen) sans device -- le device ne vit qu'au niveau
    `photo_id` (`<inv_id>_<device_type>_<i>`), pour ne pas fragmenter un
    même individu en plusieurs "spécimens" selon l'appareil (aurait cassé
    l'analyse de variance inter-device qui a justement besoin de savoir
    qu'il s'agit du même individu). Le besoin d'Adrien (nombre de photos
    par appareil) est couvert par des colonnes (`n_photos_P`/`n_photos_S`)
    sur le CSV specimen-level, pas par des lignes séparées.
  - `NOM_INVENTAIRE` dérivé de `collection_origin` via une table éditable
    à la main, `data/identification/collection_origin_codes.csv` (livrée
    cette session, 11 origines -> 11 codes courts) -- **à relire par
    Jules**, notamment si `Arthropologia` et `Arthropologia Lyon` doivent
    fusionner ou rester deux inventaires distincts (les 3 collisions
    trouvées touchent toutes des variantes du libellé `Arthropologia
    Lyon`, ce n'est peut-être pas un hasard).
  - `inv_num` gelé via une table append-only (`data/identification/
    inv_id_mapping.csv`, générée au premier run) -- jamais recalculé pour
    un spécimen déjà vu, testé explicitement (voir "Testé" plus bas).
  - Conflits résolus automatiquement par "1ère occurrence gardée" (choix
    délégué à Claude par Jules), mais **rien n'est perdu** : toutes les
    lignes des groupes en conflit sont tracées dans
    `reports/identification_conflicts.csv` pour correction manuelle
    ultérieure par Jules/Adrien.
  - Pas de renommage des fichiers sources sur le disque externe -- export
    en **lecture seule** vers un `--output-dir` séparé (images copiées et
    renommées, CSV source jamais modifiés), pour que Jules puisse renvoyer
    un dataset propre à Adrien sans toucher aux données de travail.
- **Livré :** `src/manifest/identification.py` (fonctions pures --
  restriction aux images présentes, résolution conflits/doublons,
  dérivation `NOM_INVENTAIRE`, mapping gelé, table specimen, `photo_id`)
  et `src/tools/export_clean_dataset.py` (CLI orchestrateur, positional
  `manifest`, `--collect-csv`/`--terrain-csv`/`--origin-codes`/
  `--mapping-file`/`--output-dir`, `--no-copy-images`, `--dry-run`,
  `main(argv=None)`/`--verbose`/`--quiet` via `utils.cli`).
- **Testé bout en bout dans cet environnement** sur les 3 fichiers réels
  fournis par Jules (pas d'accès aux images brutes, disque externe non
  monté ici) :
  - Run complet `--no-copy-images` : 630 spécimens propres (538 collection
    + 92 terrain), 3039 photos, 3 conflits détectés et correctement
    tracés (12 lignes brutes), compteurs d'exclusion cohérents avec le
    diagnostic ci-dessus.
  - **Gel vérifié explicitement** : run sur un sous-ensemble du CSV
    collection (retrait d'une origine), puis run sur le CSV complet --
    tous les `inv_id` déjà assignés au premier run sont restés identiques
    au second, seuls les nouveaux spécimens ont reçu un numéro, à la
    suite du max déjà utilisé dans leur groupe. Ré-exécution à l'identique
    (mapping déjà présent) : sorties strictement identiques
    (`biological_data.csv`/`manifest_clean.csv` comparés ligne à ligne).
  - Run avec copie d'images activée (pas de `--no-copy-images`) : échec
    de copie propre et tracé dans `reports/copy_failures.csv` pour les
    3034 images (chemins `/Volumes/EXT DATA/...` inaccessibles depuis cet
    environnement) -- aucun crash, comportement attendu à vérifier
    positivement une fois lancé sur le vrai disque de Jules.
  - Deux bugs trouvés et corrigés en cours de test (avant livraison) :
    (1) des conflits existant **entre deux lignes du même device** (pas
    seulement entre devices différents, ex. deux lignes `P` de
    `CD388229c` avec des espèces différentes) disparaissaient
    silencieusement -- l'ordre des opérations dans
    `resolve_identification` a été inversé (détection de conflit sur
    toutes les lignes brutes du spécimen d'abord, dédoublonnage
    par-device ensuite) ; (2) `manifest.csv` ne renseigne jamais
    `device_type` pour les lignes terrain (colonne vide, seule la
    collection l'utilise) -- corrigé en le forçant à `"S"` pour le
    terrain dans l'orchestrateur (terrain 100% smartphone, confirmé par
    Jules), documenté en commentaire dans le script.
- **Pas testé** (nécessite le vrai disque/les vraies images, à faire par
  Jules) : la copie réelle des fichiers, le `inv_id_mapping.csv` définitif
  (celui produit ici vient des CSV fournis mais n'a pas vocation à être
  celui utilisé -- Jules doit relancer chez lui pour générer le mapping
  qui fera foi).
- **Reste à faire** (voir `TODO.md` "Hors plan -- nettoyage
  identification") : premier run réel par Jules ; brancher le CSV propre
  résultant dans `tools/convert_landmarks.py --identification-csv` à la
  place des CSV bruts.

Fait (session Phase 1, 2 sept. 2026) :
- `src/core/` créé comme nouveau package (pas un renommage de `utils/` --
  `utils/` continue d'exister). Déplacés : `tps_io.py`, `gpa.py`,
  `alignment.py`, `outliers.py`, `model_io.py`. Le sort des 7 autres
  fichiers de `utils/` (`dataset.py`, `predictions.py`, `pipeline_io.py`,
  `run_io.py`, `cli.py`, `repair_images.py`, `tps_overlay.py`) reste
  **une décision ouverte, non traitée cette session** -- ils restent dans
  `utils/` en l'état, seuls leurs imports vers les 5 fichiers déplacés ont
  été corrigés (ils importaient tous `utils.tps_io`, directement ou en
  cascade).
- Tous les imports du reste du codebase vers ces 5 modules mis à jour
  (`from utils.xxx import ...` -> `from core.xxx import ...`), y compris
  les imports croisés entre les 5 fichiers eux-mêmes (`gpa.py` ->
  `core.alignment`, `outliers.py` -> `core.gpa`/`core.tps_io`) et les
  mentions en docstring/commentaire (pas seulement les lignes `import` --
  ex. le docstring d'`alignment.py` référençait `utils/gpa.py`). Détail des
  fichiers touchés dans `TODO.md` Phase 1.
- Vérifié : compilation (`py_compile`) OK sur tout `src/`+`tests/`,
  60 tests toujours verts sans modification (`from core.xxx import ...`
  dans les 5 fichiers de test concernés), import runtime de `core.*` +
  `utils.dataset`/`utils.predictions` (les deux fichiers restés dans
  `utils/` qui dépendent le plus directement de `core.tps_io`) confirmé
  sain.
- `pyproject.toml` non modifié -- le layout `src/` avec
  `packages.find(where=["src"])` découvre `core/` automatiquement au même
  titre que les autres dossiers, aucun changement nécessaire.
- Décisions Phase 2 tranchées par Jules cette session (voir "Questions
  ouvertes" ci-dessous pour le détail) : `landmarks_trainer/` reporté à la
  Phase 4 (priorité basse confirmée), app Streamlit unique, mode terrain
  garde un aperçu overlay, deux scripts distincts (dataset/terrain),
  `reconcile-review` confirmé comme nom définitif. Conception uniquement --
  aucun code Phase 2 commencé, Phase 1 pas terminée (7 fichiers `utils/`
  encore à trancher).

Fait (sessions précédentes) :
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
- ~~`tools/convert_landmarks.py` : le CSV de sortie annoncé
  (`biological_data_<split>.csv`) n'est en réalité jamais écrit~~ --
  **résolu (session 4 sept. 2026)** : script reconstruit sur `core.tps_io`/
  `utils.dataset.load_dataset()`, `bio_output.to_csv(...)` n'est plus en
  commentaire, plus de reparsing TPS à la main par regex. Voir "Où on en
  est" pour le détail complet (nouvelles sorties, réprojection, décisions
  prises).
- `extraction/heavy/`, `extraction/light/` : inspectés cette session,
  RAS niveau logique -- juste la langue + les valeurs `error_reason` à
  traduire (fait).

## Questions ouvertes

- ~~Streamlit vs Gradio pour l'outil 1~~ -- résolu (session du 2 sept.
  2026) : Streamlit. Voir "Détails outil 1" pour les raisons.
- ~~Aligner `landmarks_trainer/train.py` (et le reste du dossier) sur les
  conventions CLI/logging maintenant, ou reporter à la Phase 4~~ -- résolu
  (session Phase 1, 2 sept. 2026) : **reporté à la Phase 4**, priorité
  basse confirmée par Jules.
- ~~App Streamlit unique (page landmarking + page classification) ou deux
  apps séparées ?~~ -- résolu (session Phase 1, 2 sept. 2026) : **une
  seule app.** Jules : focus sur un premier outil (landmarking) simple et
  fonctionnel d'abord, la classification s'ajoute plus tard dans la même
  app de façon incrémentale -- pas de refonte d'archi à prévoir pour
  l'ajout.
- ~~Le mode terrain garde-t-il un aperçu overlay~~ -- résolu (session
  Phase 1, 2 sept. 2026) : **oui**, aperçu overlay conservé en mode
  terrain (image annotée avec landmarks numérotés, sans statut
  persistant/éditable -- cf. distinction avec le mode dataset qui, lui, a
  la validation complète). Le cas "plusieurs photos ad hoc" n'a pas été
  explicitement retranché de "une seule photo" dans la réponse de Jules --
  à confirmer en Phase 2 si le comportement diffère entre les deux au
  moment de l'implémentation.
- ~~Nommage/structure exacte du mode dataset vs mode terrain -- deux
  scripts distincts ou un seul script avec flag de mode ?~~ -- résolu
  (session Phase 1, 2 sept. 2026) : **deux scripts distincts**, Jules :
  "plus simple". Confirme l'option déjà provisoirement envisagée.
- ~~Nom définitif de la commande CLI de réconciliation~~ -- résolu (session
  Phase 1, 2 sept. 2026) : **`reconcile-review` confirmé**, n'est plus un
  nom provisoire.
- **Nouvelle question (session Phase 1, 2 sept. 2026), toujours ouverte :
  le sort des 7 fichiers `utils/` non déplacés en Phase 1**
  (`dataset.py`, `predictions.py`, `pipeline_io.py`, `run_io.py`,
  `cli.py`, `repair_images.py`, `tps_overlay.py`) -- à trancher avant de
  clore la Phase 1. Piste de départ pour la prochaine session : `dataset.py`
  et `predictions.py` sont a priori spécifiques à l'outil 2
  (classification -- chargement/join TPS+CSV, schéma de prédictions), donc
  candidats à finir ailleurs qu'en `core/` (probablement dans le futur
  dossier de l'outil 2, pas encore créé) plutôt qu'en `core/` lui-même ;
  `pipeline_io.py`, `run_io.py`, `cli.py` semblent au contraire assez
  génériques (statut/CSV incrémental pour `pipeline_io.py`, convention de
  sortie de run pour `run_io.py`, briques `argparse` communes pour
  `cli.py`) pour rester utiles aux 3 outils sans être de la géométrie pure
  -- pas évident que ce soit la définition de `core/` telle que posée dans
  "Décisions d'architecture" (`core` = TPS I/O, GPA, alignement, outliers).
  Hypothèse non tranchée, à valider avec Jules plutôt qu'à décider
  unilatéralement.
- **Nouvelle question (session 4 sept. 2026), ouverte : `Arthropologia`
  vs `Arthropologia Lyon` (et ses variantes `stock pratorum`/`stock
  soroeensis`) sont-elles vraiment des inventaires distincts ?** Les 3
  collisions d'identité trouvées en session touchent toutes des lignes
  étiquetées avec une variante de `Arthropologia Lyon` -- pas forcément un
  hasard. La table `data/identification/collection_origin_codes.csv`
  traite actuellement les 4 libellés comme 4 `NOM_INVENTAIRE` séparés
  (`ARTHRO`, `ARLY`, `ARLY_PRAT`, `ARLY_SORO`) ; à confirmer ou fusionner
  par Jules une fois les 3 conflits eux-mêmes réglés avec Adrien.