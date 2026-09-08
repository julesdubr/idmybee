# Conventions

Ce document fixe les conventions du projet. Il sert de référence unique ;
tout code nouveau ou retouché doit s'y conformer. Statut : **validé**
(voir chat).

## Langue

Tout passe en anglais : code, docstrings, commentaires, messages de log,
aide CLI, et les valeurs de champ type `error_reason`/`status` écrites dans
les CSV (ex: `image_illisible_ou_format_non_supporte` ->
`unreadable_image_or_unsupported_format`). Les CSV déjà produits avec des
valeurs en français ne sont pas migrés rétroactivement -- seules les
NOUVELLES lignes écrites après la bascule utilisent les nouveaux libellés.
Ça crée une incohérence entre anciennes et nouvelles données sur ce point
précis ; à garder en tête si un script filtre sur la valeur exacte d'un
`error_reason`. Les identifiants de domaine (espèce, caste...) restent tels
quels (ex: noms latins, `rupestris`).

## Docstrings

Le module `utils/tps_io.py`, `utils/alignment.py`, `utils/run_io.py` et
`utils/pipeline_io.py` sont la référence de style à suivre partout.

**Module** : description courte du rôle, puis seulement ce qui est
non-évident (convention de fichier, piège connu, référence à un bug déjà
corrigé). Pas de rationale historique complète -- un lien vers un commit ou
une phrase suffit.

**Fonction/classe** : une ligne de description, puis inputs/outputs si pas
évidents de la signature. Un exemple d'usage seulement si l'API n'est pas
auto-descriptive. Detail algorithmique uniquement pour les étapes
réellement non triviales (ex. le correctif Kabsch-Umeyama documenté dans
`alignment.py`).

**À éviter** : paraphraser la signature, expliquer *pourquoi* une fonction
existe si ce n'est pas surprenant, dupliquer un `# commentaire` qui répète
juste le nom de la variable suivante.

## Nommage CLI

- Argument requis unique et central du script (dataset, tps, model_path...) :
  **positionnel**. Tout le reste : `--flag` en kebab-case.
  → généraliser le style déjà en place dans `classifiers/train.py`,
  `classifiers/predict.py`, `analysis/variance_report.py` à
  `extraction/*.py` et `landmarks/*.py` (aujourd'hui en `--dataset`
  obligatoire).
- Variable du parser : `parser` (pas `ap`, `p`, etc.).
- Entrée de script :
  ```python
  def main(argv: list[str] | None = None) -> None:
      parser = argparse.ArgumentParser(...)
      ...
      args = parser.parse_args(argv)

  if __name__ == "__main__":
      main()
  ```
  Systématique -- même pour un script "jetable" dans `tools/` -- pour rester
  testable/appelable en dehors du CLI (utile pour la future UI Streamlit).

## Logging

- `logger = logging.getLogger(__name__)` en tête de chaque module qui
  produit des messages de statut/progression.
- `print()` réservé à la sortie destinée à l'utilisateur final en mode CLI
  (un rapport, un tableau récap) -- jamais pour du statut interne
  ("traitement de X...", "N images ok").
- Un seul point de configuration du logging, `setup_console_logging(verbose:
  bool = False)` (déjà présent dans `utils/run_io.py`, à généraliser/déplacer
  au niveau du `cli.py` commun) -- `--verbose`/`--quiet` partagés par tous
  les scripts plutôt que redéfinis à la main.

## Imports

- Le projet est installé en mode éditable (`pip install -e .`, voir
  `pyproject.toml`) -- imports absolus depuis la racine `src/`
  (`from utils.tps_io import ...`, `from landmarks.methods.base import ...`).
- Plus de `sys.path.insert(...)`, plus d'import "nu" qui ne marche qu'en
  exécution directe du script (`from constants import ...` depuis un
  fichier du même dossier). Si un module a besoin d'un autre module du même
  package, import qualifié complet.

## Fonctions core réutilisables (pas seulement CLI)

*(Ajouté session du 2 sept. 2026, suite à la conception de l'outil 1 --
voir `RESUME.md` "Détails outil 1".)*

- Toute étape de pipeline appelée par plusieurs outils (ex. la pose de
  landmarks, consommée à la fois par le mode dataset de l'outil 1 et par
  son mode terrain/single) doit exister comme **fonction Python pure**
  (entrée -> sortie en mémoire, ex. `place_landmarks(image) -> Landmarks`)
  en plus de son entrée CLI/fichier. Le CLI/l'UI appellent cette fonction
  et gèrent l'I/O disque autour -- la fonction elle-même ne lit/écrit
  jamais de fichier.
- Même principe pour le dessin d'overlay annoté (landmarks numérotés sur
  l'image, utilisé par l'étape de validation) :
  `draw_landmarks_overlay(image, landmarks) -> Image`, partagée telle
  quelle entre CLI (export d'overlays sur disque) et UI (affichage direct).
- Objectif : éviter un aller-retour disque (écrire un TPS à une ligne puis
  le relire) pour des usages "ad hoc" comme la prédiction terrain sur une
  photo isolée.

## Placement d'un fichier : `core/` vs `utils/` vs `tools/`

*(Ajouté session du 2 sept. 2026, Phase 1, suite à l'extraction de `core/`
et au tri des fichiers restés dans `utils/` -- voir `RESUME.md` Décisions
d'architecture #4.)*

- **`core/`** : géométrie/I-O pure, sans dépendance à `argparse`/CLI,
  partagée par plusieurs outils (ex. `tps_io.py`, `gpa.py`,
  `alignment.py`, `outliers.py`, `model_io.py`).
- **`utils/`** : logique partagée mais pas purement géométrique (I/O de
  run, CLI, dataset/join, etc.), **référencée (importée) par au moins un
  autre module**. Reste ici tant qu'au moins un autre module l'utilise
  comme bibliothèque -- ne pas créer de sous-modules pour une poignée de
  fichiers (pas la peine en dessous d'une dizaine).
- **`tools/`** : script CLI autonome, **lancement seul**
  (`main(argv)` + `if __name__ == "__main__"`), **aucune référence
  ailleurs dans le code** -- seulement invoqué via `python -m
  tools.mon_script`.
- Critère de décision : dès qu'un fichier n'est référencé (importé) par
  aucun autre module et n'a qu'un usage `python -m ...`, il va dans
  `tools/`. À l'inverse, si un script de `tools/` commence à être importé
  ailleurs, il doit remonter vers `utils/` (ou `core/` si la logique
  ajoutée est de la géométrie/I-O pure).

### Clôture de la question ouverte `utils/` -> `core/`, et sous-découpage de `tools/` (session 8 sept. 2026)

*(Répond à l'item resté ouvert depuis la Phase 1 -- "Décider du sort de
`utils/dataset.py`, `predictions.py`, `pipeline_io.py`, `run_io.py`,
`cli.py`, `repair_images.py`, `tps_overlay.py`" -- voir `TODO.md` Phase 1
et `RESUME.md` "Emplacement final de `utils/`".)*

- **`utils/dataset.py`, `predictions.py`, `pipeline_io.py`, `run_io.py`
  déplacés vers `core/`.** Les quatre satisfont le critère déjà écrit
  ci-dessus pour `core/` (aucune dépendance à `argparse`/CLI, partagés par
  plusieurs outils) même s'ils ne sont pas de la géométrie au sens strict
  (jointure dataset, schéma de prédictions, tracking de run/statuts,
  convention de nommage des modèles) -- le critère retenu au final est
  "pas de dépendance CLI, partagé largement", pas "géométrique
  spécifiquement". `core/` compte maintenant 9 fichiers : les 5 déjà là
  (`tps_io.py`, `gpa.py`, `alignment.py`, `outliers.py`, `model_io.py`) +
  ces 4.
- **`utils/repair_images.py` déplacé vers `tools/maintenance/`.** C'est un
  script CLI autonome (`python -m ...`, jamais importé ailleurs) --
  répondait déjà au critère `tools/` ci-dessus, resté dans `utils/` par
  oubli plutôt que par choix.
- **`utils/cli.py` et `utils/tps_overlay.py` restent dans `utils/`** :
  `cli.py` est intrinsèquement lié à `argparse` (ne peut pas aller dans
  `core/`) ; `tps_overlay.py` a un usage double (fonctions réutilisées --
  `draw_landmarks` par `app/single_image.py` et `app/build_dataset.py` --
  ET son propre CLI), ce qui l'exclut de `tools/` (référencé ailleurs).
  `utils/landmarking_pipeline.py` et le nouveau `utils/review.py` (voir
  plus bas) sont dans le même cas : glue d'orchestration partagée, pas de
  la géométrie pure, donc `utils/` plutôt que `core/`.
- **`tools/` découpé en trois sous-paquets** (`ingestion/`, `pipeline/`,
  `maintenance/`, chacun avec son propre `__init__.py` documentant son
  rôle) plutôt que laissé en vrac -- la commande CLI change en conséquence,
  ex. `python -m tools.build_manifest` devient `python -m
  tools.ingestion.build_manifest`. Répartition : `ingestion/` = brut vers
  manifest propre (`ingest_raw.py`, `export_clean_dataset.py`,
  `build_manifest.py`, `combine_manifests.py`, `prepare_dataset.py`) ;
  `pipeline/` = orchestrateurs dataset-agnostiques + outils de validation
  (`export_final_landmarks.py`, `export_review.py`, `reconcile_review.py`,
  `train_dataset.py`, `predict_dataset.py`) ; `maintenance/` = scripts
  ponctuels sans rapport avec le pipeline courant (`clean_tps.py`,
  `clean_images_from_tps.py`, `convert_heic_to_jpeg.py`,
  `drop_landmark_from_tps.py`, `flatten_image_dirs.py`, `verify_tps.py`,
  `repair_images.py`). `tools/pipeline/export_final_landmarks.py` reste le
  seul fichier de `tools/` importé ailleurs (par
  `utils/landmarking_pipeline.py::run_export`) -- exception assumée au
  critère "aucune référence ailleurs" ci-dessus, déjà le cas avant ce
  découpage.
- **`src/manifest/io.py` supprimé** : confirmé mort (déjà signalé comme
  piste de nettoyage dans `TODO.md`, aucune référence nulle part dans
  `src/`/`tests/`, reste du schéma pré-refactor `images.csv`/
  `specimens.csv`/`crops.csv`).
- Migration mécanique vérifiée par `py_compile` sur tout `src/`+`app/`,
  `--help` sans crash sur chaque script CLI déplacé, et la suite de tests
  complète (109 tests, tous verts) -- aucun changement de comportement,
  seulement des chemins d'import/de commande.

## Tests

- `pytest`, un fichier `tests/test_<module>.py` par module de `src/` qui
  contient de la logique pure (géométrie, parsing, nommage). Pas besoin de
  données réelles : données synthétiques minimales dans le test lui-même.
- Toute fonction dans `core/` (`gpa.py`, `alignment.py`, `tps_io.py`,
  `outliers.py`, `model_io.py`, `dataset.py`, `predictions.py`,
  `run_io.py`, `pipeline_io.py`) doit avoir un test qui ne dépend pas du
  dataset -- c'est le filet de sécurité du refactor. Même exigence pour
  `utils/review.py` (logique de validation partagée CLI/UI, pas
  purement géométrique mais tout aussi facile à tester en synthétique --
  voir `tests/test_review.py`).
- Un bug corrigé une fois (ex. le facteur d'échelle Kabsch-Umeyama) doit
  laisser un test de non-régression, pas seulement une note dans le
  docstring.

## Identification des spécimens (`inv_id`)

*(Ajouté session du 4 sept. 2026, suite au nettoyage des CSV
d'identification collection/terrain -- voir `RESUME.md` "Où on en est".)*

- **Granularité specimen vs photo.** Deux CSV distincts, deux granularités :
  - CSV maître d'identification (`biological_data.csv`, sortie de
    `tools/export_clean_dataset.py`) : **une ligne par spécimen**,
    `inv_id` sans device. Le besoin d'Adrien (nombre de photos par
    appareil) est couvert par des colonnes -- `n_photos_P`/`n_photos_S`
    pour la collection (deux appareils), une seule `n_photos` pour le
    terrain (smartphone uniquement, pas de distinction utile).
  - CSV associé au TPS (sortie de `tools/convert_landmarks.py`, Phase 2) :
    **une ligne par photo**, `photo_id` avec device, jointe au CSV maître
    par `inv_id`. Ne pas mélanger les deux granularités dans un même
    fichier -- c'est la source de confusion identifiée en session.
- **`inv_id`** (specimen) : `<NOM_INVENTAIRE>_<XXXX>`, **sans le device**
  -- un spécimen de collection est photographié par plusieurs appareils
  (P et S), le device ne doit donc pas apparaître à ce niveau sous peine
  de fragmenter un même individu physique en plusieurs "spécimens"
  distincts (ça casserait entre autres l'analyse de variance inter-device
  qui a justement besoin de savoir qu'il s'agit du même individu).
- **`photo_id`** : `<inv_id>_<device_type>_<i>`, avec `i` un ré-indexage
  propre 1..n par (spécimen, device) -- ne pas réutiliser tel quel le
  `shot_index` du manifest, qui n'est pas garanti contigu à partir de 1.
- **`NOM_INVENTAIRE`** :
  - Collection : dérivé de `collection_origin` via une table de
    correspondance éditable à la main,
    `data/identification/collection_origin_codes.csv` (`collection_origin
    -> nom_inventaire`). Pas de code dérivé automatiquement (slug) --
    des libellés `collection_origin` proches (ex. `Arthropologia Lyon` vs
    `Arthropologia Lyon (stock pratorum)`) ne sont pas forcément
    interchangeables (voir les collisions ci-dessous), la décision de
    fusionner ou non reste un choix humain à faire dans cette table.
  - Terrain : le nom de campagne existant sert déjà de `NOM_INVENTAIRE`
    (ex. `WB1_23`), aucun renommage -- le terrain est **déjà propre**,
    l'`inv_id` terrain est repris tel quel (parsé en `<nom>_<numéro>`).
- **`XXXX` gelé.** Une fois assigné, un `inv_num` n'est jamais recalculé.
  La table `data/identification/inv_id_mapping.csv` est la source de
  vérité, **append-only** : un nouveau run n'assigne un numéro que pour un
  spécimen encore jamais vu, toujours après le max déjà utilisé dans son
  groupe `NOM_INVENTAIRE`. Les nouveaux spécimens d'un même run sont
  ordonnés par (`yyyy`, `mm`, `dd`) ; environ la moitié des lignes
  collection n'ont pas de date complète -- celles-ci sont classées après
  les lignes datées de leur groupe, départagées par `specimen_key` (ordre
  déterministe, pas chronologique).
- **Conflits d'identité.** Deux spécimens physiquement différents peuvent
  avoir reçu la même étiquette dans les données brutes (ex. `CD388229c`
  utilisé pour un *B. soroeensis* drone ET un *B. sichelii* worker). Ce
  n'est pas un doublon de ligne anodin (celui-là se résout par simple
  dédoublonnage silencieux) : la première occurrence (ordre du fichier
  source) est retenue comme valeur canonique, mais **toutes** les lignes
  du groupe en conflit sont tracées dans un rapport
  (`reports/identification_conflicts.csv`, colonne
  `matches_first_occurrence`) -- rien n'est perdu silencieusement, à
  corriger manuellement par Jules/Adrien. Voir
  `manifest/identification.py::resolve_identification` pour
  l'implémentation exacte (le conflit peut se manifester entre deux
  lignes du même device, pas seulement entre devices différents).

## Export propre par source, un run = une source

*(Ajouté session du 5 sept. 2026, suite au passage de
`tools/export_clean_dataset.py` en "un run = une source". Mise à jour
session du 7 sept. 2026, suite 7 : ce script ne produit plus
`manifest.csv`/`biological_data.csv` directement -- sa sortie devient
`dataset.csv`, voir "Manifest découplé du nettoyage, `build_manifest.py`
créé" plus bas. Le reste de cette section (une source par run,
`--image-group-by`, etc.) reste valable tel quel.)*

- Une invocation de `tools/export_clean_dataset.py` traite **une seule**
  source (collection, terrain, ou un futur autre jeu -- ex. un autre
  pollinisateur) -- jamais de fusion de plusieurs sources dans un même
  run. Pour un dataset avec plusieurs sources, lancer le script une fois
  par source, en pointant `--mapping-file` vers le **même** fichier pour
  toutes (l'espace `inv_id` reste unique entre elles, distingué en interne
  par `source_type`).
- Deux sorties par run : `biological_data_all.csv` (tous les spécimens
  identifiés de cette source, photographiés ou non -- utile à Adrien même
  sans photo) et `biological_data.csv` (uniquement ceux avec au moins une
  image, joint à `manifest_clean.csv`). Pas de CSV combiné entre sources --
  si besoin d'une vue combinée, la faire a posteriori depuis les deux
  fichiers séparés, jamais en sortie de cet outil.
- Deux modes de dérivation de `nom_inventaire`, choisis par la présence ou
  non de `--origin-codes` (jamais par le nom de la source en dur) : table
  de correspondance + numérotation séquentielle par date (collection), ou
  parsing direct de `--key-column` déjà sous forme `<nom>_<numéro>`
  (terrain). Voir `manifest/identification.py::assign_nom_inventaire`.
- Rangement des images copiées piloté par `--image-group-by` (liste de
  colonnes, ex. `genus,species,caste` pour la collection ou `collector`
  pour le terrain) -- pas de branchement en dur sur le nom de la source.
  Une colonne demandée mais absente est une erreur explicite au lancement,
  pas un dossier `unknown_*` généralisé à toutes les lignes.
- **Piège identifié et corrigé (session du 5 sept.) :** si `--key-column`
  porte le même nom que la colonne canonique de sortie (`inv_id` -- cas du
  terrain, dont la colonne brute s'appelle déjà `inv_id`),
  `build_specimen_table` renomme la colonne brute en `<key_column>_raw`
  au lieu de la supprimer par erreur (elle était auparavant confondue avec
  un éventuel `inv_id` hérité à jeter) -- sans quoi l'identifiant complet
  de la source disparaissait de la sortie, ne laissant que le
  `nom_inventaire` (préfixe de campagne seul, pas l'id complet). `inv_id`
  est désormais toujours la première colonne de `biological_data*.csv` et
  `manifest_clean.csv`.

## Statuts harmonisés OK/SUSPECT/SKIPPED/FAILED, hors classification/landmarks

*(Ajouté session du 5 sept. 2026. Mise à jour importante le 6 sept. 2026:
`manifest/build_dataset.py` mentionné ci-dessous a depuis été **supprimé**
-- voir "Manifest indépendant du brut, `build_dataset` supprimé" plus bas.
Section laissée telle quelle pour l'historique du raisonnement sur les
statuts, qui reste valable pour `ingest_raw.py` et `export_clean_dataset.py`.)*

- `tools/ingest_raw.py` (scan brut) et `manifest/build_dataset.py`
  (manifest pipeline, sur dataset déjà propre) utilisent désormais le même
  vocabulaire de statut que le reste du pipeline (`utils/pipeline_io.py` --
  `RunCounter`, `OK`/`SUSPECT`/`SKIPPED`/`FAILED`) au lieu de valeurs
  locales (`parsed_ok`/`unparsed_name`/`unreadable`, `is_duplicate_content`
  en booléen à part). `SKIPPED` reste inutilisé par ces deux scripts --
  aucun mécanisme de reprise, rescan complet à chaque lancement -- mais
  fait partie du vocabulaire commun si une reprise incrémentale est
  ajoutée plus tard. `SUSPECT` = contenu dupliqué (même hash qu'un fichier
  déjà vu dans le run) ; `FAILED` = fichier illisible ou nom ne
  correspondant à aucune convention reconnue (`ingest_raw.py`) / à la
  convention canonique attendue (`build_dataset.py`).
- Séparation des deux étapes de scan : `tools/ingest_raw.py` (dossier brut
  potentiellement multi-convention -> `manifest.csv` brut, **sans**
  jointure avec une identification -- ce rôle appartient entièrement à
  `tools/export_clean_dataset.py` désormais) et `manifest/build_dataset.py`
  (dataset **déjà propre**, une seule convention de nommage
  `<inv_id>_<device_type>_<n>`, refuse explicitement -- `status=FAILED` --
  tout fichier qui ne la respecte pas plutôt que de deviner). Un dataset
  déjà propre dès le départ peut sauter `ingest_raw`/
  `export_clean_dataset` et aller directement à `build_dataset`.
  `manifest/build_dataset.py` peut combiner plusieurs racines déjà propres
  (ex. collection + terrain) en un seul manifest pipeline, via un config
  JSON listant `{path, identification_csv, source_type}` par racine --
  voir le docstring du module.
- `photo_id` (le nom de fichier canonique lui-même) est l'identifiant ;
  `content_hash` sert uniquement à détecter les doublons de contenu
  (`SUSPECT`), plus jamais comme identifiant. Colonnes supprimées de
  `manifest/build_dataset.py` par rapport à l'ancienne version :
  `collector`, `source_root`, `naming` -- propres au scan brut
  multi-convention, sans objet une fois sur dataset propre à convention
  unique.

## Manifest indépendant du brut, `build_dataset` supprimé

*(Ajouté session du 6 sept. 2026, suite au retour de Jules après le
premier run réel.)*

- **`manifest/build_dataset.py` supprimé.** Son rôle (re-scanner un
  dossier propre pour reconstruire un manifest) est redondant :
  `tools/export_clean_dataset.py` produit déjà directement `manifest.csv` +
  `biological_data.csv`, le couple complet attendu par la suite du
  pipeline (extraction/crop -> landmarks -> GPA -> PCA -> LDA). Pour une
  seule source, il n'y a donc **rien d'autre à faire** après
  `export_clean_dataset.py`. Pour plusieurs sources (ex. collection +
  terrain), `tools/combine_manifests.py` les concatène -- aucun rescan
  disque, aucun hash, aucun parsing de nom de fichier, puisque
  `export_clean_dataset.py` a déjà tout fait une fois par source. Lève une
  erreur explicite si `photo_id` entre en collision entre deux sources
  combinées (signe que leurs espaces `nom_inventaire` se chevauchent --
  à vérifier dans les tables de correspondance/`inv_id_mapping.csv`
  des sources concernées).
- **`manifest.csv` (sortie d'`export_clean_dataset.py`, renommé depuis
  `manifest_clean.csv`) est indépendant du dataset brut** : ne contient
  plus `raw_path` (ni `specimen_id`/`shot_index`, l'identifiant est
  `photo_id`/`inv_id`) -- rien ne pointe vers l'ancien emplacement des
  images. `content_hash`/`file_size_bytes` sont recalculés sur la copie
  fraîchement écrite, jamais hérités du scan brut -- le dataset propre est
  vérifiable seul, sans accès à l'ancien. Exception : avec
  `--no-copy-images` (aucune copie écrite), ces deux colonnes retombent
  sur les valeurs du scan brut par défaut faute de mieux, et `status` vaut
  `SKIPPED` (nouvel usage réel de ce statut, jusqu'ici inutilisé -- voir
  section précédente) plutôt que `OK`.
- Colonnes bio (`genus`/`species`/`caste`/`collector` de rangement...) ne
  sont plus jointes dans `manifest.csv` : seul `inv_id` y figure, à joindre
  à `biological_data.csv` si besoin des colonnes biologiques -- règle
  générale, pas seulement pour `--image-group-by`. `collector` reste une
  colonne du manifest quand la source la fournit (photo-level, pas
  biologique -- qui a pris la photo, pas une propriété du spécimen).
- Schéma allégé de `tools/ingest_raw.py` (`manifest.csv` brut) :
  `image_id` supprimé (doublon de `content_hash`), `source_root` supprimé
  (dérivable de `raw_path` si vraiment nécessaire), `naming` supprimé de
  la sortie (reste une notion interne au parsing -- `NAMING_PARSERS` --
  mais son résultat n'a pas besoin d'être tracé par ligne). `split`
  renommé `source_type` dans le schéma `config/roots.json` (clé
  `"splits"` renommée `"roots"`) -- cohérence avec le reste de la session
  du 5 sept., et suppression de la connotation train/test héritée. **Les
  `roots.json` existants de Jules doivent être mis à jour** (`"splits"` ->
  `"roots"`, chaque entrée `"split"` -> `"source_type"`).
- `tools/export_clean_dataset.py` filtre désormais son manifest d'entrée
  sur `--source-type` si une colonne `source_type` y est présente -- un
  seul run d'`ingest_raw.py` peut donc scanner toutes les racines brutes
  d'un coup (collection + terrain), chaque run d'`export_clean_dataset.py`
  ne prenant que sa part sans que Jules ait à pré-découper le manifest à
  la main.

## Conflits : une ligne par identité distincte, pas par device

*(Ajouté session du 6 sept. 2026, retour de Jules sur
`reports/identification_conflicts.csv`.)*

- `resolve_identification` (`manifest/identification.py`) groupe déjà la
  détection de conflit sur `specimen_key` seul (jamais sur le device), et
  ne l'a jamais fait autrement. Le vrai problème était le **rapport** :
  il renvoyait telles quelles toutes les lignes brutes du groupe en
  conflit -- si un même `specimen_key` recouvrait deux vrais spécimens
  ayant chacun une ligne par device (P+S), le rapport affichait 4 lignes
  pour 2 identités réellement en désaccord.
- Corrigé : le rapport ne garde plus qu'une ligne par **variante
  d'identité distincte** au sein du groupe (première occurrence de chaque
  variante, ordre du fichier), indépendamment du nombre de devices qui la
  portent. Principe retenu : un désaccord entre deux lignes d'un même
  `specimen_key` signifie toujours deux spécimens physiquement différents
  -- deux devices du même spécimen s'accordent par construction, le device
  n'est donc jamais un axe pertinent pour un conflit.
- `photographer` (rangement des images terrain) : la colonne/le flag
  s'appelait `collector`/`collector_subfolder` par héritage de l'ancien
  `manifest/build_dataset.py` -- renommé `photographer`/
  `photographer_subfolder` partout (`tools/ingest_raw.py`,
  `config/roots.json`, `--image-group-by`, `manifest.csv`) pour coller au
  vocabulaire du terrain plutôt qu'à celui d'une collection de musée.
  **Les `roots.json` existants de Jules ont aussi ce renommage à faire**
  (`"collector_subfolder"` -> `"photographer_subfolder"`).

## Manifest découplé du nettoyage, `build_manifest.py` créé

*(Ajouté session du 7 sept. 2026, suite 7 -- retour de Jules : l'étape
"pré-pipeline" -- copier les images localement puis les symlinker vers un
disque externe -- jugée confuse, et `export_clean_dataset.py` jugé faire
deux métiers à la fois (nettoyage d'identité + construction du manifest).)*

- **`tools/export_clean_dataset.py` ne construit plus `manifest.csv`/
  `biological_data.csv`.** Sa sortie devient `dataset.csv` (une ligne par
  photo copiée avec succès, colonnes biologiques fusionnées directement --
  dénormalisé) + `biological_data_all.csv` (inchangé, tous les spécimens y
  compris sans photo). Ce script reste **optionnel** : à lancer uniquement
  si le dataset brut a vraiment besoin d'une résolution d'identité
  (étiquette réutilisée pour deux spécimens différents, plusieurs
  conventions de nommage, etc.).
- **`src/manifest/build.py` + `tools/build_manifest.py` créés** : nouvel
  outil, **toujours lancé**, seul point d'entrée réel vers la suite du
  pipeline. Prend n'importe quel CSV par photo (une ligne par photo,
  colonnes obligatoires `inv_id`/`species`/`caste` + une colonne chemin) --
  qu'il vienne d'`export_clean_dataset.py` ou d'un dataset tiers avec ses
  propres noms de fichiers et son propre tableur -- et produit
  `manifest.csv`/`biological_data.csv`, le contrat exact attendu par
  `utils.dataset.load_dataset()`. Ne copie jamais de fichier, ne résout
  aucun conflit d'identité (validation structurelle uniquement : fichiers
  présents/lisibles, `photo_id` unique, cohérence des colonnes biologiques
  par `inv_id`) -- si le dataset ne respecte pas ce standard, il écrit
  `manifest_raw.csv` (même schéma, colonne `status`/`status_reason`)
  plutôt que de deviner.
- **Deuxième retournement, volontaire.** `manifest/build_dataset.py`
  faisait quelque chose de similaire et a été supprimé le 6 sept. 2026
  (voir "Manifest indépendant du brut" ci-dessus) car jugé redondant une
  fois qu'`export_clean_dataset.py` produisait directement le manifest.
  `build_manifest.py` n'est **pas** la même erreur répétée : contrairement
  à `build_dataset.py` (qui ne faisait que rescanner un dossier déjà nommé
  selon la convention canonique du projet), `build_manifest.py` fonctionne
  sur n'importe quel dataset compatible, y compris un qui n'est jamais
  passé par l'outillage de ce projet -- piloté par un CSV explicite, pas
  par une convention de nommage figée.
- **Symlinks supprimés.** `tools/link_external_data.py`/
  `config/derived_root.json` n'ont plus de raison d'être : `build_manifest.py`
  ne copie jamais de fichier (il référence les images là où elles se
  trouvent déjà), et `export_clean_dataset.py --output-dir` peut déjà
  écrire directement sur un volume externe sans copie locale intermédiaire.
  Voir `PIPELINE.md` "Dataset roots can live anywhere on disk".
- **`tools/combine_manifests.py` : dérive doc/code trouvée et corrigée.**
  Ce fichier était marqué fait dans `TODO.md`/`RESUME.md` depuis la session
  du 7 sept. (suite 6 et avant) mais n'existait pas réellement dans le
  dépôt -- créé pour de vrai cette fois, opère maintenant sur les
  `manifest.csv`/`biological_data.csv` déjà produits par `build_manifest.py`
  (même logique de concat pure qu'annoncé, juste jamais implémentée).

## Validation review : format des CSV, `reviewed_status` (session 8 sept. 2026)

*(Convention introduite avec `utils/review.py` -- voir `PIPELINE.md`
"Validation review" pour le mécanisme complet.)*

- Un DataFrame de review garde toujours `auto_status` (jamais modifié une
  fois construit) à côté de `reviewed_status` (éditable, initialisé à la
  même valeur) -- jamais un seul champ `status` réutilisé pour les deux :
  perdre la trace de la valeur automatique interdirait de savoir, après
  coup, ce qui a été corrigé à la main vs ce qui était déjà comme ça.
- Les fichiers réconciliés (`crops_reviewed.csv`, `landmarks_reviewed.csv`)
  gardent le **même schéma** que le fichier qu'ils remplacent
  (`crops.csv`, `landmarks_numbered.csv`) -- c'est ce qui leur permet de
  se brancher directement sur un mécanisme déjà existant (`--crops-csv`,
  `--landmarks-status-csv`) sans qu'aucun code consommateur n'ait besoin
  de savoir qu'une review a eu lieu.
- Le fichier d'audit (`<dataset>/review/<nom>_review.csv`) ne contient que
  `photo_id, auto_status, reviewed_status` -- délibérément minimal, pensé
  pour être ouvert et modifié dans un tableur (`tools.pipeline.export_review`
  / `reconcile_review`) sans risquer d'y modifier accidentellement une
  colonne technique du fichier source.

## `--model-name` : nom descriptif, jamais dans le chemin (session 8 sept. 2026)

- `core.model_io.TrainedModel.model_name` (rempli par `classifiers.train
  --model-name`) est **purement descriptif** -- jamais utilisé pour
  construire `run_id`/le chemin de sortie (`core.run_io.build_run_id`
  reste la seule source de vérité pour ça, dérivé de
  niveau/dataset/appareils/source de landmarks pour rester reproductible).
  Un sélecteur de modèle (CLI ou UI) doit toujours passer par
  `core.run_io.model_display_name(model_path)` plutôt que d'afficher le nom
  de dossier `run_id` brut -- cette fonction lit `metrics.json` (jamais le
  pickle du modèle lui-même, coût minimal) et retombe sur `run_id` si
  aucun nom n'a été donné.

## Ce qui n'est *pas* couvert ici (volontairement, à traiter dans le
découpage à venir)

- Convention finale de sortie de run (`run_io.py` vs `landmarks_trainer/
  checkpoint.py` vs convention native Ultralytics) -- attend la
  réorganisation en `landmarking/` / `classification/` / `training/`.
- ~~Emplacement final de `utils/` (deviendra probablement `core/`)~~ --
  **tranché (session 8 sept. 2026)** : voir "Clôture de la question
  ouverte `utils/` -> `core/`" plus haut.