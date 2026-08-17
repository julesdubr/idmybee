# idmybee — résumé de session : Phases 3/4 (numérotation + classification)

Document de reprise pour une nouvelle conversation. Fait suite à
`idmybee_resume_manifest_phases_0_1_2.md` (Phases 0-2, déjà fourni dans une
session précédente) — ce document couvre uniquement ce qui a changé depuis :
Phase 3 (numérotation), Phase 4 (classification), et les bugs découverts en
testant sur données réelles.

## 0. Objectif final (à ne pas perdre de vue)

Le but n'est pas seulement de valider un pipeline batch. L'objectif est,
pour **chaque étape** (crop, landmarks, numérotation, classification),
d'obtenir un modèle **optimisé par CV** sur l'ensemble des données
disponibles (peu importe la méthode retenue à chaque étape), afin de
pouvoir ensuite chaîner ces modèles dans un outil **Phase 5** qui traite
**une seule image** en entrée (interface envisagée : Streamlit) et retourne
une prédiction d'espèce/caste de bout en bout. Le travail actuel (batch,
CSV manifest, scripts CLI) est l'étape de *préparation/validation* de ces
modèles, pas le produit final.

## 1. Décisions d'architecture prises cette session

### 1.1. `Specimen` → `ImageLandmarks` (changement le plus structurant)

Le TPS liste des **photos**, pas des spécimens biologiques (un spécimen a
souvent plusieurs photos). L'ancien nom `Specimen.sid` laissait croire le
contraire et a causé plusieurs bugs de jointure. Renommé dans
`utils/tps_io.py` :

```python
@dataclass
class ImageLandmarks:
    n_points: int
    landmarks: np.ndarray
    image_path: str
    tps_id: int                     # valeur brute du TPS ID= (entier, unique par PHOTO)
    image_id: str | None = None     # hash hex canonique, si connu
    specimen_id: str | None = None  # si connu
```

- `tps_id = image_id_to_sid(image_id) = int(image_id, 16)` -- toujours dans
  ce sens (jamais `hex(sid)[2:]` pour retrouver `image_id` : perd les zéros
  initiaux, PAS fiable — vérifié empiriquement, corrige une fausse
  affirmation qui était dans l'ancien docstring de `predict_unet.py`).
- `image_id`/`specimen_id`, quand connus, sont persistés dans un
  `COMMENT=image_id=...;specimen_id=...` du fichier .tps.
  **Vérifié sans risque pour la compatibilité R** :
  `geomorph::readland.tps` ignore explicitement tout COMMENT/variable/radii
  ("all other information...is ignored", doc officielle).
- `ImageLandmarks.from_image(n_points, landmarks, image_path, image_id, specimen_id=None)`
  calcule `tps_id` automatiquement -- c'est ce qu'utilise maintenant
  `predict_unet.py` (Phase 2) au lieu de construire l'objet à la main.
- Alias `Specimen = ImageLandmarks` gardé en bas de `tps_io.py` par
  prudence (fichiers non vus dans cette conversation, ex. `manifest/io.py`,
  qui pourraient encore importer `Specimen` -- **`.sid` n'existe en
  revanche plus nulle part**, renommé `.tps_id` : à grep si un crash
  `AttributeError: 'ImageLandmarks' object has no attribute 'sid'`
  apparaît quelque part hors de cette conversation).

### 1.2. Kabsch/Umeyama : un seul cœur SVD partagé

`gpa.py` et `register.py` réimplémentaient chacun la même algèbre
(rotation optimale sans réflexion) sous deux formulations différentes.
Extrait dans `utils/alignment.py::kabsch_umeyama(source_c, target_c, estimate_scale)` --
vérifié algébriquement équivalent aux deux implémentations d'origine
(rotation seule pour la GPA, rotation+échelle pour la registration).

### 1.3. Convention CSV unifiée pour les diagnostics qualité

`numbering/reconstruct_tps.py` (statut par photo, colonnes
`tps_id, specimen_id, status, ...`) et `tools/flag_outlier_specimens.py`
(idem, `status` au lieu de l'ancien `heavy` booléen) partagent maintenant
la même convention. `--exclude-ids` de `lda.py` accepte plusieurs fichiers
à la fois (`nargs="+"`), tous filtrés sur `status != "OK"` -- un seul
mécanisme au lieu de deux à traitement spécial.

### 1.4. Filtrage automatique, non optionnel

`utils/dataset.py::load_labeled_dataset` écarte maintenant *toujours*
(pas de flag CLI) les photos dont le nombre de landmarks diffère du schéma
majoritaire -- un nombre de points incohérent ne peut jamais entrer dans
une GPA, ce n'était pas censé être une décision laissée à l'utilisateur.

## 2. Arborescence actuelle (src/)

```
src/
├── manifest/            build_manifest.py, io.py                      [INCHANGÉ cette session]
├── extraction/           crop_wings.py, detection.py, geometry.py,
│                          qa_clip.py                                    [INCHANGÉ cette session]
├── landmarks/
│   └── predict_unet.py     [MODIFIÉ] ImageLandmarks.from_image() au lieu de
│                              Specimen(sid=...) ; embarque specimen_id via COMMENT=
├── numbering/
│   ├── base.py               [NOUVEAU] contrat NumberingResult / numerate(landmarks, reference)
│   ├── hungarian_umeyama.py  [NOUVEAU, ex register.py] + détection d'ambiguïté
│   │                            d'orientation (voir §4.2, PAS calibrée)
│   └── reconstruct_tps.py    [NOUVEAU emplacement] écrit data/manifest/landmarks_numbered.csv,
│                                exclut les FAILED du TPS de sortie, --ambiguity-ratio exposé
├── classifiers/
│   ├── __init__.py           [NOUVEAU]
│   ├── lda.py                 [NOUVEAU emplacement] --level species|caste, --exclude-ids
│   │                             (multi), --exclude-species, --images-csv, --out-dir
│   └── predict.py              [NOUVEAU emplacement] colonnes tps_id/image_id/specimen_id
├── tools/
│   └── flag_outlier_specimens.py  [REECRIT] seuil MAD calculé PAR ESPÈCE (voir §3),
│                                     sortie par défaut data/manifest/outlier_specimens.csv
└── utils/
    ├── tps_io.py            [REECRIT] ImageLandmarks (voir §1.1)
    ├── alignment.py          [NOUVEAU] cœur SVD partagé (voir §1.2)
    ├── gpa.py                [MODIFIÉ] utilise utils.alignment
    ├── dataset.py             [NOUVEAU] chargement + filtres partagés (voir §1.3, §1.4)
    ├── reporting.py           [MODIFIÉ] group_summary_table/device_summary_table prennent
    │                             `groupe` directement (plus de re-dérivation depuis meta_df[level])
    ├── model_io.py            [INCHANGÉ cette session]
    └── register.py            [SUPPRIMÉ, remplacé par numbering/hungarian_umeyama.py]

data/manifest/
├── images.csv, specimens.csv, crops.csv, landmarks.csv    [Phases 0-2, inchangés]
├── landmarks_numbered.csv     [NOUVEAU] statut par photo (Phase 3)
└── outlier_specimens.csv       [NOUVEAU emplacement] statut par photo (post-GPA, Phase 3.5)

scripts/
└── run_pipeline.sh            [NOUVEAU] enchaîne Phases 0->4 (voir fichier joint)
```

**specimens.csv, colonnes utilisées** : `specimen_id, species, caste,
is_labeled` (+ `datasets_present`, `in_identification_csv`, `in_images`,
`n_images` non utilisées par le code actuel).
**images.csv, colonnes utilisées** : `image_id, specimen_id, device_type`
(repli seulement -- voir §1.1, la plupart des TPS récents n'en ont plus
besoin grâce au COMMENT=).

## 3. Bugs réels trouvés et corrigés cette session (par ordre chronologique)

1. **`ModuleNotFoundError: numbering`** -- `reconstruct_tps.py` (point
   d'entrée) n'avait pas le bootstrap `sys.path` présent dans `lda.py`/
   `predict.py`. Corrigé.
2. **`KeyError: 'species'`** -- `load_labeled_dataset` joignait `sid` (en
   fait un `image_id` déguisé) directement contre `specimen_id` de
   `specimens.csv` : count de non-appariés énorme, puis `meta_df` vide →
   `KeyError` cryptique au lieu d'une erreur claire. Cause racine : voir
   §1.1. Corrigé par la jointure en deux étapes (COMMENT= puis repli
   `images.csv`), avec erreur explicite si 0 spécimen apparié.
3. **`--level caste` classait sur la caste seule** (worker/queen/male),
   mélangeant les espèces sous un même label. Corrigé : `--level caste`
   classe sur `species_caste` (colonne `groupe`), voir
   `utils/dataset.py::target_groupe`.
4. **`ValueError: Landmark count incohérent`** -- une photo à 6 landmarks
   (détection UNet incomplète) faisait planter la GPA. Cause : un `FAILED`
   de `reconstruct_tps.py` était quand même écrit dans le TPS de sortie.
   Corrigé à deux niveaux : `reconstruct_tps.py` n'écrit plus les `FAILED`
   dans le TPS (juste dans le log), et `load_labeled_dataset` filtre en
   plus, toujours, tout nombre de landmarks minoritaire (défense en
   profondeur, §1.4).
5. **`--exclude-ids` a exclu 2858/2858 spécimens (100%)** -- un des deux
   CSV passés n'avait pas de colonne `status` (ancien format `heavy`) ;
   le code traitait alors TOUT le fichier comme à exclure au lieu de
   lever une erreur. Corrigé : colonnes `tps_id`+`status` obligatoires
   dans tout fichier `--exclude-ids`, sinon `ValueError` explicite.
6. **1097/2858 (38%) exclus par `flag_outlier_specimens.py`**, très
   au-dessus du taux historique (~2.6%). Cause : seuil médiane+MAD calculé
   sur TOUTES les espèces mélangées -- une espèce à la forme d'aile
   différente (c'est le principe même de la classification) est alors
   comparée à une forme moyenne qui ne la représente pas, et sur-marquée
   outlier. Corrigé : seuil calculé **par espèce** (`--min-group-size`,
   défaut 10, en dessous le groupe est ignoré plutôt que d'inventer un
   seuil peu fiable). A fait tomber le taux à 666/2858 (~23%), toujours
   élevé mais concentré sur *ruderarius*/*rupestris* (visible sur le plot
   GPA fourni par Jules : ces deux espèces forment des nuages de points
   dispersés au lieu de clusters serrés par landmark).
7. **Accuracy 91% → 79% → 71%** en excluant *rupestris*/*ruderarius*
   (contre-intuitif : ça aurait dû remonter). Sur le plot GPA suivant,
   plusieurs amas de points nettement séparés du reste, tournés à ~90° --
   **pour la plupart des espèces**, pas seulement les deux exclues. La GPA
   étant par construction invariante en rotation, ce n'est pas elle qui
   échoue : ce sont des photos **mal numérotées** d'une façon qui *imite*
   une rotation de 90° (mauvaise correspondance landmark-par-landmark),
   probablement parce que le multi-départ de `hungarian_umeyama.numerate()`
   (angles 0/90/180/270 × miroir) tombe parfois sur un faux optimum à un
   coût presque aussi bas que le bon. Palliatif ajouté : comparaison
   meilleur/deuxième-meilleur départ, marque `SUSPECT` si trop proches
   (`--ambiguity-ratio`, voir §4.2) -- **PAS calibré**, voir problèmes
   ouverts.

## 4. Problèmes ouverts (à traiter en priorité dans la prochaine session)

### 4.1. Le vrai goulot d'étranglement : la numérotation (Phase 3)

C'est le problème principal actuellement, pas la classification. Deux
symptômes distincts, peut-être liés, peut-être pas :
- *rupestris*/*ruderarius* : nuages dispersés, pas de cluster net par
  landmark (voir plot GPA fourni).
- Un sous-ensemble transverse à "la plupart des espèces" : amas isolés
  tournés à ~90° du reste.

Le multi-départ Hungarian+Umeyama (4 rotations × 2 miroirs, coût le plus
bas retenu) n'est visiblement pas assez robuste pour une fraction non
négligeable des photos. C'est exactement le problème que la Phase 3 par
graphe (`numbering/graph_matching.py`, jamais commencée) est censée
adresser à terme -- ce n'est peut-être plus la peine d'investir beaucoup
plus dans le réglage fin de Hungarian+Umeyama avant d'avoir essayé
l'alternative.

### 4.2. `--ambiguity-ratio` non calibré

Avec la valeur par défaut (1.3), 2577/2858 (90%) sont marqués `SUSPECT` --
inutilisable tel quel. Deux hypothèses à trancher en regardant la vraie
distribution des ratios meilleur/deuxième-meilleur coût (pas encore
exportée nulle part, à ajouter si utile) :
- Le seuil est juste mal réglé (trop permissif) → resserrer.
- Le ratio meilleur/deuxième-meilleur n'est pas un signal fiable pour ce
  problème (un miroir ou une petite variation donnent souvent un coût
  proche même sur une registration correcte) → il faudrait un autre signal
  (ex: cohérence de l'orientation prédite par CLIP en Phase 1, déjà
  utilisée pour la QA des crops -- comparer l'angle qu'elle a déterminé à
  celui du meilleur départ retenu ici).

**En attendant : ne pas utiliser `landmarks_numbered.csv` comme
`--exclude-ids`** (sur-exclut), utiliser `outlier_specimens.csv` seul.

### 4.3. Deux scripts non vérifiés dans cette conversation

`src/manifest/build_manifest.py` et `src/extraction/crop_wings.py` n'ont
jamais été collés dans cette conversation -- le script `run_pipeline.sh`
les appelle avec des options reconstruites depuis le résumé Phase 0-2,
**pas vérifiées**. A corriger au premier run si les noms d'options sont
faux.

### 4.4. `manifest/io.py` jamais vu

Utilisé par `predict_unet.py` (`manifest_io.read_table`,
`check_schema`, `load_existing_by_key`, `should_skip`). S'il référence
`Specimen`/`.sid` quelque part, il n'a pas été corrigé cette session.

## 5. Perspectives (améliorations, pas des blocants)

Dans l'ordre suggéré par l'objectif final (§0) :

1. **Diagnostiquer et fiabiliser la numérotation** (§4.1) -- priorité
   immédiate, bloque une évaluation fiable de tout le reste.
2. **`numbering/graph_matching.py`** -- méthode de numérotation
   alternative par graphe, même contrat `numbering.base.numerate()` que
   Hungarian+Umeyama, donc substituable directement dans
   `reconstruct_tps.py` sans autre changement.
3. **`detector_training/`** -- détecteur CNN dédié pour la Phase 1
   (remplacerait/compléterait YOLOE), OBB déjà loguées dans `crops.csv`
   depuis la Phase 1, split train/val par `specimen_id`.
4. **Classifieur NN** en complément/remplacement de LDA (`classifiers/`) --
   utiliserait le même `utils/dataset.py`.
5. **Phase 5** -- une fois les 3 étapes ci-dessus validées par CV chacune
   séparément : outil bout-en-bout (crop → landmarks → numérotation →
   classification) sur une image unique, interface Streamlit envisagée.
   Rien commencé sur ce point.

## 6. Pour démarrer la prochaine conversation efficacement

1. **Coller ce document en premier message.**
2. **Coller les fichiers sources actuels** des modules les plus susceptibles
   d'être retouchés en premier (probablement, vu §4.1) :
   `src/numbering/hungarian_umeyama.py`, `src/numbering/reconstruct_tps.py`,
   `src/numbering/base.py`, `src/utils/alignment.py`, `src/utils/gpa.py`.
   Le reste (`classifiers/`, `utils/dataset.py`, `tools/flag_outlier_specimens.py`)
   n'est probablement pas nécessaire tant que la numérotation n'est pas
   fiabilisée -- mais les coller aussi ne coûte rien si le sujet dérive.
3. Si tu as pu exporter la distribution des ratios meilleur/deuxième-meilleur
   coût (§4.2) ou des exemples visuels de TPS mal numérotés superposés à
   leurs images (ruderarius en particulier), c'est probablement la
   donnée la plus utile à apporter pour avancer vite sur §4.1.
4. Mentionner explicitement si `build_manifest.py`/`crop_wings.py` (§4.3)
   ont dû être corrigés pour faire tourner `run_pipeline.sh`, pour que ce
   soit su du départ plutôt que redécouvert.