# idmybee — résumé : manifest + Phases 0/1/2

Document de reprise pour une nouvelle conversation. Contexte, décisions
d'architecture, arborescence, et détail de ce que fait chaque fichier livré
jusqu'ici.

## 1. Problème de départ

Avant ce travail, le pipeline (`crop_wings.py` -> `lm_predictions.ipynb` ->
`reconstruct_tps.py`/`register.py`/`gpa.py` -> `lda.py` -> `predict.py`)
souffrait de plusieurs frictions structurelles :

- Pas de clé commune entre étapes : chaque script réinvente sa façon de
  désigner un specimen/image, jointures manuelles fragiles.
- Les échecs disparaissent : crops `SUSPECT`/`FAILED` ignorés, specimens
  exclus du TPS sans être tracés nulle part.
- Pas d'idempotence : `crop_wings.py` tournait par sous-dossier, produisant
  3 CSV séparés (`organized_wings_log.csv`, `terrain_wings_log.csv`, + vrac).
- Images brutes dispersées (`data/images/wide/`, disque externe), avec une
  sémantique de nommage mal comprise au départ (voir §3).
- Pas de couche de chargement de données partagée entre `lda.py`,
  `predict.py`, `flag_outlier_specimens.py`.

## 2. Décision d'architecture

Un **manifest relationnel en CSV consolidés** (pas de SQLite — choix
explicite de Jules), un fichier par "table", jointes par deux clés stables :

- `specimen_id` : le numéro d'inventaire (num_inv).
- `image_id` : hash du CONTENU du fichier (sha256 tronqué), stable même si
  le fichier est déplacé/copié/renommé, et qui sert aussi de déduplication.

Chaque étape du pipeline lit la sortie de l'étape précédente et écrit sa
propre table dans `data/manifest/` : plus de recalcul de fichiers entiers,
plus de redondance d'information (espèce/caste ne sont écrites qu'une fois,
dans `specimens.csv`).

Principe transversal : **ne jamais perdre une image silencieusement**.
Chaque étape logue un statut (`OK`/`SUSPECT`/`FAILED`) + une raison
d'échec explicite par image, jamais une simple absence.

## 3. Conventions de nommage des images brutes (clarifiées ensemble)

- **organized / vrac** (répertoires locaux) : `num_inv_[S|P]<n>` — S/P ne
  distingue PAS deux appareils différents comme on le pensait au début,
  mais smartphone vs appareil photo ; `<n>` est le n-ième cliché pris avec
  CET appareil pour ce specimen.
- **terrain** (répertoires locaux) : `num_inv_<n>` — pas de lettre, le
  collecteur (donc l'appareil) est déduit du sous-dossier
  (`terrain/adrien/`, `terrain/basile/`, ...) et peut varier d'un num_inv à
  l'autre (contrairement aux autres jeux de données à un seul smartphone).
- **disque externe** : `num_inv-[S|P]-01` — mêmes composantes que
  organized, mais séparées par des tirets plutôt que collées, numéro
  zero-paddé.

Ces 3 conventions sont gérées par un **registre de parseurs** dans
`build_manifest.py` (`NAMING_PARSERS`), pas par du code dupliqué —
en ajouter une 4ᵉ un jour = une fonction de plus dans ce registre.

## 4. Arborescence

```
src/
├── manifest/                     # Phase 0+ : indexation, single source of truth
│   ├── __init__.py
│   ├── build_manifest.py          [FAIT] Phase 0
│   └── io.py                        [FAIT] I/O CSV partagé (toutes phases)
│
├── extraction/                    # Étape 1 : crops d'ailes
│   ├── __init__.py
│   ├── detection.py                 [FAIT] YOLOE (VPE multi-référence)
│   ├── geometry.py                   [FAIT] lecture image + oriented_crop + letterbox
│   ├── qa_clip.py                     [FAIT] CLIP (QA + sélection de candidat)
│   └── crop_wings.py                   [FAIT] CLI Phase 1 (orchestration)
│
├── landmarks/                     # Étape 2
│   ├── __init__.py
│   └── predict_unet.py              [FAIT] Phase 2 (UNet -> landmarks.csv + TPS)
│
├── numbering/                      # Étape 3 : PROPOSÉ, PAS ENCORE FAIT
│   ├── base.py                       contrat commun : numerate(landmarks, ref) -> (numbered, status, score)
│   ├── hungarian_umeyama.py            = register.py actuel, à adapter au contrat
│   ├── graph_matching.py                nouvelle méthode (demandée par Jules), même contrat
│   └── reconstruct_tps.py                CLI : applique une méthode enregistrée, écrit TPS + statuts
│
├── classifiers/                     # Étape 4/5 : PROPOSÉ, PAS ENCORE FAIT
│   ├── lda.py                         existant, à brancher sur un loader partagé
│   ├── landmarks_knn.py                 nouveau : classifieur simple sur coordonnées (demandé par Jules)
│   └── predict.py                        existant
│
├── detector_training/                 # PROPOSÉ, PAS ENCORE FAIT
│   ├── export_obb_dataset.py            crops.csv -> dataset YOLO (les OBB sont déjà loguées en Phase 1)
│   └── train_detector.py                 entraînement du CNN dédié (demandé par Jules)
│
├── tools/                              # existant, inchangé
│   └── flag_outlier_specimens.py
│
└── utils/                               # existant, inchangé
    ├── tps_io.py                         (parse_tps / write_tps — fourni par Jules, pas modifié)
    ├── gpa.py
    ├── reporting.py
    ├── model_io.py
    └── dataset.py                        PROPOSÉ : loader partagé landmarks+labels pour classifiers/*

data/manifest/                       # tables produites/consommées par les scripts ci-dessus
├── images.csv        (Phase 0)
├── specimens.csv      (Phase 0)
├── unparsed.csv         (Phase 0, diagnostic)
├── duplicates.csv         (Phase 0, diagnostic)
├── crops.csv                (Phase 1)
└── landmarks.csv               (Phase 2)

config/
├── roots.example.json             racines locales (organized/terrain/vrac) pour build_manifest.py
└── external_roots.example.json      racine(s) disque externe, format organized_hyphen

docs/
├── manifest_schema.md              schéma détaillé des tables Phase 0
└── idmybee_resume_manifest_phases_0_1_2.md   ce document
```

## 5. Ce que fait chaque script (en détail)

### Phase 0 — `src/manifest/build_manifest.py`

Scanne les racines d'images brutes (locales + disque externe si monté),
**sans jamais déplacer/renommer/modifier un fichier**. Parse chaque nom de
fichier selon la convention de la racine (`NAMING_PARSERS`), calcule un
hash de contenu (`image_id`), joint avec le CSV d'identification
espèce/caste. Écrit :

- `images.csv` : une ligne par photo (image_id, specimen_id, dataset,
  collector, device_type, shot_index, raw_path, naming, status_ingest,
  is_duplicate_content, ...).
- `specimens.csv` : une ligne par num_inv (species, caste, is_labeled,
  n_images, datasets_present...).
- `unparsed.csv` : noms qui ne correspondent à aucune convention connue —
  à revoir à la main, rien n'est perdu.
- `duplicates.csv` : groupes de fichiers au contenu strictement identique.

Racine absente (ex: disque externe non monté) -> avertissement, pas de
plantage. Testé sur fixtures synthétiques reproduisant les 3 conventions +
un cas de racine non montée + un doublon de contenu.

### Phase 1 — `src/extraction/{crop_wings.py, detection.py, geometry.py, qa_clip.py}`

`crop_wings.py` (orchestration CLI) lit `images.csv` (+ `specimens.csv` si
`--only_labeled`), filtre (`status_ingest=parsed_ok`, doublons de contenu
exclus, `--dataset`/`--only_labeled` optionnels), et pour chaque image :
détection YOLOE (`detection.py`) -> découpe orientée + letterbox
(`geometry.py`) -> QA/sélection de candidat CLIP (`qa_clip.py`). Écrit un
**unique `crops.csv`** indexé par `image_id` (remplace les 3 CSV
organized/terrain/vrac séparés d'avant). Nom de sortie dérivé du manifest
(`<dataset>_<specimen_id>_<device><shot>_<image_id>.jpg`), pas du chemin
d'entrée. `crops.csv` est **append-only** : un consommateur doit garder la
DERNIÈRE ligne par `image_id` (mêmes conventions utilisées par
`load_target_crops` en Phase 2).

Bugs corrigés dans `crop_wings.py` par rapport à la version initiale :
- Sélection multi-candidats par `conf+similarity` (deux échelles non
  normalisées) au lieu de la similarité CLIP seule -> pouvait annuler
  l'intérêt même du mécanisme de désambiguïsation.
- Padding asymétrique (`py` utilisait `pad` complet, `px` utilisait `pad/2`).
- `.heic`/`.heif` invisibles (`IMG_EXTS` trop restreint) -> `read_image()`
  tolérant (cv2 puis repli PIL/pillow-heif), erreur explicite sinon.
- Ré-exécuter le script dupliquait le log et rejouait les échecs à
  l'infini (`idx` repartait de 0 en mode append) -> résumabilité par
  `image_id`, `--retry_failed` explicite, refus de démarrer si un
  `crops.csv` existant a un schéma différent (évite la corruption
  silencieuse en mode append).

`geometry.py`/`detection.py`/`qa_clip.py` : découpage mono-responsabilité
du script original (aucune connaissance mutuelle des CSV/du manifest).

### Phase 2 — `src/landmarks/predict_unet.py`

Repris du notebook `lm_predictions.ipynb` fourni par Jules. Lit
**seulement `crops.csv`** (pas besoin d'`images.csv`/`specimens.csv` à ce
stade). Filtre `status=OK` (`--include_suspect` pour inclure aussi
`SUSPECT`), déduplique par dernière occurrence d'`image_id`. Pour chaque
crop : UNet -> heatmap -> extraction des landmarks -> écrit dans
`landmarks.csv` (statut/raison/nombre trouvé) et dans le TPS via
`utils/tps_io.py` (`parse_tps`/`write_tps`, fournis par Jules, non modifiés).

Contrainte découverte en cours de route : `tps_io.Specimen.sid` doit être
un **entier**, et `write_tps` **réécrit tout le fichier** (pas d'append).
Conséquences sur la conception :
- `sid = int(image_id, 16)` (pas `specimen_id`, qui collisionnerait entre
  les plusieurs photos d'un même specimen ; pas une position de ligne,
  fragile sous reprise/retraitement partiel comme dans le notebook d'origine).
- Le script fonctionne par **checkpoint** : `parse_tps` recharge l'existant
  au démarrage (erreurs de blocs explicitement affichées, jamais juste
  avalées), chaque image traitée met à jour un dict en mémoire
  (`{sid: Specimen}`), et `write_tps` + réécriture complète de
  `landmarks.csv` ont lieu tous les `--log_every` + une dernière fois à la
  fin. Un retry/overwrite **remplace** proprement l'entrée existante
  (contrairement à `crops.csv`, pas de risque de doublon ici).

Bugs corrigés par rapport au notebook original :
- `get_image()` retranchait 2 caractères fixes du nom de fichier pour
  isoler le specimen_id -> cassait dès que le numéro de photo dépassait 1
  chiffre. Plus nécessaire : `specimen_id` vient déjà de `crops.csv`.
- `device='D1'` codé en dur pour tous les specimens terrain, alors qu'on a
  établi que le collecteur (donc l'appareil) varie par specimen terrain.
  Disparaît avec la simplification ci-dessous.
- **Simplification** : le notebook réécrivait sa propre `terrain.csv`
  (espèce/caste/device dérivés à la main) — entièrement supprimé, cette
  info vit déjà dans `specimens.csv`/`images.csv`.
- `local_maxima()` peut marquer plusieurs pixels adjacents pour un même pic
  (plateau) ; les traiter comme des landmarks séparés puis garder "les N
  plus hauts" pouvait faire disparaître un vrai landmark distinct au profit
  de doublons du même pic -> regroupement par composante connexe
  (`scipy.ndimage.label`) avant classement. Testé sur un heatmap
  synthétique (3 pics isolés + 1 plateau de 4 pixels).
- `maximas[-18:,:]` ne levait aucune erreur avec moins de 18 maxima trouvés
  (troncature silencieuse) -> statut `SUSPECT` explicite + raison si moins
  de landmarks que prévu.
- `ID=` du TPS était un index de ligne positionnel (`i` de
  `valids.iterrows()`), cohérent seulement tant que 2 fichiers restent
  synchronisés ligne à ligne sans jamais être filtrés indépendamment ->
  remplacé par `sid_for(image_id)`, stable sous reprise/filtrage partiel.
- TPS réécrit en entier à CHAQUE image dans la boucle d'origine (O(n²),
  aucune reprise possible) -> checkpoints périodiques (cf. ci-dessus).

## 6. État actuel / limites de ce qui a été testé

Aucun des scripts n'a pu être exécuté sur les vraies données ni sur les
vrais modèles (YOLOE/CLIP/UNet, poids `.pth`) dans cet environnement —
seulement sur des fixtures synthétiques et des versions "stubbées" de
torch/ultralytics pour valider la logique pure : parsing de noms de
fichiers, filtrage/déduplication des tables, résumabilité (skip/retry/
overwrite sur plusieurs runs simulés), géométrie (angle, padding,
letterbox), dédup des maxima locaux, et **round-trip réel avec
`utils/tps_io.py`** (write_tps -> parse_tps sans aucune erreur de
reparsing). La logique métier spécifique à YOLOE/CLIP/UNet (le contenu
exact de `detection.py`/`qa_clip.py`/l'appel au modèle UNet) reprend le
code de Jules quasiment tel quel et n'a donc pas eu besoin d'être re-testée
en profondeur, mais n'a pas tourné de bout en bout sur GPU.

## 7. Prochaines étapes (pas commencées)

Dans l'ordre suggéré, mais Jules choisit :

1. **Valider Phases 0/1/2 sur les vraies données** (avant d'aller plus
   loin) : lancer `build_manifest.py`, vérifier `unparsed.csv` (probablement
   surtout `vrac/`), lancer `crop_wings.py` puis `predict_unet.py` sur un
   sous-ensemble.
2. **Phase 3** (`numbering/`) : sortir `register.py` (Hungarian+Umeyama)
   derrière un contrat commun (`numerate(landmarks, reference) ->
   (numbered, status, score)`), écrire `landmarks_numbered.csv` avec
   statut d'alignement par specimen (au lieu d'exclure silencieusement les
   échecs du TPS de sortie comme aujourd'hui), puis ajouter la méthode par
   graphe en tant que second module respectant le même contrat.
3. **Phase 4/5** (`classifiers/`) : extraire un loader partagé
   (`utils/dataset.py`) depuis `lda.py` pour pouvoir brancher un
   classifieur simple sur coordonnées de landmarks sans dupliquer la
   logique GPA/jointure.
4. **Détecteur CNN dédié** (`detector_training/`) : les OBB sont déjà
   loguées dans `crops.csv` depuis la Phase 1 — export au format YOLO,
   split train/val par `specimen_id` (pas par image, pour éviter les fuites
   entre plusieurs photos d'un même specimen).