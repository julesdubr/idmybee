# idmybee — identifier des bourdons à partir d'une photo d'aile

Ce dépôt contient le pipeline que j'ai développé pour identifier des
bourdons (genre *Bombus*) — espèce et caste — à partir d'une simple photo
d'aile antérieure, en utilisant de la morphométrie géométrique automatisée
par deep learning.

Ce document est volontairement long et détaillé : il ne sert pas
seulement de mode d'emploi, mais aussi de support pour expliquer la
démarche à quelqu'un qui n'a jamais fait de morphométrie géométrique
(mon directeur de stage, en particulier). Si vous cherchez juste "comment
je lance truc", allez directement à la section **Scénarios d'usage**. Pour
le détail technique exact de chaque étape (entrées/sorties/CLI), voir
[`PIPELINE.md`](PIPELINE.md), qui est la référence à jour ; ce README en
est la version pédagogique, en français.

## Sommaire

1. [Contexte et objectif](#1-contexte-et-objectif)
2. [Idée générale de l'approche](#2-idée-générale-de-lapproche)
3. [Notions de morphométrie géométrique](#3-notions-de-morphométrie-géométrique)
4. [Le pipeline, étape par étape](#4-le-pipeline-étape-par-étape)
5. [Comment lire les scores de l'appli (registration cost, distance de Procrustes)](#5-comment-lire-les-scores-de-lappli)
6. [Organisation du dépôt](#6-organisation-du-dépôt)
7. [Scénarios d'usage](#7-scénarios-dusage)
8. [Points d'attention et limites connues](#8-points-dattention-et-limites-connues)
9. [Pour aller plus loin](#9-pour-aller-plus-loin)

## 1. Contexte et objectif

Distinguer les espèces de bourdons (et leur caste — fondatrice/reine,
ouvrière, mâle) est un exercice classique mais qui demande de l'expertise :
il faut souvent examiner des critères fins (pilosité, genitalia pour les
mâles...) sur un spécimen en main, ce qui n'est pas toujours possible sur
le terrain ou à partir d'une simple photo. La forme de la nervation des
ailes, en revanche, est un caractère morphologique connu pour être
discriminant entre espèces, stable, et surtout : visible sur une photo
ordinaire, sans dissection.

L'idée du projet est de partir d'une collection de référence de
spécimens déjà identifiés (actuellement 13 espèces de *Bombus*, ~2600
spécimens de la collection Arthropologia Lyon, avec plusieurs photos par
individu) pour entraîner un modèle qui apprend à discriminer les espèces
(et castes) à partir de la seule forme de l'aile, puis à appliquer ce
modèle sur de nouvelles photos — y compris des photos de terrain, prises
dans des conditions moins contrôlées qu'en collection (éclairage, angle,
appareil photo différents).

Contraintes de départ :
- ne pas s'appuyer sur un placement manuel des points de repère sur
  l'aile (trop lent pour être utilisable en routine) — il faut l'automatiser ;
- fonctionner à partir d'une photo "brute" (aile non pré-détourée,
  cadrage variable), pas d'une image déjà isolée et alignée ;
- rester utilisable par quelqu'un qui n'a pas écrit le code : d'où
  l'application Streamlit pour la prédiction sur une image.

## 2. Idée générale de l'approche

La discipline qui étudie la forme des organismes indépendamment de leur
taille et de leur orientation s'appelle la **morphométrie géométrique**.
Le principe général (bien établi en biologie, utilisé historiquement avec
des logiciels comme `geomorph` sous R) est :

1. Placer un ensemble de points anatomiques homologues (des **landmarks**)
   sur chaque spécimen — les mêmes points, dans le même ordre, sur chaque
   aile.
2. Superposer toutes ces configurations de points entre elles en ne
   gardant que ce qui relève de la **forme** (on retire la position, la
   taille et l'orientation, qui n'ont pas de sens biologique ici).
3. Analyser statistiquement les formes ainsi normalisées pour voir ce qui
   les distingue — ici, discriminer les espèces/castes.

Ce que ce projet ajoute par rapport à un usage "manuel" classique de la
morphométrie géométrique, c'est l'automatisation complète des étapes 1 et
2 à partir d'une photo brute, via deux modèles de deep learning :

- un **détecteur** qui localise l'aile dans la photo (elle peut être prise
  sous n'importe quel angle, avec le reste de l'insecte ou une règle
  visible à côté, etc.) et en extrait un recadrage standardisé ;
- un **UNet** (réseau de segmentation) qui, sur ce recadrage, prédit la
  position de chacun des 19 landmarks anatomiques définis par Tancrède
  (voir `docs/tancrède_roger_rapport.pdf`), sous forme de cartes de
  chaleur (heatmaps) — un pic de probabilité par landmark.

Une fois les 19 points obtenus (mais pas encore *numérotés* : le réseau ne
sait pas dans quel ordre les rendre), le reste de la chaîne est de la
morphométrie géométrique "classique" : renumérotation, superposition de
Procrustes (GPA), réduction de dimension (PCA), puis classification
supervisée (LDA) pour prédire l'espèce (ou la caste).

Schéma d'ensemble :

```
photo brute
   -> détection de l'aile (YOLO, boîte orientée)
   -> recadrage normalisé (crop)
   -> placement des 19 landmarks (UNet)
   -> renumérotation (association aux 19 points de référence)
   -> alignement de Procrustes (GPA) sur une forme de référence
   -> projection PCA puis LDA (modèle déjà entraîné)
   -> prédiction : espèce (ou caste) la plus probable, + 2 alternatives,
      + scores de confiance et de qualité
```

## 3. Notions de morphométrie géométrique

Cette section explique le vocabulaire qu'on retrouve dans le code et dans
l'appli, pour quelqu'un qui découvre la méthode.

**Landmark (point-repère)** — Un point anatomique précis et reconnaissable
d'un individu à l'autre (une intersection de nervures, par exemple), placé
au même endroit biologique sur chaque aile. Ici, 19 landmarks définis sur
la nervation de l'aile antérieure. Un landmark n'a de sens que parce
qu'il est *homologue* : le landmark n°7 sur le spécimen A doit correspondre
à la même intersection de nervures que le landmark n°7 sur le spécimen B.

**Fichier TPS** — Format standard (utilisé notamment par le logiciel
`geomorph` sous R) pour stocker des coordonnées de landmarks : un bloc de
texte par photo, avec la liste des coordonnées (x, y), le chemin de
l'image, et un identifiant. C'est le format d'échange utilisé à toutes
les étapes du pipeline (`landmarks.tps`, `landmarks_numbered.tps`, les
exports finaux `landmarks_19lm_*.tps`...).

**Forme (shape) vs taille (size)** — Une configuration de points bruts
mélange trois choses qui n'intéressent pas la biologie ici (où l'aile est
photographiée, sous quel angle, à quelle distance) et une seule chose qui
nous intéresse (sa forme). La *taille du centroïde* (`centroid_size`,
distance quadratique moyenne des points à leur barycentre) sert justement
à factoriser la taille pour ne garder que la forme.

**GPA — Generalized Procrustes Analysis (superposition de Procrustes
généralisée)** — L'algorithme qui, à partir d'un ensemble de
configurations de landmarks, calcule pour chacune la translation, la
rotation et le facteur d'échelle qui la superposent le mieux possible aux
autres, en minimisant la somme des écarts au carré. Le résultat converge
vers une **forme consensus** (`mean_shape` dans le code) : la forme
moyenne du groupe, une fois toute variation de position/taille/orientation
retirée. Techniquement, l'implémentation (`core/gpa.py`) est une version
2D sans réflexion autorisée (une aile gauche superposée en miroir sur une
aile droite donnerait un résultat biologiquement absurde) — équivalent de
la fonction `gpagen()` du package R `geomorph`.

**Distance de Procrustes** — Une fois deux formes superposées au mieux
(même échelle, même orientation, même position), la distance de
Procrustes est simplement l'écart géométrique résiduel entre les deux
configurations de points. C'est la mesure de "à quel point deux formes se
ressemblent" une fois qu'on a exclu tout ce qui n'est pas la forme. Dans
ce projet, on l'utilise pour comparer la forme d'*un nouveau spécimen* à
la forme de référence *d'un modèle déjà entraîné* — voir section 5 pour
l'interprétation concrète du chiffre affiché dans l'appli.

**PCA (analyse en composantes principales)** — Appliquée aux coordonnées
alignées par GPA, elle réduit la dimension de l'espace des formes tout en
gardant l'essentiel de la variation. Avec *n* landmarks en 2D, l'espace des
formes après GPA a `2n - 4` dimensions utiles (on a retiré 2 degrés de
liberté pour la translation, 1 pour la rotation, 1 pour l'échelle) — c'est
la formule utilisée dans `classifiers/train.py`.

**LDA (analyse discriminante linéaire)** — Contrairement à la PCA (non
supervisée, ne "sait" pas quelles espèces existent), la LDA est entraînée
avec les étiquettes d'espèce (ou de caste) connues, et cherche les
combinaisons de composantes principales qui séparent le mieux les groupes.
C'est elle qui produit la prédiction finale et les probabilités par
classe.

**LOOCV (validation croisée leave-one-out)** — Pour estimer la précision
du modèle sans avoir à sacrifier une partie des données comme "jeu de
test", chaque spécimen est tour à tour retiré, le modèle réentraîné sans
lui, puis utilisé pour le prédire. Le taux de bonnes réponses obtenu ainsi
(`top-1`/`top-3`, voir plus bas) est l'estimation de précision rapportée
par `train.py`. Nuance à garder en tête : cette LOOCV retire une *photo* à
la fois, pas un *individu* entier — un individu qui a plusieurs photos
reste donc partiellement "vu" pendant l'entraînement même quand une de ses
photos est celle qu'on évalue, ce qui gonfle légèrement la précision
estimée par rapport à une vraie généralisation à un individu totalement
inconnu.

**Top-1 / Top-3** — Le modèle ne renvoie pas une seule réponse mais un
classement de probabilités sur toutes les classes connues. "Top-1" = la
prédiction la plus probable est correcte. "Top-3" = la bonne réponse fait
partie des trois plus probables. Utile ici car certaines espèces de
bourdons ont des formes d'aile très proches — le top-3 donne une mesure
plus tolérante, réaliste pour un usage d'aide à l'identification (pas de
remplacement pur et simple d'un⋅e expert⋅e).

**Caste** — Trois valeurs possibles dans ce jeu de données : `fondatrice`
(reine), `worker` (ouvrière), `drone` (mâle). Point important à ne pas
manquer en utilisant `--level caste` : le modèle ne discrimine pas la
caste seule, il discrimine la combinaison **espèce + caste** (colonne
interne `groupe` = `species_caste`, ex. `rupestris_fondatrice`) — parce
qu'une ouvrière et une reine de la même espèce n'ont pas forcément la même
forme d'aile, et qu'on ne veut pas mélanger cette variation avec la
variation inter-espèces.

## 4. Le pipeline, étape par étape

Description complète et à jour, avec les scripts qui implémentent chaque
étape. Le détail exact des colonnes de CSV, des statuts, et des options
CLI est dans [`PIPELINE.md`](PIPELINE.md) — je ne le duplique pas ici,
seulement le "pourquoi" de chaque étape.

**Étape 0 — Jeu de données propre** (`tools/ingestion/build_manifest.py`, et en
amont si besoin `tools/ingestion/ingest_raw.py` + `tools/ingestion/export_clean_dataset.py`).
Avant toute chose, il faut un dossier avec des photos et un tableur
associant chaque photo à un individu (`inv_id`) et ses données
biologiques (espèce, caste...). Cette étape produit deux fichiers pivots,
utilisés par tout le reste du pipeline : `manifest.csv` (une ligne par
photo) et `biological_data.csv` (une ligne par individu). Voir le
scénario 1 plus bas pour le détail pratique.

**Étape 1 — Détection de l'aile** (`extraction/detect_wing.py`). Sur
chaque photo, un détecteur YOLO (mode "light", entraîné sur ce projet)
localise l'aile antérieure sous forme d'une boîte englobante *orientée*
(elle suit l'axe de l'aile, pas horizontale/verticale comme une boîte
classique) — indispensable puisque les photos ne sont pas prises sous un
angle standard.

**Étape 2 — Recadrage normalisé** (`extraction/normalize_crop.py`). La
boîte détectée sert à faire pivoter l'image pour aligner l'aile
horizontalement, puis à découper un recadrage de taille fixe (512x256 px
par défaut, en niveaux de gris) autour d'elle, avec une marge. Toutes les
photos, quelle que soit leur résolution ou leur cadrage d'origine, sont
ainsi ramenées à une géométrie comparable avant le placement des
landmarks.

**Étape 2b — Détection des landmarks** (`landmarks/predict.py`). Un UNet
(réseau de segmentation) prend le recadrage et produit une carte de
chaleur par landmark attendu (19 par défaut) ; le pic de chaque carte
donne la position (x, y) du point correspondant. À ce stade, les points
sont juste une liste de coordonnées **sans ordre garanti** — le réseau ne
sait pas "ceci est le landmark n°7".

**Étape 3 — Renumérotation** (`landmarks/renumber.py`, après avoir
construit une référence une fois avec `landmarks/build_reference.py`). Un
algorithme (association hongroise + alignement de Procrustes, voir
`landmarks/methods/hungarian_umeyama.py`) associe chaque point détecté au
landmark de référence qui lui correspond le mieux, en testant plusieurs
orientations/reflets de départ (l'aile peut avoir été photographiée dans
n'importe quel sens). C'est cette étape qui produit le **coût
d'enregistrement** (`registration_cost` / "Registration cost" dans
l'appli) — voir section 5. Un diagnostic complémentaire (`core/outliers.py`)
compare chaque spécimen renuméroté à la position médiane du même landmark
*au sein de son espèce* pour repérer les points aberrants (bruit de
détection localisé vs échec de renumérotation généralisé à tout le
spécimen).

**Étape 4 — Export final** (`tools/pipeline/export_final_landmarks.py`). Produit un
paquet TPS + CSV autonome, compatible avec un usage sous R/`geomorph` si
besoin, à la fois dans l'espace du recadrage et dans l'espace de la photo
d'origine (reprojection).

**Étape 5 — Entraînement du modèle de classification**
(`classifiers/train.py`, cœur : `core/gpa.py`). Sur l'ensemble d'un jeu de
données (typiquement la collection de référence), calcule le GPA (forme
consensus), projette en PCA, puis entraîne la LDA. C'est ce qui produit le
`model.joblib` utilisé pour toute prédiction ultérieure — voir section 3
pour ce que chaque étape signifie.

**Étape 6 — Prédiction** (`classifiers/predict.py`). Deux modes :
`batch` (on connaît déjà la vérité, sert à évaluer un modèle sur un autre
jeu de données que celui d'entraînement — typiquement collection ->
terrain) et `single` (une seule photo, aucune vérité connue, usage
terrain réel — c'est la fonction que l'appli Streamlit appelle).

**Étape 7 — Analyse de variance** (`analysis/variance_report.py`,
optionnelle, réservée à l'analyse/diagnostic plutôt qu'à la prédiction).
Fait une ANOVA emboîtée (espèce ⊃ caste ⊃ individu ⊃ appareil photo) sur
les formes GPA pour comparer, par exemple, la variance biologique entre
individus d'une même espèce à la variance introduite par l'appareil photo
utilisé — utile pour savoir si le bruit de mesure (matériel photo) reste
en dessous du signal biologique qu'on cherche à détecter.

Deux scripts "orchestrateurs" (`tools/pipeline/train_dataset.py`,
`tools/pipeline/predict_dataset.py`) enchaînent les étapes 1 à 6 automatiquement
sur un jeu de données donné — voir scénario 1.

## 5. Comment lire les scores de l'appli

C'est la question qui revient le plus souvent en utilisant
`app/single_image.py`, donc je la détaille à part. L'appli affiche **deux
scores différents**, qui ne mesurent pas la même chose et n'interviennent
pas au même moment du pipeline :

### Registration cost

Calculé à l'**étape de renumérotation** (avant toute classification),
indépendamment de tout modèle entraîné. Il répond à la question : *"les
points bruts détectés par le UNet ont-ils pu être associés de façon
cohérente aux 19 positions anatomiques attendues ?"*. Concrètement, c'est
l'écart quadratique moyen, par point, entre la configuration détectée
(une fois réalignée au mieux) et le patron de référence, après avoir testé
plusieurs rotations et le cas miroir pour ne pas se faire piéger par
l'orientation de la photo.

- **Bas** = le placement des points est structurellement cohérent (le bon
  nombre de points, à peu près au bon endroit relatif les uns par rapport
  aux autres).
- **Haut** = quelque chose s'est mal passé *avant même de parler de forme
  biologique* : landmarks mal détectés, aile partiellement occluse, photo
  de mauvaise qualité, mauvaise orientation... Dans ce cas, ne faites pas
  confiance à la numérotation elle-même, et donc pas à ce qui suit
  (l'alignement GPA et la prédiction reposent entièrement sur le fait que
  chaque landmark a été mis au bon numéro).

Il n'y a pas encore de seuil calibré statistiquement pour ce score dans
l'appli (il est juste affiché) — un chantier à faire serait de fixer un
seuil à partir de la distribution observée sur le jeu d'entraînement,
comme cela a été fait pour la distance de Procrustes ci-dessous.

### Distance de Procrustes (`procrustes_distance`)

Calculée **au moment de la prédiction**, une fois les landmarks numérotés
et le modèle chargé. Le spécimen est aligné (en un seul passage, sans
recalculer un nouveau consensus) sur la forme moyenne (`mean_shape`) *du
modèle entraîné*, puis on mesure l'écart résiduel entre les deux. Elle
répond à une question différente de la précédente : *"une fois qu'on est
sûrs que les points sont bien numérotés, est-ce que la forme de cette aile
ressemble à ce que le modèle a appris ?"*.

- **Bas** = la forme de l'aile est dans la gamme de variation déjà vue à
  l'entraînement — la prédiction d'espèce/caste peut être considérée avec
  la confiance habituelle du modèle.
- **Haut** = la forme est atypique par rapport à la population
  d'entraînement. Plusieurs causes possibles, indissociables par ce seul
  chiffre : une espèce/caste absente du jeu d'entraînement, un spécimen
  biologiquement inhabituel (aile abîmée, malformation), ou un problème de
  landmark passé inaperçu au stade précédent (registration cost bas mais
  quand même une erreur de placement fine). Dans l'appli, un seuil
  empirique (0.3, choisi à l'œil sur les cas observés jusqu'ici, **pas
  calibré statistiquement**) déclenche un avertissement visuel.

### En résumé

| | Registration cost | Distance de Procrustes |
|---|---|---|
| Calculé à | l'étape de renumérotation | l'étape de prédiction |
| Dépend d'un modèle entraîné ? | non | oui (relatif à `mean_shape` du modèle chargé) |
| Répond à | "le placement des points est-il cohérent ?" | "cette forme d'aile ressemble-t-elle à ce que le modèle connaît ?" |
| Si haut | ne pas faire confiance à la numérotation ni à ce qui suit | la numérotation est OK mais la prédiction d'espèce/caste est à prendre avec précaution |

Les deux chiffres ne sont **pas sur la même échelle** (l'un est une
moyenne quadratique par point dans l'espace normalisé du patron de
référence, l'autre une somme quadratique sur l'ensemble des points par
rapport à la forme moyenne du modèle) : ne pas les comparer directement
entre eux, chacun s'interprète par rapport à sa propre distribution
(petit/grand par rapport aux valeurs habituellement observées, pas par
rapport à une valeur absolue universelle).

## 6. Organisation du dépôt

```
app/                    Applications Streamlit
  build_dataset.py         -> préparer un jeu de données, construire un modèle de référence,
                              et prédire avec (voir scénario 1) -- les deux étapes de validation
                              (recadrage, landmarks) sont des checkpoints humains dans cette appli
  single_image.py          -> l'outil de prédiction pour une seule photo, terrain (scénario 2)
  annotate_wings.py        outil d'annotation manuelle (usage ponctuel, pas la voie principale)

src/
  core/                  Brique commune : TPS, GPA/Procrustes, alignement, détection d'aberrants,
                          jointure dataset (`dataset.py`), schéma des prédictions (`predictions.py`),
                          I/O de run et convention de nommage des modèles (`run_io.py`,
                          `pipeline_io.py`) -- géométrie/I-O pure, sans dépendance CLI/argparse,
                          partagée par plusieurs outils (voir CONVENTIONS.md "Placement d'un fichier")
  extraction/            Détection de l'aile + recadrage normalisé
  landmarks/             Placement + renumérotation des landmarks (utilise le UNet déjà entraîné)
  landmarks_trainer/      (Ré-)entraînement du UNet de landmarks (GPU, avancé)
  obb_trainer/            (Ré-)entraînement du détecteur YOLO-OBB (GPU, avancé)
  classifiers/            Entraînement/prédiction du modèle GPA-PCA-LDA (`--model-name` pour un nom
                          humain, voir scénario 1 étape 4)
  analysis/               ANOVA / analyse de variance, comparaison de runs
  manifest/               Construction du manifest et résolution d'identité (source brute désordonnée)
  utils/                  CLI partagé (`cli.py`), orchestration multi-étapes (`landmarking_pipeline.py`),
                          validation manuelle recadrage/landmarks (`review.py`), overlay TPS
                          (`tps_overlay.py`) -- logique partagée mais pas purement géométrique
  tools/                  Scripts CLI autonomes, groupés par rôle :
    ingestion/              nettoyage brut -> manifest.csv/biological_data.csv, y compris
                             `prepare_dataset.py` (orchestrateur "étape 0", config JSON)
    pipeline/               orchestrateurs dataset-agnostiques (`train_dataset.py`,
                             `predict_dataset.py`), export final, et les outils de validation
                             CLI (`export_review.py`/`reconcile_review.py`, pendant CLI des
                             étapes de validation de l'appli)
    maintenance/            scripts ponctuels (nettoyage TPS, conversion HEIC, réparation
                             d'images...), sans rapport avec le pipeline courant

data/
  Bombus/collection/     Jeu de données de référence (collection identifiée)
  Bombus/terrain/        Jeu de données terrain (évaluation "en conditions réelles")
  models/                Modèles entraînés (LDA, UNet, YOLO-OBB) -- voir note plus bas
  references/             Forme de référence GPA figée, gabarits de landmarks
  identification/         Sources brutes d'identification biologique

docs/                    Rapports de stage des personnes ayant travaillé sur des briques
                         antérieures du projet (contexte historique, définition des landmarks
                         par Tancrède, etc.)

PIPELINE.md              Référence technique exacte du pipeline (entrées/sorties/CLI/statuts)
CONVENTIONS.md           Conventions de code du projet
RESUME.md / TODO.md      Journal de session du refactor (utile seulement si on retouche le code)
```

**Note importante** : les poids de modèles lourds (`*.pt` — UNet, YOLO-OBB)
et les photos elles-mêmes ne sont **pas versionnés dans git** (voir
`.gitignore`) : ils sont trop volumineux et/ou concernent des données non
publiques. Seuls les petits artefacts (CSV, `model.joblib` de la LDA,
JSON de métriques) le sont. Voir le scénario 0 pour ce que ça implique
concrètement à l'installation.

## 7. Scénarios d'usage

### Scénario 0 — Récupérer le projet et installer l'environnement

```bash
git clone https://github.com/julesdubr/idmybee.git
cd idmybee

mamba create -n idmybee python=3.11
mamba activate idmybee
pip install -e ".[dev]"
```

(`pip install -e .` installe le projet en mode éditable : chaque dossier
de `src/` — `core`, `extraction`, `landmarks`, `classifiers`... — devient
un package importable directement, sans manipuler `sys.path` ni se placer
dans un sous-dossier précis pour lancer les scripts. `.[dev]` ajoute
`pytest`.)

Vérification rapide que tout s'installe bien :

```bash
pytest
```

**Ce que le clone git ne contient PAS**, et qu'il faut récupérer à part
(demandez-moi directement, ou récupérez-les depuis l'endroit où on stocke
les données du projet) :
- les photos brutes (`data/*/collection`, `data/*/terrain`... — dossiers
  `images/`) ;
- les poids des modèles déjà entraînés : `data/models/unet_landmarks/*/weights.pt`
  et `data/models/yolon_obb/best.pt` (nécessaires pour le scénario 2).

En revanche, le modèle de classification déjà entraîné
(`data/models/lda/species_collection/train/model.joblib`) et les petits
CSV/JSON associés SONT dans le dépôt : après avoir récupéré au moins les
poids UNet + YOLO-OBB ci-dessus, l'appli de prédiction (scénario 2) peut
tourner sans tout réentraîner.

### Scénario 1 — Préparer un jeu de données, construire un modèle de référence, et prédire dessus

Cette partie concerne la préparation de *nouvelles* données (par exemple
un nouveau lot de photos de collection ou de terrain) et l'entraînement
d'un modèle dessus, pas juste l'utilisation de l'appli. Deux façons de la
mener, qui font exactement la même chose en coulisses (mêmes fonctions
Python, voir "Fonctions core réutilisables" dans `CONVENTIONS.md`) :

- **En ligne de commande**, étape par étape — décrit en détail ci-dessous,
  utile pour scripter/automatiser ou pour un jeu de données bien connu.
- **Avec l'interface graphique** (`streamlit run app/build_dataset.py`) —
  un assistant pas-à-pas qui enchaîne les mêmes étapes, avec en plus deux
  **points de contrôle visuels** (recadrage, puis placement des landmarks)
  où l'on peut vérifier et corriger à la main le statut automatique
  (OK/SUSPECT/FAILED) de chaque photo avant de continuer — voir "Interface
  graphique" plus bas pour le détail. Recommandé pour un premier passage
  sur un nouveau jeu de données, où l'on veut *voir* ce qui se passe avant
  de faire confiance au modèle qui en sortira.

**1. Format attendu en entrée.** Il faut un dossier de photos et un CSV
"par photo" avec au minimum les colonnes `inv_id` (identifiant de
l'individu), `species`, `caste`, et une colonne donnant le chemin de
l'image (`path` par défaut). Si vos données sont déjà propres dans ce
format (peu importe la convention de nommage des fichiers), passez
directement à l'étape 3.

**2. (Optionnel) Nettoyer des données brutes désordonnées.** Si vos photos
n'ont pas encore d'identifiant `inv_id` cohérent (plusieurs conventions de
nommage, doublons, identifiants bruts venant d'un musée...), deux outils
enchaînés s'en chargent :

```bash
python -m tools.ingestion.ingest_raw config/roots_collection.json --name ma_source
python -m tools.ingestion.export_clean_dataset data/ma_source/manifest.csv \
    --identification-csv chemin/vers/identifications.csv \
    --source-type collection \
    --key-column mon_identifiant_brut \
    --mapping-file data/mapping_inv_id.csv \
    --output-dir data/clean/ma_source
```

Ça produit une copie propre, renommée de façon canonique, des images +
un `dataset.csv` prêt pour l'étape suivante. Voir `PIPELINE.md` section 1a
pour le détail complet des options (c'est la partie la plus "avancée" du
pipeline — je ne la relance moi-même que quand j'intègre une nouvelle
source de données brute).

Pour plusieurs sources à la fois (collection + terrain, par exemple), ou
pour rejouer cette étape + la suivante sans retaper toutes les options à
la main, `tools.ingestion.prepare_dataset` enchaîne ingestion brute ->
nettoyage par source -> manifest par source -> combinaison, depuis un seul
fichier de config JSON (voir le docstring du module pour le schéma exact) :

```bash
python -m tools.ingestion.prepare_dataset config/prepare_ma_source.json
```

**3. Construire le manifest (étape obligatoire, toujours).**

```bash
python -m tools.ingestion.build_manifest data/clean/ma_source/dataset.csv \
    --output-dir data/MonJeuDeDonnees
```

(`dataset.csv` est soit celui produit à l'étape 2 ci-dessus, soit
n'importe quel CSV "par photo" déjà conforme au format décrit au point 1
si vos données étaient déjà propres.)

Produit `manifest.csv` + `biological_data.csv` dans
`data/MonJeuDeDonnees/` — ce sont les deux fichiers que tout le reste du
pipeline attend en entrée. Si le CSV d'entrée ne respecte pas le format
attendu, l'outil écrit `manifest_raw.csv` avec la raison de l'échec par
ligne plutôt que de deviner.

**4. Construire le modèle de référence (entraînement complet).**

```bash
python -m tools.pipeline.train_dataset data/MonJeuDeDonnees \
    --unet-model data/models/unet_landmarks/2026-08-29_131929/weights.pt \
    --model-name "Identification bourdons (collection)"
```

Ce script enchaîne automatiquement détection -> recadrage -> landmarks ->
renumérotation -> export -> entraînement GPA-PCA-LDA. Résultat :
`data/models/lda/<nom_du_jeu_de_données>/train/model.joblib`, plus
`metrics.json` (précision top-1/top-3 en LOOCV) et `loocv_predictions.csv`
(le détail spécimen par spécimen). C'est ce `model.joblib` qui sert
ensuite de "référence" pour toute prédiction (c'est lui qui contient la
forme moyenne GPA, la PCA et la LDA entraînées).

`--model-name` (optionnel) donne un nom lisible au modèle — par exemple
"Identification bourdons à abdomen rouge (collection)" plutôt que l'id
technique `species_collection` généré automatiquement à partir du niveau
et du nom du dataset. Ce nom est purement descriptif (il n'affecte pas
l'emplacement du fichier, toujours dérivé de façon reproductible du
niveau/dataset/appareils — voir `core/run_io.py`) : à défaut, l'id
technique reste utilisé. Il est repris tel quel dans les listes
déroulantes de modèle, aussi bien dans `app/single_image.py` que dans
`app/build_dataset.py` (voir `core.run_io.model_display_name`).

**5. Prédire sur un autre jeu de données (évaluation).**

```bash
python -m tools.pipeline.predict_dataset data/UnAutreJeuDeDonnees \
    --model data/models/lda/MonJeuDeDonnees/train/model.joblib \
    --unet-model data/models/unet_landmarks/2026-08-29_131929/weights.pt
```

Même enchaînement de traitement, mais applique le modèle déjà entraîné à
l'étape 4 plutôt que d'en réentraîner un nouveau. Utile pour évaluer un
modèle sur un jeu indépendant (typiquement : entraîné sur la collection,
évalué sur le terrain) : `data/models/lda/MonJeuDeDonnees/predict/<nom_du_jeu_évalué>/predictions.csv`
contient une ligne par spécimen avec la prédiction, la confiance, et les
deux scores de la section 5.

*Exemple réel, déjà exécuté sur ce projet* : modèle entraîné sur
`data/Bombus/collection` (2600 spécimens, 19 landmarks) -> 94.8% top-1 /
98.7% top-3 en LOOCV ; le même modèle évalué sur `data/Bombus/terrain`
(265 spécimens de vérité connue) -> 82.3% top-1 / 97.0% top-3. L'écart
entre les deux illustre bien pourquoi évaluer sur un jeu terrain distinct
compte : la précision en LOOCV sur la collection est optimiste par rapport
à des photos prises dans des conditions réelles.

**Étapes de validation (recadrage, landmarks).** Entre les étapes 4/5
(détection+recadrage) et 5/6 (placement des landmarks), le statut
automatique de chaque photo (OK/SUSPECT/FAILED) peut être vérifié et
corrigé à la main avant de poursuivre — voir `utils/review.py`. Deux
façons d'y accéder :
- **Interface graphique** (`app/build_dataset.py`) : galerie filtrable
  (par statut, par recherche `photo_id`/`inv_id`), aperçu de la photo ou
  de l'overlay de landmarks numérotés, statut éditable directement dans
  le tableau. "Enregistrer et continuer" écrit la correction et
  l'applique automatiquement à la suite du pipeline (recadrage rejeté ->
  jamais passé par le placement de landmarks ; landmark rejeté/récupéré ->
  exclu/inclus de l'export et du modèle).
- **Ligne de commande**, pour parité fonctionnelle sans interface :
  `tools.pipeline.export_review` écrit deux CSV éditables
  (`<dataset>/review/{crops,landmarks}_review.csv`, avec une colonne
  `reviewed_status` à modifier dans un tableur) plus, avec `--overlays`,
  les images annotées triées par statut ; une fois corrigé à la main,
  `tools.pipeline.reconcile_review <dataset>` applique le résultat.

**Interface graphique (`app/build_dataset.py`).**

```bash
streamlit run app/build_dataset.py
```

Un assistant en 7 étapes : choix de l'objectif (construire un modèle ou
prédire avec un modèle existant) puis du jeu de données (racine déjà
propre, ou construction du manifest depuis un CSV) ; paramètres du
pipeline (mêmes options que la ligne de commande, sous forme de
formulaire) ; détection + recadrage ; validation du recadrage (voir
ci-dessus) ; placement des landmarks ; validation des landmarks ; export ;
puis, selon l'objectif choisi à la première étape, soit la construction du
modèle (nom du modèle, niveau espèce/caste), soit la classification avec
un modèle existant (sélection dans la liste, par nom si `--model-name` a
été utilisé). Chaque étape "Run" a un bouton "Skip -- already done" pour
reprendre un jeu de données déjà partiellement traité (par la ligne de
commande ou une session précédente de l'appli) sans tout relancer.

### Scénario 2 — Identifier une aile à partir d'une photo (l'appli)

C'est l'outil pensé pour un usage "sur le terrain", sans ligne de
commande.

```bash
streamlit run app/single_image.py
```

Dans la barre latérale :
- **Classification model** : le `model.joblib` à utiliser (celui produit
  au scénario 1, étape 4 — la liste déroulante détecte automatiquement
  ceux présents sous `data/models/lda/`).
- **Wing detector** / **UNet landmark weights** : les poids des deux
  réseaux (détection + landmarks). Là aussi auto-détectés s'ils sont au
  bon endroit (`data/models/yolon_obb/`, `data/models/unet_landmarks/`).
- Les paramètres du pipeline (padding, taille du recadrage, seuil de
  confiance de détection...) ont des valeurs par défaut raisonnables — je
  n'y touche en pratique que pour déboguer un cas particulier.

Ensuite : charger une photo, cliquer sur "Run pipeline". L'appli affiche,
dans l'ordre : la boîte de détection sur la photo d'origine, le recadrage
avec les 19 landmarks numérotés en overlay, le registration cost, puis le
classement des 3 espèces (ou espèce+caste) les plus probables avec leur
confiance, et enfin la distance de Procrustes — voir la section 5 pour
interpréter ces deux derniers chiffres avant de conclure quoi que ce soit.

### Scénario 3 — (Ré-)entraîner les modèles

Il y a en réalité trois modèles distincts dans ce pipeline, avec des coûts
très différents à réentraîner :

**a) Le modèle de classification (GPA-PCA-LDA)** — c'est le plus léger
(quelques secondes à quelques minutes, CPU seul, pas de GPU). C'est celui
qu'on réentraîne le plus souvent : dès qu'on ajoute des spécimens à la
collection de référence, ou qu'on veut tester une variante (par exemple
`--level caste` plutôt que `--level species`, ou un autre nombre de
composantes LDA). C'est exactement le scénario 1, étape 4
(`tools/pipeline/train_dataset.py`), ou directement :

```bash
python -m classifiers.train data/MonJeuDeDonnees --level species
```

**b) Le UNet de placement des landmarks** — beaucoup plus lourd (GPU
recommandé), et rarement nécessaire : seulement si le placement
automatique des landmarks se dégrade sur un nouveau type de photo non
représenté à l'entraînement (nouvel appareil, nouvelles conditions de
prise de vue très différentes), ou si le patron de landmarks change (par
exemple passer de 18 à 19 points). Le sous-projet
[`src/landmarks_trainer/`](src/landmarks_trainer/README.md) documente
cette procédure en détail (fine-tuning à partir des poids existants ou
entraînement depuis zéro, évaluation de la précision de localisation en
pixels).

**c) Le détecteur d'aile (YOLO-OBB)** — également lourd (GPU), rarement
nécessaire pour la même raison que (b). Documenté dans
[`src/obb_trainer/`](src/obb_trainer/README.md).

En pratique, pour un usage courant du projet (ajouter des spécimens,
évaluer sur un nouveau jeu terrain, faire des prédictions), seul (a) est
à connaître — (b) et (c) sont des interventions plus rares, réservées à
faire évoluer les briques de détection/localisation elles-mêmes.

## 8. Points d'attention et limites connues

- **Le code est en anglais** (docstrings, logs, messages CLI) — c'est un
  choix de convention du projet (voir `CONVENTIONS.md`), ce README est
  volontairement le seul document en français, pour rester lisible sans
  avoir à lire le code.
- **`--level caste` classe en réalité `espèce + caste`**, pas la caste
  seule (voir section 3, "Caste") — à garder en tête pour ne pas
  mal interpréter un résultat.
- **Les seuils affichés dans l'appli (0.3 pour la distance de Procrustes)
  sont empiriques, pas calibrés statistiquement** — à traiter comme un
  ordre de grandeur indicatif, pas une valeur validée.
- **La précision LOOCV rapportée par `train.py` est légèrement optimiste**
  puisqu'elle retire une photo à la fois, pas un individu entier (voir
  section 3, "LOOCV") — l'évaluation sur un jeu terrain indépendant
  (scénario 1, étape 5) reste la mesure la plus honnête de la précision
  réelle.
- **Un `model.joblib` entraîné avant la suppression du champ `split`**
  (courant sept. 2026) n'est plus compatible avec le code actuel —
  réentraîner plutôt que de chercher à charger un vieux modèle qui plante.
- **`obb_trainer/` n'a pas suivi le dernier refactor du schéma de données**
  (il attend encore un ancien format de CSV `image_id`/`specimen_id`) —
  concerne uniquement le réentraînement du détecteur (scénario 3c), pas
  l'usage courant.
- **Les seuils de la distance de Procrustes/du registration cost dans
  `app/build_dataset.py` (validation des landmarks) sont, comme dans
  `single_image.py`, des repères empiriques** — la décision finale (statut
  éditable OK/SUSPECT/FAILED) reste humaine, l'appli ne fait qu'aider à la
  prendre plus vite (tri, aperçu annoté).

## 9. Pour aller plus loin

- [`PIPELINE.md`](PIPELINE.md) — référence technique exacte (schémas de
  CSV, arguments CLI, vocabulaire de statut) pour chaque étape du
  pipeline. À consulter dès qu'on modifie ou relance une étape précise.
- [`CONVENTIONS.md`](CONVENTIONS.md) — conventions de style de code du
  projet.
- [`RESUME.md`](RESUME.md) / [`TODO.md`](TODO.md) — journal de session du
  refactor du pipeline : utile seulement si on retouche le code, pas pour
  un usage du projet en l'état.
- `docs/*.pdf` — rapports de stage des personnes ayant travaillé sur des
  briques antérieures du projet (dont la définition initiale des 19
  landmarks, et les premiers prototypes de détection/segmentation), pour
  le contexte biologique et historique.
