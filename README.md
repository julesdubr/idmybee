# idmybee — identifier des bourdons à partir d'une photo d'aile

Pipeline pour identifier des bourdons (genre *Bombus* — espèce et caste) à
partir d'une simple photo d'aile antérieure, par morphométrie géométrique
automatisée par deep learning.

Pour le détail technique exact de chaque étape (entrées/sorties/CLI), voir
[`docs/PIPELINE.md`](docs/PIPELINE.md).

## Méthode

1. **Détection** : un détecteur YOLO localise l'aile sur la photo brute
   (n'importe quel angle) et en extrait un recadrage normalisé.
2. **Landmarks** : un UNet place 19 points anatomiques homologues sur ce
   recadrage, puis un algorithme de renumérotation les associe à un
   patron de référence (calcule au passage un **coût d'enregistrement**,
   qui indique si le placement des points est structurellement cohérent).
3. **Morphométrie géométrique classique** : superposition de Procrustes
   (GPA) des landmarks sur une forme consensus, réduction de dimension
   (PCA), puis classification supervisée (LDA) pour prédire l'espèce (ou
   la caste), avec un score de confiance et une **distance de Procrustes**
   à la forme moyenne du modèle (indique si l'aile ressemble à ce que le
   modèle a appris).

```
photo -> détection (YOLO) -> recadrage -> landmarks (UNet)
      -> renumérotation -> GPA -> PCA -> LDA -> espèce/caste prédite
```

## Installation

```bash
git clone https://github.com/julesdubr/idmybee.git
cd idmybee

mamba create -n idmybee python=3.11
mamba activate idmybee
pip install -e ".[dev]"

pytest   # vérifie que tout s'installe correctement
```

Le clone git ne contient pas les photos (`data/*/collection`,
`data/*/terrain`) ni les poids des modèles déjà entraînés
(`models/unet_landmarks/*/weights.pt`, `models/yolon_obb/best.pt`) — à
récupérer séparément (demandez-moi, ou depuis l'endroit où sont stockées
les données du projet). Le modèle de classification déjà entraîné
(`models/lda/species_collection/model.joblib`) est en revanche versionné :
une fois les poids UNet + YOLO-OBB récupérés, l'appli de prédiction peut
tourner sans tout réentraîner.

## Utilisation

### Identifier une photo (usage terrain)

```bash
streamlit run app/single_image.py
```

Charger une photo, cliquer sur "Lancer le pipeline". L'appli affiche la
détection, les 19 landmarks numérotés, puis les 3 espèces (ou
espèce+caste) les plus probables avec leur confiance.

### Préparer un jeu de données, entraîner, prédire en lot

Trois outils Streamlit, chacun pour une étape :

```bash
streamlit run app/setup_dataset.py    # préparer/nettoyer un jeu de données
                                       # (détection -> recadrage -> landmarks -> export,
                                       # avec deux points de contrôle humains)
streamlit run app/train_model.py      # entraîner un modèle (GPA-PCA-LDA) sur un export
streamlit run app/predict_dataset.py  # classifier un export avec un modèle existant
```

Équivalents en ligne de commande (utile pour scripter) :

```bash
python -m tools.ingestion.build_manifest mon_dataset.csv --output-dir data/MonJeu
python -m tools.pipeline.train_dataset data/MonJeu --model-name "Mon modèle"
python -m tools.pipeline.predict_dataset data/UnAutreJeu --model-name "Mon modèle"
```

Voir [`docs/PIPELINE.md`](docs/PIPELINE.md) pour le détail complet de chaque étape,
des options CLI et des formats de fichiers.

## Premiers résultats

Modèle entraîné sur `data/Bombus/collection` (13 espèces, 526 spécimens,
2600 photos, 19 landmarks), précision en validation croisée leave-one-out
(LOOCV) :

| Jeu de données | Photos | Top-1 | Top-3 |
|---|---|---|---|
| Collection (LOOCV, groupée par spécimen) | 2600 | ~92.8 % | ~98.3 % |
| Terrain (évaluation indépendante) | 265 | 81.9 % | 97.0 % |

L'écart entre les deux illustre pourquoi une évaluation sur un jeu terrain
distinct compte : la précision en LOOCV sur la collection reste optimiste
par rapport à des photos prises dans des conditions réelles, même corrigée
(voir ci-dessous).

**Biais du LOOCV et correction.** Un individu a souvent plusieurs photos
(en moyenne ~5 dans la collection). Un LOOCV naïf retire une photo à la
fois : les autres photos du même individu restent dans l'ensemble
d'entraînement et facilitent artificiellement la prédiction de la photo
retirée, ce qui surestimait le top-1 collection (~94.8 % avant correction).
`classifiers/train.py` fait maintenant un LOOCV **groupé par spécimen**
(`inv_id`, via `sklearn.model_selection.LeaveOneGroupOut`) : chaque fold
retire TOUTES les photos d'un même individu, ce qui fait redescendre le
top-1 collection à ~92.8 % (chiffre plus honnête).

Cette correction introduit son propre biais résiduel, dans l'autre sens :
une espèce représentée par un seul spécimen (ex. *B. lucorum* dans la
collection actuelle, avec un seul individu) est totalement absente de
l'entraînement du fold qui évalue ses photos -- le classifieur n'a alors
jamais vu cette espèce et la rate systématiquement (0 % pour cette espèce en
LOOCV), ce qui sous-estime la précision réelle pour les espèces
sous-représentées (le modèle final, lui, entraîné sur l'ensemble complet,
a bien vu ces espèces). `classifiers/train.py` logue un avertissement
listant les espèces concernées (voir `singleton_classes()`).

## Pour aller plus loin

- [`docs/PIPELINE.md`](docs/PIPELINE.md) — référence technique du pipeline (CSV, CLI, statuts)
- `docs/*.pdf` — rapports de stage sur les briques antérieures du projet
  (définition des 19 landmarks, premiers prototypes)
