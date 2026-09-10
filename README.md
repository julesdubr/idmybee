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

Modèle entraîné sur `data/Bombus/collection` (13 espèces, ~2600 spécimens,
19 landmarks), précision en validation croisée leave-one-out (LOOCV) :

| Jeu de données | Spécimens | Top-1 | Top-3 |
|---|---|---|---|
| Collection (LOOCV) | 2600 | 94.8 % | 98.7 % |
| Terrain (évaluation indépendante) | 265 | 82.3 % | 97.0 % |

L'écart entre les deux illustre pourquoi une évaluation sur un jeu terrain
distinct compte : la précision en LOOCV sur la collection est optimiste
par rapport à des photos prises dans des conditions réelles.

## Pour aller plus loin

- [`docs/PIPELINE.md`](docs/PIPELINE.md) — référence technique du pipeline (CSV, CLI, statuts)
- `docs/*.pdf` — rapports de stage sur les briques antérieures du projet
  (définition des 19 landmarks, premiers prototypes)
