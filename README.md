# idmybee — Pipeline de détection automatique de landmarks sur ailes

Scaffold de départ pour automatiser la pose des 19 points morphométriques,
en remplacement de l'annotation manuelle (TPSdig2) réalisée par Tancrède.

## Structure

```
src/
  tps_parser.py       parsing des fichiers .tps -> objets Python
  utils.py            heatmaps gaussiennes, letterbox resize, crop, métrique NME
  dataset.py          Dataset PyTorch (augmentation via albumentations)
  model.py            U-Net avec encodeur ResNet18 pré-entraîné
  train.py            boucle d'entraînement (k-fold, mixed precision)
  predict_export.py   inférence sur nouvelles images + export .tps
scripts/
  verify_tps_annotations.py   sanity check visuel du parsing (À LANCER EN PREMIER)
```

## Étape 0 — à faire avant tout le reste

Avec ton vrai fichier .tps et quelques photos P1 :

```bash
pip install -r requirements.txt
python scripts/verify_tps_annotations.py --tps annotations.tps --images-root photos/ --specimen-idx 0
```

Regarder `check.png` : si les points ne tombent pas sur les bonnes
intersections de nervures, essayer avec `--flip-y` (problème classique de
convention d'axe Y dans TPSdig2). Répéter sur 5-10 spécimens différents pour
être sûr que le chemin d'image et l'ordre des points sont cohérents partout.

## Étape 1 — entraînement

```bash
python src/train.py --tps annotations.tps --images-root photos/ \
    --image-size 256 --batch-size 16 --epochs 100
```

Réglages pensés pour la RTX 2070 (8 Go) : `image_size=256`, encodeur
ResNet18, mixed precision activée automatiquement (CUDA détecté). Sur le
MacBook M2, le même script tourne (CPU ou MPS) pour du debug rapide sur un
petit sous-ensemble — pas pour l'entraînement complet.

À surveiller : `val_nme` (erreur normalisée par la distance entre les
landmarks de référence définis dans `train.py`, `REF_LANDMARK_A`/`B` —
actuellement points 1 et 9, **à ajuster si ce ne sont pas les bons indices
de référence sur vos données**).

## Étape 2 — Phase 2 (pas encore implémentée)

Pour l'instant, `dataset.py` et `predict_export.py` recadrent l'image autour
d'une bounding box (soit celle des landmarks connus pour l'entraînement,
soit à fournir manuellement pour l'inférence). Pour un pipeline entièrement
automatique sur photo brute (fond sombre + bande damier + étiquettes), il
faudra ajouter une étape de détection/localisation de l'aile en amont —
soit une heuristique (contraste, exclusion de la bande damier par sa
régularité), soit un petit détecteur entraîné sur les bounding box dérivées
des annotations existantes.

## Points à valider avec Adrien / en comparant au travail d'Exbrayat

- Écart entre 18 et 19 canaux en sortie (un point exclu ou fusionné ?)
- Indices des landmarks de référence pour la métrique NME
- Convention d'axe Y du fichier .tps (à confirmer avec `verify_tps_annotations.py`)
- Éventuel gain à ajouter une tête DSNT (soft-argmax différentiable) plutôt
  que l'extraction argmax + raffinement local actuellement dans `utils.py`
