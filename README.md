# idmybee — Pipeline d'identification de pollinisateurs à partir de leur ailes

Outil d'identification des pollinisateurs à partir de photos d'aile antérieure.
La classification se base sur les positions de 17 à 19 landmarks sur les nervures des ailes.

L'outil réalise 4 étapes :

1. Autocrop de l'aile
2. Pose automatique des landmarks
3. Numérotation des landmarks
4. Identification de l'espèce-caste

## 1. Autocrop de l'aile

Construction d'un ground-truth avec YOLOE, qui permet d'auto-segmenter les ailes, à partir d'un embedding de prompts.
À partir de cette ground truth, apprentissage d'un CNN spécifique à la reconnaissance d'ailes antérieures de pollinisateurs,
à partir de l'OBB prédites sur les segmentations de YOLOE.

## 2. Pose automatique des landmarks



## 3. Numérotation des landmarks



## 4. Identification de l'espèce-caste