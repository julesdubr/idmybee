# 1. Autocrop des ailes

## 1.1 Construction d'une ground truth

- [x] Segmentation des ailes via YOLOE (à partir d'un fichier de réferences ref.json)
- [x] Alignement horizontal des l'aile
- [x] OBB en sortie

Tester sur jeux de données :
- [x] organized
- [ ] terrain
- [ ] vrac

## 1.2. Entrainement d'un CNN dédié



# 2. Pose automatique des points

- [x] Tester best model sur données adrien et tancrède
- [x] Générer un fichier TPS à partir des prédictions
- [ ] En fonction des résultats fine-tuner le modèle avec les données de tancrède augmentées

**Autres pistes :** _Descripteurs DinoV3 à la place du modèle (model fondateur), résolution de l'image en fonction du champs perspectif du modèle._


# 3. Renommer les ids de manière croissante

Trouver comment numéroter les points.
- photos 2 ailes, les flagger, est-ce qu’on arrive à avoir les points ? plus de variabilité par rapport aux autres ?
- connection entre les points par couleur des pixels
- tracer droite/courbe entre points, sliding landmarks pour la gpa
- nervous post furcal ? pre furcal ? neverulus a 2 liaisons : avant pose des points, identifié 2 ou 3 cellules sub marginales et choisir un modle specifique a chaque (2 ou 3)


# 4. Identification

CSV des données biologiques (espèces, caste) pour validation (tester hypothèses). Dans R :
1. charger tps dans un ordre, avec tableau avec même ordre les infos associées à ce spécimen (sexe, espèces, sous-genre)
2. a partir de cette compatibilité d’ordre, on regroupe les co-formations d’ailes entre elles pour analyse multivariée, canonique pour faire les groupes, et finir les axes discriminants en fonction des groupes (ACP)
3. variabilité supérieur caste entre espèces, variabilité entre sexe
4. variation biologique (au sens de l’analyse morpho, cf thèse adrien, shiny mise en application r module) au sein d’un caste (plus petite variation) → variabilité entre caste, entre espèces, entre sous-genre….
5. voir si l’erreur de mesure di



# 5. Analyse

- ANOVA tous les points espèces / type appareil photo
- intra-espece, inter-espece (variations biologiques)
- inter-appareil (variance méthodologique), les comparer
==> on espère V_interesp > V_intraesp > V_interapp

- si pas 100% dû aux variation biologiques proches => pas problème de méthode