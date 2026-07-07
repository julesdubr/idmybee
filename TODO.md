# TODO — Pipeline de crop serré d'ailes de bourdon (pour rendre exploitable le modèle de Gabriel)

Rappel d'objectif : produire un dataset d'images cropées **serrées** sur l'aile, format compatible avec le modèle UNet de landmarks déjà pré-entraîné par Gabriel (qui ne fonctionne pas sur nos crops actuels, trop larges/non normalisés).

Pas de cas de "photo groupée" dans ce dataset. Les sous-dossiers `pomorum (reine)`, `ruderarius (reine)`, `sichelii (reine)` sont juste des dossiers présents dans `orga_cropping` sans image cropée ni wide — inexploitables, à documenter et ignorer.

**Ordre de traitement révisé** : on resserre d'abord les crops (indépendamment du wide), puis on localise seulement les crops serrés obtenus dans le wide — puisque c'est cette paire (wide, bbox serrée dans le repère wide) qui sert directement à l'entraînement du détecteur final. Localiser le crop large brut dans le wide avant resserrement n'apporte rien en soi.

---

## Phase 0 — Manifest & inventaire

- [ ] Construire un `pandas.DataFrame` au niveau **image**, colonnes minimales :
  `id`, `espece`, `role` (`male` / `worker` / `queen`), `chemin_wide` (nullable), `chemin_crop_large` (nullable), `statut`.
- [ ] Peupler `statut` avec 5 valeurs possibles :
  - `paire_exploitable` : crop large réel (sous-crop franc) + wide présent (confusus male/worker, pomorum male/worker, ruderarius male/worker, rupestris male/queen, sichelii male/worker, sylvarum male, wurlfenii male).
  - `crop_sans_wide` : cropé mais pas de wide d'origine (confusus queen, sylvarum queen, wurlfenii worker).
  - `wide_non_crope` : image identique dans les deux dossiers, pas de crop réel (lapidarius, monticola, pratorum, pyrenaus, soroeensis — **à confirmer au cas par cas**, voir statut suivant).
  - `wide_recadre_leger` : image ni identique ni franchement cropée — recadrage léger (marge coupée, léger zoom), à distinguer de `wide_non_crope` et de `paire_exploitable`. Concerne potentiellement une partie de lapidarius/monticola/pratorum/pyrenaus/soroeensis, à vérifier image par image plutôt que supposer un statut uniforme par espèce.
  - `inexploitable` : ni crop ni wide utilisables (pomorum queen, ruderarius queen, sichelii queen).
- [ ] Vérifier par `groupby(['espece','role'])` que le décompte colle à la liste validée par Jules.
- [ ] Distinguer `wide_non_crope` de `wide_recadre_leger` par une mesure quantitative plutôt qu'à l'œil sur tout le dataset : différence pixel directe (`cv2.absdiff` après alignement des dimensions) — écart quasi nul → `wide_non_crope` ; écart significatif mais image globalement similaire (même sujet, cadrage proche) → `wide_recadre_leger`.
- [ ] Échantillonner ~50-100 paires `paire_exploitable`, inspection visuelle rapide (`matplotlib`, grille de vignettes) pour estimer la fréquence des cas à 2 ailes dans les crops larges. Consigner dans le manifest.

---

## Phase 1 — Resserrer le crop existant (crop large/recadré → crop serré + padding)

Travaille uniquement sur le crop lui-même (`paire_exploitable` et `wide_recadre_leger`), indépendamment de toute localisation dans le wide.

- [ ] **1.1 — Piste A : segmentation classique**
  - Neutraliser d'abord la bande quadrillée si présente (détection par texture/fréquence locale, pas de coordonnées fixes codées en dur).
  - Niveaux de gris + `cv2.threshold` (Otsu) ou `cv2.adaptiveThreshold`.
  - Nettoyage morphologique (`cv2.morphologyEx`) pour réduire le bruit de texture du papier.
  - `cv2.findContours` → garder le/les plus grands contours plausibles.
  - `cv2.boundingRect` (ou `cv2.minAreaRect` si rotation nécessaire) + léger padding fixe ou proportionnel à la taille de l'aile détectée.
- [ ] **1.2 — Piste B : SAM**
  - `SamAutomaticMaskGenerator` sur le crop (fenêtre déjà réduite par rapport au wide complet).
  - Filtre à repenser pour gérer proprement le cas 2 ailes — pas de "garder seulement le plus grand masque".
  - Extraction bbox + padding, même logique qu'en 1.1.
- [ ] **1.3 — Comparaison des deux pistes**
  - Sur un échantillon commun, calculer l'IoU entre les bbox produites par 1.1 et 1.2.
  - Revue visuelle manuelle des désaccords (`matplotlib`, grille avec overlay des deux bbox superposées).
  - Décider d'une méthode par défaut (ou combinaison : fallback, vote, moyenne) selon les résultats.
- [ ] **1.4 — Gestion multi-instances** : si la Phase 0 indique une fréquence non négligeable de 2 ailes, adapter les deux pistes pour ressortir une bbox par instance plutôt qu'une seule bbox globale.
- [ ] **1.5 — Cas `wide_recadre_leger`** : appliquer la même logique 1.1-1.4 directement sur ces images (elles jouent ici le même rôle qu'un crop large, même si elles n'ont pas été explicitement cropées par un humain).
- [ ] Contrôle qualité final : échantillon stratifié par espèce/rôle, overlay bbox + padding sur l'image, revue visuelle rapide.

---

## Phase 2 — Localiser le crop serré dans le wide

Fait uniquement *après* la Phase 1, et uniquement pour les images ayant un wide associé (`paire_exploitable`, `wide_recadre_leger`).

- [ ] Template matching pour retrouver l'offset du crop serré dans le wide :
  - `cv2.matchTemplate(wide_gray, crop_serre_gray, method=cv2.TM_CCOEFF_NORMED)`
  - `cv2.minMaxLoc` → coin haut-gauche + score de confiance.
- [ ] Fallback recalage par features si corrélation faible (cas attendu en priorité sur `wide_recadre_leger`, où un simple matching de template peut être moins fiable à cause du zoom/recadrage) :
  - `cv2.ORB_create()` / `cv2.AKAZE_create()` + `cv2.BFMatcher(cv2.NORM_HAMMING)` + ratio test de Lowe.
  - `cv2.estimateAffinePartial2D` (RANSAC) → offset et échelle, gère aussi un léger zoom contrairement au template matching pur.
- [ ] Stocker `(x_offset, y_offset, échelle, méthode, score_confiance)` dans le manifest.
- [ ] Isoler les échecs de recalage dans une liste à part pour revue manuelle groupée — ne pas les laisser silencieusement polluer le jeu d'entraînement de la Phase 3.
- [ ] Composer la bbox serrée (repère crop) avec l'offset/échelle trouvés pour obtenir la bbox finale dans le repère du wide.

---

## Phase 3 — Entraîner le détecteur wide → crop serré

- [ ] Construire les paires d'entraînement : (wide complet, bbox serrée dans le repère wide) — résultat direct de la Phase 2.
- [ ] Split train/val/test stratifié par espèce et par rôle, pas par image brute.
- [ ] Choix d'architecture à trancher selon préférence d'infrastructure :
  - Réutilisation du squelette UNet existant (heatmap de centre + régression largeur/hauteur, angle si la Phase 0 montre des ailes non horizontales).
  - `ultralytics` (YOLO ou YOLO-OBB selon besoin de rotation), infrastructure prête à l'emploi.
- [ ] Augmentations via `albumentations` : luminosité/contraste, flips, rotation si pertinent.
- [ ] Inférence sur les images sans wide du tout (`crop_sans_wide`, rien à prédire, sert seulement de test qualitatif en aval) et sur toute image `orga_widecrop` dépourvue de version cropée (`wide_non_crope` confirmé).

---

## Phase 4 — Validation finale (objectif réel du pipeline)

- [ ] Faire tourner le modèle pré-entraîné de Gabriel (UNet landmarks, degré 100) sur les crops serrés produits (Phase 1 pour les images avec crop/recadrage existant, Phase 3 pour les cas sans wide).
- [ ] Comparer le score de fiabilité / l'erreur en pixels obtenus à ceux de son test set d'origine (0.83 px, 97% accuracy) — c'est le critère de succès réel du pipeline, pas seulement l'IoU du détecteur.
- [ ] Si l'écart reste important : vérifier en priorité l'échelle/résolution du crop serré (le UNet de Gabriel attend probablement une taille ou un ratio proche de son jeu d'entraînement Base 1) avant de remettre en cause la qualité de la bbox elle-même.

---

## Librairies à prévoir

`opencv-python` (template matching, recalage, segmentation, bbox), `numpy`, `pandas` (manifest), `albumentations` (augmentations), `torch` / `torchvision` (option UNet-maison), `ultralytics` (option YOLO/YOLO-OBB), `matplotlib` (QC visuel), `segment-anything` (piste B de la Phase 1).