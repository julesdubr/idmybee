# Résumé — Pipeline de crop d'ailes de bourdon (contexte du TODO.md)

## Objectif réel

Le but final n'est pas seulement un détecteur d'ailes autonome : c'est de produire un dataset d'images **cropées serrées sur l'aile** pour rendre utilisable le modèle déjà pré-entraîné de Gabriel (UNet de prédiction de landmarks, 18 points, degré 100, ~0.83 px d'erreur sur son propre test set). Ce modèle ne fonctionne pas correctement sur nos images actuelles car nos crops ne sont pas assez serrés/normalisés par rapport au format sur lequel il a été entraîné.

## Pourquoi l'ancienne méthode (Gabriel) ne suffit pas

Sa méthode génère tous les masques SAM, garde ceux dont le ratio largeur/hauteur optimal tombe dans une fenêtre fixe [2.2, 2.8], puis prend le plus grand masque valide. Deux défaillances identifiées sur nos données :
- Deux ailes sur une photo → SAM les fusionne en un masque hors fenêtre (crash), ou les deux passent le filtre et une seule est gardée silencieusement.
- Fichiers `.heic` : `cv2.imread` renvoie un tableau vide plutôt que `None`, ce qui fait planter `cv2.cvtColor` en aval.

## État réel des données

**Paires wide/crop large exploitables** (vrai sous-crop pixel) :
confusus (mâle/ouvrière), pomorum (mâle/ouvrière), ruderarius (mâle/ouvrière), rupestris (mâle/reine), sichelii (mâle/ouvrière), sylvarum (mâle), wurlfenii (mâle).

**Types cropés sans wide d'origine** (inutilisables pour l'entraînement, réservés à l'inférence qualitative) :
confusus (reine), sylvarum (reine), wurlfenii (ouvrière).

**Espèces sans crop franc dans `orga_cropping`** : lapidarius, monticola, pratorum, pyrenaus, soroeensis. Ce groupe recouvre en réalité **deux sous-cas à distinguer image par image**, pas un seul statut uniforme :
- image réellement identique au wide (`wide_non_crope`) ;
- image légèrement recadrée par rapport au wide — marge coupée ou léger zoom, sans être un crop franc (`wide_recadre_leger`). Ce cas nécessite le même travail de resserrement que les crops larges classiques, et une localisation dans le wide plus délicate (le simple template matching peut échouer à cause du zoom).

**Sous-dossiers sans crop ni wide exploitables** (inexploitables, ni source ni cible) :
pomorum (reine), ruderarius (reine), sichelii (reine). Pas de cas de "photos groupées multi-spécimens" dans ce dataset.

## Pipeline révisé — ordre des étapes

Point clé : localiser le crop dans le wide **avant** de le resserrer n'apporte rien en soi, puisque ce qui sert réellement à l'entraînement du détecteur final est la paire (wide, bbox **serrée**) dans le repère du wide. La localisation n'est donc utile qu'une fois le resserrement fait.

**Étape 1 — Resserrer le crop existant (ou recadré léger) → crop serré + léger padding.** Travaille uniquement sur le crop lui-même, indépendamment du wide. Deux familles de méthodes à comparer en parallèle : segmentation classique (seuillage/contours) et SAM (comme Gabriel, filtre à repenser). Les images `wide_recadre_leger` suivent le même traitement que les crops larges classiques.

**Étape 2 — Localiser le crop serré dans le wide.** Template matching en première intention ; fallback recalage par features (`estimateAffinePartial2D`) attendu plus souvent sur les cas `wide_recadre_leger`, où un simple matching de template peut être moins fiable à cause du zoom/recadrage. Le résultat donne directement la bbox serrée dans le repère du wide.

**Étape 3 — Entraîner un détecteur wide → crop serré**, à partir des paires (wide, bbox serrée) obtenues en étape 2. Utile pour couvrir les images sans version cropée du tout.

## Problèmes encore ouverts

- Distinguer systématiquement `wide_non_crope` de `wide_recadre_leger` (pas supposer un statut uniforme par espèce, vérifier image par image via une mesure de différence pixel).
- Choix final de méthode pour l'étape de resserrement : comparaison prévue entre segmentation classique et SAM avant de trancher (métrique retenue : IoU inter-méthodes + revue visuelle des désaccords).
- Fréquence des cas à 2 ailes dans les crops — conditionne si le resserrement doit gérer nativement le multi-instance.
- Fiabilité du recalage par features sur les cas `wide_recadre_leger` : à vérifier en priorité, car c'est là que le template matching simple est le plus susceptible d'échouer.
- Validation finale : mesurer si le nouveau format de crop rend effectivement le modèle de Gabriel exploitable (comparaison de son score de fiabilité/erreur en pixels à son test set d'origine).