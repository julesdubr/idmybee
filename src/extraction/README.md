# Extraction des ailes

Le pipeline d'extraction est séparé en deux méthodes de détection et une étape commune de normalisation.
Les deux méthodes partagent désormais une seule boucle principale (`extract_wings.py`) : detection -> normalisation candidate, avec écriture incrémentale du CSV par lots de 50 images.

```text
images.csv
   |
   +--------------------+
   |                    |
   v                    v
heavy                light
YOLOE + SAM           YOLO-OBB
   |                    |
   +---------+----------+
             |
             v
      extract_wings.py
             |
             v
        detections.csv
             |
             v
       normalize_crop.py
             |
             v
          512 x 256
```

## Arborescence

```text
src/extraction/
├── detection_io.py
├── extract_wings.py
├── normalize_crop.py
├── heavy/
│   ├── __init__.py
│   ├── detection.py
│   ├── qa_clip.py
│   └── vpe.py
└── light/
    ├── __init__.py
    └── detection.py
```

`extract_wings.py` est l'unique point d'entrée CLI pour la détection. Il porte la boucle
commune (lecture de `images.csv`, timing, écriture par lots) et délègue à un backend :

- `heavy/detection.py` : `load_model(args)` charge YOLOE (avec ses références baked-in) et
  CLIP ; `process_one(ctx, image, row, args)` fait la détection YOLOE -> top-k candidats ->
  score CLIP -> OBB du meilleur candidat.
- `light/detection.py` : `load_model(args)` charge le modèle YOLO-OBB ; `process_one(ctx,
  image, row, args)` fait la détection directe.

Chaque backend déclare ses propres arguments CLI via `add_arguments(parser)`, ajoutés
dynamiquement selon `--mode`.

## Méthode heavy

```powershell
python .\src\extraction\extract_wings.py --mode heavy `
    --images-csv .\data\manifest\images.csv `
    --output-csv .\data\extraction\heavy\detections.csv `
    --ref .\data\references\ref-obb.json `
    --ref-crops .\data\references\ref-crops `
    --model yoloe-11s-seg.pt `
    --imgsz 1024 `
    --conf 0.05
```

Les références YOLOE et les crops de référence CLIP restent des paramètres spécifiques à la
méthode heavy (`--ref`, `--ref-crops`), requis uniquement en mode heavy.

## Méthode light

```powershell
python .\src\extraction\extract_wings.py --mode light `
    --images-csv .\data\manifest\images.csv `
    --output-csv .\data\extraction\light\detections.csv `
    --model .\runs\obb\weights\best.pt `
    --imgsz 1024 `
    --conf 0.10
```

Le mode light ne connaît ni `ref.json`, ni `ref-crops`, ni CLIP.

## Écriture incrémentale

`extract_wings.py` accumule les lignes en mémoire par lots de 50 images, puis les ajoute
(append) au CSV de sortie — l'entête n'est écrite qu'une seule fois, au premier lot. À chaque
flush, la console affiche le temps total écoulé et le temps moyen par image :

```text
[50/2700] temps écoulé : 1m12.3s — moyenne : 1.446 s/image
[100/2700] temps écoulé : 2m25.1s — moyenne : 1.451 s/image
```

Cela limite l'empreinte mémoire sur les gros lots et permet de reprendre un suivi visuel de
la progression sans attendre la fin du traitement complet.

## Normalisation commune

```powershell
python .\src\extraction\normalize_crop.py `
    --images-csv .\data\manifest\images.csv `
    --detections-csv .\data\extraction\light\detections.csv `
    --output-root .\data\crops\light `
    --csv .\data\extraction\light\normalized.csv
```

Pour heavy, remplacer simplement `light` par `heavy`.

La normalisation :

1. convertit les quatre points OBB en pixels ;
2. redresse le grand axe ;
3. ajoute le padding autour de l'aile ;
4. agrandit le rectangle dans sa petite dimension en récupérant les pixels réels de l'image ;
5. vise un ratio 2:1 ;
6. redimensionne directement en 512x256 ;
7. écrit le crop en niveaux de gris.

Il n'y a plus de letterbox avec bandes blanches. Une bordure réfléchie peut uniquement être utilisée en dernier recours lorsqu'un rectangle 2:1 dépasse réellement les limites de l'image.