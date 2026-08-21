# Extraction des ailes avec le détecteur OBB spécialisé

Deux fichiers ont une responsabilité distincte.

- `obb_geometry.py` reproduit la géométrie du pipeline lourd et ne fait que la lecture image, le redressement OBB et le letterbox.
- `extract_wings_obb.py` sélectionne les images, lance le modèle YOLO-OBB, écrit les crops 512x256 et le CSV.

## 1. Comparaison strictement équivalente au pipeline lourd

Cette méthode reprend uniquement les `image_id` dont la dernière entrée dans `crops.csv` est `OK`.

```powershell
python .\src\bombus_obb_detector\extract_wings_obb.py `
    --model .\runs\obb\weights\best.pt `
    --images_csv .\data\manifest\images.csv `
    --reference_csv .\data\manifest\crops.csv `
    --output_root .\data\crops_obb `
    --csv .\data\crops_obb\crops_obb.csv `
    --imgsz 1024 `
    --conf 0.10 `
    --padding 0.10
```

Le résultat est donc directement comparable aux crops produits par la méthode lourde.

Les images finales sont des JPEG grayscale de taille exacte `512x256`.

## 2. Traitement d'un dossier

```powershell
python .\src\bombus_obb_detector\extract_wings_obb.py `
    --model .\runs\obb\weights\best.pt `
    --input .\data\bombus_obb_dataset\images `
    --output_root .\data\crops_obb `
    --imgsz 1024 `
    --conf 0.10
```

`--input` peut être répété et les sous-dossiers sont parcourus récursivement.

## 3. Principe de normalisation

La géométrie est la même que dans `crop_wings.py` :

1. le grand axe de l'OBB est déterminé ;
2. l'image est tournée ;
3. la zone correspondant à l'OBB est recadrée ;
4. un padding de `10 %` est appliqué, comme dans le pipeline lourd ;
5. le crop est letterboxé en `512x256`, sans déformation ;
6. l'image finale est convertie en niveaux de gris.

La seule différence expérimentale est la source de l'OBB :
- méthode lourde : YOLOE + masque ;
- nouvelle méthode : modèle YOLO-OBB spécialisé.

## 4. CSV produit

Colonnes :

`image_id,specimen_id,dataset,status,error_reason,confidence,aspect_ratio,x1,y1,x2,y2,x3,y3,x4,y4,output_path,processing_time_s,processed_at`

Les coordonnées `x1...y4` sont normalisées dans l'image source et correspondent aux quatre sommets de l'OBB détectée.
