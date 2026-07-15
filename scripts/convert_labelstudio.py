#!/usr/bin/env python3
"""
Convertit le JSON de Label Studio au format attendu :
  [{"image": image_path, "boxes": [[x1,y1,x2,y2],...]}]

Entrée Label Studio :
  - x, y, width, height en pourcentage (0-100)
  - original_width, original_height en pixels

Sortie attendue :
  - x1, y1, x2, y2 en pixels (coordonnées absolues)
"""

import json
import sys
from pathlib import Path


def convert_labelstudio_to_boxes(
    labelstudio_json,
    output_json="annotations.json",
    image_root="data/images",
    dataset_pattern="organized"
):
    """
    Convertit un JSON Label Studio au format boxes.
    
    Args:
        labelstudio_json: chemin du fichier JSON de Label Studio
        output_json: chemin du fichier JSON de sortie
        image_root: répertoire racine des images (ex: data/images)
        dataset_pattern: sous-dossier du dataset (ex: organized)
    
    Returns:
        Nombre d'images converties
    """
    
    # Charger le JSON Label Studio
    with open(labelstudio_json, "r") as f:
        labelstudio_data = json.load(f)
    
    output_data = []
    image_root_path = Path(image_root)
    
    print(f"🔍 Conversion de {len(labelstudio_data)} tâches...")
    
    for idx, task in enumerate(labelstudio_data, 1):
        # Extraire le nom de fichier de l'image
        file_upload = task.get("file_upload", "")
        
        if not file_upload:
            print(f"  ⚠️  Tâche {idx}: pas de fichier image trouvé")
            continue
        
        # Extraire le nom du fichier (dernière partie après /)
        full_filename = Path(file_upload).name
        
        # Label Studio renomme les fichiers en "id-nom_original"
        # On extrait le vrai nom après le tiret
        if "-" in full_filename:
            image_filename = full_filename.split("-", 1)[1]
        else:
            image_filename = full_filename
        
        # Chercher l'image dans organized/
        image_path = None
        for organized_dir in image_root_path.rglob(dataset_pattern):
            if organized_dir.is_dir():
                # Chercher récursivement dans organized/
                for found_image in organized_dir.rglob(image_filename):
                    image_path = found_image
                    break
            if image_path:
                break
        
        if not image_path:
            print(f"  ⚠️  Tâche {idx}: image '{image_filename}' non trouvée")
            continue
        
        # Extraire les annotations
        annotations = task.get("annotations", [])
        if not annotations:
            print(f"  ⚠️  Tâche {idx}: pas d'annotations")
            continue
        
        # Traiter la première annotation complétée
        annotation = annotations[0]
        results = annotation.get("result", [])
        
        boxes = []
        
        for result in results:
            # Vérifier que c'est une bounding box
            if result.get("type") != "rectanglelabels":
                continue
            
            # Extraire dimensions
            original_width = result.get("original_width")
            original_height = result.get("original_height")
            
            if not original_width or not original_height:
                print(f"    ⚠️  Dimensions originales manquantes")
                continue
            
            # Extraire coordonnées en pourcentage
            value = result.get("value", {})
            x_pct = value.get("x")  # 0-100
            y_pct = value.get("y")  # 0-100
            w_pct = value.get("width")  # 0-100
            h_pct = value.get("height")  # 0-100
            
            if None in (x_pct, y_pct, w_pct, h_pct):
                print(f"    ⚠️  Coordonnées manquantes")
                continue
            
            # Convertir en pixels
            x1 = int((x_pct / 100) * original_width)
            y1 = int((y_pct / 100) * original_height)
            x2 = int(((x_pct + w_pct) / 100) * original_width)
            y2 = int(((y_pct + h_pct) / 100) * original_height)
            
            boxes.append([x1, y1, x2, y2])
        
        if boxes:
            output_data.append({
                "image": str(image_path.resolve()),
                "boxes": boxes
            })
            print(f"  ✅ {image_filename} ({len(boxes)} box{'es' if len(boxes) > 1 else ''})")
        else:
            print(f"  ⚠️  Tâche {idx}: aucune bounding box valide")
    
    # Sauvegarder
    with open(output_json, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n📁 Résultat sauvegardé : {output_json}")
    print(f"📊 {len(output_data)} image(s) avec annotations")
    
    return len(output_data)


def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_labelstudio_to_boxes.py <labelstudio.json> [output.json] [image_root]")
        print()
        print("Exemple:")
        print("  python convert_labelstudio_to_boxes.py project.json annotations.json data/images")
        sys.exit(1)
    
    labelstudio_json = sys.argv[1]
    output_json = sys.argv[2] if len(sys.argv) > 2 else "annotations.json"
    image_root = sys.argv[3] if len(sys.argv) > 3 else "data/images"
    
    if not Path(labelstudio_json).exists():
        print(f"❌ Fichier '{labelstudio_json}' non trouvé")
        sys.exit(1)
    
    convert_labelstudio_to_boxes(
        labelstudio_json,
        output_json,
        image_root
    )


if __name__ == "__main__":
    main()