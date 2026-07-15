#!/usr/bin/env python3
"""
Convertit les images HEIC en JPEG et les place dans le dossier role parent.

Architecture attendue:
  organized/species/role/images.jpg
  organized/species/role/heic/images.heic

Les images HEIC sont converties et stockées directement dans le dossier role.

Dépendances :
  pip install pillow pillow-heif
"""

import os
import sys
from pathlib import Path
from PIL import Image

# Enregistrer le codec HEIF
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    print("❌ pillow-heif n'est pas installé.")
    print("   Installez avec : pip install pillow-heif")
    sys.exit(1)

def find_and_convert_heic(root_dir="organized"):
    """
    Parcourt l'arborescence et convertit les HEIC en JPEG.
    Supprime ensuite les dossiers heic vidés.
    
    Args:
        root_dir: répertoire racine à scanner (défaut: "organized")
    
    Returns:
        dict contenant les statistiques de conversion
    """
    root_path = Path(root_dir)
    
    if not root_path.exists():
        print(f"❌ Le répertoire '{root_dir}' n'existe pas.")
        return {"error": f"Directory {root_dir} not found"}
    
    stats = {
        "converted": 0,
        "failed": 0,
        "skipped": 0,
        "total_found": 0,
        "deleted_dirs": 0
    }
    
    # Lister les dossiers heic à supprimer après conversion
    heic_dirs_to_delete = []
    
    # Parcourir : organized/species/role/heic/
    for species_dir in root_path.iterdir():
        if not species_dir.is_dir():
            continue
        
        for role_dir in species_dir.iterdir():
            if not role_dir.is_dir():
                continue
            
            heic_dir = role_dir / "heic"
            
            # Vérifier si le dossier heic existe
            if not heic_dir.exists():
                continue
            
            # Chercher les fichiers HEIC
            heic_files = list(heic_dir.glob("*.heic")) + list(heic_dir.glob("*.HEIC"))
            
            if not heic_files:
                continue
            
            # Marquer ce dossier pour suppression potentielle
            heic_dirs_to_delete.append(heic_dir)
            
            print(f"\n📁 {species_dir.name}/{role_dir.name}/heic ({len(heic_files)} fichier(s))")
            
            for heic_file in heic_files:
                stats["total_found"] += 1
                jpeg_filename = heic_file.stem + ".jpg"
                jpeg_path = role_dir / jpeg_filename
                
                # Vérifier si le fichier JPEG existe déjà
                if jpeg_path.exists():
                    print(f"  ⏭️  {heic_file.name} → {jpeg_filename} (existe déjà)")
                    stats["skipped"] += 1
                    continue
                
                try:
                    # Ouvrir et convertir
                    img = Image.open(heic_file)
                    # Convertir RGBA/autres en RGB si nécessaire
                    if img.mode in ("RGBA", "LA", "P"):
                        rgb_img = Image.new("RGB", img.size, (255, 255, 255))
                        rgb_img.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
                        img = rgb_img
                    
                    # Sauvegarder en JPEG
                    img.save(jpeg_path, "JPEG", quality=95)
                    print(f"  ✅ {heic_file.name} → {jpeg_filename}")
                    stats["converted"] += 1
                
                except Exception as e:
                    print(f"  ❌ {heic_file.name} : {e}")
                    stats["failed"] += 1
    
    # Supprimer les dossiers heic (uniquement s'il n'y a pas eu d'erreurs)
    if stats["failed"] == 0:
        print(f"\n🗑️  Suppression des dossiers heic...")
        for heic_dir in heic_dirs_to_delete:
            try:
                import shutil
                shutil.rmtree(heic_dir)
                print(f"  ✅ {heic_dir.relative_to(Path(root_dir))}/")
                stats["deleted_dirs"] += 1
            except Exception as e:
                print(f"  ⚠️  {heic_dir.relative_to(Path(root_dir))}/ : {e}")
    else:
        print(f"\n⚠️  Conversions échouées détectées. Dossiers heic conservés pour inspection.")
    
    return stats


def main():
    root_dir = sys.argv[1] if len(sys.argv) > 1 else "organized"
    
    print(f"🔍 Scan du répertoire '{root_dir}'...")
    stats = find_and_convert_heic(root_dir)
    
    if "error" in stats:
        print(f"\n{stats['error']}")
        sys.exit(1)
    
    # Résumé
    print("\n" + "="*50)
    print("📊 Résumé de conversion")
    print("="*50)
    print(f"Total trouvé        : {stats['total_found']}")
    print(f"Converties          : {stats['converted']}")
    print(f"Ignorées (existent) : {stats['skipped']}")
    print(f"Erreurs             : {stats['failed']}")
    print(f"Dossiers supprimés  : {stats['deleted_dirs']}")
    print("="*50)
    
    if stats["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()