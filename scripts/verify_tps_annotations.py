"""À lancer EN PREMIER, avant tout entraînement : vérifie que le parsing du
fichier .tps est correct en superposant les points sur l'image d'origine.

Usage:
    python scripts/verify_tps_annotations.py --tps annotations.tps \
        --images-root photos/ --specimen-idx 0 --out check.png

Si les points apparaissent décalés verticalement (symétrie haut/bas), c'est
le problème classique de convention d'axe Y de TPSdig2 -- relancer avec
--flip-y.
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from tps_parser import parse_tps_file, resolve_image_path, index_by_id  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tps", required=True)
    parser.add_argument("--images-root", required=True)
    parser.add_argument(
        "--specimen-idx", type=int, default=None,
        help="Position dans la liste parsée (ATTENTION : ne correspond pas à l'ID déclaré dans le .tps)",
    )
    parser.add_argument(
        "--specimen-id", type=str, default=None,
        help="ID déclaré dans le .tps (ex. --specimen-id 17) -- utiliser de préférence à --specimen-idx",
    )
    parser.add_argument("--flip-y", action="store_true")
    parser.add_argument("--out", default="check.png")
    args = parser.parse_args()

    specimens = parse_tps_file(args.tps)
    print(f"{len(specimens)} spécimens trouvés dans le fichier .tps")

    if args.specimen_id is not None:
        spec = index_by_id(specimens)[args.specimen_id]
    elif args.specimen_idx is not None:
        spec = specimens[args.specimen_idx]
    else:
        spec = specimens[0]
    img_path = resolve_image_path(spec, args.images_root)
    
    print(f"Image du specimen numero {args.specimen_idx} : {img_path}")

    image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    landmarks = spec.landmarks.copy()
    if args.flip_y:
        landmarks[:, 1] = h - landmarks[:, 1]

    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], c="red", s=25)
    for i, (x, y) in enumerate(landmarks):
        plt.annotate(str(i + 1), (x, y), color="blue", fontsize=9, xytext=(3, 3), textcoords="offset points")
    plt.title(
        f"ID={spec.specimen_id} (OrigNum={spec.orig_num}) — image {img_path.name} ({w}x{h})"
    )
    plt.axis("off")
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Vérification sauvegardée dans {args.out}")
    print("-> Vérifier visuellement que chaque point rouge tombe bien sur une intersection de nervure.")


if __name__ == "__main__":
    main()