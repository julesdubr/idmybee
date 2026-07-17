"""Vérifie que le parsing du fichier .tps est correct en superposant les points sur
l'image d'origine.

Usage:
    python3 scripts/verify_tps.py --tps annotations.tps --sid 0 --out out/check.png --flip-y
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from utils.tps_parser import parse_tps, resolve_image_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tps", required=True)
    parser.add_argument("--sid", type=int, default=0)
    parser.add_argument("--flip-y", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    specimens = parse_tps(args.tps)
    print(f"{len(specimens)} spécimens trouvés dans le fichier .tps")

    spec = specimens[args.sid]
    img_path = Path.cwd().parent / spec.image_path

    print(f"Image du specimen numero {args.sid} : {img_path}")

    image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    landmarks = spec.landmarks.copy()
    if args.flip_y:
        landmarks[:, 1] = h - landmarks[:, 1]

    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], c="red", s=20)
    plt.title(
        f"ID={spec.sid} — image {img_path.name} ({w}x{h})"
    )
    plt.axis("off")
    plt.show()

    if args.out is not None:
        plt.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"Vérification sauvegardée dans {args.out}")

if __name__ == "__main__":
    main()
