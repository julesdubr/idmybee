"""Applique le modèle entraîné sur de nouvelles images et exporte les
landmarks prédits au format .tps, compatible avec le pipeline de calcul des
indicateurs morphométriques existant.

Note (Phase 1 -> Phase 2) : ce script suppose encore un crop grossier fourni
(bounding box approximative de l'aile). Tant que le détecteur automatique de
Phase 2 n'existe pas, `bbox` peut être fournie manuellement ou par une
heuristique simple (à ajouter dans utils.py).

Usage:
    python predict_export.py --checkpoint checkpoints/best_model.pt \
        --images-dir chemin/vers/nouvelles_photos --out predictions.tps
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from model import ResNetUNet
from utils import letterbox_resize, heatmaps_to_coords, unletterbox_coords
from dataset import IMAGENET_MEAN, IMAGENET_STD


def preprocess(image: np.ndarray, image_size: int):
    resized, _dummy_landmarks, transform = letterbox_resize(image, np.zeros((1, 2)), image_size)
    tensor = resized.astype(np.float32) / 255.0
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(tensor.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor, transform


def write_tps(records: list[dict], out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            lm = rec["landmarks"]
            f.write(f"LM={len(lm)}\n")
            for x, y in lm:
                f.write(f"{x:.4f} {y:.4f}\n")
            f.write(f"IMAGE={rec['image_path']}\n")
            f.write(f"ID={rec['specimen_id']}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-keypoints", type=int, default=19)
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetUNet(n_keypoints=args.n_keypoints).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    records = []
    image_paths = sorted(Path(args.images_dir).glob("*.jpg")) + sorted(Path(args.images_dir).glob("*.JPG"))

    for img_path in image_paths:
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        tensor, transform = preprocess(image, args.image_size)
        with torch.no_grad():
            heatmaps = model(tensor.to(device))[0].cpu().numpy()

        coords_resized_space = heatmaps_to_coords(heatmaps, args.image_size)
        coords_original = unletterbox_coords(coords_resized_space, transform)

        records.append(
            {
                "landmarks": coords_original,
                "image_path": str(img_path),
                "specimen_id": img_path.stem,
            }
        )
        print(f"{img_path.name}: OK")

    write_tps(records, args.out)
    print(f"{len(records)} prédictions écrites dans {args.out}")


if __name__ == "__main__":
    main()
