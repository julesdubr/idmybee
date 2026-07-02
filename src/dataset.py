"""Dataset PyTorch : image d'aile -> heatmaps de landmarks.

Phase 1 : on recadre autour de la bounding box des landmarks connus (avec
marge) avant le letterbox resize. C'est un raccourci qui suppose qu'on
connaît déjà approximativement où est l'aile -- suffisant pour valider le
modèle de régression de points, mais à remplacer en Phase 2 par un détecteur
automatique pour l'inférence sur photo brute complète.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A

from utils import crop_around_landmarks, letterbox_resize, generate_heatmaps

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_augmentations(train: bool) -> A.Compose | None:
    if not train:
        return None  # pas d'augmentation en validation/inférence
    transforms = [
        A.Rotate(
            limit=15, border_mode=cv2.BORDER_CONSTANT, fill=(255, 255, 255), p=0.7
        ),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.GaussNoise(std_range=(0.02, 0.08), p=0.3),
        A.Affine(scale=(0.9, 1.1), translate_percent=(0.0, 0.05), p=0.5),
    ]
    return A.Compose(
        transforms,
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


class WingKeypointDataset(Dataset):
    def __init__(
        self,
        samples: list[dict],  # [{"image_path": ..., "landmarks": np.ndarray(N,2)}, ...]
        image_size: int = 256,
        heatmap_size: int = 64,
        sigma: float = 1.5,
        margin_ratio: float = 0.25,
        train: bool = True,
    ):
        self.samples = samples
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.margin_ratio = margin_ratio
        self.train = train
        self.aug = build_augmentations(train)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = cv2.imread(sample["image_path"])
        if image is None:
            raise FileNotFoundError(sample["image_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        landmarks = sample["landmarks"].astype(float)

        # Phase 1 : crop grossier basé sur les landmarks connus
        cropped, landmarks, _offset = crop_around_landmarks(
            image, landmarks, self.margin_ratio
        )

        # Augmentation (avant letterbox, sur l'image recadrée)
        if self.aug is not None:
            augmented = self.aug(image=cropped, keypoints=landmarks.tolist())
            cropped, landmarks = augmented["image"], np.array(
                augmented["keypoints"], dtype=float
            )

        # Mise à taille fixe avec conservation du ratio
        resized, landmarks, transform = letterbox_resize(
            cropped, landmarks, self.image_size
        )

        heatmaps = generate_heatmaps(
            landmarks, self.image_size, self.heatmap_size, self.sigma
        )

        img_tensor = resized.astype(np.float32) / 255.0
        img_tensor = (img_tensor - IMAGENET_MEAN) / IMAGENET_STD
        img_tensor = torch.from_numpy(img_tensor.transpose(2, 0, 1)).float()

        return {
            "image": img_tensor,
            "heatmaps": torch.from_numpy(heatmaps).float(),
            "landmarks": torch.from_numpy(
                landmarks
            ).float(),  # espace image_size, pour calcul de métrique
            "specimen_id": sample.get("specimen_id", str(idx)),
        }
