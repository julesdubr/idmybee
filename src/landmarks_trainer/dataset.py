"""
Training manifest -> PyTorch Dataset.

Manifest CSV format (produced by export_dataset.py):
    image_id,crop_path,x0,y0,x1,y1,...,x{N-1},y{N-1}
- image_id:  matches the pipeline's image_id convention (joins back to
             images.csv/crops.csv if needed)
- crop_path: path to the crop image file, resolved relative to the
             manifest's own directory if not absolute (see resolve_path
             usage elsewhere in the pipeline for the same pattern)
- x{i},y{i}: landmark i coordinates in (x, y) -- the TPS/manifest
             convention. Converted to (row, col) once, in
             LandmarkHeatmapDataset.__getitem__, before touching
             heatmap.py or augment.py (see heatmap.py docstring for why
             this boundary matters).

This file owns that CSV <-> in-memory representation. It does not know
anything about TPS files or crops.csv -- that join happens in
export_dataset.py, once, ahead of time, so training itself only ever reads
one flat, self-contained manifest.
"""

import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from landmarks_trainer.constants import N_LANDMARKS, IMG_HEIGHT, IMG_WIDTH
from landmarks_trainer.heatmap import encode_heatmap
from landmarks_trainer.augment import augment as augment_fn


def load_manifest(csv_path: str, n_landmarks: int = N_LANDMARKS) -> pd.DataFrame:
    """Load and validate a training manifest CSV. Fails loudly (not
    silently) if expected columns are missing -- an incomplete manifest
    must never be trained on as if it were complete."""
    df = pd.read_csv(csv_path)

    required = ["image_id", "crop_path"]
    for i in range(n_landmarks):
        required += [f"x{i}", f"y{i}"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Manifest {csv_path} is missing columns {missing}. "
            f"Expected image_id, crop_path, and x0..y{n_landmarks - 1}."
        )

    manifest_dir = Path(csv_path).resolve().parent
    df["crop_path"] = df["crop_path"].apply(
        lambda p: p if os.path.isabs(p) else str((manifest_dir / p).resolve())
    )
    return df


def split_manifest(df: pd.DataFrame, val_fraction: float = 0.1,
                    test_fraction: float = 0.1, seed: int = 58,
                    group_col: str = None):
    """Split a manifest into train/val/test. If group_col is given (e.g.
    'specimen_id'), all rows sharing a group value stay in the same split --
    use this if a specimen can appear more than once in the manifest, to
    avoid leaking the same wing into both train and val."""
    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    n = len(df)
    idx = np.arange(n)

    if group_col is not None and group_col in df.columns:
        groups = df[group_col].values
        splitter1 = GroupShuffleSplit(n_splits=1, test_size=val_fraction + test_fraction,
                                       random_state=seed)
        train_idx, rest_idx = next(splitter1.split(idx, groups=groups))
        rest_groups = groups[rest_idx]
        rel_test = test_fraction / (val_fraction + test_fraction)
        splitter2 = GroupShuffleSplit(n_splits=1, test_size=rel_test, random_state=seed)
        val_idx_rel, test_idx_rel = next(splitter2.split(rest_idx, groups=rest_groups))
        val_idx, test_idx = rest_idx[val_idx_rel], rest_idx[test_idx_rel]
    else:
        train_idx, rest_idx = train_test_split(idx, test_size=val_fraction + test_fraction,
                                                 random_state=seed)
        rel_test = test_fraction / (val_fraction + test_fraction)
        val_idx, test_idx = train_test_split(rest_idx, test_size=rel_test, random_state=seed)

    return df.iloc[train_idx].reset_index(drop=True), \
        df.iloc[val_idx].reset_index(drop=True), \
        df.iloc[test_idx].reset_index(drop=True)


class LandmarkHeatmapDataset(Dataset):
    """Yields (image_tensor[3,H,W], heatmap_tensor[1,H,W]) pairs.

    When train_augment=True, a fresh random zoom/shift/quality augmentation
    is applied on every __getitem__ call (i.e. every epoch sees a different
    variant), rather than pre-generating a fixed augmented set once as the
    original notebook did.
    """

    def __init__(self, manifest: pd.DataFrame, n_landmarks: int = N_LANDMARKS,
                 img_shape: tuple = (IMG_HEIGHT, IMG_WIDTH),
                 train_augment: bool = False, heatmap_kwargs: dict = None):
        self.manifest = manifest.reset_index(drop=True)
        self.n_landmarks = n_landmarks
        self.img_shape = img_shape
        self.train_augment = train_augment
        self.heatmap_kwargs = heatmap_kwargs or {}

    def __len__(self):
        return len(self.manifest)

    def _load_points_xy(self, row) -> np.ndarray:
        return np.array(
            [[row[f"x{i}"], row[f"y{i}"]] for i in range(self.n_landmarks)],
            dtype=np.float64,
        )

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]

        image = cv2.imread(row["crop_path"])
        if image is None:
            raise FileNotFoundError(f"Could not read crop image: {row['crop_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.shape[:2] != tuple(self.img_shape):
            raise ValueError(
                f"{row['crop_path']} has shape {image.shape[:2]}, expected {self.img_shape}. "
                f"Crops must already be at the model's input size (no resize happens here, "
                f"matching landmarks/predict.py's inference path) -- either fix the crop, or "
                f"pass the right --img-height/--img-width if the extraction pipeline's output "
                f"size has changed."
            )

        points_xy = self._load_points_xy(row)
        points_rc = points_xy[:, ::-1]  # (x, y) -> (row, col), see heatmap.py docstring

        if self.train_augment:
            image, points_rc = augment_fn(image, points_rc)

        heatmap = encode_heatmap(points_rc, shape=self.img_shape, **self.heatmap_kwargs)

        image_t = torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32) / 255.0)
        heatmap_t = torch.from_numpy(heatmap).unsqueeze(0)

        return image_t, heatmap_t
