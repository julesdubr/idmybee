"""CLIP quality assurance for the heavy YOLOE + SAM backend.

This module is intentionally isolated from the light detector.

Backend selection is automatic:
    1. OpenAI `clip` package if installed
    2. `open_clip` if installed

The public API is kept small so the extraction script does not depend on the
chosen CLIP implementation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


_BACKEND = None


def cuda_available() -> bool:
    return bool(torch.cuda.is_available())


def _load_openai_clip(device: str):
    import clip

    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    return model, preprocess


def _load_open_clip(device: str):
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="openai",
    )
    model = model.to(device)
    model.eval()
    return model, preprocess


def load_clip(device: str | None = None):
    """Load CLIP using an installed backend."""
    global _BACKEND

    device = device or ("cuda" if cuda_available() else "cpu")

    try:
        model, preprocess = _load_openai_clip(device)
        _BACKEND = "clip"
        return model, preprocess
    except ImportError:
        pass

    try:
        model, preprocess = _load_open_clip(device)
        _BACKEND = "open_clip"
        return model, preprocess
    except ImportError as exc:
        raise RuntimeError(
            "Aucun backend CLIP disponible. Installe `clip` ou `open_clip_torch`."
        ) from exc


def _encode_image(image, model, preprocess, device):
    tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        if _BACKEND == "clip":
            features = model.encode_image(tensor)
        elif _BACKEND == "open_clip":
            features = model.encode_image(tensor)
        else:
            raise RuntimeError("Backend CLIP non initialisé.")

    features = features.float()
    return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def load_ref_embeddings(ref_crop_paths, model, preprocess, device):
    """Encode all reference crops into normalized CLIP embeddings."""
    embeddings = []

    for path in ref_crop_paths:
        path = Path(path)
        if not path.is_file():
            continue

        image = Image.open(path).convert("RGB")
        embeddings.append(
            _encode_image(image, model, preprocess, device)
        )

    if not embeddings:
        raise ValueError("Aucun crop de référence CLIP exploitable.")

    return torch.cat(embeddings, dim=0)


def clip_similarity(image_bgr, ref_embeddings, model, preprocess, device):
    """Return the highest cosine similarity against reference crops."""
    import cv2

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(np.asarray(image_rgb))

    embedding = _encode_image(image, model, preprocess, device)
    scores = embedding @ ref_embeddings.T
    return float(scores.max().item())
