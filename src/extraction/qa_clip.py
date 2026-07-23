"""
QA / sélection de candidat : similarité CLIP à des crops de référence.

Dépend de `geometry.read_image` pour charger les crops de référence (mêmes
garanties de tolérance de format), sinon aucune dépendance à YOLOE ou au
manifest.
"""

import torch
from PIL import Image

try:
    from . import geometry
except ImportError:
    import geometry


def load_clip(device):
    """Même modèle que celui déjà utilisé pour le QA du pipeline classique
    (open_clip ViT-B-16-plus-240 / laion400m_e32)."""
    import open_clip  # dépendance optionnelle, chargée seulement ici
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16-plus-240", pretrained="laion400m_e32"
    )
    model.eval().to(device)
    return model, preprocess


def embed(bgr_img, model, preprocess, device):
    import cv2
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    x = preprocess(Image.fromarray(rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode_image(x)
    return feat / feat.norm(dim=-1, keepdim=True)


def load_ref_embeddings(ref_crop_paths, model, preprocess, device):
    embs = []
    for p in ref_crop_paths:
        img = geometry.read_image(p)
        if img is None:
            print(f"ATTENTION: référence introuvable/illisible: {p}")
            continue
        embs.append(embed(img, model, preprocess, device))
    if not embs:
        raise ValueError("Aucune image de référence valide dans --ref_crops")
    return embs


def clip_similarity(crop_512x256, ref_embs, model, preprocess, device):
    """Similarité max du crop aux références connues. Sert de signal qualité
    (un corps/doigt mal détecté ne ressemblera à aucune référence) et, si
    plusieurs candidats sont comparés, de critère de sélection."""
    emb = embed(crop_512x256, model, preprocess, device)
    return max((emb @ r.T).item() for r in ref_embs)


def cuda_available():
    return torch.cuda.is_available()