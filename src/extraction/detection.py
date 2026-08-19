"""
Détection YOLOE : VPE multi-référence (une ou plusieurs images d'exemple).

Aucune dépendance à la géométrie de découpe, à CLIP, ou au manifest : ce
module ne sait faire que "calculer et injecter le VPE dans le modèle YOLOE".
"""

from pathlib import Path

import numpy as np
import torch
import sys


def load_references(ref_json_path: str):
    """Retourne une liste de (image_path, boxes) -- 1 ou plusieurs entrées."""
    import json
    with open(ref_json_path) as f:
        data = json.load(f)

    base_root = Path(data["base_root"][sys.platform])
    return [(base_root / Path(entry["image"]), np.array(entry["boxes"], dtype=np.float32)) for entry in data["references"]]


def compute_vpe(model, image_path: str, boxes: np.ndarray, predictor_cls, imgsz: int = 1024, device=None):
    """VPE (visual prompt embedding) d'UNE image de référence.

    `refer_image=` du predict() haut niveau ne prend qu'une image par appel
    (assertion dans get_vpe), donc pour moyenner plusieurs références on
    reproduit ici ce que YOLOE.predict() fait en interne pour une seule.
    Le predicteur doit être construit avec les mêmes overrides (task/batch/
    device/imgsz) que le code source, sinon le VPE résultant est dégradé.
    """
    from ultralytics.cfg import get_cfg

    visual_prompts = dict(bboxes=boxes, cls=np.zeros(len(boxes), dtype=int))
    if type(model.predictor) is not predictor_cls:
        args = get_cfg(overrides={**model.overrides, "imgsz": imgsz, "device": device})
        model.predictor = predictor_cls(
            overrides={
                "task": model.model.task, "mode": "predict", "save": False, "verbose": False,
                "batch": 1, "device": args.device, "quantize": args.quantize, "imgsz": args.imgsz,
            },
            _callbacks=model.callbacks,
        )
    model.model.model[-1].nc = 1
    model.model.names = ["object0"]
    model.predictor.set_prompts(visual_prompts.copy())
    model.predictor.setup_model(model=model.model, verbose=False)
    return model.predictor.get_vpe(image_path)


def bake_references(model, references, predictor_cls, imgsz: int = 1024, device=None):
    """Calcule le VPE de chaque référence, moyenne, injecte dans le modèle.
    Après appel, model.predict(image, ...) fonctionne directement (VPE déjà
    "baked in", plus besoin de refer_image/visual_prompts).

    Note : set_classes doit être appelé sur le nn.Module sous-jacent
    (model.model.set_classes), pas sur le wrapper YOLOE.set_classes -- ce
    dernier est pensé pour du texte et ne met pas à jour le bon état interne.
    """
    vpes = [compute_vpe(model, img, boxes, predictor_cls, imgsz, device) for img, boxes in references]
    avg_vpe = torch.mean(torch.stack(vpes, dim=0), dim=0)
    avg_vpe = avg_vpe / avg_vpe.norm(dim=-1, keepdim=True)  # les VPE individuels sont unitaires
    model.model.set_classes(["object0"], avg_vpe)
    model.model.names = ["object0"]
    model.predictor = None  # un predicteur standard sera recréé au 1er predict() réel