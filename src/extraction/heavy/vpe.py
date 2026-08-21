"""YOLOE visual-prompt detection helpers.

This module is deliberately specific to the heavy YOLOE + segmentation path.
It contains no crop normalization and no CLIP logic.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch


def load_references(ref_json_path: str | Path):
    """Load one or more YOLOE visual references."""
    with Path(ref_json_path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    base_root = Path(data["base_root"][sys.platform])
    return [
        (
            base_root / Path(entry["image"]),
            np.asarray(entry["boxes"], dtype=np.float32),
        )
        for entry in data["references"]
    ]


def compute_vpe(model, image_path, boxes, predictor_cls, imgsz=1024, device=None):
    """Compute the YOLOE visual prompt embedding for one reference image."""
    from ultralytics.cfg import get_cfg

    visual_prompts = {
        "bboxes": boxes,
        "cls": np.zeros(len(boxes), dtype=int),
    }

    if type(model.predictor) is not predictor_cls:
        args = get_cfg(
            overrides={
                **model.overrides,
                "imgsz": imgsz,
                "device": device,
            }
        )
        model.predictor = predictor_cls(
            overrides={
                "task": model.model.task,
                "mode": "predict",
                "save": False,
                "verbose": False,
                "batch": 1,
                "device": args.device,
                "quantize": args.quantize,
                "imgsz": args.imgsz,
            },
            _callbacks=model.callbacks,
        )

    model.model.model[-1].nc = 1
    model.model.names = ["forewing"]
    model.predictor.set_prompts(visual_prompts.copy())
    model.predictor.setup_model(model=model.model, verbose=False)

    return model.predictor.get_vpe(str(image_path))


def bake_references(model, references, predictor_cls, imgsz=1024, device=None):
    """Average reference VPEs and inject the resulting class into YOLOE."""
    if not references:
        raise ValueError("Aucune référence YOLOE fournie.")

    vpes = [
        compute_vpe(
            model,
            image_path,
            boxes,
            predictor_cls,
            imgsz=imgsz,
            device=device,
        )
        for image_path, boxes in references
    ]

    average_vpe = torch.mean(torch.stack(vpes, dim=0), dim=0)
    average_vpe = average_vpe / average_vpe.norm(dim=-1, keepdim=True)

    model.model.set_classes(["forewing"], average_vpe)
    model.model.names = ["forewing"]
    model.predictor = None
