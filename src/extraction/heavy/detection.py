"""Détection des ailes avec la méthode heavy (YOLOE + SAM/segmentation).

La logique de sélection est la suivante :
YOLOE segmentation -> top-k candidats -> score CLIP -> OBB du meilleur candidat.

Ce module expose deux points d'entrée utilisés par `extract_wings.py` :

- `load_model(args)` : charge YOLOE, les références visuelles et CLIP.
- `process_one(ctx, target, args)` : détecte l'aile sur une image.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

import vpe


def add_arguments(parser) -> None:
    """Déclare les arguments CLI spécifiques au backend heavy."""
    parser.add_argument("--ref", required=True, help="JSON des références YOLOE.")
    parser.add_argument("--ref-crops", required=True, help="Dossier des crops de référence CLIP.")
    parser.add_argument("--model", default="yoloe-11s-seg.pt")
    parser.add_argument("--topk-candidates", type=int, default=1)
    parser.add_argument("--min-aspect-ok", type=float, default=1.3)
    parser.add_argument("--min-similarity", type=float, default=0.0)


def mask_to_obb(mask_points: np.ndarray, image_width: int, image_height: int):
    if mask_points is None or len(mask_points) < 3:
        return None

    rect = cv2.minAreaRect(mask_points.astype(np.float32))
    (center_x, center_y), (width, height), _ = rect
    if width < 1 or height < 1:
        return None

    box = cv2.boxPoints(rect).astype(np.float32)

    # Même convention que crops.csv : quatre coins x,y, normalisés.
    box[:, 0] /= image_width
    box[:, 1] /= image_height
    box = box.clip(0., 1.)

    aspect = max(width, height) / max(min(width, height), 1e-6)
    return box, aspect


def load_model(args):
    """Charge YOLOE (avec ses références baked-in)."""
    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

    references = vpe.load_references(args.ref)
    model = YOLOE(args.model)
    vpe.bake_references(
        model,
        references,
        YOLOEVPSegPredictor,
        imgsz=args.imgsz,
        device=args.device,
    )
    return {"model": model}


def process_one(ctx: dict, image: np.ndarray, row: dict, args) -> dict:
    """Détecte l'aile sur `image` et complète `row` (status/box/scores).

    `image` a déjà été chargée par l'appelant. `row` contient déjà les champs
    communs (image_id, specimen_id, dataset, ...) initialisés à leurs valeurs
    par défaut ; cette fonction ne fait que les mettre à jour.
    """
    model = ctx["model"]

    try:
        results = model.predict(
            image,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )
        result = results[0]
        count = len(result.boxes) if result.boxes is not None else 0
        row["n_detections"] = str(count)

        if count == 0:
            row["error_reason"] = "aucune_detection"
            return row

        if result.masks is None:
            row["error_reason"] = "pas_de_masque"
            return row

        confidences = result.boxes.conf.cpu().numpy()
        order = np.argsort(-confidences)
        topk = order[:max(1, args.topk_candidates)]

        candidates = []
        for i in topk:
            candidate = mask_to_obb(
                result.masks.xy[i],
                image.shape[1],
                image.shape[0],
            )
            if candidate is None:
                continue

            box, aspect = candidate

            candidates.append(
                {
                    "confidence": float(confidences[i]),
                    "aspect": float(aspect),
                    "box": box,
                }
            )

        if not candidates:
            row["error_reason"] = "obb_degenere"
            return row

        best = max(candidates, key=lambda item: item["confidence"])

        if best["confidence"] < args.min_conf:
            row["error_reason"] = "confidence_insuffisante"
        elif best["aspect"] < args.min_aspect_ok:
            row["error_reason"] = "aspect_ratio_insuffisant"
        else:
            row["status"] = "OK"

        row["confidence"] = f"{best['confidence']:.4f}"
        row["selection_score"] = f"{best['similarity']:.4f}"
        values = best["box"].reshape(-1)
        for i in range(4):
            row[f"x{i + 1}"] = f"{values[2 * i]:.8f}"
            row[f"y{i + 1}"] = f"{values[2 * i + 1]:.8f}"

    except Exception as exc:
        row["error_reason"] = f"exception: {exc}"

    return row
