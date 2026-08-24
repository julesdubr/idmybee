"""Backend de détection light (YOLO-OBB spécialisé).

Entrée : une image BGR (np.ndarray).
Sortie : `detect_one()` -> dict avec status/box/scores (voir detect_wing.py).
"""

from __future__ import annotations

import numpy as np


def add_arguments(parser) -> None:
    """Déclare les arguments CLI spécifiques au backend light."""
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-det", type=int, default=10)


def load_model(args):
    """Charge le modèle YOLO-OBB. Retourne un contexte (dict)."""
    from ultralytics import YOLO

    return {"model": YOLO(args.model)}


def detect_one(ctx: dict, image: np.ndarray, args) -> dict:
    """Détecte l'aile sur une image unique.

    Retourne un dict : `status` (OK/FAILED), `error_reason`, `confidence`,
    `n_detections`, `box` (4x2 np.ndarray normalisé [0,1] ou None).
    """
    result = {
        "status": "FAILED",
        "error_reason": "",
        "confidence": None,
        "n_detections": 0,
        "box": None,
    }

    model = ctx["model"]

    try:
        results = model.predict(
            image,
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            max_det=args.max_det,
            verbose=False,
        )
        prediction = results[0]
        count = len(prediction.obb) if prediction.obb is not None else 0
        result["n_detections"] = count

        if prediction.obb is None or count == 0:
            result["error_reason"] = "aucune_detection"
            return result

        confidences = prediction.obb.conf.detach().cpu().numpy()
        best = int(np.argmax(confidences))

        points = prediction.obb.xyxyxyxy[best].detach().cpu().numpy().reshape(4, 2)
        h, w = image.shape[:2]
        points[:, 0] /= w
        points[:, 1] /= h
        points = np.clip(points, 0.0, 1.0)

        result["status"] = "OK"
        result["confidence"] = float(confidences[best])
        result["box"] = points

    except Exception as exc:
        result["error_reason"] = f"exception: {exc}"

    return result