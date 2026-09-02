"""Light detection backend (specialized YOLO-OBB).

Input: a BGR image (np.ndarray).
Output: `detect_one()` -> dict with status/box/scores (see detect_wing.py).
"""
from __future__ import annotations

import numpy as np


def add_arguments(parser) -> None:
    """Declares the light backend's specific CLI arguments."""
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-det", type=int, default=10)


def load_model(args):
    """Loads the YOLO-OBB model. Returns a context (dict)."""
    from ultralytics import YOLO

    return {"model": YOLO(args.model)}


def detect_one(ctx: dict, image: np.ndarray, args) -> dict:
    """Detects the wing on a single image.

    Returns a dict: `status` (OK/FAILED), `error_reason`, `confidence`,
    `n_detections`, `box` (4x2 np.ndarray normalized [0,1] or None).
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
            result["error_reason"] = "no_detection"
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
