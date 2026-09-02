"""Heavy detection backend (YOLOE segmentation).

Input: a BGR image (np.ndarray).
Output: `detect_one()` -> dict with status/box/scores (see detect_wing.py).
"""
from __future__ import annotations

import cv2
import numpy as np

from . import vpe


def add_arguments(parser) -> None:
    """Declares the heavy backend's specific CLI arguments."""
    parser.add_argument("--ref", required=True, help="JSON file of YOLOE references.")
    parser.add_argument("--model", default="yoloe-11s-seg.pt")
    parser.add_argument("--min-aspect-ok", type=float, default=1.3)


def mask_to_obb(mask_points: np.ndarray, image_width: int, image_height: int):
    """Converts a mask to a normalized [0,1] OBB + aspect ratio."""
    if mask_points is None or len(mask_points) < 3:
        return None

    rect = cv2.minAreaRect(mask_points.astype(np.float32))
    (_, _), (width, height), _ = rect
    if width < 1 or height < 1:
        return None

    box = cv2.boxPoints(rect).astype(np.float32)
    box[:, 0] /= image_width
    box[:, 1] /= image_height
    box = np.clip(box, 0.0, 1.0)

    aspect = max(width, height) / max(min(width, height), 1e-6)
    return box, aspect


def load_model(args):
    """Loads YOLOE with its baked-in references. Returns a context (dict)."""
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
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )
        prediction = results[0]
        count = len(prediction.boxes) if prediction.boxes is not None else 0
        result["n_detections"] = count

        if count == 0:
            result["error_reason"] = "no_detection"
            return result

        if prediction.masks is None:
            result["error_reason"] = "no_mask"
            return result

        confidences = prediction.boxes.conf.cpu().numpy()
        best = int(np.argmax(confidences))

        candidate = mask_to_obb(
            prediction.masks.xy[best],
            image.shape[1],
            image.shape[0],
        )
        if candidate is None:
            result["error_reason"] = "degenerate_obb"
            return result

        box, aspect = candidate
        result["confidence"] = float(confidences[best])

        if aspect < args.min_aspect_ok:
            result["error_reason"] = "insufficient_aspect_ratio"
        else:
            result["status"] = "OK"
            result["box"] = box

    except Exception as exc:
        result["error_reason"] = f"exception: {exc}"

    return result
