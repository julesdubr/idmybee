#!/usr/bin/env python3
"""Single-image inference and OBB-aligned crop extraction."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--conf", type=float, default=0.45)
    p.add_argument("--device", default="0")
    p.add_argument("--pad", type=float, default=0.03)
    p.add_argument("--long-side", type=int, default=1400)
    return p.parse_args()


def order_points(pts: np.ndarray) -> np.ndarray:
    # Convex quadrilateral ordering by angle around centroid.
    c = pts.mean(axis=0)
    a = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    return pts[np.argsort(a)]


def rotate_crop(image: np.ndarray, pts: np.ndarray, out_long_side: int, pad: float) -> np.ndarray:
    pts = order_points(pts.astype(np.float32))
    # Estimate adjacent side lengths.
    edges = np.roll(pts, -1, axis=0) - pts
    lengths = np.linalg.norm(edges, axis=1)
    width = max(lengths[0], lengths[2])
    height = max(lengths[1], lengths[3])
    if height > width:
        width, height = height, width
        # Rotate source corner ordering so long side becomes horizontal.
        pts = np.roll(pts, -1, axis=0)

    width_out = max(16, int(round(out_long_side)))
    height_out = max(16, int(round(out_long_side * height / width)))

    # Destination is a canonical horizontal rectangle.
    dst = np.array([
        [0, 0],
        [width_out - 1, 0],
        [width_out - 1, height_out - 1],
        [0, height_out - 1],
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    crop = cv2.warpPerspective(image, M, (width_out, height_out), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    if pad > 0:
        px = int(round(width_out * pad))
        py = int(round(height_out * pad))
        crop = cv2.copyMakeBorder(crop, py, py, px, px, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return crop


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model))

    result = model.predict(
        source=str(args.image),
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        verbose=False,
    )[0]
    if result.obb is None or len(result.obb) == 0:
        raise RuntimeError("No forewing detected above confidence threshold")

    i = int(result.obb.conf.argmax().item())
    conf = float(result.obb.conf[i].item())
    pts = result.obb.xyxyxyxy[i].cpu().numpy().astype(np.float32)

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(args.image)

    crop = rotate_crop(image, pts, args.long_side, args.pad)
    stem = args.image.stem
    crop_path = args.out / f"{stem}_wing.jpg"
    cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

    print(f"confidence={conf:.4f}")
    print(f"crop={crop_path}")
    print("corners_px=")
    print(pts)


if __name__ == "__main__":
    main()
