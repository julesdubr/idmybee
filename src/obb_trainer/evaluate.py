#!/usr/bin/env python3
"""Evaluate a trained OBB model and optionally export per-image predictions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--device", default="0")
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--pred-csv", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(str(args.model))
    metrics = model.val(
        data=str(args.data),
        split=args.split,
        imgsz=args.imgsz,
        device=args.device,
        plots=True,
        verbose=True,
    )

    print("\nOBB metrics")
    print(f"mAP50-95: {metrics.box.map:.5f}")
    print(f"mAP50:    {metrics.box.map50:.5f}")
    print(f"mAP75:    {metrics.box.map75:.5f}")

    if args.pred_csv:
        image_dir = args.data.parent / "images" / args.split
        rows = []
        for path in sorted(image_dir.iterdir()):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            result = model.predict(source=str(path), imgsz=args.imgsz, device=args.device, verbose=False)[0]
            if result.obb is None or len(result.obb) == 0:
                rows.append({"image_id": path.stem, "confidence": "", "x1": "", "y1": "", "x2": "", "y2": "", "x3": "", "y3": "", "x4": "", "y4": ""})
                continue
            i = int(result.obb.conf.argmax().item())
            conf = float(result.obb.conf[i].item())
            pts = result.obb.xyxyxyxy[i].cpu().numpy().reshape(-1).tolist()
            rows.append({
                "image_id": path.stem,
                "confidence": conf,
                **{f"x{j}": pts[2*(j-1)] for j in range(1, 5)},
                **{f"y{j}": pts[2*(j-1)+1] for j in range(1, 5)},
            })
        args.pred_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.pred_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else ["image_id"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Predictions written to: {args.pred_csv}")


if __name__ == "__main__":
    main()
