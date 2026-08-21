#!/usr/bin/env python3
"""Train a compact YOLO OBB detector on the prepared dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--model", default="yolo26n-obb.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--batch", type=int, default=-1)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--project", type=Path, default=Path("bombus"))
    p.add_argument("--name", default="yolo26n_1024")
    p.add_argument("--patience", type=int, default=25)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)

    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(args.project),
        name=args.name,
        pretrained=True,
        patience=args.patience,
        # Dataset-specific augmentation policy.
        degrees=180.0,
        translate=0.10,
        scale=0.20,
        shear=3.0,
        perspective=0.0005,
        fliplr=0.5,
        flipud=0.0,
        hsv_h=0.015,
        hsv_s=0.40,
        hsv_v=0.30,
        # Camera/field robustness.
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        erasing=0.05,
        # For this task, preserve full-object geometry rather than relying heavily
        # on aggressive multi-image composition.
        close_mosaic=0,
        cache=False,
        plots=True,
        save=True,
        verbose=True,
    )

    print(f"Training finished: {results.save_dir}")
    print(f"Best weights: {Path(results.save_dir) / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
