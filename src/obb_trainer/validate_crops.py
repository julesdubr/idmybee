#!/usr/bin/env python3
"""Validate crops.csv directly by drawing its normalized OBBs on source images.

The expected crop columns are x1,y1,...,x4,y4, already normalized to [0,1].
No geometric reconstruction is performed.
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

POINT_COLUMNS = ["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images-csv", type=Path, required=True)
    p.add_argument("--crops-csv", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--split", choices=["organized", "basile_m1", "vrac", "terrain", "all"], default="all")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"OpenCV cannot decode {path}")
    return img


def normalized_points(row):
    pts = np.asarray([float(row[c]) for c in POINT_COLUMNS], dtype=np.float64).reshape(4, 2)
    if not np.all(np.isfinite(pts)):
        raise ValueError("NaN/Inf in OBB")
    if np.any(pts < 0) or np.any(pts > 1):
        raise ValueError(f"OBB outside [0,1]: {pts.tolist()}")
    return pts


def draw(img, pts_norm, label):
    h, w = img.shape[:2]
    pts = pts_norm.copy()
    pts[:, 0] *= w
    pts[:, 1] *= h
    q = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [q], True, (0, 220, 0), max(2, int(max(h, w) / 1500)), cv2.LINE_AA)
    for i, (x, y) in enumerate(pts):
        cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(img, str(i + 1), (int(x) + 7, int(y) - 7), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(img, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, .75, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(img, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def fit(img, max_side=1400):
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s == 1:
        return img
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def main():
    a = parse_args()
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    crops = pd.read_csv(a.crops_csv)
    images = pd.read_csv(a.images_csv)
    crops = crops[crops["status"].astype(str).eq("OK")].copy()
    if a.split != "all":
        crops = crops[crops["dataset"].astype(str).eq(a.split)]
    df = crops.merge(images[["image_id", "raw_path"]], on="image_id", how="left", validate="one_to_one")
    df = df.sample(min(a.n, len(df)), random_state=a.seed)

    rows = []
    errors = []
    for _, r in df.iterrows():
        try:
            src = Path(str(r["raw_path"]))
            with Image.open(src) as im:
                im.verify()
            pts = normalized_points(r)
            img = load_image(src)
            label = f"{r.image_id} | {r.specimen_id} | status=OK"
            img = fit(draw(img, pts, label))
            dst = out / "individual" / f"{r.image_id}.jpg"
            dst.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            rows.append((dst, r["image_id"]))
        except Exception as e:
            errors.append({"image_id": r.get("image_id", ""), "error": repr(e)})

    if rows:
        make_sheet([p for p, _ in rows], out / "contact_sheet.jpg", cols=4)
    pd.DataFrame(errors).to_csv(out / "errors.csv", index=False)
    print(f"Validated/visualized: {len(rows)}")
    print(f"Errors: {len(errors)}")
    print(f"Output: {out}")


def make_sheet(files, output, cols=4, thumb_w=420):
    from PIL import ImageDraw
    thumbs = []
    for f in files:
        im = Image.open(f).convert("RGB")
        scale = thumb_w / im.width
        im = im.resize((thumb_w, max(1, int(im.height * scale))), Image.Resampling.LANCZOS)
        thumbs.append((f.name, im))
    if not thumbs:
        return
    cell_h = max(im.height for _, im in thumbs) + 30
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * thumb_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (name, im) in enumerate(thumbs):
        x = (i % cols) * thumb_w
        y = (i // cols) * cell_h
        draw.text((x + 4, y + 3), name, fill="black")
        sheet.paste(im, (x, y + 24))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=95)


if __name__ == "__main__":
    main()
