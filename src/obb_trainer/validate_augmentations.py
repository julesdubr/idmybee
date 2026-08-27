#!/usr/bin/env python3
"""Generate visually inspectable affine/photometric augmentations from a YOLO-OBB dataset.

The original labels are read in YOLO-OBB format, converted to pixels, transformed
with the exact same affine matrix as the image, then converted back to normalized
coordinates only for drawing. This validates the geometry independently of the
training framework.
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="train")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def read_label(path, w, h):
    v = np.fromstring(path.read_text(encoding="utf-8").strip(), sep=" ")
    if v.size != 9:
        raise ValueError(f"Invalid label: {path}")
    pts = v[1:].reshape(4, 2).astype(np.float64)
    pts[:, 0] *= w
    pts[:, 1] *= h
    return int(v[0]), pts


def transform_points(pts, M):
    hp = np.hstack([pts, np.ones((4, 1), dtype=np.float64)])
    return hp @ M.T


def photometric(img, rng, np_rng):
    out = img.astype(np.float32)
    out = out * rng.uniform(0.75, 1.25) + rng.uniform(-25, 25)
    if rng.random() < 0.35:
        k = rng.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), 0)
    if rng.random() < 0.30:
        noise_sigma = rng.uniform(1, 6)
        noise = np_rng.normal(0, noise_sigma, out.shape).astype(np.float32)
        out += noise
    return np.clip(out, 0, 255).astype(np.uint8)


def draw(img, pts, text):
    q = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [q], True, (0, 220, 0), max(2, int(max(img.shape[:2]) / 1500)), cv2.LINE_AA)
    for i, (x, y) in enumerate(pts):
        cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(img, str(i + 1), (int(x) + 7, int(y) - 7), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255), 2)
    cv2.putText(img, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 3)
    cv2.putText(img, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,0), 1)
    return img


def make_sheet(files, output, cols=4, thumb_w=420):
    from PIL import ImageDraw
    thumbs=[]
    for f in files:
        im=Image.open(f).convert("RGB")
        s=thumb_w/im.width
        im=im.resize((thumb_w,max(1,int(im.height*s))),Image.Resampling.LANCZOS)
        thumbs.append(im)
    if not thumbs: return
    cell_h=max(im.height for im in thumbs)+20
    rows=math.ceil(len(thumbs)/cols)
    sheet=Image.new("RGB",(cols*thumb_w,rows*cell_h),"white")
    for i,im in enumerate(thumbs):
        sheet.paste(im,((i%cols)*thumb_w,(i//cols)*cell_h+20))
    ImageDraw.Draw(sheet).text((4,2), "augmentation examples", fill="black")
    output.parent.mkdir(parents=True,exist_ok=True)
    sheet.save(output,quality=95)


def main():
    a=parse_args()
    rng=random.Random(a.seed)
    np_rng=np.random.default_rng(a.seed)
    img_dir=a.dataset/"images"/a.split
    lab_dir=a.dataset/"labels"/a.split
    files=[p for p in sorted(img_dir.iterdir()) if p.suffix.lower() in EXTS]
    rng.shuffle(files)
    files=files[:min(a.n,len(files))]
    out=a.out/a.split
    out.mkdir(parents=True,exist_ok=True)
    saved=[]
    for image_path in files:
        label_path=lab_dir/f"{image_path.stem}.txt"
        img=cv2.imread(str(image_path),cv2.IMREAD_COLOR)
        if img is None: continue
        h,w=img.shape[:2]
        _,pts=read_label(label_path,w,h)
        angle=rng.uniform(-180,180)
        scale=rng.uniform(.85,1.15)
        tx=rng.uniform(-.08,.08)*w
        ty=rng.uniform(-.08,.08)*h
        M=cv2.getRotationMatrix2D((w/2,h/2),angle,scale)
        M[:,2]+=[tx,ty]
        aug=cv2.warpAffine(img,M,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(255,255,255))
        aug=photometric(aug,rng,np_rng)
        new_pts=transform_points(pts,M)
        # Keep visualisation even when part of the transformed polygon is clipped.
        text=f"rot={angle:+.1f} deg scale={scale:.2f} shift=({tx/w:+.2f},{ty/h:+.2f})"
        aug=draw(aug,new_pts,text)
        dst=out/f"{image_path.stem}_aug.jpg"
        cv2.imwrite(str(dst),aug,[cv2.IMWRITE_JPEG_QUALITY,95])
        saved.append(dst)
    make_sheet(saved,out/"contact_sheet.jpg")
    print(f"Saved {len(saved)} augmented examples to {out}")


if __name__ == "__main__":
    main()
