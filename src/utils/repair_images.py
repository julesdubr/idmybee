#!/usr/bin/env python3
"""Scan source/dataset images and optionally repair recoverable JPEG files."""
from __future__ import annotations
import argparse
from pathlib import Path
from PIL import Image, ImageOps

from tqdm import tqdm

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True, help='Dataset root or image directory')
    return p.parse_args()


def iter_images(root):
    exts = {'.jpg','.jpeg','.png','.webp','.bmp','.JPG','.JPEG'}
    return [p for p in root.rglob('*') if p.is_file() and p.suffix in exts]


def main():
    a = parse_args()
    paths = iter_images(a.root)

    for im_file in tqdm(paths):
        try:
            ImageOps.exif_transpose(Image.open(im_file)).save(
                im_file, "JPEG", subsampling=0, quality=100
            )
        except Exception as e:
            continue

if __name__=='__main__':
    main()
