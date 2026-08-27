#!/usr/bin/env python3
"""Build a YOLO-OBB dataset from images.csv + crops.csv.

Only status == OK is used. The train/val/test split is done at specimen level,
so different photographs of the same specimen cannot leak across splits.

Expected crops.csv columns:
    image_id,specimen_id,status,x1,y1,x2,y2,x3,y3,x4,y4,...

The eight OBB coordinates are already normalized to [0, 1] in crops.csv.
They are used DIRECTLY. No x/y/w/h/theta reconstruction is performed.

Expected manifest.csv columns used by this script:
    image_id,specimen_id,raw_path

The output label format is YOLO-OBB:
    class_id x1 y1 x2 y2 x3 y3 x4 y4

where all coordinates remain normalized to [0, 1].
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from tqdm import tqdm

CLASS_ID = 0
CLASS_NAME = "forewing"
POINT_COLUMNS = [
    "x1", "y1",
    "x2", "y2",
    "x3", "y3",
    "x4", "y4",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest-csv", type=Path, required=True)
    p.add_argument("--crops-csv", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train", type=float, default=0.70)
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--test", type=float, default=0.15)
    p.add_argument(
        "--copy-mode",
        choices=["copy", "hardlink"],
        default="copy",
        help="copy is safest; hardlink saves disk when source is on the same volume.",
    )
    p.add_argument(
        "--clip",
        action="store_true",
        help="Clip OBB coordinates to [0,1]. By default, out-of-range coordinates raise an error.",
    )
    p.add_argument(
        "--allow-missing",
        action="store_true",
        help="Allow missing source images instead of failing.",
    )
    return p.parse_args()


def split_groups(
    specimens: list[str],
    train: float,
    val: float,
    test: float,
    seed: int,
):
    if not np.isclose(train + val + test, 1.0):
        raise ValueError("train + val + test must equal 1.0")
    if len(specimens) < 3:
        raise ValueError("At least 3 specimens are required for train/val/test splitting")

    rng = random.Random(seed)
    specimens = list(specimens)
    rng.shuffle(specimens)
    n = len(specimens)

    n_train = int(round(n * train))
    n_val = int(round(n * val))

    n_train = min(max(n_train, 1), n - 2)
    n_val = min(max(n_val, 1), n - n_train - 1)

    train_ids = set(specimens[:n_train])
    val_ids = set(specimens[n_train:n_train + n_val])
    test_ids = set(specimens[n_train + n_val:])
    return train_ids, val_ids, test_ids


def validate_and_get_points(row: pd.Series, clip: bool = False) -> np.ndarray:
    """Read x1..y4 directly from crops.csv and return shape (4,2)."""
    values = []
    for col in POINT_COLUMNS:
        value = row[col]
        if pd.isna(value):
            raise ValueError(f"Missing OBB coordinate {col} for image_id={row['image_id']}")
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid coordinate {col}={value!r} for image_id={row['image_id']}"
            ) from exc
        if not np.isfinite(value):
            raise ValueError(f"Non-finite coordinate {col} for image_id={row['image_id']}")
        values.append(value)

    pts = np.asarray(values, dtype=np.float64).reshape(4, 2)

    if clip:
        pts = np.clip(pts, 0.0, 1.0)
    else:
        bad = (pts < 0.0) | (pts > 1.0)
        if bad.any():
            bad_cols = [POINT_COLUMNS[i] for i in np.flatnonzero(bad.reshape(-1))]
            raise ValueError(
                f"OBB coordinates outside [0,1] for image_id={row['image_id']}: {bad_cols}; "
                f"values={pts.reshape(-1).tolist()}"
            )

    # Polygon should have a non-zero area. This catches corrupted labels.
    x = pts[:, 0]
    y = pts[:, 1]
    area2 = float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    if abs(area2) < 1e-10:
        raise ValueError(f"Degenerate OBB for image_id={row['image_id']}: {pts.tolist()}")

    return pts


def copy_file(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    crops = pd.read_csv(args.crops_csv)
    images = pd.read_csv(args.manifest_csv)

    required_crop = {"image_id", "specimen_id", "status", *POINT_COLUMNS}
    required_images = {"image_id", "specimen_id", "raw_path"}

    missing_crop = required_crop - set(crops.columns)
    missing_images = required_images - set(images.columns)
    if missing_crop:
        raise ValueError(f"Missing crop columns: {sorted(missing_crop)}")
    if missing_images:
        raise ValueError(f"Missing image columns: {sorted(missing_images)}")

    # Only confirmed annotations are used.
    crops = crops.loc[crops["status"].astype(str).str.upper().eq("OK")].copy()
    if crops.empty:
        raise RuntimeError("No rows with status == OK were found in crops.csv")

    crops["image_id"] = crops["image_id"].astype(str)
    crops["specimen_id"] = crops["specimen_id"].astype(str)
    images["image_id"] = images["image_id"].astype(str)
    images["specimen_id"] = images["specimen_id"].astype(str)

    duplicated = crops["image_id"].duplicated(keep=False)
    if duplicated.any():
        ids = crops.loc[duplicated, "image_id"].tolist()[:20]
        raise RuntimeError(
            f"Found duplicated image_id values in OK crops.csv. Examples: {ids}"
        )

    df = crops.merge(
        images[["image_id", "specimen_id", "raw_path"]],
        on="image_id",
        how="left",
        suffixes=("", "_img"),
        validate="one_to_one",
    )

    if df["raw_path"].isna().any():
        missing = int(df["raw_path"].isna().sum())
        examples = df.loc[df["raw_path"].isna(), "image_id"].head(20).tolist()
        raise RuntimeError(
            f"{missing} OK crops have no matching raw image in manifest.csv. Examples: {examples}"
        )

    # Crop CSV specimen_id is authoritative, but require exact agreement with manifest.csv.
    if "specimen_id_img" in df.columns:
        bad = ~df["specimen_id"].eq(df["specimen_id_img"])
        if bad.any():
            examples = df.loc[bad, ["image_id", "specimen_id", "specimen_id_img"]].head(20)
            raise RuntimeError(
                "specimen_id mismatch between crops.csv and manifest.csv:\n"
                + examples.to_string(index=False)
            )
        df = df.drop(columns=["specimen_id_img"])

    specimens = sorted(df["specimen_id"].unique())
    train_ids, val_ids, test_ids = split_groups(
        specimens, args.train, args.val, args.test, args.seed
    )

    split_map = {sid: "train" for sid in train_ids}
    split_map.update({sid: "val" for sid in val_ids})
    split_map.update({sid: "test" for sid in test_ids})
    df["split"] = df["specimen_id"].map(split_map)

    # Save an exact manifest with the source OBB coordinates.
    manifest_columns = [
        "image_id", "specimen_id", "raw_path", "split", "status",
        *POINT_COLUMNS,
    ]
    df[manifest_columns].to_csv(out / "manifest.csv", index=False)

    # Validate all OBBs before copying any data.
    print(f"Validating {len(df)} OK annotations...")
    for row in df.itertuples(index=False):
        # Build a lightweight Series for the shared validation function.
        row_dict = {"image_id": row.image_id}
        for col in POINT_COLUMNS:
            row_dict[col] = getattr(row, col)
        validate_and_get_points(pd.Series(row_dict), clip=args.clip)
    print("All OBB annotations are valid.")

    stats = []
    missing_sources = []

    from PIL import Image, UnidentifiedImageError

    for split in ["train", "val", "test"]:
        rows = df[df["split"].eq(split)]
        img_dir = out / "images" / split
        label_dir = out / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        copied = 0
        missing = 0
        unreadable = 0

        for _, row in tqdm(rows.iterrows(), total=rows.shape[0], desc=split):
            src = Path(str(row["raw_path"]))
            if not src.exists():
                missing += 1
                missing_sources.append({
                    "split": split,
                    "image_id": row["image_id"],
                    "raw_path": str(src),
                    "reason": "missing",
                })
                continue

            # Verify that Pillow can actually read the source image now.
            try:
                with Image.open(src) as im:
                    im.verify()
                with Image.open(src) as im:
                    width, height = im.size
                    if width <= 0 or height <= 0:
                        raise ValueError("invalid image dimensions")
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                unreadable += 1
                missing_sources.append({
                    "split": split,
                    "image_id": row["image_id"],
                    "raw_path": str(src),
                    "reason": f"unreadable: {exc}",
                })
                continue

            suffix = src.suffix.lower() or ".jpg"
            dst_name = f"{row['image_id']}{suffix}"
            dst = img_dir / dst_name
            copy_file(src, dst, args.copy_mode)

            # Use coordinates directly from crops.csv. They are already normalized.
            pts = validate_and_get_points(row, clip=args.clip)
            label_path = label_dir / f"{row['image_id']}.txt"
            values = [CLASS_ID] + pts.reshape(-1).tolist()
            label_path.write_text(
                " ".join(f"{v:.8f}" for v in values) + "\n",
                encoding="utf-8",
            )
            copied += 1

        stats.append({
            "split": split,
            "annotations": len(rows),
            "images_written": copied,
            "missing_source": missing,
            "unreadable_source": unreadable,
            "specimens": rows["specimen_id"].nunique(),
        })

        if missing:
            print(f"WARNING: {missing} missing source images in split={split}")
        if unreadable:
            print(f"WARNING: {unreadable} unreadable source images in split={split}")

    if missing_sources:
        pd.DataFrame(missing_sources).to_csv(out / "image_errors.csv", index=False)
        if not args.allow_missing:
            raise RuntimeError(
                f"Found {len(missing_sources)} missing/unreadable source images. "
                "Fix them or rerun with --allow-missing. See image_errors.csv."
            )

    yaml_text = (
        f"path: {out.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        f"  0: {CLASS_NAME}\n"
    )
    (out / "dataset.yaml").write_text(yaml_text, encoding="utf-8")

    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(out / "split_stats.csv", index=False)

    print("\nDataset created")
    print(stats_df.to_string(index=False))
    print(f"Specimens: {len(specimens)}")
    print(f"OK annotations: {len(df)}")
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
