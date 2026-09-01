"""
Fine-tune (or train from scratch) the landmark UNet.

    python train.py \
        --manifest data/models/unet_landmarks/train_manifest.csv \
        --init-weights data/models/unet_landmarks/legacy_baseline/weights.pth \
        --output-root data/models --epochs 150 --patience 20

Produces data/models/<family>/<run_id>/weights.pth, train_config.json,
metrics.csv, and appends a summary row to data/models/<family>/runs.csv.

Fixes two issues present in the original UNet_training.ipynb loop:
- patience_count was read before being initialized (NameError risk on a
  lucky first epoch); it's now initialized once, outside the loop.
- best_avg_val_loss started at a hardcoded 1 (wrong if the first epoch's
  loss happens to exceed 1); it now starts at +inf.
"""

import argparse
import json
import time
from datetime import datetime, timezone

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from landmarks_trainer.constants import (DEFAULT_HEATMAP_RADIUS, DEFAULT_HEATMAP_POWER,
                        IMG_HEIGHT, IMG_WIDTH, N_LANDMARKS, MODEL_FAMILY)
from landmarks_trainer.dataset import load_manifest, split_manifest, LandmarkHeatmapDataset
from landmarks_trainer.model import UNet, load_weights
from landmarks_trainer.checkpoint import make_run_dir, save_run, append_runs_summary


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for images, heatmaps in loader:
        images, heatmaps = images.to(device), heatmaps.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, heatmaps)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    for images, heatmaps in loader:
        images, heatmaps = images.to(device), heatmaps.to(device)
        outputs = model(images)
        running_loss += criterion(outputs, heatmaps).item()
    return running_loss / len(loader)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--init-weights", default=None,
                         help="Existing weights.pth to fine-tune from; omit to train from scratch")
    parser.add_argument("--output-root", default="data/models")
    parser.add_argument("--family", default=MODEL_FAMILY)
    parser.add_argument("--run-id", default=None, help="Default: UTC timestamp")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--group-col", default="specimen_id",
                         help="Column to group-split on so the same specimen never spans "
                              "train/val/test; ignored if absent from the manifest")
    parser.add_argument("--seed", type=int, default=58)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--heatmap-radius", type=float, default=DEFAULT_HEATMAP_RADIUS)
    parser.add_argument("--heatmap-power", type=float, default=DEFAULT_HEATMAP_POWER)
    parser.add_argument("--img-height", type=int, default=IMG_HEIGHT)
    parser.add_argument("--img-width", type=int, default=IMG_WIDTH)
    parser.add_argument("--n-landmarks", type=int, default=N_LANDMARKS)
    args = parser.parse_args()

    device = resolve_device(args.device)
    torch.manual_seed(args.seed)

    full_manifest = load_manifest(args.manifest, n_landmarks=args.n_landmarks)
    train_df, val_df, test_df = split_manifest(
        full_manifest, val_fraction=args.val_fraction, test_fraction=args.test_fraction,
        seed=args.seed, group_col=args.group_col,
    )
    if len(train_df) == 0 or len(val_df) == 0:
        raise ValueError(f"Empty train ({len(train_df)}) or val ({len(val_df)}) split -- "
                          f"manifest too small for the requested val/test fractions.")

    heatmap_kwargs = {"radius": args.heatmap_radius, "power": args.heatmap_power}
    img_shape = (args.img_height, args.img_width)
    train_ds = LandmarkHeatmapDataset(train_df, img_shape=img_shape, train_augment=True,
                                       heatmap_kwargs=heatmap_kwargs)
    val_ds = LandmarkHeatmapDataset(val_df, img_shape=img_shape, train_augment=False,
                                     heatmap_kwargs=heatmap_kwargs)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    if args.init_weights:
        model = load_weights(args.init_weights, device=device)
    else:
        model = UNet(in_channels=3, out_channels=1).to(device)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    best_state_dict = None
    patience_count = 0
    metrics_rows = []

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        metrics_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"No improvement for {args.patience} epochs, stopping early.")
                break

    model.load_state_dict(best_state_dict)
    elapsed = time.time() - start_time

    config = {
        "manifest": args.manifest,
        "init_weights": args.init_weights,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "n_landmarks": args.n_landmarks,
        "img_height": args.img_height,
        "img_width": args.img_width,
        "epochs_requested": args.epochs,
        "epochs_run": len(metrics_rows),
        "lr": args.lr,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "seed": args.seed,
        "heatmap_radius": args.heatmap_radius,
        "heatmap_power": args.heatmap_power,
        "best_val_loss": best_val_loss,
        "elapsed_seconds": elapsed,
        "device": device,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    run_dir = make_run_dir(args.output_root, args.family, args.run_id)
    save_run(run_dir, model, config, metrics_rows)

    test_path = run_dir / "test_manifest.csv"
    test_df.to_csv(test_path, index=False)  # held out for evaluate.py, never touched during training

    append_runs_summary(args.output_root, args.family, {
        "run_id": run_dir.name,
        "timestamp": config["timestamp"],
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "epochs_run": len(metrics_rows),
        "best_val_loss": best_val_loss,
        "init_weights": args.init_weights or "",
    })

    print(f"Saved run -> {run_dir}")


if __name__ == "__main__":
    main()
