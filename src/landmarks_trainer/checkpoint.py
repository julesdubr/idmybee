"""
models/<family>/<run_id>/ artifact writer.

Shared by train.py (real fine-tuning runs) and migrate_legacy_weights.py
(the one-off import of Gabriel's pickled baseline), so both produce runs
that look identical to anything reading models/ later -- same layout
as build_reference.py's references/ artifacts.

Each run directory gets:
    weights.pth       -- model.state_dict(), NOT the full pickled object
    train_config.json -- hyperparameters, dataset/source, seed, timestamp
    metrics.csv       -- one row per epoch: epoch,train_loss,val_loss
                          (single row for a migrated legacy model, marked as such)

A family-level runs.csv (models/<family>/runs.csv) gets one summary
row appended per run, so you can compare runs without opening every folder.
"""

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch


def make_run_dir(output_root: str, family: str, run_id: str = None) -> Path:
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    run_dir = Path(output_root) / family / run_id
    run_dir.mkdir(parents=True, exist_ok=False)  # fail loudly on accidental run_id collision
    return run_dir


def save_run(run_dir: Path, model, config: dict, metrics_rows: list):
    """model: the trained UNet (state_dict is what gets saved, never the object)
    config: dumped as-is to train_config.json (must be JSON-serializable)
    metrics_rows: list of dicts, e.g. [{"epoch": 1, "train_loss": .., "val_loss": ..}, ...]
    """
    torch.save(model.state_dict(), run_dir / "weights.pt")

    with open(run_dir / "train_config.json", "w") as f:
        json.dump(config, f, indent=2)

    if metrics_rows:
        fieldnames = list(metrics_rows[0].keys())
        with open(run_dir / "metrics.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metrics_rows)


def append_runs_summary(output_root: str, family: str, summary_row: dict):
    """Append one row to models/<family>/runs.csv, creating it (with
    header) on the first call. Column set is fixed by the first row written
    -- keep summary_row's keys consistent across callers."""
    runs_csv = Path(output_root) / family / "runs.csv"
    runs_csv.parent.mkdir(parents=True, exist_ok=True)

    file_exists = runs_csv.exists()
    with open(runs_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(summary_row)
