"""
One-off migration: convert an old full-pickle .pth (torch.save(model)) into
the new state_dict format, saved as a proper data/models/unet_landmarks/
run so it shows up next to real fine-tuning runs (e.g. as the --init-weights
baseline for the first real train.py run).

    python migrate_legacy_weights.py \
        --old-pth landmarks/best_model/UNet_150_epoch_lr=0.001_seed=58_func=pow_param=30.pth \
        --old-module-dir landmarks \
        --run-id legacy_baseline

--old-module-dir must still contain UNet_class_and_functions.py at
migration time (it's added to sys.path so the pickle can find the class it
was saved with). Once every legacy .pth has been migrated, that file can be
deleted from landmarks/ -- this module (model.py) is the new owner.

Verified on the existing best_model/*.pth: migrated weights reproduce
bit-identical outputs to the original pickled model.
"""

import argparse
import re
import sys
from pathlib import Path

import torch

from checkpoint import make_run_dir, save_run, append_runs_summary
from constants import MODEL_FAMILY


def parse_hparams_from_filename(name: str) -> dict:
    """Best-effort extraction of hyperparameters from Gabriel's filename
    convention, e.g. 'UNet_150_epoch_lr=0.001_seed=58_func=pow_param=30'.
    Missing pieces are left out rather than guessed."""
    hparams = {}
    patterns = {
        "epochs": r"(\d+)_epoch",
        "lr": r"lr=([\d.]+)",
        "seed": r"seed=(\d+)",
        "heatmap_function": r"func=([a-zA-Z]+)",  # letters only: stops before the next "_key=" segment
        "heatmap_power": r"param=(\d+)",
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, name)
        if m:
            value = m.group(1)
            hparams[key] = float(value) if key == "lr" else (
                value if key == "heatmap_function" else int(value)
            )
    return hparams


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--old-pth", required=True)
    parser.add_argument("--old-module-dir", required=True,
                         help="Directory still containing UNet_class_and_functions.py")
    parser.add_argument("--output-root", default="data/models")
    parser.add_argument("--family", default=MODEL_FAMILY)
    parser.add_argument("--run-id", default="legacy_baseline")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(args.old_module_dir).resolve()))
    from landmarks.UNet_class_and_functions import UNet as LegacyUNet  # noqa: F401 -- needed for unpickling

    old_model = torch.load(args.old_pth, weights_only=False, map_location="cpu")
    old_model.eval()

    config = {
        "source": "migrated from legacy full-pickle .pth",
        "original_path": str(Path(args.old_pth).resolve()),
        **parse_hparams_from_filename(Path(args.old_pth).stem),
    }
    metrics_rows = []  # per-epoch history wasn't saved alongside the legacy .pth, nothing to carry over

    run_dir = make_run_dir(args.output_root, args.family, args.run_id)
    save_run(run_dir, old_model, config, metrics_rows)

    append_runs_summary(args.output_root, args.family, {
        "run_id": run_dir.name,
        "timestamp": "",
        "n_train": "",
        "n_val": "",
        "n_test": "",
        "epochs_run": config.get("epochs", ""),
        "best_val_loss": "",
        "init_weights": "",
    })

    print(f"Migrated {args.old_pth} -> {run_dir / 'weights.pth'}")


if __name__ == "__main__":
    main()
