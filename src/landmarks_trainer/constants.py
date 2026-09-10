"""
Shared constants for landmarks_trainer.

Kept in one place so every file (dataset, heatmap, train, evaluate...) agrees
on image size, landmark count and heatmap shape parameters. Override via CLI
flags where it matters (train.py, export_dataset.py) rather than editing
these -- these are just the defaults that match Gabriel's original model.
"""

# UNet input/output spatial size (H, W). Matches Gabriel's original crop
# convention (256 tall x 512 wide). If the current extraction pipeline's
# normalize_crop.py outputs a different size, either resize crops to this
# before export, or override --img-height/--img-width consistently across
# export_dataset.py and train.py (and re-check init weights still make sense).
IMG_HEIGHT = 256
IMG_WIDTH = 512

# Number of landmarks the UNet is trained to localize. Currently 19 --
# Tancrede's full blueprint (LM3 included). Earlier fine-tuning used
# Gabriel's 18-point scheme (LM3 dropped, no UNet counterpart at the time --
# see reconstruct_tps.py --drop 3); this run trains the model to predict
# LM3 too, so downstream steps can use Tancrede's reference directly without
# the drop/reconcile step.
N_LANDMARKS = 19

# Ground-truth heatmap shape: for each landmark point, intensity falls off
# linearly with distance within `radius` pixels, raised to `power` (a sharp
# power curve keeps the bump tight around the point). Values match the best
# legacy model (UNet_150_epoch_..._func=pow_param=30).
DEFAULT_HEATMAP_RADIUS = 60.0 * (2 ** 0.5)  # ~84.85 px
DEFAULT_HEATMAP_POWER = 30

# Model family name under models/<family>/<run_id>/
MODEL_FAMILY = "unet_landmarks"
