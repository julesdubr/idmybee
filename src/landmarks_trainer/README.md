# landmarks_trainer

Fine-tuning module for the UNet that predicts wing landmark heatmaps
(18 unordered points per crop). Sibling to `extraction/`, `landmarks/`,
`obb_detector/` under `src/`.

## Pipeline order

1. **`migrate_legacy_weights.py`** (one-off) -- convert Gabriel's existing
   pickled `.pth` into `data/models/unet_landmarks/legacy_baseline/weights.pth`.
   Run this once; verified to reproduce bit-identical outputs to the original.
2. **`export_dataset.py`** -- build `train_manifest.csv` (crop path + 18
   ground-truth points per specimen) from Tancrede's reference TPS +
   `crops.csv`. **Has unverified assumptions about repo internals -- read
   its docstring, and look at the debug overlays it generates before
   trusting the output.**
3. **`train.py`** -- fine-tune (from `legacy_baseline` or from scratch),
   early stopping on val loss, saves a full run under
   `data/models/unet_landmarks/<run_id>/`.
4. **`evaluate.py`** -- raw point-localization metrics (mean/median pixel
   error, detection rate) on a held-out manifest, e.g. the `test_manifest.csv`
   a training run sets aside automatically.

## Files

| File | Responsibility |
|---|---|
| `constants.py` | Shared sizes/defaults (image shape, landmark count, heatmap radius/power) |
| `model.py` | UNet architecture + `load_weights()`. New owner of the model class. |
| `heatmap.py` | Ground-truth heatmap encoding + peak decoding |
| `augment.py` | Joint image/point zoom-shift + quality augmentation |
| `dataset.py` | Manifest CSV <-> PyTorch `Dataset`, train/val/test split |
| `checkpoint.py` | `data/models/<family>/<run_id>/` artifact writer, shared by train + migration |
| `export_dataset.py` | TPS + crops.csv -> training manifest |
| `train.py` | Fine-tuning CLI |
| `evaluate.py` | Localization evaluation CLI |
| `migrate_legacy_weights.py` | One-off pickle -> state_dict conversion |

## Integration with `landmarks/predict.py`

`predict.py` should import the model class from here instead of carrying
its own copy, and load state_dict instead of the full pickled object:

```python
from landmarks_trainer.model import load_weights

model = load_weights("data/models/unet_landmarks/<run_id>/weights.pth", device=device)
```

Once every `.pth` currently loaded by `predict.py` has been migrated,
`landmarks/UNet_class_and_functions.py` can be deleted -- this module's
`model.py` is the new single owner of the architecture.

## Coordinate convention (read before touching landmark arrays)

- **TPS files, manifest CSVs, `export_dataset.py`**: `(x, y)`, matching the
  rest of the pipeline.
- **`heatmap.py`, `augment.py`, and inside `dataset.py`'s `__getitem__`**:
  `(row, col)` i.e. `(y, x)`, matching numpy/opencv image indexing.

The conversion happens exactly once, in `LandmarkHeatmapDataset.__getitem__`.
Don't let `(x, y)` leak into `heatmap.py`/`augment.py`, or vice versa --
that mixup is what made Gabriel's original coordinate handling hard to
follow.

## Not done here (out of scope for this pass)

- `obb_detector` was **not** touched or renamed. Aligning it to these same
  `data/models/<family>/<run_id>/` conventions is a separate pass.
- `landmarks/predict.py` was **not** edited (not provided). See the
  integration snippet above.
