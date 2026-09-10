# landmarks_trainer

Fine-tuning module for the UNet that predicts wing landmark heatmaps.
Sibling to `extraction/`, `landmarks/`, `obb_detector/` under `src/`.

**Current goal of this fine-tuning round**: train the model to predict
Tancrede's full 19-point blueprint (LM3 included), instead of Gabriel's
original 18-point scheme (LM3 dropped for lack of a UNet counterpart --
see `reconstruct_tps.py --drop 3`). `N_LANDMARKS` in `constants.py`
defaults to 19 accordingly. The architecture itself doesn't hardcode a
landmark count anywhere (single-channel heatmap, blobs aren't counted
until decode time), so this is purely a training-data + CLI-arg change,
not an architecture change.

## Pipeline order

0. **`reproject_reference.py`** (nouveau) -- reprojette la référence brute de
   Tancrede (espace image non-croppée, 19 landmarks) vers l'espace crop
   final, en réutilisant la géométrie exacte de
   `extraction/normalize_crop.py` (voir `compute_wing_transform`,
   vérifiée bit-identique sur 150 cas synthétiques). Produit une nouvelle
   référence directement utilisable par `export_dataset.py` (image_id déjà
   renseigné), plus un `review.csv` + des overlays numérotés pour repérer et
   exclure les specimens mal alignés (photos que Tancrede a annotées dans un
   cadrage différent de celui que l'OBB automatique produit aujourd'hui).
   Flux en 2 passes : un premier run produit tout avec `keep` vide, tu
   annotes les mauvais dans review.csv, un second run avec `--exclude-csv`
   les retire de la référence finale.
1. **`migrate_legacy_weights.py`** (one-off) -- convert Gabriel's existing
   pickled `.pth` into `models/unet_landmarks/legacy_baseline/weights.pth`.
   Verified to reproduce bit-identical outputs to the original pickle.
2. **`export_dataset.py`** -- build `train_manifest.csv` (crop path + 19
   ground-truth points per specimen) from Tancrede's reference TPS +
   `crops.csv`, reusing `landmarks.predict.load_target_crops` and
   `core.pipeline_io.resolve_path` rather than re-deriving that logic.
   **Still has one unverified assumption -- read its docstring, and look
   at the debug overlays it generates before trusting the output.**
3. **`train.py`** -- fine-tune (from `legacy_baseline` or from scratch),
   early stopping on val loss, saves a full run under
   `models/unet_landmarks/<run_id>/`.
4. **`evaluate.py`** -- point-localization metrics (mean/median pixel
   error, detection rate) on a held-out manifest, computed through the
   *actual* `landmarks.predict.predict_landmarks_from_path` inference
   path -- not a separate decode implementation -- so the numbers reflect
   real deployment behaviour, plateau-grouping bugfixes included.

## Files

| File | Responsibility |
|---|---|
| `constants.py` | Shared sizes/defaults (image shape, landmark count, heatmap radius/power) |
| `model.py` | UNet architecture + `load_weights()`. New owner of the model class. |
| `heatmap.py` | Ground-truth heatmap encoding only -- decode lives in `landmarks/predict.py` |
| `augment.py` | Joint image/point zoom-shift + quality augmentation |
| `dataset.py` | Manifest CSV <-> PyTorch `Dataset`, train/val/test split |
| `checkpoint.py` | `models/<family>/<run_id>/` artifact writer, shared by train + migration |
| `reproject_reference.py` | Raw-space Tancrede TPS -> crop-space reference + review workflow |
| `export_dataset.py` | TPS + crops.csv -> training manifest |
| `train.py` | Fine-tuning CLI |
| `evaluate.py` | Localization evaluation CLI (via `landmarks.predict`) |
| `migrate_legacy_weights.py` | One-off pickle -> state_dict conversion |

All CLIs (`export_dataset.py`, `train.py`, `evaluate.py`,
`migrate_legacy_weights.py`) insert `src/` onto `sys.path` the same way
`landmarks/predict.py` already does, so they're meant to be run from
inside `src/landmarks_trainer/`, exactly like `predict.py` is run from
inside `src/landmarks/`.

## Integration with `landmarks/predict.py` -- done

The provided `predict.py` has been patched (included in this delivery)
to load a state_dict instead of the full pickled object:

```python
from landmarks_trainer.model import load_weights

model = load_weights(args.model, device=device)
model.eval()
```

`--n-landmarks` default changed from 18 to 19 (Tancrede's full blueprint) --
`--n-landmarks 18` still works for running an older model. Everything else
in `predict.py` is untouched: `extract_top_landmarks` was already fully
generic in landmark count, no changes needed there.

Once every `.pth` still in the old pickle format has been migrated,
`landmarks/UNet_class_and_functions.py` can be deleted -- `model.py` is now
the single owner of the architecture.

## Coordinate convention (read before touching landmark arrays)

- **TPS files, manifest CSVs, `export_dataset.py`, `landmarks/predict.py`'s
  public functions**: `(x, y)`, matching the rest of the pipeline.
- **`heatmap.py`, `augment.py`, and inside `dataset.py`'s `__getitem__`**:
  `(row, col)` i.e. `(y, x)`, matching numpy/opencv image indexing.

The conversion happens exactly once, in `LandmarkHeatmapDataset.__getitem__`.
Don't let `(x, y)` leak into `heatmap.py`/`augment.py`, or vice versa --
that mixup is what made Gabriel's original coordinate handling hard to
follow.

## Downstream ripple, not handled here

Moving the UNet from 18 to 19 landmarks means `build_reference.py`'s frozen
GPA consensus and `renumber.py`'s blueprint will need rebuilding against
Tancrede's full 19-point reference once a 19-landmark model exists --
arguably the point of this change, since it removes the drop/reconcile
step entirely rather than just working around it. Out of scope here (not
provided), just flagging it so it's not a surprise.

## Not done here (out of scope for this pass)

- `obb_detector` was **not** touched or renamed. Aligning it to these same
  `models/<family>/<run_id>/` conventions is a separate pass.

## Testing notes

Everything except `export_dataset.py`/`evaluate.py`'s exact TPS/crops.csv
column names (see their docstrings) was validated end-to-end: migrated
legacy weights reproduce the original pickle's output bit-for-bit,
`train.py` runs (from scratch and from migrated weights) and writes valid
run artifacts, and `evaluate.py` was exercised against the real, unmodified
`predict.py` functions (`load_target_crops`, `predict_landmarks_from_path`)
using synthetic fixtures standing in for `core.tps_io`/`core.pipeline_io`.
