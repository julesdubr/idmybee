# PIPELINE -- end-to-end data flow

Reference document: exact inputs/outputs/CLI/status vocabulary for every
stage, from raw photos to a fitted GPA-PCA-LDA classification model. See
`RESUME.md`/`TODO.md` for session history and open questions,
`CONVENTIONS.md` for style rules. Written after the "unify on
`photo_id`/`inv_id`" refactor (session 7 sept. 2026) -- the schema
described here is the CURRENT, single one: no more split between a "clean
dataset" pipeline and a separate, incompatible extraction/classification
pipeline.

Canonical identifiers, used everywhere below:
- `inv_id`: canonical specimen id (`<inv_name>_<num:04d>`), frozen
  once assigned. One specimen may have several photos.
- `photo_id`: canonical per-photo id (`<inv_id>_<device_type>_<n>`),
  globally unique, human-readable.
- `original_id`: the specimen's raw, source-specific identifier (e.g. a
  museum catalog number) -- never used as a join key past the clean
  dataset stage, kept only for traceability back to the source.

## Dataset roots can live anywhere on disk

There is no local-vs-external distinction to manage: a dataset root's
images can sit anywhere (a repo folder, an external volume, a third-party
location) -- `tools/ingestion/export_clean_dataset.py --output-dir` can point
straight at an external mount, and `tools/ingestion/build_manifest.py`'s path column
is used exactly as given (absolute, or resolved against `--base-dir`).
Every downstream stage already reads `manifest.csv`'s `path` via
`core.pipeline_io.resolve_path`, unchanged. There is no copy-then-symlink
step to run and nothing to configure for it. In practice: point
`tools/ingestion/export_clean_dataset.py --output-dir` at the external volume (its
`images/` subfolder is the only thing that scales into GBs), but give
`tools/ingestion/build_manifest.py --output-dir` a local, tracked path under
`data/clean/<source>/` -- `manifest.csv`/`biological_data.csv` are a few
KB-MB and belong in the repo like any other small pipeline output; the
`path` column inside them can still point at the external volume just
fine.

## 1. Clean dataset (`tools/ingestion/ingest_raw.py` -> `tools/ingestion/export_clean_dataset.py`
[optional] -> `tools/ingestion/build_manifest.py` [-> `tools/ingestion/combine_manifests.py`])

Two independent jobs, split across three tools:
- **Cleaning identity** (`ingest_raw.py`/`export_clean_dataset.py`) --
  OPTIONAL, only needed when the raw data itself is messy (a raw label
  reused across two different specimens, multiple filename conventions,
  ...). Produces a clean, canonically-named copy plus a per-photo
  `dataset.csv`.
- **Building the manifest** (`build_manifest.py`) -- ALWAYS run, the sole
  entry point into stage 2 onward. Takes ANY per-photo CSV (cleaned by
  `export_clean_dataset.py` or not -- a third-party dataset with its own
  filenames and its own spreadsheet works too) that meets a simple
  standard, and produces `manifest.csv`/`biological_data.csv`. It never
  copies files or resolves identity conflicts -- purely structural
  validation. If the input doesn't meet the standard, it writes
  `manifest_raw.csv` instead (same shape, `status`/`status_reason`
  explaining what's wrong) rather than guessing.

A dataset that's already compliant (canonical or not, as long as it has
images + a per-photo CSV with the required columns) can skip
`ingest_raw`/`export_clean_dataset` entirely and go straight to
`build_manifest.py`.

### 1a. `tools/ingestion/ingest_raw.py` -> `tools/ingestion/export_clean_dataset.py` (optional cleaning)

**Input**: a local raw image root (`config/roots.json`) for `ingest_raw.py`;
its raw manifest.csv + a raw identification CSV for `export_clean_dataset.py`.

**Output**, one run per source (collection, terrain, ...), under
`--output-dir` (e.g. `data/clean/collection/`):
- `dataset.csv` -- one row per successfully copied **photo**: `photo_id,
  inv_id, device_type, device, photo_index, [photographer], path` +
  biological columns merged in directly (genus/species/caste/etc.,
  `original_id` if `--origin-codes` was given). `path` is the freshly
  copied, renamed image under `<output-dir>/images/...` -- independent of
  wherever the raw file lived. This is the compliant per-photo CSV
  `tools/ingestion/build_manifest.py` expects next, with zero extra steps. `device`
  is the specific camera/phone used for that photo (may be NaN if
  unresolved) -- distinct from `device_type`, a coarse code (`P`/`S`) that
  can cover several physical devices over time.
- `biological_data_all.csv` -- one row per **specimen** (`inv_id`),
  including those with no photo at all (no manifest equivalent downstream
  -- useful on its own, e.g. for a collaborator who wants the full
  specimen list regardless of what's been digitized).
- `device_types.csv` -- dataset-wide `device_type`<->`device` name pairs
  (reference only, not per-photo -- see `dataset.csv`'s own `device`
  column for that).
- `reports/identification_conflicts.csv`, `reports/excluded_identification_rows.csv`,
  `reports/missing_biological_data.csv`, `reports/copy_failures.csv` --
  diagnostics, one row per specimen (not per device) for the first three,
  see `manifest/identification.py`.

**Key CLI args**: `--key-column` (raw CSV's specimen key), `--device-column`/
`--device-name-column`, `--origin-codes` (collection-style) or omitted
(terrain-style, embedded numbering), `--mapping-file` (frozen, append-only
`original_id -> inv_id` mapping), `--image-group-by`.

**Status vocabulary** (internal to this stage, not carried into
`dataset.csv`): `OK`/`FAILED` (copy outcome), `SKIPPED` (`--no-copy-images`:
no clean copy written -- excluded from `dataset.csv`, which needs a valid
`path`).

### 1b. `tools/ingestion/build_manifest.py` (always run)

**Input**: one per-photo CSV (`--path-column`, default `path`), mandatory
columns `inv_id`/`species`/`caste` + the path column. Any other column is
carried into `biological_data.csv`; recognized photo-level columns
(`photo_id`, `device_type`, `device`, `photo_index`, `photographer`,
`source_type`) stay in `manifest.csv` instead, derived automatically where
absent.

**Output**: `manifest.csv` (same schema as before: `photo_id, inv_id,
source_type, device_type, device, photo_index, photographer, ext,
content_hash, file_size_bytes, path, status, status_reason`) +
`biological_data.csv` (`inv_id` + biological columns +
`n_photos_<device_type>`) if every row validates OK. Otherwise
`manifest_raw.csv` only (same shape) + `reports/biological_inconsistencies.csv`
if the failure was a biological-data conflict.

**Status vocabulary**: `OK`, `FAILED` (missing/unreadable image file,
duplicate `photo_id`), `SUSPECT` (biological columns disagree across rows
sharing an `inv_id` -- reported, not arbitrated).

**Key CLI args**: `--path-column`, `--base-dir` (resolve relative paths --
default: the input CSV's own parent directory), `--default-device-type`.

Several sources for one dataset: run `build_manifest.py` once per source,
then `tools/ingestion/combine_manifests.py` concatenates their `manifest.csv`/
`biological_data.csv` (no rescan, no rejoining -- each source is already
validated and independent).

**Not part of this stage, deliberately, nor of any other stage anymore**
(session 7 sept. 2026, suite 5 -- `--split`/`--splits-csv` removed
pipeline-wide): train/test split assignment. Each clean dataset root
(collection, terrain, ...) is entirely used for whatever it's loaded for --
see "Orchestrator scripts" below.

## 2. Wing detection (`extraction/detect_wing.py`)

**Input**: `<dataset>/manifest.csv` (`photo_id, inv_id, path, status`,
stage 1's schema directly -- `path` is read as-is, no separate "raw"
location). Always processes the WHOLE dataset root -- there is no
train/test split within one dataset (see "Known gaps": each clean dataset
root, e.g. collection vs terrain, is itself the unit of separation now).

**Output**: `<dataset>/extraction/<mode>/detection.csv` -- `photo_id,
inv_id, status, error_reason, confidence, n_detections, x1..y4
(normalized 0..1 OBB corners), processing_time_s, processed_at`.
`<dataset>/pipeline_stats.csv` (step `detection`).

**Key CLI args**: `--mode {heavy,light}` (backend), `--imgsz`, `--conf`,
`--device`, `--base-dir` (root to resolve `manifest.csv`'s `path` if
relative).

**Status**: row starts `FAILED`, overwritten by the backend's result.
`error_reason=unreadable_image_or_unsupported_format` if the image can't
even be opened. No `SUSPECT`/`SKIPPED` here.

## 3. Crop normalization (`extraction/normalize_crop.py`)

**Input**: `<dataset>/manifest.csv` + stage 2's `detection.csv` (only
`status=="OK"` rows are normalized).

**Output**: crop images at `<dataset>/extraction/<mode>/images/<photo_id>.jpg`
(grayscale, `out_width`x`out_height`, default 512x256) +
`<dataset>/extraction/<mode>/crops.csv` (`photo_id, inv_id, status,
error_reason, aspect_ratio, output_path, processing_time_s, processed_at`).

**Key CLI args**: `--padding` (default 0.10), `--out-width`/`--out-height`
(default 512/256), `--overwrite`.

**Geometric contract** (pure math, reused by stage 7 for reprojection):
`rotate_image()` aligns the wing horizontally (OBB long edge), `crop_with_context()`
adds padding then extends the short axis to the target ratio using real
pixels (no letterbox). `compute_wing_transform()` recomputes the exact
same geometry from just the OBB + image dimensions (no pixel decode) into
a `WingTransform`; `apply_wing_transform_to_points[_inverse]()` map points
between raw-image space and crop-pixel space, exactly.

**Status**: `SKIPPED` here means "file already on disk from a prior run"
(not `--overwrite`) -- **the opposite sense of stage 1's `SKIPPED`**
("no copy written at all"). Don't conflate the two. `OK`/`FAILED`
otherwise (`write_failed` if `cv2.imwrite` itself fails).

## 4. Landmark prediction (`landmarks/predict.py`)

**Input**: stage 3's `crops.csv` (`status in ("OK","SKIPPED")` kept) + a
trained UNet `.pth`.

**Output**: `<dataset>/landmarks/<tps>` (default `landmarks.tps`, crop-pixel
space, `COMMENT=photo_id=...;inv_id=...` on every entry -- **required**,
the sole join mechanism, no more hash-derived `ID=`) + `<dataset>/landmarks/landmarks.csv`
(`photo_id, inv_id, status, error_reason, n_landmarks_found, model_name,
processing_time_s, processed_at`).

**Key CLI args**: `--model` (required), `--n-landmarks` (default 19,
Tancrede's full blueprint), `--overwrite`, `--retry-failed`.

**Status**: `OK` (all `n_landmarks` points found), `SUSPECT` (fewer than
expected but >0 -- will automatically become `FAILED` at stage 5, since
its point count won't match the reference), `FAILED` (no peak found, or
unreadable crop).

**Checkpointing**: resumable via an in-memory `{photo_id: ImageLandmarks}`
dict reloaded from any existing TPS/CSV at startup; `ID=` (`tps_id`) is
just a TPS-format requirement, reassigned sequentially
(`core.tps_io.assign_sequential_ids`) at every checkpoint write -- it
carries no identity across runs, `photo_id` does.

## 5. Renumbering (`landmarks/build_reference.py` once, then `landmarks/renumber.py`)

**`build_reference.py`** (run once): a ground-truth ordered TPS ->
GPA consensus shape, frozen to a `.npz` (`--drop <n>` to exclude a
landmark the detector doesn't predict, e.g. LM3).

**`renumber.py`**:
**Input**: stage 4's TPS + the frozen reference `.npz` + optionally
`<dataset>/biological_data.csv` (`--biological-data`, enables the
per-species outlier diagnostic; "labeled" = non-empty `species`, there is
no `is_labeled` column in this schema).

**Output**: `<tps stem>_numbered.tps` (still crop-space, only `OK`+`SUSPECT`
specimens, landmarks reordered -- `photo_id`/`inv_id` carried through
unchanged) + `landmarks_numbered.csv` (`photo_id, inv_id, tps_id,
image_path, status, registration_cost, n_outlier_landmarks, error_reason,
processing_time_s` -- every INPUT specimen, including `FAILED` ones).

**Status**: `FAILED` (landmark count incompatible with the reference --
every stage-4 `SUSPECT` ends up here), `SUSPECT` (numbered but a
per-species post-GPA outlier, needs `--biological-data`), `OK`.

## 6. Final export (`tools/pipeline/export_final_landmarks.py`)

The terminal, R-facing export: reads a dataset already through stages 1-5,
produces a self-contained TPS+CSV package.

**Input**: `core.dataset.load_dataset()` (stage 7's join contract) +
stage 2's `detection.csv` (for reprojection) + `<dataset>/manifest.csv`.
Exports the WHOLE dataset root given in one call -- no train/test split,
run it once per dataset root (e.g. once for `data/Bombus/collection`, once
for `data/Bombus/terrain`).

**Output**, in `--output-dir` (default: `<dataset>/export/` -- see
`core.pipeline_io.dataset_export_dir` -- kept alongside `extraction/` and
`landmarks/` under the dataset root rather than an ad hoc path the caller
has to invent each time):
- `landmarks_<n>lm_crop.tps` -- crop-space, sequential `ID=1,2,...`,
  no `COMMENT=` (unlike the intermediate TPS files -- this is the final,
  R-facing export, row-order-aligned to the biological CSV instead).
- `landmarks_<n>lm_original.tps` -- raw-image-space, via the same
  `WingTransform`/`apply_wing_transform_to_points_inverse` machinery as
  stage 3 (omitted with `--no-original-space`).
- `biological_data.csv` -- `id, inv_id, species, caste, device, device_tag`,
  same row order/count as both TPS files.
- `failed.csv` -- `photo_id, inv_id, stage, reason` (stages:
  `landmark_placement`, `outlier_registration`, `biological_metadata`,
  `other`, `reprojection`).

**Key CLI args**: `--tps` (19- vs 18-landmark variant),
`--devices`/`--species`/`--castes`/`--include-outliers`, `--no-original-space`,
`--mode`, `--padding`/`--out-width`/`--out-height` (must match stage 3's).

**Reprojection**: reads the SAME image (`manifest.csv`'s `path`) stage 2
already used to run detection -- no separate "raw" file, no "is the
external drive mounted" dependency anymore. Aborts after 5 consecutive
unreadable images (likely a misconfigured `--base-dir`), otherwise logs
each failure to `failed.csv` and continues.

## 7. Classification fit (`classifiers/train.py`, via `utils/dataset.py`)

**`utils/dataset.py::load_dataset()`** -- the one join contract every
downstream tool shares (stages 6, 8, 9, plus `train.py` itself). Requires
under `root`: `biological_data.csv` (`inv_id, species, caste, ...`),
`manifest.csv` (`photo_id, inv_id, device_type, device, photo_index,
path, status`), `landmarks/landmarks_numbered.{tps,csv}` (overridable via
`--tps`/`--landmarks-status-csv`). Join: TPS `COMMENT=` (`photo_id`/`inv_id`)
is required -- a TPS without it can't be joined. One output row = one
photo. There is no train/test split filter anymore: each dataset root
(collection, terrain, ...) is entirely used for whatever it's loaded for --
typically collection to fit a model (`train.py`), terrain to evaluate one
(`predict.py batch`), never a split of the SAME dataset.

**`train.py`**: **Input**: `load_dataset(root)` on the WHOLE dataset root
(no split filter -- see `--devices`/`--species`/`--castes` to restrict by
something else). **Algorithm**: `core.gpa.gpagen()` (2D Procrustes, no
reflection) on all loaded specimens' landmarks -> `two_d_array()` -> PCA
(`n_components = 2*n_points-4`) -> LDA (`n_components` capped by
`--lda-components`, default 2, for the saved projection only). Accuracy
estimated by a SEPARATE LOOCV LDA (per-photo, not per-specimen -- an
individual with several photos stays partly in training when one photo is
held out, inflating accuracy slightly).
**Output**, under `data/models/lda/<run_id>/train/` (`run_id` now built
from the dataset root's own name, e.g. `species_collection`, instead of a
split -- see `core.run_io.build_run_id`): `model.joblib`
(a `core.model_io.TrainedModel`: `mean_shape, n_points, pca, lda, level,
classes, dataset_label, devices, source_tps, n_train`),
`loocv_predictions.csv`, `metrics.json`, `params.json`, `run.log`.

**Key CLI args**: `--level {species,caste}`, `--lda-components`,
`--no-save-model`, plus the shared dataset args above.

## 8. Prediction (`classifiers/predict.py`)

**`batch`** (known truth, evaluation): same `load_dataset()` filters as
`train.py`, inherits `--tps`/`--landmarks-status-csv` from the model's own
training run's `params.json` unless overridden. Normally run on a
DIFFERENT dataset root than the one used to train the model (e.g. a model
trained on `data/Bombus/collection`, evaluated on `data/Bombus/terrain`) --
`eval_tag` is keyed off that dataset root's own name (see
`core.run_io.build_eval_tag`), not a split, so evaluating the same model
against two different datasets never collides.
**Output**: `data/models/<family>/<run_id>/predict/<eval_tag>/predictions.csv`
(`tps_id, photo_id, inv_id, image_path, predicted_<level>, confidence,
second_choice, second_confidence, third_choice, third_confidence,
procrustes_distance, true_<level>, correct_top1, correct_top3`) +
`metrics.json`/`params.json`/`run.log`.

**`single`** (field use, no truth): one TPS with exactly one specimen, NO
biological join at all (`core.dataset.load_unlabeled_tps`) -- pure
geometry-in, prediction-out. Console output only, unless `--out` given.
This is the function the future single-image UI tool (see `TODO.md` Phase
3) will call directly.

Every prediction reuses the SAVED `pca`/`lda` (never refit) and a
single-pass alignment onto `model.mean_shape` (`core.gpa.align_to_reference`,
not a fresh GPA consensus) -- plus a `procrustes_distance` to that
reference, flagging atypical shapes/landmark problems.

## 9. Variance analysis (`analysis/variance_report.py`)

Independent of any saved model -- fits its own fresh GPA on the filtered
specimens. **Input**: `load_dataset()` (same shared filters) + `--levels`
(nested, broadest to finest, from `species, caste, inv_id, device,
device_tag`) + `--n-perm` (permutation p-values) + `--balanced-devices`
(keep only specimens with every requested device_tag, via
`core.dataset.restrict_to_complete_devices` -- avoids biasing the device
effect estimate by uneven photo coverage per individual).
**Output**, under `data/analysis/variance/<variance_id>/`: `anova.csv`
(`SS, df, MS, F, p (permutation)` per level + `Residual`), `anova.png`
(MS per level, log scale), `params.json`, `run.log`.

Interpretation worked example (from the module's own docstring):
`MS(inv_id)` = biological floor (variance between individuals of the same
species/caste); `MS(device) > MS(inv_id)` -> measurement noise exceeds
real biological variation, questioning the pipeline's ability to
discriminate below that threshold.

## Cross-stage data flow (exact chaining)

```
[tools/ingestion/ingest_raw.py --> tools/ingestion/export_clean_dataset.py, optional]
  dataset.csv (photo_id, inv_id, device_type, device, path, species, caste, ...)
   |
   v  tools/ingestion/build_manifest.py [--> tools/ingestion/combine_manifests.py]
manifest.csv (photo_id, inv_id, device_type, device, path, status)
biological_data.csv (inv_id, species, caste, n_photos_<device_type>, ...)
   |
   v  extraction/detect_wing.py --mode {heavy,light}
extraction/<mode>/detection.csv (photo_id, inv_id, status, x1..y4 normalized OBB)
   |
   v  extraction/normalize_crop.py --mode {heavy,light}
extraction/<mode>/images/<photo_id>.jpg
extraction/<mode>/crops.csv (photo_id, inv_id, status, output_path)
   |
   v  landmarks/predict.py --mode {heavy,light} --model <unet.pth>
landmarks/<tps>            (crop-space, COMMENT=photo_id=...;inv_id=...)
landmarks/landmarks.csv    (photo_id, inv_id, status, ...)
   |
   v  landmarks/renumber.py --tps <tps> --reference <ref.npz> [--biological-data biological_data.csv]
   |      (reference built once: landmarks/build_reference.py --ref <ground-truth.tps> --drop <n> --out <ref.npz>)
landmarks/<tps stem>_numbered.tps   (crop-space, reordered, OK+SUSPECT only)
landmarks/landmarks_numbered.csv    (ALL input specimens incl. FAILED)
   |
   +--> tools/pipeline/export_final_landmarks.py --tps <numbered.tps> --output-dir <dir> [--mode] [--base-dir]
   |      landmarks_<n>lm_crop.tps / _original.tps
   |      biological_data.csv / failed.csv
   |
   +--> classifiers/train.py <dataset> --level {species,caste}
          data/models/lda/<run_id>/train/model.joblib   (run_id includes the dataset root's name)
           |
           +--> classifiers/predict.py batch <model.joblib> <another dataset>
           |      data/models/lda/<run_id>/predict/<eval_tag>/predictions.csv  (eval_tag = that dataset's name)
           +--> classifiers/predict.py single <model.joblib> <field_photo.tps>
           |      console top-N species/caste + confidence
           +--> analysis/variance_report.py <dataset> --levels species caste inv_id device
                  data/analysis/variance/<variance_id>/anova.csv
```

The two steps `detect_wing.py -> normalize_crop.py -> predict.py ->
renumber.py` (stages 2-5) followed by the export (stage 6) and either
`train.py` or `predict.py batch` are each chained end to end by a single,
DATASET-AGNOSTIC orchestrator script -- see "Orchestrator scripts" below.

## Orchestrator scripts (session 7 sept. 2026, suite 5/6; split + validation hooks added session 8 sept. 2026)

Two `tools/pipeline/` scripts chain stages 2-6 plus the classification
step. Both take the dataset root as a REQUIRED positional argument with no
default -- nothing in either script or in the shared helper they call
(`utils/landmarking_pipeline.py`) is specific to "collection" or "terrain";
any clean dataset root (`manifest.csv` + `biological_data.csv` + images,
stage 1's output) works. `data/Bombus/collection`/`data/Bombus/terrain`
below are just the two concrete datasets this project happens to have --
not special-cased anywhere in the code.

**`utils/landmarking_pipeline.py`** -- shared, not duplicated between the
two scripts below, or with `app/build_dataset.py` (see "Validation review"
below):
- `add_landmarking_args(parser)`: every flag for stages 2-6 (detection,
  crop, landmark placement, renumbering, export) in one place.
- `run_detection_and_crop(args)`: stages 1-2 (`extraction.detect_wing` ->
  `extraction.normalize_crop`).
- `run_landmark_placement(args)`: stages 3-4 (`landmarks.predict` ->
  `landmarks.renumber`). Honors `args.crops_csv` if set (an override
  written by a validation review, see below) as `landmarks.predict
  --crops-csv`, skipping a human-rejected crop instead of the default
  `extraction/<mode>/crops.csv`.
- `run_landmarking(args)`: `run_detection_and_crop` then
  `run_landmark_placement` in one call -- what `train_dataset.py`/
  `predict_dataset.py` use; a validation-gated caller (`app/build_dataset.py`)
  calls the two halves separately, with a review step of its own in between.
- `run_export(args)`: stage 6 (`tools.pipeline.export_final_landmarks`),
  writing to `<dataset>/export/` by default (`--export-dir` to override).
  Forwards `--devices`/`--species`/`--castes`/`--include-outliers`/`--tps`/
  `--landmarks-status-csv` (see `dataset_filter_argv`) -- so a landmark
  review override (`args.landmarks_status_csv` pointing at a
  `landmarks_reviewed.csv`) is reflected in the exported package the same
  way it already was in the fitted model, rather than the export silently
  using the unreviewed statuses (a real gap before session 8 sept. 2026,
  fixed alongside the review mechanism itself).
- `dataset_filter_argv(args)`: the same seven dataset-filter flags as CLI
  argv, for a caller that invokes another script's `main(argv)` rather than
  passing a `Namespace` through directly -- used by `run_export` and
  `tools/pipeline/train_dataset.py` (previously duplicated in both).

- **`tools/pipeline/train_dataset.py`**: `run_landmarking` -> `run_export` ->
  `classifiers.train`, producing a `model.joblib` fitted on the given
  dataset (GPA mean shape + PCA + LDA).
  ```
  python -m tools.pipeline.train_dataset data/Bombus/collection \
      --unet-model data/models/unet_landmarks/2026-08-29_131929/weights.pt \
      --model-name "Identification bourdons (collection)"
  ```
- **`tools/pipeline/predict_dataset.py`**: `run_landmarking` -> `run_export` ->
  `classifiers.predict batch` against an already-trained `--model`
  (typically one produced by `train_dataset.py` on a DIFFERENT dataset).
  ```
  python -m tools.pipeline.predict_dataset data/Bombus/terrain \
      --model data/models/lda/species_collection/train/model.joblib \
      --unet-model data/models/unet_landmarks/2026-08-29_131929/weights.pt
  ```

Both require `--unet-model` (landmark placement weights, no default -- two
runs currently coexist, `2026-08-29_131929` and `legacy_baseline`, an
explicit choice is deliberate) and default `--detector-model` to
`data/models/yolon_obb/best.pt` (the light-mode YOLO-OBB detector already
trained in this project) and `--reference` to
`data/references/shapes/reference_shape_<n-landmarks>.npz` (the frozen GPA
reference already built from Tancrede's ground-truth blueprint -- see
stage 5 -- no need to rerun `landmarks/build_reference.py`). Each of the
first three stages they call is independently resumable (`--overwrite`/
`--retry-failed` passed through where each stage actually supports them);
rerunning either script after a partial run only reprocesses what's
missing.

Neither script does true GPU-batch parallelism beyond what each stage
already does on its own -- they are sequencing convenience, not a new
execution engine.

**Run end to end on real data (session 7 sept. 2026, suite 6)**:
`tools/pipeline/train_dataset.py data/Bombus/collection` (2600 specimens, 19
landmarks) -> LOOCV top-1 = 94.77%, top-3 = 98.73%. `tools/pipeline/predict_dataset.py
data/Bombus/terrain` with that model -> top-1 = 82.26%, top-3 = 96.98% on
the 265 terrain specimens with known truth. First real confirmation the
whole chain works end to end past stage 1, not just on synthetic fixtures.

## Stage 0 orchestrator (session 8 sept. 2026)

`tools/ingestion/prepare_dataset.py` closes the previously-flagged "no
orchestrator covers stage 1" gap: driven by a single JSON config (one
block per source, since stage 1 genuinely needs per-source identification
CSV/columns -- see the module's own docstring for the exact schema), it
chains, in-process, exactly the tools already documented in stage 1 above:

```
[config.ingest, optional]  tools.ingestion.ingest_raw           (once)
[per source, "raw" type]   tools.ingestion.export_clean_dataset (identity resolution)
[per source, always]       tools.ingestion.build_manifest       (structural validation)
[once, over N sources]     tools.ingestion.combine_manifests    (N=1 works too)
```

A source can be `"raw"` (needs `export_clean_dataset`, e.g. a museum
collection with messy identifiers) or `"compliant"` (already a per-photo
CSV meeting stage 1b's standard -- straight to `build_manifest`). A
source whose `build_manifest` run only produces `manifest_raw.csv` (not
`manifest.csv`) is reported and excluded from the final combine, rather
than aborting the whole run for one bad source. `--skip-ingest` reuses an
already-scanned raw manifest instead of a full rescan.

```
python -m tools.ingestion.prepare_dataset config/prepare_ma_source.json
```

## Validation review (session 8 sept. 2026)

A human checkpoint between stages 1-2 (cropping) and 3-4 (landmark
placement): review the auto OK/SUSPECT/FAILED status per photo, correct it
if needed, and have the correction flow through to everything downstream
with no other code change -- see `utils/review.py` for the shared
implementation (used identically by the CLI pair below and by
`app/build_dataset.py`'s two validation steps).

Mechanism: a review is a DataFrame (`photo_id, inv_id, auto_status,
reviewed_status, ...`) built from a stage's own status log
(`build_crop_review_df`/`build_landmark_review_df`, reading
`extraction/<mode>/crops.csv`/`landmarks/landmarks_numbered.csv`).
Persisting it (`write_crop_review`/`write_landmarks_review`) never mutates
the original log (kept intact for audit) -- it writes a schema-compatible
override file the rest of the pipeline already knows how to consume:
- `extraction/<mode>/crops_reviewed.csv` -> `landmarks.predict --crops-csv`
  (a human-rejected crop is skipped before spending UNet compute on it,
  and everything downstream of it).
- `landmarks/landmarks_reviewed.csv` -> `--landmarks-status-csv` (see
  `utils.cli.add_dataset_args`), consumed by `run_export`/`classifiers.train`/
  `classifiers.predict batch` exactly as `landmarks_numbered.csv` would be.

Both also get an audit trail at `<dataset>/review/{crops,landmarks}_review.csv`
(`photo_id, auto_status, reviewed_status`).

CLI counterpart, for parity without the UI (`tools/pipeline/`):
```
python -m tools.pipeline.export_review data/Bombus/collection --overlays
#   -> <dataset>/review/{crops,landmarks}_review.csv (reviewed_status column
#      ready for hand-editing in a spreadsheet), plus numbered-landmark
#      overlays sorted by status if --overlays.
# ... edit reviewed_status by hand ...
python -m tools.pipeline.reconcile_review data/Bombus/collection
#   -> re-applies the edited review CSV(s), producing crops_reviewed.csv/
#      landmarks_reviewed.csv exactly as write_crop_review/write_landmarks_review would.
```

`app/build_dataset.py` (see also README.md "Scenario 1") does the same
thing interactively: a filterable, searchable, editable table
(`st.data_editor`, `reviewed_status` column) plus a preview of the crop or
the numbered-landmark overlay for the selected photo, "Save & continue"
calling `write_crop_review`/`write_landmarks_review` and setting
`args.crops_csv`/`args.landmarks_status_csv` on the shared pipeline
`Namespace` before the next stage runs.

## Model naming (session 8 sept. 2026)

`classifiers/train.py --model-name "..."` (also exposed by
`tools/pipeline/train_dataset.py` and `app/build_dataset.py`'s training
step) attaches a human-facing name to a model, stored as
`core.model_io.TrainedModel.model_name` and in the run's `metrics.json`.
Purely descriptive: it never affects `model.joblib`'s path, still derived
reproducibly from level/dataset_label/devices/landmarks-source (see
`core/run_io.py::build_run_id`) -- omit it and the run_id is used as
before. `core.run_io.model_display_name(model_path)` resolves the name to
show in a model picker (reads the run's `metrics.json`, never unpickles
the model just for a label) -- used by both `app/single_image.py` and
`app/build_dataset.py`'s model selectors, and by
`classifiers/predict.py`'s console banner.

## Known gaps (verbatim, not glossed over)

- `tools/` is split into three subpackages by responsibility
  (`ingestion/`, `pipeline/`, `maintenance/` -- see each `__init__.py`),
  replacing the previous flat layout (session 8 sept. 2026). Every
  `python -m tools.<script>` command written before that session needs its
  package inserted, e.g. `tools.build_manifest` -> `tools.ingestion.build_manifest`.
  `core/` absorbed the four `utils/` files with no CLI/argparse dependency
  that were shared by several modules (`dataset.py`, `predictions.py`,
  `pipeline_io.py`, `run_io.py` -- the "Décider du sort de..." item open
  since Phase 1, see `TODO.md`); `utils/repair_images.py` moved to
  `tools/maintenance/` (standalone CLI, matches that package's own
  criterion). `utils/` now holds only CLI-entangled or orchestration glue
  (`cli.py`, `tps_overlay.py`, `landmarking_pipeline.py`, `review.py`).
- `data/Bombus/collection` and `data/Bombus/terrain` are the current,
  photo_id/inv_id-clean dataset roots (output of `tools/ingestion/export_clean_dataset.py`
  chained into `tools/ingestion/build_manifest.py`, see stage 1) -- the OLD flat
  `data/Bombus/` (a single `manifest.csv` with an
  `image_id`/`specimen_id`/`shot_index`/in-manifest `split` column) has
  been replaced, not merely deprecated; nothing in this pipeline reads
  that old schema anymore.
- `obb_trainer/` (the YOLO-OBB detector trainer feeding stage 2's `--mode light`
  backend) still expects an older `image_id`/`specimen_id` CSV schema and
  was not part of this refactor -- out of scope, flagged in `TODO.md`.
- Existing `model.joblib` files saved before the `--split` removal (session
  7 sept. 2026, suite 5) still have the old `TrainedModel.split` field --
  loading them with the current `core.model_io.TrainedModel` (now
  `dataset_label`, no `split`) raises `AttributeError` on `model.dataset_label`
  or `model.split` wherever that's read. Not a bug to fix: retrain with
  `tools/pipeline/train_dataset.py` (or `classifiers/train.py` directly) to get a
  model matching the current schema.
