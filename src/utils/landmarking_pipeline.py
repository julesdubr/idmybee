"""Shared landmarking + export stages for tools/pipeline/train_dataset.py and
tools/pipeline/predict_dataset.py, plus the in-memory single-image counterpart used
by app/single_image.py.

File-based path -- runs, in order, each stage's own main(argv) in-process
(no subprocess, no duplicated logic -- see PIPELINE.md):
    extraction.detect_wing
    extraction.normalize_crop
    landmarks.predict
    landmarks.renumber
    tools.pipeline.export_final_landmarks   -> <dataset>/export/

Dataset-agnostic: any clean dataset root (manifest.csv + biological_data.csv
+ images/) is a valid input. Source names like "collection"/"terrain" are
just folder names, never baked into these tools.

In-memory path -- `place_landmarks()`: the same 4 stages (detection ->
crop -> UNet landmarks -> renumbering) on ONE already-loaded image, calling
the exact same pure, disk-free functions the file-based path uses under the
hood (detect_one_image, normalize_one, predict_landmarks, numerate_one) --
no TPS/CSV ever written. See CONVENTIONS.md "Fonctions core réutilisables".
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from extraction.detect_wing import detect_one_image, main as detect_wing_main
from extraction.normalize_crop import (
    apply_wing_transform_to_points_inverse,
    compute_wing_transform,
    main as normalize_crop_main,
    normalize_one,
)
from landmarks.predict import main as landmarks_predict_main, predict_landmarks
from landmarks.renumber import main as renumber_main, numerate_one
from tools.pipeline.export_final_landmarks import main as export_final_landmarks_main
from utils.cli import verbosity_argv
from core.pipeline_io import dataset_export_dir

DEFAULT_DETECTOR_MODEL = Path("models/yolon_obb/Bombus_612.pt")
REFERENCE_SHAPES_DIR = Path("references/shapes")


def add_landmarking_args(parser: argparse.ArgumentParser) -> None:
    """Flags shared by train_dataset.py and predict_dataset.py for stages
    detect -> crop -> landmarks -> renumber -> export."""
    parser.add_argument("--mode", default="light", choices=["heavy", "light"],
                         help="Detection/crop/landmark-placement backend (shared across stages).")
    parser.add_argument("--detector-model", type=Path, default=DEFAULT_DETECTOR_MODEL,
                         help=f"Wing detector weights: required for --mode light, YOLO-OBB .pt "
                              f"(default: {DEFAULT_DETECTOR_MODEL}). Not required for --mode heavy "
                              "(uses the YOLOE backend's own bundled weights unless explicitly "
                              "overridden here) -- see --heavy-ref instead.")
    parser.add_argument("--heavy-ref", type=Path, default=None,
                         help="--mode heavy only: JSON file of YOLOE references (required in that mode).")
    parser.add_argument("--imgsz", type=int, default=1024, help="detect_wing --imgsz.")
    parser.add_argument("--conf", type=float, default=0.10, help="detect_wing --conf.")
    parser.add_argument("--padding", type=float, default=0.10, help="normalize_crop --padding.")
    parser.add_argument("--out-width", type=int, default=512)
    parser.add_argument("--out-height", type=int, default=256)
    parser.add_argument("--unet-model", type=Path, required=True, help="UNet weights (.pt) for landmark placement.")
    parser.add_argument("--n-landmarks", type=int, default=19,
                         help="19 = Tancrede's full blueprint (default). 18 for an older/legacy UNet model.")
    parser.add_argument(
        "--landmarks-tag", default=None,
        help="Optional label -- writes this run's landmark-stage files to "
             "<dataset>/landmarks_<tag>/ instead of the default <dataset>/landmarks/ (see "
             "landmarks_dirname()). Set this when re-running landmark placement on an already-cropped "
             "dataset with a different --unet-model/--n-landmarks (e.g. 'tag=18lm'), so the earlier "
             "run's files aren't overwritten. Point classifiers.train/predict/export_final_landmarks "
             "at this run's output with their own --tps <dataset>/landmarks_<tag>/landmarks_numbered.tps.",
    )
    parser.add_argument("--reference", type=Path, required=True,
                         help="Frozen GPA reference shape, a plain .tps (see landmarks.build_reference) -- "
                              f"e.g. one of {REFERENCE_SHAPES_DIR}/*.tps. Its own landmark count is read "
                              "directly from the file, not encoded in its name.")
    parser.add_argument("--base-dir", default=None, help="Root to resolve manifest.csv's path, if relative.")
    parser.add_argument("--device", default=None, help="'cpu'/'cuda' for detection AND landmark placement.")
    parser.add_argument("--overwrite", action="store_true",
                         help="Reprocess crops/landmarks already logged (detection has no resume state).")
    parser.add_argument("--retry-failed", action="store_true",
                         help="Retry photos logged FAILED by a previous landmarks.predict run.")
    parser.add_argument("--no-original-space", action="store_true",
                         help="Skip original-image-space reprojection in the export package.")
    parser.add_argument("--export-dir", type=Path, default=None,
                         help="Where to write the R-facing landmarks package "
                              "(default: <dataset>/export/ -- see core.pipeline_io.dataset_export_dir).")


def resolve_gpa_reference(args: argparse.Namespace) -> Path:
    return args.reference


def landmarks_dirname(args: argparse.Namespace) -> str:
    """<dataset>/-relative folder name for this run's landmark-stage files
    (landmarks.tps/landmarks.csv/landmarks_numbered.*) -- "landmarks" by
    default, or "landmarks_<tag>" when --landmarks-tag is set (see
    add_landmarking_args), so re-running placement with a different
    --unet-model/--n-landmarks on an already-cropped dataset doesn't
    overwrite the previous run's files."""
    tag = getattr(args, "landmarks_tag", None)
    return "landmarks" if not tag else f"landmarks/{tag}"


def resolve_export_dir(args: argparse.Namespace) -> Path:
    return args.export_dir or dataset_export_dir(args.dataset)


def dataset_filter_argv(args: argparse.Namespace) -> list[str]:
    """--devices/--species/--castes/--include-outliers/--non-strict/--tps/
    --landmarks-status-csv/--run-label (see utils.cli.add_dataset_args) as
    CLI flags, for a caller that invokes another script's main(argv) rather
    than passing a Namespace straight through (classifiers.predict.run_batch
    takes args directly and has no need for this). Shared by run_export()
    and tools.pipeline.train_dataset, which both used to duplicate these
    same seven if-blocks."""
    argv: list[str] = []
    if getattr(args, "devices", None):
        argv += ["--devices", *args.devices]
    if getattr(args, "species", None):
        argv += ["--species", *args.species]
    if getattr(args, "castes", None):
        argv += ["--castes", *args.castes]
    if not getattr(args, "exclude_outliers", True):
        argv += ["--include-outliers"]
    if getattr(args, "non_strict", False):
        argv += ["--non-strict"]
    if getattr(args, "landmarks_tps", None):
        argv += ["--tps", str(args.landmarks_tps)]
    if getattr(args, "landmarks_status_csv", None):
        argv += ["--landmarks-status-csv", str(args.landmarks_status_csv)]
    if getattr(args, "run_label", None):
        argv += ["--run-label", args.run_label]
    return argv


def run_detection_and_crop(args: argparse.Namespace) -> None:
    """Stages 1-2: detection -> crop normalization.

    Writes under the dataset root:
        extraction/<mode>/detection.csv
        extraction/<mode>/images/, crops.csv

    Split out from run_landmarking() so a caller can pause here for a
    validation step (review the crops, see utils.review) before spending
    UNet compute in run_landmark_placement() -- see app/setup_dataset.py.
    """
    if args.mode == "heavy" and args.heavy_ref is None:
        raise SystemExit("--mode heavy requires --heavy-ref (JSON file of YOLOE references).")

    dataset = str(args.dataset)
    verbosity = verbosity_argv(args)

    print(f"\n=== Wing detection ({args.mode}) ===")
    detect_wing_argv = [
        dataset, "--mode", args.mode,
        "--imgsz", str(args.imgsz), "--conf", str(args.conf), *verbosity,
    ]
    if args.mode == "heavy":
        detect_wing_argv += ["--ref", str(args.heavy_ref)]
        # No detector weights required for heavy: the YOLOE backend has its own
        # sensible default (yoloe-11s-seg.pt) -- only forward --model if the
        # caller explicitly overrode it away from the shared light-mode default.
        if Path(args.detector_model) != DEFAULT_DETECTOR_MODEL:
            detect_wing_argv += ["--model", str(args.detector_model)]
    else:
        detect_wing_argv += ["--model", str(args.detector_model)]
    if args.device:
        detect_wing_argv += ["--device", args.device]
    if args.base_dir:
        detect_wing_argv += ["--base-dir", str(args.base_dir)]
    detect_wing_main(detect_wing_argv)

    print(f"\n=== Crop normalization ({args.mode}) ===")
    normalize_crop_argv = [
        dataset, "--mode", args.mode, "--padding", str(args.padding),
        "--out-width", str(args.out_width), "--out-height", str(args.out_height), *verbosity,
    ]
    if args.base_dir:
        normalize_crop_argv += ["--base-dir", str(args.base_dir)]
    if args.overwrite:
        normalize_crop_argv += ["--overwrite"]
    normalize_crop_main(normalize_crop_argv)


def run_landmark_placement(args: argparse.Namespace) -> None:
    """Stages 3-4: UNet landmark placement -> renumbering.

    Writes under the dataset root, in landmarks_dirname(args) (default
    "landmarks", or "landmarks_<tag>" with --landmarks-tag -- see that
    function):
        <landmarks-dir>/landmarks.tps, landmarks.csv
        <landmarks-dir>/landmarks_numbered.tps, landmarks_numbered.csv

    `args.crops_csv`, if set (e.g. a reviewed override written by
    utils.review.write_crop_review), is forwarded as landmarks.predict's
    --crops-csv -- skips a human-rejected crop here without mutating
    extraction/<mode>/crops.csv itself. Split out from run_landmarking() --
    see run_detection_and_crop()'s docstring."""
    dataset = str(args.dataset)
    reference = str(resolve_gpa_reference(args))
    verbosity = verbosity_argv(args)
    landmarks_dir = landmarks_dirname(args)

    print("\n=== Landmark placement (UNet) ===")
    landmarks_predict_argv = [
        dataset, "--mode", args.mode, "--model", str(args.unet_model),
        "--n-landmarks", str(args.n_landmarks), "--landmarks-dir", landmarks_dir, *verbosity,
    ]
    if args.device:
        landmarks_predict_argv += ["--device", args.device]
    if args.base_dir:
        landmarks_predict_argv += ["--base-dir", str(args.base_dir)]
    if args.overwrite:
        landmarks_predict_argv += ["--overwrite"]
    if args.retry_failed:
        landmarks_predict_argv += ["--retry-failed"]
    if getattr(args, "crops_csv", None):
        landmarks_predict_argv += ["--crops-csv", str(args.crops_csv)]
    landmarks_predict_main(landmarks_predict_argv)

    print("\n=== Renumbering ===")
    renumber_main([
        dataset, "--reference", reference, "--landmarks-dir", landmarks_dir,
        "--biological-data", str(Path(args.dataset) / "biological_data.csv"), *verbosity,
    ])


def run_landmarking(args: argparse.Namespace) -> None:
    """Stages 1-4: detection -> crop -> UNet landmarks -> renumbering, in
    one call -- see run_detection_and_crop()/run_landmark_placement() for
    the two halves, used independently by a validation-gated caller."""
    run_detection_and_crop(args)
    run_landmark_placement(args)


def run_export(args: argparse.Namespace) -> Path:
    """Stage 5: R-facing package into <dataset>/export/ (or --export-dir).

    Writes:
        landmarks_<n>lm_crop.tps
        landmarks_<n>lm_original.tps           (unless --no-original-space)
        landmarks_<n>lm_biological_data.csv    (photo-level, row-aligned to the TPS)
        failed.csv

    Forwards the same --devices/--species/--castes/--include-outliers/--tps/
    --landmarks-status-csv filters as classifiers.train (see
    dataset_filter_argv) -- so a landmark-review override
    (--landmarks-status-csv pointing at a landmarks_reviewed.csv) is
    reflected in the exported package the same way it is in the fitted
    model, instead of the export silently using the unreviewed statuses.
    Also defaults --tps to this run's own landmarks_dirname(args)/
    landmarks_numbered.tps when --landmarks-tag was used and the caller
    hasn't already set args.landmarks_tps itself (e.g. via a review step) --
    otherwise the export would silently fall back to the untagged
    <dataset>/landmarks/ default.
    """
    output_dir = resolve_export_dir(args)
    verbosity = verbosity_argv(args)
    print(f"\n=== Export landmarks package -> {output_dir} ===")
    export_argv = [
        str(args.dataset), "--output-dir", str(output_dir), "--mode", args.mode,
        "--padding", str(args.padding), "--out-width", str(args.out_width),
        "--out-height", str(args.out_height), *verbosity, *dataset_filter_argv(args),
    ]
    if not getattr(args, "landmarks_tps", None) and getattr(args, "landmarks_tag", None):
        default_tps = Path(args.dataset) / landmarks_dirname(args) / "landmarks_numbered.tps"
        export_argv += ["--tps", str(default_tps)]
    if args.base_dir:
        export_argv += ["--base-dir", str(args.base_dir)]
    if args.no_original_space:
        export_argv += ["--no-original-space"]
    export_final_landmarks_main(export_argv)
    return output_dir


@dataclass
class PlacementResult:
    """Result of running ONE in-memory image through detection -> crop ->
    UNet landmark placement -> renumbering. `status` is OK/FAILED, reusing
    each stage's own vocabulary (see PIPELINE.md) -- there is no SUSPECT
    here: the only population-level SUSPECT signal (renumber.py's
    per-species outlier diagnostic, core.outliers.flag_by_species) needs a
    group to compare against, which a lone unlabeled specimen doesn't have."""
    status: str
    stage: str | None                          # "detection"|"crop"|"landmarks"|"renumbering", None if OK
    error_reason: str
    crop_image: np.ndarray | None = None       # BGR, out_width x out_height -- set once a crop exists
    landmarks: np.ndarray | None = None        # (n_landmarks, 2), crop space, canonically numbered
    original_landmarks: np.ndarray | None = None  # (n_landmarks, 2), reprojected into the ORIGINAL image
    detection_box: np.ndarray | None = None    # 4x2 pixel corners in the ORIGINAL image
    registration_score: float | None = None    # renumbering cost, see landmarks.methods.base.NumberingResult


def place_landmarks(
    image_bgr: np.ndarray,
    *,
    detector_mode: str,
    detector_ctx: dict,
    detector_args: argparse.Namespace,
    padding: float,
    out_width: int,
    out_height: int,
    unet_model,
    unet_device: str,
    n_landmarks: int,
    reference_zones: np.ndarray,
    numbering_method: str = "hungarian_umeyama",
) -> PlacementResult:
    """Field/single-image counterpart to run_landmarking(): places
    canonically-numbered landmarks on ONE already-loaded image, entirely in
    memory, no file written anywhere. Reuses the same per-specimen pure
    functions the batch pipeline calls under the hood (detect_one_image,
    normalize_one, predict_landmarks, numerate_one), so a single photo and a
    dataset row go through identical geometry/model code -- only the I/O
    around them differs.

    `detector_ctx`/`unet_model`/`reference_zones` are already-loaded models
    (see extraction.detect_wing.get_backend(mode).load_model,
    landmarks_trainer.model.load_weights, landmarks.build_reference.load_reference)
    -- loading them is the caller's job (e.g. cached once per Streamlit
    session), never repeated on each call here.

    A predict_landmarks() SUSPECT (fewer landmarks than expected, but not
    zero) is not special-cased: it is passed through to numerate_one() as
    normal, which fails it there for a point-count mismatch against the
    reference -- exactly what the batch pipeline already documents (see
    landmarks/renumber.py's module docstring).
    """
    detection = detect_one_image(detector_mode, detector_ctx, image_bgr, detector_args)
    if detection["status"] != "OK":
        return PlacementResult(
            status="FAILED", stage="detection",
            error_reason=detection["error_reason"] or "no_detection",
        )

    height, width = image_bgr.shape[:2]
    box_pixels = detection["box"].copy()
    box_pixels[:, 0] *= width
    box_pixels[:, 1] *= height

    crop, _aspect_ratio = normalize_one(
        image_bgr, box_pixels, pad=padding, out_width=out_width, out_height=out_height,
    )
    if crop is None:
        return PlacementResult(
            status="FAILED", stage="crop", error_reason="normalization_failed",
            detection_box=box_pixels,
        )

    _lm_status, lm_error_reason, coords_xy, _n_found = predict_landmarks(
        crop, unet_model, unet_device, n_landmarks,
    )
    if coords_xy is None:
        return PlacementResult(
            status="FAILED", stage="landmarks", error_reason=lm_error_reason,
            crop_image=crop, detection_box=box_pixels,
        )

    numbering = numerate_one(coords_xy, reference_zones, method=numbering_method)
    if numbering.status == "FAILED":
        return PlacementResult(
            status="FAILED", stage="renumbering", error_reason=numbering.reason,
            crop_image=crop, detection_box=box_pixels,
        )

    # Reprojects crop-space landmarks back into the original image, via the
    # same WingTransform/apply_wing_transform_to_points_inverse machinery
    # tools.pipeline.export_final_landmarks.reproject_to_raw_space uses for
    # the file-based path -- degenerate OBB geometry (already ruled out by
    # the successful normalize_one() call above) is the only way this
    # returns None, so original_landmarks is left unset rather than failing
    # a pipeline whose crop-space result is otherwise valid.
    original_landmarks = None
    transform = compute_wing_transform(
        image_bgr.shape, box_pixels, pad=padding, target_ratio=out_width / out_height,
    )
    if transform is not None:
        original_landmarks = apply_wing_transform_to_points_inverse(
            numbering.numbered.astype(np.float32), transform, out_width, out_height,
        )

    return PlacementResult(
        status="OK", stage=None, error_reason="",
        crop_image=crop, landmarks=numbering.numbered, original_landmarks=original_landmarks,
        detection_box=box_pixels, registration_score=numbering.score,
    )
