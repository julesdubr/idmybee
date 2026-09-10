"""predict.py
Classifies specimens with a GPA -> PCA -> LDA model trained by train.py.

  batch  : evaluates the model on a dataset with known truth (top-1/top-3
           accuracy, predictions CSV). Same filters as train.py
           (--devices/--species/--castes/--include-outliers/--tps). Usually
           a DIFFERENT dataset root than the one the model was trained on
           (e.g. a model trained on data/Bombus/collection, evaluated on
           data/Bombus/terrain) -- see build_eval_tag, keyed off the
           dataset root's own name rather than a train/test split.
  single : classifies a single photo (no known truth, field use).

The input TPS must have the same landmark scheme (count and order) as the
one used at training time -- go through reconstruct_tps.py for landmarks
coming from Gabriel's UNet. Only the point count is checked here, not the
order: a wrong scheme gives wrong predictions with no error.

`batch` automatically inherits --tps/--landmarks-status-csv from the
train.py run that produced the model (read from its params.json), to avoid
having to re-enter them and the risk of a mismatch between the training and
evaluation schemes. Pass --tps explicitly here to deliberately evaluate
against a different landmark source (the console shows which one is used
and where it came from).

Each prediction includes a Procrustes distance to the model's reference
(procrustes_distance): a value markedly higher than the training set's
signals an atypical shape or a landmark problem.

To compare several landmark sources against each other, train one model per
source with train.py --tps (see classifiers/train.py), then compare with
analysis/compare_runs.py.

Usage:
    python -m classifiers.predict batch models/lda/species_collection/train/model.joblib data/Bombus/terrain
    python -m classifiers.predict single models/lda/species_collection/train/model.joblib data/Bombus/terrain/landmarks/new_photo.tps
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from utils.cli import add_dataset_args, add_logging_args, dataset_kwargs, log_level_from_args
from core.dataset import load_dataset, load_unlabeled_tps
from core.gpa import align_to_reference, procrustes_distance, two_d_array
from core.model_io import TrainedModel, load_model
from core.predictions import accuracy_summary, build_predictions_df, print_predictions_report
from core.run_io import (
    build_eval_tag, read_params, result_path, run_id_from_model_path, run_path, setup_console_logging,
    write_metrics, write_params, write_run_log,
)
from core.tps_io import ImageLandmarks

logger = logging.getLogger(__name__)


def predict_specimens(
    model: TrainedModel, specimens: list[ImageLandmarks], truth_by_tps_id: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Aligns each specimen to the model's reference, projects it into
    PCA/LDA space, and returns a predictions DataFrame (see
    core.predictions.build_predictions_df for the schema)."""
    valid: list[ImageLandmarks] = []
    aligned_list: list[np.ndarray] = []
    dist_list: list[float] = []
    skipped: list[ImageLandmarks] = []

    for sp in specimens:
        if sp.n_points != model.n_points:
            skipped.append(sp)
            continue
        aligned = align_to_reference(sp.landmarks, model.mean_shape)
        aligned_list.append(aligned)
        dist_list.append(procrustes_distance(aligned, model.mean_shape))
        valid.append(sp)

    if skipped:
        logger.warning(
            "%d specimen(s) skipped (landmark count differs from the model's: "
            "expected %d): tps_id=%s%s",
            len(skipped), model.n_points, [s.tps_id for s in skipped][:10],
            ", ..." if len(skipped) > 10 else "",
        )

    if not valid:
        raise ValueError(
            "No specimen in the TPS has the same landmark count as the model "
            f"({model.n_points}). Check the landmark scheme (e.g. go through "
            "reconstruct_tps.py if the landmarks come from Gabriel's UNet)."
        )

    X = two_d_array(np.stack(aligned_list))
    scores = model.pca.transform(X)
    proba = model.lda.predict_proba(scores)
    classes = model.lda.classes_
    predicted = classes[np.argmax(proba, axis=1)]

    return build_predictions_df(
        valid, model.level, predicted, proba, classes,
        truth_by_tps_id=truth_by_tps_id, procrustes_distances=dist_list,
    )


def _print_model_info(model: TrainedModel) -> None:
    devices_str = f", devices={model.devices}" if model.devices else ""
    dataset_str = f", dataset={model.dataset_label}" if model.dataset_label else ""
    name_str = f"{model.model_name!r} " if model.model_name else ""
    print(
        f"Model loaded: {name_str}(level={model.level}, {len(model.classes)} classes, {model.n_points} landmarks, "
        f"trained on {model.n_train} specimens{dataset_str}{devices_str} from {model.source_tps})"
    )


def run_batch(args: argparse.Namespace) -> None:
    setup_console_logging(log_level_from_args(args))
    model = load_model(args.model_path)
    _print_model_info(model)

    family, run_id = run_id_from_model_path(args.model_path)
    train_params = read_params(result_path(family, run_id, "train"))

    tps_explicit = args.landmarks_tps is not None
    if not tps_explicit:
        args.landmarks_tps = train_params.get("landmarks_tps")
    if args.landmarks_status_csv is None:
        args.landmarks_status_csv = train_params.get("landmarks_status_csv")
    source_note = "explicit --tps" if tps_explicit else f"inherited from train ({run_id})"
    print(f"Landmarks source ({source_note}): {args.landmarks_tps or 'landmarks_numbered.tps (default)'}")

    ds_kwargs = dataset_kwargs(args)
    specimens, meta_df = load_dataset(args.dataset, labeled_only=True, **ds_kwargs)
    truth_col = "groupe" if model.level == "caste" else "species"
    truth_by_tps_id = dict(zip((sp.tps_id for sp in specimens), meta_df[truth_col]))

    df = predict_specimens(model, specimens, truth_by_tps_id=truth_by_tps_id)
    acc = accuracy_summary(df, model.level)
    print(f"\nEvaluation on {acc['n']} specimen(s): top-1 = {acc['accuracy_top1']:.4f} | top-3 = {acc['accuracy_top3']:.4f}")

    eval_tag = build_eval_tag(args.dataset.name, args.devices, args.landmarks_tps, args.run_label)
    out_dir = run_path(family, run_id, "predict", eval_tag)

    predictions_path = out_dir / "predictions.csv"
    df.to_csv(predictions_path, index=False)

    metrics = {
        "run_id": run_id,
        "eval_tag": eval_tag,
        "model_path": str(args.model_path),
        "level": model.level,
        "landmarks_source": str(args.landmarks_tps) if args.landmarks_tps else "landmarks_numbered.tps (default)",
        **acc,
    }
    write_metrics(out_dir, metrics)
    write_params(out_dir, args, extra={
        "run_id": run_id, "eval_tag": eval_tag, "family": family,
        "model_path": str(args.model_path),
    })

    header = f"run_id={run_id} | eval_tag={eval_tag} | model={args.model_path}"
    log_text = (
        f"{header}\n" + "=" * len(header) + "\n"
        f"Landmarks source: {metrics['landmarks_source']}\n"
        f"Evaluation: top-1 = {acc['accuracy_top1']:.4f} | top-3 = {acc['accuracy_top3']:.4f} (n={acc['n']})\n"
        f"Predictions -> {predictions_path}\n"
    )
    write_run_log(out_dir, log_text)

    print_predictions_report(df, model.level, args.low_confidence_threshold)
    print(f"\nRun -> {out_dir}")


def run_single(args: argparse.Namespace) -> None:
    setup_console_logging(log_level_from_args(args))
    model = load_model(args.model_path)
    _print_model_info(model)

    specimens = load_unlabeled_tps(args.tps_path, strict=not args.non_strict)
    if len(specimens) != 1:
        raise SystemExit(
            f"{args.tps_path} contains {len(specimens)} specimen(s) -- 'single' mode expects "
            "exactly one photo (a new field photo not yet merged into data/Bombus). "
            "To classify several specimens at once, use 'batch' mode."
        )

    df = predict_specimens(model, specimens)
    row = df.iloc[0]
    pred_col = f"predicted_{model.level}"
    print(
        f"\npredicted {model.level}: {row[pred_col]}  (confidence {row['confidence']:.3f})"
        f"\n  2nd choice: {row['second_choice']} ({row['second_confidence']})"
        f"\n  3rd choice: {row['third_choice']} ({row['third_confidence']})"
        f"\n  Procrustes distance to reference: {row['procrustes_distance']:.4f}"
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"\n-> {args.out}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify specimens with a model trained by train.py"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    batch = subparsers.add_parser("batch", help="Validate the model on a data folder (with known truth)")
    batch.add_argument("model_path", type=Path, help="Saved model (e.g. models/lda/<run_id>/train/model.joblib)")
    batch.add_argument("dataset", type=Path, help="Root folder (e.g. data/Bombus/terrain) -- see core.dataset.load_dataset")
    add_dataset_args(batch)
    batch.add_argument("--low-confidence-threshold", type=float, default=0.6,
                        help="Confidence threshold below which a prediction is listed for manual review (default: 0.6)")
    add_logging_args(batch)
    batch.set_defaults(func=run_batch)

    single = subparsers.add_parser("single", help="Classify a single photo (field use)")
    single.add_argument("model_path", type=Path, help="Saved model")
    single.add_argument("tps_path", type=Path, help=".tps file for a single photo")
    single.add_argument("--out", type=Path, default=None, help="Optional: also save the result as CSV")
    single.add_argument("--non-strict", action="store_true", help="Tolerate malformed TPS blocks")
    add_logging_args(single)
    single.set_defaults(func=run_single)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
