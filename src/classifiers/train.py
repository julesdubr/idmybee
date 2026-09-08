"""train.py
Fits a GPA -> PCA -> LDA classification model on a dataset (see
utils.dataset.load_dataset) and evaluates its accuracy by LOOCV.

Writes the model and raw predictions to data/models/lda/<run_id>/train/.
Detailed figures and tables are produced separately by
analysis/classification_report.py; shape variance (ANOVA/PERMANOVA) stays
in analysis/variance_report.py.

--level species : discriminates by species.
--level caste   : discriminates by (species, caste) -- see utils.dataset.target_groupe.

Usage:
    python -m classifiers.train data/Bombus --level species
    python -m classifiers.train data/Bombus --level species --devices P1 S1
    python -m classifiers.train data/Bombus --level species \\
        --tps data/Bombus/landmarks/tancrede_reference_19lm.tps --run-label tancrede19lm
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut, cross_val_predict

from utils.cli import add_dataset_args, add_dataset_positional, add_logging_args, dataset_kwargs, log_level_from_args
from utils.dataset import load_dataset, target_groupe
from core.gpa import gpagen, two_d_array
from core.model_io import TrainedModel, save_model
from utils.predictions import build_predictions_df, accuracy_summary
from utils.run_io import (
    FAMILY_LDA, build_run_id, run_path, setup_console_logging, write_metrics, write_params, write_run_log,
)
from core.tps_io import ImageLandmarks

logger = logging.getLogger(__name__)


def run_gpa_pca(specimens: list[ImageLandmarks]):
    """GPA then PCA. Returns (scores, gpa_result, pca). Assumes a
    homogeneous landmark count (guaranteed by utils.dataset.load_dataset)."""
    if not specimens:
        raise ValueError(
            "No specimen to process (empty list after loading/filtering) -- check "
            "--devices/--species/--castes, or that the TPS contains valid landmarks."
        )
    n_points = specimens[0].n_points
    gpa_result = gpagen([sp.landmarks for sp in specimens])
    X = two_d_array(gpa_result.aligned)  # (n_specimens, 2*n_points)

    n_components = 2 * n_points - 4  # degrees of freedom remaining after 2D GPA
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X)

    logger.info(
        "PCA: %d component(s) kept over %d specimen(s) (cumulative explained variance = %.1f%%)",
        n_components, len(specimens), 100 * pca.explained_variance_ratio_.sum(),
    )
    return scores, gpa_result, pca


def fit_lda(scores: np.ndarray, groupe: pd.Series, n_components: int = 2):
    """Fits the final LDA (on the whole set, saved in the model).
    n_components only limits .transform() (projection); predict()/
    predict_proba() are unaffected."""
    n_components = min(n_components, len(np.unique(groupe)) - 1)
    if n_components < 1:
        raise ValueError("LDA needs at least two classes to compute a projection.")
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    lda.fit(scores, groupe)
    return lda


def loocv_lda(scores: np.ndarray, groupe: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Leave-one-out cross-validated LDA. Returns (proba, predicted, classes).

    Note: LOOCV holds out one photo, not one specimen -- an individual with
    several photos stays partly in the training set when one of its photos
    is held out, which artificially inflates accuracy."""
    clf = LinearDiscriminantAnalysis()
    loo = LeaveOneOut()
    proba = cross_val_predict(clf, scores, groupe, cv=loo, method="predict_proba")
    classes = np.unique(groupe)
    predicted = classes[np.argmax(proba, axis=1)]
    return proba, predicted, classes


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GPA -> PCA -> LDA on bumblebee landmarks -- model fitting + LOOCV evaluation"
    )
    add_dataset_positional(parser, help="Root folder (e.g. data/Bombus/collection) -- see utils.dataset.load_dataset")
    add_dataset_args(parser)
    parser.add_argument("--level", type=str, default="species", choices=["species", "caste"],
                         help="'species': discriminate by species. 'caste': discriminate by "
                              "(species, caste) -- see utils.dataset.target_groupe.")
    parser.add_argument("--lda-components", type=int, default=2,
                         help="LDA components kept in the saved model, for the projection "
                              "(default: 2 -- doesn't affect predict()/predict_proba()).")
    parser.add_argument("--no-save-model", action="store_true",
                         help="Don't write model.joblib (saved by default -- no reason not to, "
                              "each run lives in its own folder).")
    add_logging_args(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    setup_console_logging(log_level_from_args(args))

    ds_kwargs = dataset_kwargs(args)
    specimens, meta_df = load_dataset(args.dataset, labeled_only=True, **ds_kwargs)
    groupe = target_groupe(meta_df, args.level)

    dataset_label = args.dataset.name
    run_id = build_run_id(args.level, dataset_label, args.devices, args.landmarks_tps, args.run_label)
    out_dir = run_path(FAMILY_LDA, run_id, "train")

    scores, gpa_result, pca = run_gpa_pca(specimens)
    n_components = scores.shape[1]

    lda_final = fit_lda(scores, groupe, n_components=args.lda_components)

    proba, predicted, classes = loocv_lda(scores, groupe)
    truth_by_tps_id = dict(zip((sp.tps_id for sp in specimens), groupe))
    df = build_predictions_df(specimens, args.level, predicted, proba, classes, truth_by_tps_id=truth_by_tps_id)
    acc = accuracy_summary(df, args.level)

    predictions_path = out_dir / "loocv_predictions.csv"
    df.to_csv(predictions_path, index=False)

    metrics = {
        "run_id": run_id,
        "level": args.level,
        "landmarks_source": str(args.landmarks_tps) if args.landmarks_tps else "landmarks_numbered.tps (default)",
        "n_specimens": len(specimens),
        "n_points": specimens[0].n_points,
        "n_pca_components": n_components,
        "pca_explained_variance_cum": float(pca.explained_variance_ratio_.sum()),
        "gpa_n_iterations": gpa_result.n_iterations,
        **acc,  # n, accuracy_top1, accuracy_top3
    }
    write_metrics(out_dir, metrics)
    write_params(out_dir, args, extra={"run_id": run_id, "family": FAMILY_LDA})

    header = f"run_id={run_id} | level={args.level} | n={len(specimens)} specimen(s) | {specimens[0].n_points} landmarks"
    log_text = (
        f"{header}\n" + "=" * len(header) + "\n"
        f"Landmarks source: {metrics['landmarks_source']}\n"
        f"GPA: converged in {gpa_result.n_iterations} iteration(s)\n"
        f"PCA: {n_components} component(s) kept "
        f"(cumulative explained variance = {100 * metrics['pca_explained_variance_cum']:.1f}%)\n"
        f"LDA (LOOCV): top-1 = {acc['accuracy_top1']:.4f} | top-3 = {acc['accuracy_top3']:.4f}\n"
        f"Predictions -> {predictions_path}\n"
    )
    write_run_log(out_dir, log_text)
    print("\n" + log_text)

    if not args.no_save_model:
        model = TrainedModel(
            mean_shape=gpa_result.mean_shape,
            n_points=specimens[0].n_points,
            pca=pca,
            lda=lda_final,
            level=args.level,
            classes=sorted(groupe.unique()),
            dataset_label=dataset_label,
            devices=args.devices,
            source_tps=str(args.landmarks_tps) if args.landmarks_tps else str(args.dataset / "landmarks" / "landmarks_numbered.tps"),
            n_train=len(specimens),
        )
        save_model(model, out_dir / "model.joblib")

    print(f"Run -> {out_dir}")


if __name__ == "__main__":
    main()
