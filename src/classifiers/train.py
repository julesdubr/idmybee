"""train.py
Fits a GPA -> PCA -> LDA classification model on a dataset (see
core.dataset.load_dataset) and evaluates its accuracy by LOOCV.

Writes the model to models/lda/<run_id>/model.joblib and the LOOCV
performance record (metrics.json/params.json/run.log/loocv_predictions.csv)
to runs/lda/<run_id>/train/ -- see core.run_io module docstring for why
these are two separate trees. Detailed figures and tables are produced
separately by analysis/classification_report.py; shape variance
(ANOVA/PERMANOVA) stays in analysis/variance_report.py.

--level species : discriminates by species.
--level caste   : discriminates by (species, caste) -- see core.dataset.target_groupe.

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
from typing import NamedTuple, Sequence

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

from utils.cli import add_dataset_args, add_dataset_positional, add_logging_args, dataset_kwargs, log_level_from_args
from core.dataset import load_dataset, target_groupe
from core.gpa import gpagen, two_d_array
from core.model_io import TrainedModel, save_model
from core.predictions import build_predictions_df, accuracy_summary
from core.run_io import (
    FAMILY_LDA, MODELS_ROOT, RUNS_ROOT, build_run_id, resolve_model_slug, run_path, setup_console_logging, slugify,
    write_metrics, write_params, write_run_log,
)
from core.tps_io import ImageLandmarks

logger = logging.getLogger(__name__)


class TrainOutput(NamedTuple):
    """train.py's own return value: the model artifact and its performance
    record live in two separate trees (see core.run_io module docstring),
    so a caller (app/train_model.py, tools/pipeline/train_dataset.py) needs
    both paths rather than a single output folder."""
    runs_dir: Path  # runs/lda/<run_id>/train/ -- metrics.json/params.json/run.log/loocv_predictions.csv
    model_path: Path  # models/lda/<run_id>/model.joblib


def run_gpa_pca(specimens: list[ImageLandmarks]):
    """GPA then PCA. Returns (scores, gpa_result, pca). Assumes a
    homogeneous landmark count (guaranteed by core.dataset.load_dataset)."""
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


def singleton_classes(groupe: pd.Series, groups: Sequence) -> list[str]:
    """Classes (species, or species_caste at --level=caste) backed by only
    ONE specimen (one unique `groups` value, e.g. one inv_id) -- every photo
    of such a class is guaranteed a wrong LOOCV prediction under
    loocv_lda()'s grouped CV: the fold that holds out its one-and-only
    specimen removes the class from training entirely, so the classifier
    has never seen it and cannot predict it. Used only to log a clear
    warning identifying which classes' LOOCV accuracy is structurally 0%
    for this reason (not a modeling flaw), rather than leaving a caller to
    puzzle out a misleadingly low overall accuracy_top1 on its own."""
    return list(pd.Series(list(groups)).groupby(list(groupe)).nunique()[lambda s: s == 1].index)


def loocv_lda(scores: np.ndarray, groupe: pd.Series, groups: Sequence) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group leave-one-out cross-validated LDA. Returns (proba, predicted, classes).

    Holds out one SPECIMEN per fold (`groups`, e.g. inv_id) -- ALL of its
    photos together -- rather than one photo: a plain per-photo LeaveOneOut
    leaves the other photos of the same individual in the training set when
    one of its photos is held out, which artificially inflates accuracy
    (see README.md "Premiers résultats"). A specimen with a single photo is
    unaffected: its group has one member, so its fold is identical to what
    plain LeaveOneOut would have done for that photo anyway.

    This isn't bias-free either: a class (species, or species_caste) backed
    by a single specimen is entirely absent from every fold that evaluates
    it (see singleton_classes()) -- sklearn pads its predict_proba columns
    with 0 for that fold rather than erroring, but the class is then never
    predicted correctly even once, which *understates* accuracy for that
    class specifically (the opposite direction from the per-photo bias this
    replaces) -- a model actually trained on the full dataset (as the saved
    model.joblib is) has seen that class, this LOOCV estimate for it just
    hasn't seen it in isolation."""
    clf = LinearDiscriminantAnalysis()
    cv = LeaveOneGroupOut()
    proba = cross_val_predict(clf, scores, groupe, groups=groups, cv=cv, method="predict_proba")
    classes = np.unique(groupe)
    predicted = classes[np.argmax(proba, axis=1)]
    return proba, predicted, classes


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GPA -> PCA -> LDA on bumblebee landmarks -- model fitting + LOOCV evaluation"
    )
    add_dataset_positional(parser, help="Root folder (e.g. data/Bombus/collection) -- see core.dataset.load_dataset")
    add_dataset_args(parser)
    parser.add_argument("--level", type=str, default="species", choices=["species", "caste"],
                         help="'species': discriminate by species. 'caste': discriminate by "
                              "(species, caste) -- see core.dataset.target_groupe.")
    parser.add_argument("--lda-components", type=int, default=2,
                         help="LDA components kept in the saved model, for the projection "
                              "(default: 2 -- doesn't affect predict()/predict_proba()).")
    parser.add_argument("--model-name", type=str, default=None,
                         help="Human-facing name for this model (e.g. 'Red-rumped bumblebee "
                              "identifier') -- shown by a model picker (CLI or UI), and now also what "
                              "the output folder under models/lda/ is named (slugified). A "
                              "'_<n>LM_<level>' tag (landmark count in the TPS + --level) is always "
                              "appended, and a name already in use on top of that gets an automatic "
                              "_v2/_v3/... suffix rather than overwriting the earlier run (see "
                              "core.run_io.resolve_model_slug). Defaults to "
                              "level_dataset_label[_devices][_source] if omitted.")
    parser.add_argument("--no-save-model", action="store_true",
                         help="Don't write model.joblib (saved by default -- no reason not to, "
                              "each run lives in its own folder).")
    add_logging_args(parser)
    return parser


def main(argv: list[str] | None = None) -> TrainOutput:
    """Returns (runs_dir, model_path) for this exact run -- lets a caller
    (e.g. app/train_model.py) report on it without recomputing/guessing
    either path."""
    args = build_arg_parser().parse_args(argv)
    setup_console_logging(log_level_from_args(args))

    ds_kwargs = dataset_kwargs(args)
    specimens, meta_df = load_dataset(args.dataset, labeled_only=True, **ds_kwargs)
    groupe = target_groupe(meta_df, args.level)

    scores, gpa_result, pca = run_gpa_pca(specimens)
    n_components = scores.shape[1]

    # Landmark count + classification level, always appended to the model
    # name (typed or defaulted) so two models fit on different landmark
    # schemes (e.g. 19LM vs. 18LM, see app/setup_dataset.py) or different
    # --level are never confusable in a model picker -- computed only now
    # since it needs specimens[0], guaranteed non-empty past run_gpa_pca().
    dataset_label = args.dataset.name
    name_root = args.model_name or build_run_id(args.level, dataset_label, args.devices, args.landmarks_tps, args.run_label)
    base_name = f"{name_root}_{specimens[0].n_points}LM_{args.level}"
    run_id = resolve_model_slug(FAMILY_LDA, base_name)
    out_dir = run_path(FAMILY_LDA, run_id, "train", root=RUNS_ROOT)
    model_dir = run_path(FAMILY_LDA, run_id, root=MODELS_ROOT)
    model_path = model_dir / "model.joblib"

    lda_final = fit_lda(scores, groupe, n_components=args.lda_components)

    groups = [sp.inv_id for sp in specimens]
    singletons = singleton_classes(groupe, groups)
    if singletons:
        logger.warning(
            "%d class(es) backed by a single specimen -- their LOOCV accuracy is structurally 0%% "
            "(see loocv_lda docstring), not a sign the fitted model itself can't identify them: %s",
            len(singletons), singletons,
        )

    proba, predicted, classes = loocv_lda(scores, groupe, groups)
    truth_by_tps_id = dict(zip((sp.tps_id for sp in specimens), groupe))
    df = build_predictions_df(specimens, args.level, predicted, proba, classes, truth_by_tps_id=truth_by_tps_id)
    acc = accuracy_summary(df, args.level)

    predictions_path = out_dir / "loocv_predictions.csv"
    df.to_csv(predictions_path, index=False)

    # run_id may carry a "_v{n}" suffix resolve_model_slug added on a name
    # collision -- append that same suffix to the human-facing model_name
    # too (base_name, itself typed or defaulted, plus the _<n>LM_<level>
    # tag above), so two versions of the same name stay visually distinct
    # in a model picker instead of showing the identical label twice.
    version_suffix = run_id[len(slugify(base_name)):]
    model_name = base_name + version_suffix

    metrics = {
        "run_id": run_id,
        "model_name": model_name,
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

    header = f"model_name={model_name} | run_id={run_id} | level={args.level} | n={len(specimens)} specimen(s) | {specimens[0].n_points} landmarks"
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
            model_name=model_name,
        )
        save_model(model, model_path)

    print(f"Run -> {out_dir}")
    print(f"Model -> {model_path}")
    return TrainOutput(runs_dir=out_dir, model_path=model_path)


if __name__ == "__main__":
    main()
