"""train.py
Ajuste un modèle de classification GPA -> PCA -> LDA sur un dataset (voir
utils.dataset.load_dataset) et évalue sa précision par LOOCV.

Écrit le modèle et les prédictions brutes dans
data/models/lda/<run_id>/train/. Figures et tableaux détaillés sont produits
séparément par analysis/classification_report.py ; la variance de forme
(ANOVA/PERMANOVA) reste dans analysis/variance_report.py.

--level species : discrimine par espèce.
--level caste   : discrimine par (espèce, caste) -- voir utils.dataset.target_groupe.

Usage :
    python -m classifiers.train data/Bombus --level species
    python -m classifiers.train data/Bombus --level species --devices P1 S1
    python -m classifiers.train data/Bombus --level species \\
        --tps data/Bombus/landmarks/tancrede_reference_19lm.tps --run-label tancrede19lm
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut, cross_val_predict

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parent))
from utils.cli import add_dataset_args, dataset_kwargs
from utils.dataset import load_dataset, target_groupe
from utils.gpa import gpagen, two_d_array
from utils.model_io import TrainedModel, save_model
from utils.predictions import build_predictions_df, accuracy_summary
from utils.run_io import (
    FAMILY_LDA, build_run_id, setup_console_logging, step_dir, write_metrics, write_params, write_run_log,
)
from utils.tps_io import ImageLandmarks

logger = logging.getLogger(__name__)


def run_gpa_pca(specimens: list[ImageLandmarks]):
    """GPA puis PCA. Retourne (scores, gpa_result, pca). Suppose un nombre
    de landmarks homogène (garanti par utils.dataset.load_dataset)."""
    if not specimens:
        raise ValueError(
            "Aucun spécimen à traiter (liste vide après chargement/filtrage) -- vérifier "
            "--split/--devices/--species/--castes, ou que le TPS contient des landmarks valides."
        )
    n_points = specimens[0].n_points
    gpa_result = gpagen([sp.landmarks for sp in specimens])
    X = two_d_array(gpa_result.aligned)  # (n_specimens, 2*n_points)

    n_components = 2 * n_points - 4  # ddl restants après GPA en 2D
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X)

    logger.info(
        "PCA : %d composantes conservées sur %d spécimens (variance expliquée cumulée = %.1f%%)",
        n_components, len(specimens), 100 * pca.explained_variance_ratio_.sum(),
    )
    return scores, gpa_result, pca


def fit_lda(scores: np.ndarray, groupe: pd.Series, n_components: int = 2):
    """Ajuste la LDA finale (sur tout le jeu, sauvée dans le modèle).
    n_components ne limite que .transform() (projection) ; predict()/
    predict_proba() ne sont pas affectés."""
    n_components = min(n_components, len(np.unique(groupe)) - 1)
    if n_components < 1:
        raise ValueError("LDA nécessite au moins deux classes pour calculer une projection.")
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    lda.fit(scores, groupe)
    return lda


def loocv_lda(scores: np.ndarray, groupe: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """LDA en validation croisée leave-one-out. Retourne (proba, predicted, classes).

    Note : le LOOCV retire une photo, pas un spécimen -- un individu avec
    plusieurs photos reste partiellement dans le train quand une de ses
    photos est testée, ce qui optimise artificiellement l'accuracy."""
    clf = LinearDiscriminantAnalysis()
    loo = LeaveOneOut()
    proba = cross_val_predict(clf, scores, groupe, cv=loo, method="predict_proba")
    classes = np.unique(groupe)
    predicted = classes[np.argmax(proba, axis=1)]
    return proba, predicted, classes


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GPA -> PCA -> LDA sur landmarks de bourdons -- ajustement du modèle + évaluation LOOCV"
    )
    parser.add_argument("dataset", type=Path, help="Dossier racine (ex: data/Bombus) -- voir utils.dataset.load_dataset")
    add_dataset_args(parser, default_split="train")
    parser.add_argument("--level", type=str, default="species", choices=["species", "caste"],
                         help="'species' : discrimination par espèce. 'caste' : discrimination par "
                              "(espèce, caste) -- voir utils.dataset.target_groupe.")
    parser.add_argument("--lda-components", type=int, default=2,
                         help="Composantes LDA conservées dans le modèle sauvé, pour la projection "
                              "(défaut: 2 -- n'affecte pas predict()/predict_proba()).")
    parser.add_argument("--no-save-model", action="store_true",
                         help="Ne pas écrire model.joblib (par défaut, toujours sauvé -- pas de raison de "
                              "s'en priver, chaque run vit dans son propre dossier).")
    return parser


def main(argv: list[str] | None = None) -> None:
    setup_console_logging()
    args = build_arg_parser().parse_args(argv)

    ds_kwargs = dataset_kwargs(args, default_split="train")
    specimens, meta_df = load_dataset(args.dataset, labeled_only=True, **ds_kwargs)
    groupe = target_groupe(meta_df, args.level)

    run_id = build_run_id(args.level, ds_kwargs["split"], args.devices, args.landmarks_tps, args.run_label)
    out_dir = step_dir(run_id, "train", family=FAMILY_LDA)

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
        "landmarks_source": str(args.landmarks_tps) if args.landmarks_tps else "landmarks_numbered.tps (défaut)",
        "n_specimens": len(specimens),
        "n_points": specimens[0].n_points,
        "n_pca_components": n_components,
        "pca_explained_variance_cum": float(pca.explained_variance_ratio_.sum()),
        "gpa_n_iterations": gpa_result.n_iterations,
        **acc,  # n, accuracy_top1, accuracy_top3
    }
    write_metrics(out_dir, metrics)
    write_params(out_dir, args, extra={"run_id": run_id, "family": FAMILY_LDA, "resolved_split": ds_kwargs["split"]})

    header = f"run_id={run_id} | level={args.level} | n={len(specimens)} spécimen(s) | {specimens[0].n_points} landmarks"
    log_text = (
        f"{header}\n" + "=" * len(header) + "\n"
        f"Source landmarks : {metrics['landmarks_source']}\n"
        f"GPA : convergence en {gpa_result.n_iterations} itération(s)\n"
        f"PCA : {n_components} composante(s) conservée(s) "
        f"(variance expliquée cumulée = {100 * metrics['pca_explained_variance_cum']:.1f}%)\n"
        f"LDA (LOOCV) : top-1 = {acc['accuracy_top1']:.4f} | top-3 = {acc['accuracy_top3']:.4f}\n"
        f"Prédictions -> {predictions_path}\n"
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
            split=ds_kwargs["split"],
            devices=args.devices,
            source_tps=str(args.landmarks_tps) if args.landmarks_tps else str(args.dataset / "landmarks" / "landmarks_numbered.tps"),
            n_train=len(specimens),
        )
        save_model(model, out_dir / "model.joblib")

    print(f"Run -> {out_dir}")


if __name__ == "__main__":
    main()
