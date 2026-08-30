"""predict.py
Classe des spécimens avec un modèle GPA -> PCA -> LDA entraîné par train.py.

  batch  : évalue le modèle sur un dataset avec vérité connue (accuracy
           top-1/top-3, CSV de prédictions). Mêmes filtres que train.py
           (--split/--devices/--species/--castes/--exclude-outliers/--tps).
  single : classe une seule photo (pas de vérité connue, usage terrain).

Le TPS d'entrée doit avoir le même schéma de landmarks (nombre et ordre) que
celui utilisé à l'entraînement -- passer par reconstruct_tps.py pour des
landmarks issus du UNet de Gabriel. Seul le nombre de points est vérifié
ici, pas l'ordre : un mauvais schéma donne des prédictions fausses sans
erreur.

Chaque prédiction inclut une distance de Procrustes à la référence du
modèle (procrustes_distance) : une valeur nettement supérieure à celles du
jeu d'entraînement signale une forme atypique ou un problème de landmarks.

Pour comparer plusieurs sources de landmarks entre elles, utiliser train.py
avec --tps (voir classifiers/train.py) : predict.py sert à appliquer un
modèle déjà entraîné à de nouvelles données.

Usage :
    python -m classifiers.predict batch data/models/lda/species_train/train/model.joblib data/Bombus --split test
    python -m classifiers.predict single data/models/lda/species_train/train/model.joblib data/Bombus/landmarks/nouvelle_photo.tps
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parent))
from utils.cli import add_dataset_args, dataset_kwargs
from utils.dataset import load_dataset, load_unlabeled_tps
from utils.gpa import align_to_reference, procrustes_distance, two_d_array
from utils.model_io import TrainedModel, load_model
from utils.predictions import accuracy_summary, build_predictions_df, print_predictions_report
from utils.run_io import (
    FAMILY_LDA, build_run_id, setup_console_logging, step_dir, write_metrics, write_params, write_run_log,
)
from utils.tps_io import ImageLandmarks


def predict_specimens(
    model: TrainedModel, specimens: list[ImageLandmarks], truth_by_tps_id: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Aligne chaque spécimen sur la référence du modèle, le projette dans
    l'espace PCA/LDA, et retourne un DataFrame de prédictions (voir
    utils.predictions.build_predictions_df pour le schéma)."""
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
        print(
            f"{len(skipped)} spécimen(s) ignoré(s) (nombre de landmarks différent du "
            f"modèle : attendu {model.n_points}) : tps_id={[s.tps_id for s in skipped][:10]}"
            f"{', ...' if len(skipped) > 10 else ''}"
        )

    if not valid:
        raise ValueError(
            "Aucun spécimen du TPS n'a le même nombre de landmarks que le modèle "
            f"({model.n_points}). Vérifier le schéma de landmarks (ex: passer par "
            "reconstruct_tps.py si les landmarks viennent du UNet de Gabriel)."
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
    split_str = f", split={model.split}" if model.split else ""
    print(
        f"Modèle chargé : niveau={model.level}, {len(model.classes)} classes, {model.n_points} landmarks, "
        f"entraîné sur {model.n_train} spécimens{split_str}{devices_str} depuis {model.source_tps}"
    )


def run_batch(args: argparse.Namespace) -> None:
    setup_console_logging()
    model = load_model(args.model_path)
    _print_model_info(model)

    ds_kwargs = dataset_kwargs(args, default_split="test")
    specimens, meta_df = load_dataset(args.dataset, labeled_only=True, **ds_kwargs)
    truth_col = "groupe" if model.level == "caste" else "species"
    truth_by_tps_id = dict(zip((sp.tps_id for sp in specimens), meta_df[truth_col]))

    df = predict_specimens(model, specimens, truth_by_tps_id=truth_by_tps_id)
    acc = accuracy_summary(df, model.level)
    print(f"\nÉvaluation sur {acc['n']} spécimen(s) : top-1 = {acc['accuracy_top1']:.4f} | top-3 = {acc['accuracy_top3']:.4f}")

    run_id = build_run_id(model.level, ds_kwargs["split"], args.devices, args.landmarks_tps, args.run_label)
    out_dir = step_dir(run_id, "predict", family=FAMILY_LDA)

    predictions_path = out_dir / "predictions.csv"
    df.to_csv(predictions_path, index=False)

    metrics = {
        "run_id": run_id,
        "model_path": str(args.model_path),
        "level": model.level,
        "landmarks_source": str(args.landmarks_tps) if args.landmarks_tps else "landmarks_numbered.tps (défaut)",
        **acc,
    }
    write_metrics(out_dir, metrics)
    write_params(out_dir, args, extra={
        "run_id": run_id, "family": FAMILY_LDA, "model_path": str(args.model_path), "resolved_split": ds_kwargs["split"],
    })

    header = f"run_id={run_id} | modèle={args.model_path}"
    log_text = (
        f"{header}\n" + "=" * len(header) + "\n"
        f"Source landmarks : {metrics['landmarks_source']}\n"
        f"Évaluation : top-1 = {acc['accuracy_top1']:.4f} | top-3 = {acc['accuracy_top3']:.4f} (n={acc['n']})\n"
        f"Prédictions -> {predictions_path}\n"
    )
    write_run_log(out_dir, log_text)

    print_predictions_report(df, model.level, args.low_confidence_threshold)
    print(f"\nRun -> {out_dir}")


def run_single(args: argparse.Namespace) -> None:
    setup_console_logging()
    model = load_model(args.model_path)
    _print_model_info(model)

    specimens = load_unlabeled_tps(args.tps_path, strict=not args.non_strict)
    if len(specimens) != 1:
        raise SystemExit(
            f"{args.tps_path} contient {len(specimens)} spécimen(s) -- le mode 'single' attend "
            "exactement une photo (une nouvelle photo terrain pas encore intégrée à data/Bombus). "
            "Pour classer plusieurs spécimens d'un coup, utiliser le mode 'batch'."
        )

    df = predict_specimens(model, specimens)
    row = df.iloc[0]
    pred_col = f"predicted_{model.level}"
    print(
        f"\n{model.level} prédit : {row[pred_col]}  (confiance {row['confidence']:.3f})"
        f"\n  2e choix : {row['second_choice']} ({row['second_confidence']})"
        f"\n  3e choix : {row['third_choice']} ({row['third_confidence']})"
        f"\n  distance de Procrustes à la référence : {row['procrustes_distance']:.4f}"
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"\n-> {args.out}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classifie des spécimens avec un modèle entraîné par train.py"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    batch = subparsers.add_parser("batch", help="Valider le modèle sur un dossier de données (avec vérité connue)")
    batch.add_argument("model_path", type=Path, help="Modèle sauvegardé (ex: data/models/lda/<run_id>/train/model.joblib)")
    batch.add_argument("dataset", type=Path, help="Dossier racine (ex: data/Bombus) -- voir utils.dataset.load_dataset")
    add_dataset_args(batch, default_split="test")
    batch.add_argument("--low-confidence-threshold", type=float, default=0.6,
                        help="Seuil de confiance sous lequel une prédiction est listée pour relecture manuelle (défaut: 0.6)")
    batch.set_defaults(func=run_batch)

    single = subparsers.add_parser("single", help="Classer une seule photo (usage terrain)")
    single.add_argument("model_path", type=Path, help="Modèle sauvegardé")
    single.add_argument("tps_path", type=Path, help="Fichier .tps d'une seule photo")
    single.add_argument("--out", type=Path, default=None, help="Optionnel : sauvegarder aussi le résultat en CSV")
    single.add_argument("--non-strict", action="store_true", help="Tolérer les blocs TPS malformés")
    single.set_defaults(func=run_single)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
