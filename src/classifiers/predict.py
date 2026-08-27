"""predict.py
Classification de spécimens à partir d'un modèle GPA -> PCA -> LDA entraîné
par train.py --save-model. Deux sous-commandes :

  batch  : valider le modèle sur un dossier de données (ex: data/Bombus
           --split test) -- accuracy top-1/top-3 contre la vérité connue,
           CSV de toutes les prédictions. Réutilise utils.dataset.load_dataset
           (mêmes filtres --devices/--species/--castes/--exclude-outliers
           que train.py).
  single : classer UNE photo terrain (un seul bloc TPS, pas encore dans
           data/Bombus -- juste ses landmarks). Pas d'évaluation possible
           (pas de vérité connue) : imprime la prédiction directement,
           pensé pour un usage rapide sur le terrain une fois le modèle
           validé.

Le TPS d'entrée doit avoir le même nombre de landmarks, dans le même ordre /
schéma, que celui utilisé à l'entraînement du modèle. En pratique :
  - landmarks issus du UNet de Gabriel -> passer d'abord par
    reconstruct_tps.py (renumérotation dans le schéma canonique) ;
  - landmarks digitalisés à la main dans le même ordre que Tancrède -> le
    TPS peut être utilisé directement.
Un mauvais schéma de landmarks (mauvais ordre ou nombre de points différent)
donne des prédictions silencieusement fausses sans le détour par
reconstruct_tps.py : seul le nombre de points est vérifié ici, pas l'ordre.

Chaque spécimen aligné reçoit aussi une distance de Procrustes à la forme
de référence du modèle (`procrustes_distance`) : une valeur très supérieure
à celles observées sur le jeu d'entraînement signale une forme atypique, un
problème de landmarks, ou un spécimen hors distribution.

Note : predict_specimens() fait un *transform* (projection d'un spécimen sur
un modèle déjà figé) -- ce n'est pas le même calcul que run_gpa_pca() dans
train.py, qui *fit* un consensus GPA et un PCA sur tout un jeu
d'entraînement. Les deux restent volontairement séparés ; seul le
chargement des données est partagé (voir utils/dataset.py).

Usage:
    python -m classifiers.train data/Bombus --split train --level species --save-model out/model_species.joblib
    python -m classifiers.predict batch out/model_species.joblib data/Bombus --split test --out out/predictions.csv
    python -m classifiers.predict single out/model_species.joblib data/Bombus/landmarks/nouvelle_photo.tps
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parent))
from utils.dataset import load_dataset, load_unlabeled_tps
from utils.gpa import align_to_reference, procrustes_distance, two_d_array
from utils.model_io import TrainedModel, load_model
from utils.tps_io import ImageLandmarks

CHOICE_RANKS = ("second", "third")  # au-delà du top-1 (predicted/confidence)


def predict_specimens(model: TrainedModel, specimens: list[ImageLandmarks]) -> pd.DataFrame:
    """Aligne chaque spécimen sur la référence du modèle, le projette dans
    l'espace PCA/LDA entraîné, et retourne un DataFrame de prédictions
    (une ligne par spécimen dont le nombre de landmarks correspond au modèle).

    Rapporte les 3 meilleurs choix (predicted = top-1, second_choice,
    third_choice) plutôt que le seul top-1 : nécessaire pour l'évaluation
    top-3 de evaluate_predictions(), et utile tel quel en relecture
    manuelle (une confusion fréquente entre deux espèces proches saute aux
    yeux directement dans le CSV)."""
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

    predicted = model.lda.predict(scores)
    proba = model.lda.predict_proba(scores)
    classes = model.lda.classes_
    confidence = proba.max(axis=1)

    order = np.argsort(-proba, axis=1)
    columns = {
        "tps_id": [s.tps_id for s in valid],
        "image_id": [s.image_id for s in valid],
        "specimen_id": [s.specimen_id for s in valid],
        "image_path": [s.image_path for s in valid],
        f"predicted_{model.level}": predicted,
        "confidence": confidence,
    }
    for rank, name in enumerate(CHOICE_RANKS, start=1):  # rank 1 = 2e choix, rank 2 = 3e choix
        if len(classes) > rank:
            idx = order[:, rank]
            columns[f"{name}_choice"] = classes[idx]
            columns[f"{name}_confidence"] = proba[np.arange(len(valid)), idx]
        else:
            columns[f"{name}_choice"] = [None] * len(valid)
            columns[f"{name}_confidence"] = [None] * len(valid)
    columns["procrustes_distance"] = dist_list

    return pd.DataFrame(columns)


def evaluate_predictions(df: pd.DataFrame, truth_by_tps_id: dict[int, str], level: str) -> pd.DataFrame:
    """Compare les prédictions à la vérité connue (`truth_by_tps_id`, construit
    par predict_batch depuis meta_df -- voir load_dataset). Ajoute
    `true_<level>`, `correct_top1`, `correct_top3` à `df`, imprime les deux
    accuracies.

    Top-3 = la vraie espèce est parmi (predicted, second_choice,
    third_choice) -- se dégrade proprement si le modèle a moins de 3
    classes (colonnes second/third_choice à None, jamais égales à la
    vérité, donc jamais comptées comme un hit)."""
    true_col = f"true_{level}"
    df = df.copy()
    df[true_col] = df["tps_id"].map(truth_by_tps_id)

    choice_cols = [f"predicted_{level}"] + [f"{name}_choice" for name in CHOICE_RANKS]
    top1 = df[f"predicted_{level}"] == df[true_col]
    top3 = df.apply(
        lambda row: row[true_col] in {row[c] for c in choice_cols if pd.notna(row[c])}, axis=1
    )
    df["correct_top1"] = top1
    df["correct_top3"] = top3

    print(
        f"\nÉvaluation sur {len(df)} spécimen(s) :"
        f"\n  Top-1 accuracy : {top1.mean():.4f}"
        f"\n  Top-3 accuracy : {top3.mean():.4f}"
    )
    return df


def _print_model_info(model: TrainedModel) -> None:
    devices_str = f", devices={model.devices}" if model.devices else ""
    split_str = f", split={model.split}" if model.split else ""
    print(
        f"Modèle chargé : niveau={model.level}, {len(model.classes)} classes, {model.n_points} landmarks, "
        f"entraîné sur {model.n_train} spécimens{split_str}{devices_str} depuis {model.source_tps}"
    )


def _report_predictions(df: pd.DataFrame, level: str, low_confidence_threshold: float) -> None:
    print(f"\nRépartition des prédictions ({level}) :")
    print(df[f"predicted_{level}"].value_counts())
    print(
        f"\nConfiance moyenne : {df['confidence'].mean():.3f} "
        f"(min={df['confidence'].min():.3f}, max={df['confidence'].max():.3f})"
    )
    low_conf = df[df["confidence"] < low_confidence_threshold]
    if len(low_conf):
        cols = ["tps_id", "specimen_id", "image_path", f"predicted_{level}", "confidence", "second_choice"]
        print(
            f"\n{len(low_conf)} prédiction(s) sous le seuil de confiance "
            f"({low_confidence_threshold}) -- à vérifier manuellement :"
        )
        print(low_conf[cols].to_string(index=False))


def run_batch(args: argparse.Namespace) -> None:
    model = load_model(args.model_path)
    _print_model_info(model)

    specimens, meta_df = load_dataset(
        args.dataset, split=args.split, devices=args.devices, species=args.species,
        castes=args.castes, exclude_outliers=args.exclude_outliers, strict=not args.non_strict,
    )
    truth_col = "groupe" if model.level == "caste" else "species"
    truth_by_tps_id = dict(zip((sp.tps_id for sp in specimens), meta_df[truth_col]))

    df = predict_specimens(model, specimens)
    df = evaluate_predictions(df, truth_by_tps_id, model.level)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\n{len(df)} prédiction(s) -> {args.out}")

    _report_predictions(df, model.level, args.low_confidence_threshold)


def run_single(args: argparse.Namespace) -> None:
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
        description="Classifie des spécimens avec un modèle entraîné par train.py --save-model"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    batch = subparsers.add_parser("batch", help="Valider le modèle sur un dossier de données (avec vérité connue)")
    batch.add_argument("model_path", type=Path, help="Modèle sauvegardé (ex: out/model_species.joblib)")
    batch.add_argument("dataset", type=Path, help="Dossier racine (ex: data/Bombus) -- voir utils.dataset.load_dataset")
    batch.add_argument("--split", type=str, default="test",
                        help="Valeur de la colonne 'dataset' de manifest.csv à garder (défaut: test).")
    batch.add_argument("--devices", type=str, nargs="+", default=None, help="Ne garder que ces photos (ex: P1 S1).")
    batch.add_argument("--species", type=str, nargs="+", default=None, help="Ne garder que ces espèces.")
    batch.add_argument("--castes", type=str, nargs="+", default=None, help="Ne garder que ces castes.")
    batch.add_argument("--exclude-outliers", action="store_true", help="Exclut les photos SUSPECT/FAILED.")
    batch.add_argument("--out", type=Path, default=Path("out/predictions.csv"))
    batch.add_argument("--non-strict", action="store_true", help="Tolérer les blocs TPS malformés")
    batch.add_argument("--low-confidence-threshold", type=float, default=0.6,
                        help="Seuil de confiance sous lequel une prédiction est listée pour relecture manuelle (défaut: 0.6)")
    batch.set_defaults(func=run_batch)

    single = subparsers.add_parser("single", help="Classer une seule photo (usage terrain)")
    single.add_argument("model_path", type=Path, help="Modèle sauvegardé (ex: out/model_species.joblib)")
    single.add_argument("tps_path", type=Path, help="Fichier .tps d'une seule photo")
    single.add_argument("--out", type=Path, default=None, help="Optionnel : sauvegarder aussi le résultat en CSV")
    single.add_argument("--non-strict", action="store_true", help="Tolérer les blocs TPS malformés")
    single.set_defaults(func=run_single)

    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()