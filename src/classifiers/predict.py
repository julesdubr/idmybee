"""predict.py
Classification de nouveaux spécimens (.tps seul, sans CSV de métadonnées
biologiques) à partir d'un modèle GPA -> PCA -> LDA entraîné par lda.py.

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

--specimens-csv (optionnel) permet d'évaluer le modèle : si certains
spécimens du TPS d'entrée s'avèrent déjà labellisés (is_labeled=True) dans
ce CSV -- ex: en prédisant sur landmarks_numbered_all.tps, ou sur un lot
qui vient de recevoir une détermination -- leur vraie espèce est comparée
aux prédictions (top-1 et top-3 accuracy), sans que cela n'exclue les
spécimens non labellisés de la prédiction elle-même.

Note : predict_specimens() fait un *transform* (projection d'un spécimen sur
un modèle déjà figé) -- ce n'est pas le même calcul que run_gpa_pca() dans
lda.py, qui *fit* un consensus GPA et un PCA sur tout un jeu d'entraînement.
Les deux restent volontairement séparés ; seul le chargement du TPS est
partagé (voir utils/dataset.py).

Usage:
    python -m classifiers.lda data/annotations/landmarks_numbered_labeled.tps data/manifest/specimens.csv \\
        --level species --save-model out/model_species.joblib
    python -m classifiers.predict out/model_species.joblib data/annotations/landmarks_numbered_unlabeled.tps \\
        --out out/predictions.csv
    # évaluation sur un TPS partiellement labellisé :
    python -m classifiers.predict out/model_species.joblib data/annotations/landmarks_numbered_all.tps \\
        --specimens-csv data/manifest/specimens.csv --out out/predictions_all.csv
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
from utils.dataset import add_groupe_column, load_unlabeled_tps
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
    top-3 de evaluate_against_truth(), et utile tel quel en relecture
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


def evaluate_against_truth(df: pd.DataFrame, model: TrainedModel, specimens_csv: Path) -> pd.DataFrame:
    """Compare les prédictions à la vérité terrain pour les spécimens de
    `df` qui s'avèrent labellisés (is_labeled=True) dans specimens_csv.
    Ajoute `true_<level>` (NaN si non labellisé), `correct_top1`,
    `correct_top3` (NaN si non évalué) à `df`, imprime les deux accuracies.

    Top-3 = la vraie espèce est parmi (predicted, second_choice,
    third_choice) -- se dégrade proprement si le modèle a moins de 3
    classes (colonnes second/third_choice à None, jamais égales à la
    vérité, donc jamais comptées comme un hit -- correct sans cas
    particulier à écrire)."""
    truth_df = pd.read_csv(specimens_csv)
    required = {"specimen_id", "species", "caste", "is_labeled"}
    missing = required - set(truth_df.columns)
    if missing:
        raise SystemExit(f"Colonnes manquantes dans {specimens_csv} : {missing}")

    truth_df = truth_df[truth_df["is_labeled"].astype(bool)]
    truth_df = add_groupe_column(truth_df)
    truth_col = "groupe" if model.level == "caste" else "species"
    truth_by_id = dict(zip(truth_df["specimen_id"], truth_df[truth_col]))

    pred_col = f"predicted_{model.level}"
    true_col = f"true_{model.level}"
    df = df.copy()
    df[true_col] = df["specimen_id"].map(truth_by_id)

    evaluated = df[df[true_col].notna()]
    if evaluated.empty:
        print(f"\nAucun spécimen labellisé (is_labeled=True dans {specimens_csv}) parmi les prédictions "
              "-- pas d'évaluation possible.")
        return df

    choice_cols = [pred_col] + [f"{name}_choice" for name in CHOICE_RANKS]
    top1 = evaluated[pred_col] == evaluated[true_col]
    top3 = evaluated.apply(
        lambda row: row[true_col] in {row[c] for c in choice_cols if pd.notna(row[c])}, axis=1
    )
    df.loc[evaluated.index, "correct_top1"] = top1
    df.loc[evaluated.index, "correct_top3"] = top3

    print(
        f"\nÉvaluation sur {len(evaluated)}/{len(df)} spécimen(s) déjà labellisé(s) parmi les prédictions :"
        f"\n  Top-1 accuracy : {top1.mean():.4f}"
        f"\n  Top-3 accuracy : {top3.mean():.4f}"
    )
    return df


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classifie de nouveaux spécimens (.tps seul) avec un modèle entraîné par lda.py --save-model"
    )
    parser.add_argument("model_path", type=Path, help="Modèle sauvegardé (ex: out/model_species.joblib)")
    parser.add_argument("tps_path", type=Path, help="Fichier .tps des spécimens à classer")
    parser.add_argument("--specimens-csv", type=Path, default=None,
                         help="data/manifest/specimens.csv -- optionnel. Si fourni, évalue le modèle "
                              "(top-1/top-3 accuracy) sur les spécimens du TPS déjà labellisés "
                              "(is_labeled=True), sans changer les prédictions des autres.")
    parser.add_argument("--out", type=Path, default=Path("out/predictions.csv"))
    parser.add_argument("--non-strict", action="store_true", help="Tolérer les blocs TPS malformés")
    parser.add_argument(
        "--low-confidence-threshold", type=float, default=0.6,
        help="Seuil de confiance sous lequel une prédiction est listée pour relecture manuelle (défaut: 0.6)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args = build_arg_parser().parse_args(argv)

    model = load_model(args.model_path)
    print(
        f"Modèle chargé : niveau={model.level}, {len(model.classes)} classes, {model.n_points} landmarks, "
        f"entraîné sur {model.n_train} spécimens"
        f"{' (device=' + model.device + ')' if model.device else ''}"
        f"{' (dataset=' + model.dataset + ')' if model.dataset else ''} depuis {model.source_tps}"
    )

    specimens = load_unlabeled_tps(args.tps_path, strict=not args.non_strict)

    df = predict_specimens(model, specimens)
    if args.specimens_csv:
        df = evaluate_against_truth(df, model, args.specimens_csv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\n{len(df)} prédiction(s) -> {args.out}")

    print(f"\nRépartition des prédictions ({model.level}) :")
    print(df[f"predicted_{model.level}"].value_counts())

    print(
        f"\nConfiance moyenne : {df['confidence'].mean():.3f} "
        f"(min={df['confidence'].min():.3f}, max={df['confidence'].max():.3f})"
    )

    low_conf = df[df["confidence"] < args.low_confidence_threshold]
    if len(low_conf):
        cols = ["tps_id", "specimen_id", "image_path", f"predicted_{model.level}", "confidence", "second_choice"]
        print(
            f"\n{len(low_conf)} prédiction(s) sous le seuil de confiance "
            f"({args.low_confidence_threshold}) -- à vérifier manuellement :"
        )
        print(low_conf[cols].to_string(index=False))


if __name__ == "__main__":
    main()