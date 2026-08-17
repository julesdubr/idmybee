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

Note : predict_specimens() fait un *transform* (projection d'un spécimen sur
un modèle déjà figé) -- ce n'est pas le même calcul que run_gpa_pca() dans
lda.py, qui *fit* un consensus GPA et un PCA sur tout un jeu d'entraînement.
Les deux restent volontairement séparés ; seul le chargement du TPS est
partagé (voir utils/dataset.py).

Usage:
    python classifiers/lda.py data/annotations/gabriel_reordered.tps data/manifest/specimens.csv \\
        --level species --save-model out/model_species.joblib
    python classifiers/predict.py out/model_species.joblib data/annotations/nouveaux.tps \\
        --out out/predictions.csv
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
from utils.dataset import load_unlabeled_tps
from utils.gpa import align_to_reference, procrustes_distance, two_d_array
from utils.model_io import TrainedModel, load_model
from utils.tps_io import ImageLandmarks


def predict_specimens(model: TrainedModel, specimens: list[ImageLandmarks]) -> pd.DataFrame:
    """Aligne chaque spécimen sur la référence du modèle, le projette dans
    l'espace PCA/LDA entraîné, et retourne un DataFrame de prédictions
    (une ligne par spécimen dont le nombre de landmarks correspond au modèle)."""
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

    if proba.shape[1] > 1:
        order = np.argsort(-proba, axis=1)
        second_choice = classes[order[:, 1]]
        second_confidence = proba[np.arange(len(valid)), order[:, 1]]
    else:
        second_choice = [None] * len(valid)
        second_confidence = [None] * len(valid)

    return pd.DataFrame({
        "tps_id": [s.tps_id for s in valid],
        "image_id": [s.image_id for s in valid],
        "specimen_id": [s.specimen_id for s in valid],
        "image_path": [s.image_path for s in valid],
        f"predicted_{model.level}": predicted,
        "confidence": confidence,
        "second_choice": second_choice,
        "second_confidence": second_confidence,
        "procrustes_distance": dist_list,
    })


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Classifie de nouveaux spécimens (.tps seul) avec un modèle entraîné par lda.py --save-model"
    )
    parser.add_argument("model_path", type=Path, help="Modèle sauvegardé (ex: out/model_species.joblib)")
    parser.add_argument("tps_path", type=Path, help="Fichier .tps des nouveaux spécimens (sans CSV associé)")
    parser.add_argument("--out", type=Path, default=Path("out/predictions.csv"))
    parser.add_argument("--non-strict", action="store_true", help="Tolérer les blocs TPS malformés")
    parser.add_argument(
        "--low-confidence-threshold", type=float, default=0.6,
        help="Seuil de confiance sous lequel une prédiction est listée pour relecture manuelle (défaut: 0.6)",
    )
    args = parser.parse_args()

    model = load_model(args.model_path)
    print(
        f"Modèle chargé : niveau={model.level}, {len(model.classes)} classes, {model.n_points} landmarks, "
        f"entraîné sur {model.n_train} spécimens"
        f"{' (device=' + model.device + ')' if model.device else ''} depuis {model.source_tps}"
    )

    specimens = load_unlabeled_tps(args.tps_path, strict=not args.non_strict)

    df = predict_specimens(model, specimens)

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