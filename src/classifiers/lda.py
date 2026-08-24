"""lda.py
GPA -> PCA -> LDA (LOOCV), à partir d'un TPS de landmarks et de
data/manifest/specimens.csv (specimen_id, species, caste).

--level species -> discrimine par espèce.
--level caste   -> discrimine par (species, caste) : la caste seule
mélangerait des espèces différentes sous un même label "worker"/"queen"/
"male", donc le vrai groupe utilisé est "species_caste" (voir utils.dataset
.target_groupe).

--dataset restreint l'entraînement à un jeu de données d'origine (ex:
organized, le plus fiable car identifié par Adrien -- voir specimens.csv
'datasets_present') : pratique pour comparer un modèle entraîné sur les
données les plus propres à un modèle entraîné sur tout, ou pour exporter un
modèle dédié à un sous-ensemble précis via --save-model.

Usage:
    python -m classifiers.lda data/annotations/landmarks_numbered_labeled.tps data/manifest/specimens.csv \\
        --level species --exclude-ids data/manifest/landmarks_numbered.csv --save-model out/model_species.joblib
"""
from __future__ import annotations

import argparse
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut, cross_val_predict

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parent))
from utils.dataset import load_labeled_dataset, apply_filters, target_groupe
from utils.gpa import gpagen, two_d_array
from utils.model_io import TrainedModel, save_model
from utils.tps_io import ImageLandmarks
from analysis import reporting

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def run_gpa_pca(specimens: list[ImageLandmarks]):
    """GPA puis PCA, retourne (pca_scores, gpa_result, pca_model).

    Suppose un nombre de landmarks homogène -- déjà garanti par
    utils.dataset.load_labeled_dataset, qui écarte automatiquement les
    schémas incohérents avant que les données n'arrivent ici."""
    if not specimens:
        raise ValueError(
            "Aucun spécimen à traiter (liste vide après chargement/filtrage) -- vérifier "
            "--exclude-ids/--device, ou que le TPS contient des landmarks valides."
        )
    n_points = specimens[0].n_points
    gpa_result = gpagen([sp.landmarks for sp in specimens])
    X = two_d_array(gpa_result.aligned)  # (n_specimens, 2*n_points)

    n_components = 2 * n_points - 4  # ddl restants après GPA en 2D
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X)

    logger.info(
        "PCA : %d composantes conservées sur %d spécimens (variance expliquée cumulée = %.1f%%)",
        n_components,
        len(specimens),
        100 * pca.explained_variance_ratio_.sum(),
    )
    return scores, gpa_result, pca


def loocv_lda(scores: np.ndarray, groupe: pd.Series) -> tuple[np.ndarray, float]:
    """LDA avec validation croisée leave-one-out, équivalent de lda(..., CV=TRUE).

    Retourne (predictions, accuracy). La matrice de confusion et le reste du
    résumé sont construits séparément par utils.reporting à partir de ces
    prédictions.
    """
    clf = LinearDiscriminantAnalysis()
    loo = LeaveOneOut()
    predicted = cross_val_predict(clf, scores, groupe, cv=loo)
    accuracy = float(np.mean(np.asarray(predicted) == groupe.values))
    return predicted, accuracy


def fit_lda(scores: np.ndarray, groupe: pd.Series, n_components: int = 2):
    """Ajuste une LDA sur les composantes PCA et retourne la projection."""
    n_components = min(n_components, len(np.unique(groupe)) - 1)
    if n_components < 1:
        raise ValueError("LDA nécessite au moins deux classes pour calculer une projection.")
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    projection = lda.fit_transform(scores, groupe)
    return lda, projection


def plot_gpa_alignment(gpa_result, groupe: pd.Series, out_path: Path,
                        title: str = "Résultat de la GPA"):
    """Affiche les formes alignées par GPA en 2D, coloriées par groupe."""
    aligned = np.asarray(gpa_result.aligned)
    if aligned.ndim != 3 or aligned.shape[2] != 2:
        raise ValueError("gpa_result.aligned doit être un tableau de forme (n_specimens, n_points, 2).")

    unique_groups = sorted(groupe.unique())
    cmap = plt.get_cmap("tab20")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    for i, group in enumerate(unique_groups):
        mask = groupe == group
        specimens_coords = aligned[mask]

        if specimens_coords.size == 0:
            continue

        group_coords = np.vstack(specimens_coords)
        color = cmap(i % 20)
        ax1.scatter(group_coords[:, 0], group_coords[:, 1], color=color, alpha=0.05, label=group, s=10)
        ax1.set_xlabel("Coordonnée X")
        ax1.set_ylabel("Coordonnée Y")
        ax1.set_title(f"{title} (all)")
        ax1.legend(title="Groupe", loc="best", fontsize="small")
        ax1.set_aspect("equal", adjustable="box")

        centroids = specimens_coords.mean(axis=0)
        ax2.scatter(centroids[:, 0], centroids[:, 1], color=color, s=20, label=group, marker='x')
        ax2.set_xlabel("Coordonnée X")
        ax2.set_ylabel("Coordonnée Y")
        ax2.set_title(f"{title} (centroids)")
        ax2.legend(title="Groupe", loc="best", fontsize="small")
        ax2.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"GPA alignment -> {out_path}")


def plot_lda(lda_scores: np.ndarray, groupe: pd.Series, out_path: Path,
             title: str = "Projection LDA"):
    """Trace la projection LDA sur deux axes et annote chaque point avec son groupe."""
    if lda_scores.shape[1] == 1:
        x = lda_scores[:, 0]
        y = np.zeros_like(x)
    else:
        x, y = lda_scores[:, 0], lda_scores[:, 1]

    plt.figure(figsize=(10, 8))
    unique_groups = sorted(groupe.unique())

    if len(unique_groups) > 20:
        cmap_combined = np.vstack([plt.cm.tab20c(np.linspace(0, 1, 20)), plt.cm.tab20b(np.linspace(0, 1, 20))])
        colors = cmap_combined[:len(unique_groups)]
    else:
        colors = [plt.cm.tab20(i) for i in range(len(unique_groups))]

    for i, group in enumerate(unique_groups):
        mask = groupe == group
        plt.scatter(x[mask], y[mask], label=group, color=colors[i], alpha=0.25, s=20)

    plt.xlabel("LDA 1")
    plt.ylabel("LDA 2" if lda_scores.shape[1] > 1 else "Constante")
    plt.title(title)
    plt.legend(title="Groupe", loc="best", fontsize="small")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Projection LDA -> {out_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GPA -> PCA -> LDA (LOOCV) sur landmarks de bourdons")
    parser.add_argument("tps_path", type=Path, help="Fichier .tps de landmarks labellisés (ex: landmarks_numbered_labeled.tps)")
    parser.add_argument("csv_path", type=Path, help="data/manifest/specimens.csv (specimen_id, species, caste, ...)")
    parser.add_argument("--images-csv", type=Path, default=None,
                         help="data/manifest/images.csv (image_id, specimen_id, device_type, ...). "
                              "Nécessaire pour --device (device_type est une info par photo, pas par "
                              "spécimen, jamais dans specimens.csv seul).")
    parser.add_argument("--level", type=str, default="species", choices=["species", "caste"],
                         help="'species' : discrimination par espèce. 'caste' : discrimination par "
                              "(espèce, caste) -- voir utils.dataset.target_groupe.")
    parser.add_argument("--non-strict", action="store_true", help="Tolérer les blocs TPS malformés")
    parser.add_argument("--exclude-ids", type=Path, nargs="+", default=None,
                         help="Un ou plusieurs CSV avec colonnes 'tps_id'+'status' des spécimens à "
                              "exclure (SUSPECT/FAILED) -- typiquement data/manifest/landmarks_numbered.csv.")
    parser.add_argument("--device", type=str, default=None,
                         help="Ne garder que les spécimens de cet appareil (ex: S). Nécessite --images-csv.")
    parser.add_argument("--dataset", type=str, default=None,
                         help="Ne garder que les spécimens d'un jeu de données d'origine "
                              "(ex: organized, terrain, vrac, basile_m1 -- voir specimens.csv "
                              "'datasets_present').")
    parser.add_argument("--exclude-species", type=str, nargs="+", default=None,
                         help="Exclut entièrement une ou plusieurs espèces (ex: --exclude-species "
                              "rupestris ruderarius) -- utile le temps d'investiguer un problème de "
                              "numérotation propre à une espèce, sans construire de CSV --exclude-ids.")
    parser.add_argument("--save-model", type=Path, default=None,
                         help="Chemin où sauvegarder le modèle entraîné (GPA+PCA+LDA), "
                              "ex: out/model_species.joblib. Utilisable ensuite par predict.py "
                              "pour classer de nouveaux spécimens sans données biologiques (.tps seul).")
    parser.add_argument("--out-dir", type=Path, default=Path("out"),
                         help="Dossier de sortie racine (plots, modèle, résumés) -- défaut: out/, "
                              "relatif au répertoire d'exécution.")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    Path("out").mkdir(exist_ok=True)

    args = build_arg_parser().parse_args(argv)

    specimens, meta_df = load_labeled_dataset(
        args.tps_path, args.csv_path, images_csv=args.images_csv, strict=not args.non_strict
    )
    specimens, meta_df = apply_filters(
        specimens, meta_df, args.exclude_ids, device=args.device, dataset=args.dataset,
        exclude_species=args.exclude_species,
    )
    groupe = target_groupe(meta_df, args.level)

    # Suffixe unique par run (niveau + appareil + dataset), pour que deux runs
    # successifs (species vs caste, --device S vs P, --dataset organized vs
    # tout) n'écrasent pas leurs résultats.
    tag = args.level + (f"_{args.device}" if args.device else "") + (f"_{args.dataset}" if args.dataset else "")
    out_dir = args.out_dir / tag
    out_dir.mkdir(exist_ok=True, parents=True)

    scores, gpa_result, pca = run_gpa_pca(specimens)
    n_components = scores.shape[1]

    plot_gpa_alignment(gpa_result, groupe, out_dir / f"gpa_alignment_{tag}.png",
                        title=f"Résultat de la GPA -- {tag}")

    lda_final, lda_projection = fit_lda(scores, groupe)
    plot_lda(lda_projection, groupe, out_dir / f"lda_projection_{tag}.png",
              title=f"Projection LDA des spécimens -- {tag}")

    predicted, accuracy = loocv_lda(scores, groupe)
    cm_df = reporting.confusion_matrix_df(groupe, predicted)
    group_table = reporting.group_summary_table(args.level, groupe, gpa_result, predicted)
    device_table = reporting.device_summary_table(meta_df, groupe, predicted, gpa_result)

    reporting.print_summary(out_dir, args.level, meta_df, gpa_result, pca, n_components,
                             accuracy, cm_df, group_table, device_table, args.device)
    reporting.write_summary_csv(out_dir, args.level, tag, group_table, device_table, cm_df)
    reporting.plot_confusion_matrix(cm_df, out_dir / f"confusion_matrix_{tag}.png",
                                     title=f"Matrice de confusion -- {tag}")

    if args.save_model:
        model = TrainedModel(
            mean_shape=gpa_result.mean_shape,
            n_points=specimens[0].n_points,
            pca=pca,
            lda=lda_final,
            level=args.level,
            classes=sorted(groupe.unique()),
            device=args.device,
            dataset=args.dataset,
            source_tps=str(args.tps_path),
            n_train=len(specimens),
        )
        save_model(model, args.save_model)


if __name__ == "__main__":
    main()