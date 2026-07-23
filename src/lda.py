"""Transcription Python du pipeline R : GPA -> PCA -> LDA (LOOCV)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut, cross_val_predict

from utils.gpa import gpagen, two_d_array
from utils.model_io import TrainedModel, save_model
from utils.tps_io import Specimen, parse_tps
from utils import reporting

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def load_labeled_dataset(
    tps_path: str | Path, csv_path: str | Path, strict: bool = True
) -> tuple[list[Specimen], pd.DataFrame]:
    """Charge le TPS de référence + le CSV de métadonnées et les aligne par id/sid.

    Équivalent de la lecture de MyExcelFile + readland.tps(specID="imageID")
    dans le script R, mais la jointure ici se fait explicitement sur
    id (CSV) == sid (TPS ID=), au lieu de compter sur un ordre identique
    des deux fichiers.
    """
    specimens, errors = parse_tps(tps_path, strict=strict)
    if errors:
        logger.warning("%d erreur(s) de parsing TPS (voir ci-dessus)", len(errors))

    meta = pd.read_csv(csv_path)
    required_cols = {"id", "espece", "caste"}
    missing = required_cols - set(meta.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV : {missing}")

    meta = meta.set_index("id", drop=False)

    kept_specimens: list[Specimen] = []
    kept_rows: list[pd.Series] = []
    unmatched: list[int] = []
    for sp in specimens:
        if sp.sid not in meta.index:
            unmatched.append(sp.sid)
            continue
        kept_specimens.append(sp)
        kept_rows.append(meta.loc[sp.sid])

    if unmatched:
        logger.warning(
            "%d spécimen(s) du TPS sans entrée CSV correspondante (id manquants: %s%s)",
            len(unmatched),
            unmatched[:10],
            ", ..." if len(unmatched) > 10 else "",
        )

    meta_df = pd.DataFrame(kept_rows).reset_index(drop=True)
    meta_df["groupe"] = meta_df["espece"].astype(str) + "_" + meta_df["caste"].astype(str)

    logger.info(
        "%d spécimens appariés TPS<->CSV sur %d landmarks dans le TPS",
        len(kept_specimens),
        len(specimens),
    )
    return kept_specimens, meta_df


def run_gpa_pca(specimens: list[Specimen]):
    """GPA puis PCA, retourne (pca_scores, gpa_result, pca_model)."""
    n_points = specimens[0].n_points
    for sp in specimens:
        if sp.n_points != n_points:
            raise ValueError(
                f"Landmark count incohérent : spécimen sid={sp.sid} a {sp.n_points} points, "
                f"attendu {n_points}. Tous les spécimens doivent partager le même schéma de landmarks."
            )

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

    # plt.figure(figsize=(10, 5))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,10), sharex=True)

    for i, group in enumerate(unique_groups):
        mask = groupe == group
        specimens_coords = aligned[mask]

        if specimens_coords.size == 0:
            continue

        group_coords = np.vstack(specimens_coords)
        color = cmap(i % 20)
        ax1.scatter(
            group_coords[:, 0],
            group_coords[:, 1],
            color=color,
            alpha=0.05,
            label=group,
            s=10
        )
        ax1.set_xlabel("Coordonnée X")
        ax1.set_ylabel("Coordonnée Y")
        ax1.set_title(f"{title} (all)")
        ax1.legend(title="Espèce", loc="best", fontsize="small")
        ax1.set_aspect("equal", adjustable="box")

        centroids = specimens_coords.mean(axis=0)
        ax2.scatter(
            centroids[:, 0],
            centroids[:, 1],
            color=color,
            s=20,
            label=group,
            marker='x'
        )
        ax2.set_xlabel("Coordonnée X")
        ax2.set_ylabel("Coordonnée Y")
        ax2.set_title(f"{title} (centroids)")
        ax2.legend(title="Espèce", loc="best", fontsize="small")
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
        plt.scatter(
            x[mask],
            y[mask],
            label=group,
            color=colors[i],
            alpha=0.25,
            s=20,
        )

    plt.xlabel("LDA 1")
    plt.ylabel("LDA 2" if lda_scores.shape[1] > 1 else "Constante")
    plt.title(title)
    plt.legend(title="Espèce", loc="best", fontsize="small")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Projection LDA -> {out_path}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    Path("out").mkdir(exist_ok=True)

    parser = argparse.ArgumentParser(description="GPA -> PCA -> LDA (LOOCV) sur landmarks de bourdons")
    parser.add_argument("tps_path", type=Path, help="Fichier .tps de référence (ex: Nest2_mappedDig2.tps)")
    parser.add_argument("csv_path", type=Path, help="CSV associé (id, image, espece, caste, device)")
    parser.add_argument("--level", type=str, default="espece")
    parser.add_argument("--non-strict", action="store_true", help="Tolérer les blocs TPS malformés")
    parser.add_argument("--exclude-ids", type=Path, default=None,
                         help="CSV avec colonne 'id' (et 'heavy' optionnelle) des spécimens à exclure, "
                              "ex: out/outlier_specimens.csv produit par tools/flag_outlier_specimens.py")
    parser.add_argument("--device", type=str, default=None,
                         help="Ne garder que les spécimens de cet appareil (ex: S1). "
                              "Omis = analyse complète (tous appareils, résumé inclut les scores par appareil).")
    parser.add_argument("--save-model", type=Path, default=None,
                         help="Chemin où sauvegarder le modèle entraîné (GPA+PCA+LDA), "
                              "ex: out/model_espece_S1.joblib. Utilisable ensuite par predict.py "
                              "pour classer de nouveaux spécimens sans données biologiques (.tps seul).")
    args = parser.parse_args()

    specimens, meta_df = load_labeled_dataset(args.tps_path, args.csv_path, strict=not args.non_strict)

    if args.exclude_ids:
        exclude_df = pd.read_csv(args.exclude_ids)
        if "heavy" in exclude_df.columns:
            exclude_df = exclude_df[exclude_df["heavy"]]
        exclude_set = set(exclude_df["id"])
        keep_mask = [sp.sid not in exclude_set for sp in specimens]
        specimens = [sp for sp, keep in zip(specimens, keep_mask) if keep]
        meta_df = meta_df[keep_mask].reset_index(drop=True)
        print(f"Exclus {sum(not k for k in keep_mask)} spécimen(s) via {args.exclude_ids}")

    if args.device:
        keep_mask = (meta_df["device"] == args.device).tolist()
        specimens = [sp for sp, keep in zip(specimens, keep_mask) if keep]
        meta_df = meta_df[keep_mask].reset_index(drop=True)
        print(f"Filtré sur device={args.device} : {len(specimens)} spécimen(s) restants")

    # Suffixe unique par run (niveau + appareil), pour que deux runs successifs
    # (espece vs caste, ou --device S1 vs S2) n'écrasent pas leurs résultats.
    tag = args.level + (f"_{args.device}" if args.device else "")
    out_dir = Path(f"../out/{tag}")

    out_dir.mkdir(exist_ok=True, parents=True)
    
    scores, gpa_result, pca = run_gpa_pca(specimens)
    n_components = scores.shape[1]

    plot_gpa_alignment(gpa_result, meta_df[args.level], out_dir / f"gpa_alignment_{tag}.png",
                        title=f"Résultat de la GPA -- {tag}")

    lda_final, lda_projection = fit_lda(scores, meta_df[args.level])
    plot_lda(lda_projection, meta_df[args.level], out_dir / f"lda_projection_{tag}.png",
              title=f"Projection LDA des spécimens -- {tag}")

    predicted, accuracy = loocv_lda(scores, meta_df[args.level])
    cm_df = reporting.confusion_matrix_df(meta_df[args.level], predicted)
    group_table = reporting.group_summary_table(args.level, meta_df, gpa_result, predicted)
    device_table = reporting.device_summary_table(meta_df, args.level, predicted, gpa_result)

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
            classes=sorted(meta_df[args.level].unique()),
            device=args.device,
            source_tps=str(args.tps_path),
            n_train=len(specimens),
        )
        save_model(model, args.save_model)


if __name__ == "__main__":
    main()