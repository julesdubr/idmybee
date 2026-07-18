"""Transcription Python du pipeline R d'Adrien : GPA -> PCA -> LDA (LOOCV).

Équivalences avec le script R :

    R                                                    Python (ici)
    ----------------------------------------------------  ------------------------------------
    readland.tps(...)                                     utils.tps_parser.parse_tps(...)
    gpagen(N2)                                             gpa.gpagen(...)
    two.d.array(gpaN2$coords)                              gpa.two_d_array(...)
    prcomp(...)$x[, 1:34]                                  sklearn PCA, n_components = 2p - 4
    lda(pcaN2, DataP1$Groupe, CV=TRUE)                      LeaveOneOut + LinearDiscriminantAnalysis
    table(DataP1$Groupe, ldaN2$class)                       sklearn.metrics.confusion_matrix

Le nombre de composantes PCA conservées (2p - 4, où p = nombre de
landmarks) correspond exactement aux degrés de liberté restants après
GPA en 2D : 2 pour la translation, 1 pour l'échelle, 1 pour la rotation
sont retirés de l'espace des coordonnées brutes (2p dimensions). Pour
p=19 (comme dans le script d'Adrien), cela donne bien 34 -- la valeur
utilisée en dur dans le R.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneOut, cross_val_predict

from gpa_old import gpagen, two_d_array
from utils.tps_parser import Specimen, parse_tps

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
    required_cols = {"id", "image", "espece", "caste", "device"}
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


def loocv_lda(scores: np.ndarray, groupe: pd.Series) -> tuple[np.ndarray, float, pd.DataFrame]:
    """LDA avec validation croisée leave-one-out, équivalent de lda(..., CV=TRUE).

    Retourne (predictions, accuracy, confusion_matrix_df).
    """
    clf = LinearDiscriminantAnalysis()
    loo = LeaveOneOut()
    predicted = cross_val_predict(clf, scores, groupe, cv=loo)

    labels = sorted(groupe.unique())
    cm = confusion_matrix(groupe, predicted, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    accuracy = float(np.trace(cm) / cm.sum())
    return predicted, accuracy, cm_df


def fit_lda(scores: np.ndarray, groupe: pd.Series, n_components: int = 2):
    """Ajuste une LDA sur les composantes PCA et retourne la projection."""
    n_components = min(n_components, len(np.unique(groupe)) - 1)
    if n_components < 1:
        raise ValueError("LDA nécessite au moins deux classes pour calculer une projection.")
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    projection = lda.fit_transform(scores, groupe)
    return lda, projection


def plot_gpa_alignment(gpa_result, groupe: pd.Series, title: str = "Résultat de la GPA"):
    """Affiche les formes alignées par GPA en 2D, coloriées par groupe."""
    aligned = np.asarray(gpa_result.aligned)
    if aligned.ndim != 3 or aligned.shape[2] != 2:
        raise ValueError("gpa_result.aligned doit être un tableau de forme (n_specimens, n_points, 2).")

    unique_groups = sorted(groupe.unique())
    cmap = plt.get_cmap("tab20")

    plt.figure(figsize=(10, 8))
    for i, group in enumerate(unique_groups):
        mask = groupe == group
        specimens_coords = aligned[mask]

        if specimens_coords.size == 0:
            continue

        group_coords = np.vstack(specimens_coords)
        color = cmap(i % 10)
        plt.scatter(
            group_coords[:, 0],
            group_coords[:, 1],
            color=color,
            alpha=0.25,
            s=15,
            label=group,
        )

    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("Coordonnée X")
    plt.ylabel("Coordonnée Y")
    plt.title(title)
    plt.legend(title="Espèce", loc="best", fontsize="small")
    plt.tight_layout()
    plt.savefig("out/gpa_alignment.png", dpi=300)


def plot_lda(lda_scores: np.ndarray, groupe: pd.Series, title: str = "Projection LDA"):
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
    plt.savefig("out/lda_projection.png", dpi=300)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="GPA -> PCA -> LDA (LOOCV) sur landmarks de bourdons")
    parser.add_argument("tps_path", type=Path, help="Fichier .tps de référence (ex: Nest2_mappedDig2.tps)")
    parser.add_argument("csv_path", type=Path, help="CSV associé (id, image, espece, caste, device)")
    parser.add_argument("--precision", type=str, default="espece")
    parser.add_argument("--non-strict", action="store_true", help="Tolérer les blocs TPS malformés")
    args = parser.parse_args()

    specimens, meta_df = load_labeled_dataset(args.tps_path, args.csv_path, strict=not args.non_strict)
    scores, gpa_result, pca = run_gpa_pca(specimens)

    plot_gpa_alignment(gpa_result, meta_df[args.precision], title="Résultat de la GPA")

    _, lda_projection = fit_lda(scores, meta_df[args.precision])
    plot_lda(lda_projection, meta_df[args.precision], title="Projection LDA des spécimens")

    predicted, accuracy, cm_df = loocv_lda(scores, meta_df[args.precision])

    print("\nMatrice de confusion (lignes = vérité terrain, colonnes = prédiction LOOCV) :")
    print(cm_df)
    print(f"\nProportion bien classée (LOOCV) : {accuracy:.4f}")


if __name__ == "__main__":
    main()