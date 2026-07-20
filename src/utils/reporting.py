"""Rapport de résultats pour lda.py : tableaux par groupe/appareil, matrice
de confusion (texte + heatmap), affichage terminal et écriture des CSV.

Toute la présentation vit ici pour que lda.py reste concentré sur le calcul
(GPA -> PCA -> LDA) ; ce module ne fait que résumer/afficher/dessiner des
résultats déjà calculés.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda v: f"{v:.4f}")


def shape_variance_by_group(aligned: np.ndarray, groupe: pd.Series) -> pd.Series:
    """Variance de Procrustes par groupe : distance quadratique moyenne
    de chaque spécimen au centroïde de forme de son groupe."""
    X = aligned.reshape(len(aligned), -1)
    out = {}
    for g in groupe.unique():
        mask = (groupe == g).values
        centroid = X[mask].mean(axis=0)
        out[g] = float(((X[mask] - centroid) ** 2).sum(axis=1).mean())
    return pd.Series(out, name="shape_variance")


def confusion_matrix_df(truth: pd.Series, predicted: np.ndarray) -> pd.DataFrame:
    labels = sorted(truth.unique())
    cm = confusion_matrix(truth, predicted, labels=labels)
    return pd.DataFrame(cm, index=labels, columns=labels)


def group_summary_table(level: str, meta_df: pd.DataFrame, gpa_result, predicted: np.ndarray) -> pd.DataFrame:
    """Par valeur du niveau classé (espece ou caste) : n, precision/recall/f1
    (LOOCV) et variance de forme intra-groupe."""
    groupe = meta_df[level]
    labels = sorted(groupe.unique())

    report = classification_report(groupe, predicted, labels=labels, output_dict=True, zero_division=0)
    scores = pd.DataFrame(report).T.loc[labels, ["precision", "recall", "f1-score"]]
    scores.columns = ["precision", "recall", "f1"]

    variance = shape_variance_by_group(gpa_result.aligned, groupe)
    n = groupe.value_counts()

    table = pd.DataFrame({
        "n": n.reindex(labels).astype(int),
        "precision": scores["precision"],
        "recall": scores["recall"],
        "f1": scores["f1"],
        "shape_variance": variance.reindex(labels),
    })
    table.index.name = level
    return table


def device_summary_table(meta_df: pd.DataFrame, level: str, predicted: np.ndarray, gpa_result) -> pd.DataFrame | None:
    """Par appareil : n, accuracy LOOCV (au niveau classé) et variance de
    forme intra-appareil. None si un seul appareil (filtré via --device :
    la ventilation par appareil serait triviale)."""
    if "device" not in meta_df.columns or meta_df["device"].nunique() <= 1:
        return None

    groupe = meta_df[level]
    device = meta_df["device"]
    wrong = np.asarray(predicted) != groupe.values
    variance = shape_variance_by_group(gpa_result.aligned, device)
    n = device.value_counts()

    rows = []
    for d in sorted(device.unique()):
        mask = (device == d).values
        rows.append({
            "device": d,
            "n": int(n[d]),
            "accuracy": float(1 - wrong[mask].mean()),
            "shape_variance": variance[d],
        })
    return pd.DataFrame(rows).set_index("device")


def plot_confusion_matrix(cm_df: pd.DataFrame, out_path: str | Path,
                           title: str = "Matrice de confusion (LOOCV)") -> None:
    """Heatmap de la matrice de confusion, comptes annotés dans chaque case."""
    n = len(cm_df)
    fig, ax = plt.subplots(figsize=(max(6.0, 0.6 * n), max(5.0, 0.55 * n)))
    im = ax.imshow(cm_df.values, cmap="Blues")

    ax.set_xticks(range(len(cm_df.columns)))
    ax.set_xticklabels(cm_df.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(cm_df.index)))
    ax.set_yticklabels(cm_df.index)
    ax.set_xlabel("Prédiction (LOOCV)")
    ax.set_ylabel("Vérité terrain")
    ax.set_title(title)

    values = cm_df.values
    thresh = values.max() / 2 if values.max() > 0 else 0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if v:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=8,
                        color="white" if v > thresh else "black")

    fig.colorbar(im, ax=ax, shrink=0.8, label="n spécimens")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Heatmap matrice de confusion -> {out_path}")


def print_summary(out_dir: Path, level: str, meta_df: pd.DataFrame, gpa_result, pca, n_components: int,
                   accuracy: float, cm_df: pd.DataFrame, group_table: pd.DataFrame,
                   device_table: pd.DataFrame | None, device_filter: str | None) -> None:
    """Affiche l'essentiel du résumé dans le terminal."""
    n_points = gpa_result.aligned.shape[1]
    header = (
        f"niveau={level} | device={device_filter or 'tous'} | "
        f"n={len(meta_df)} spécimens | {n_points} landmarks"
    )

    output = "=" * len(header)
    output += "\n" + header
    output += "\n" + "=" * len(header)
    output += f"\nGPA : convergence en {gpa_result.n_iterations} itération(s)"
    output += f"\nPCA : {n_components} composante(s) conservée(s) " \
              f"(variance expliquée cumulée = {100 * pca.explained_variance_ratio_.sum():.1f}%)"
    output += f"\nLDA (LOOCV) : proportion bien classée = {accuracy:.4f}"

    output += f"\n\n--- Scores par {level} (LOOCV) ---"
    output += "\n" + str(group_table)

    if device_table is not None:
        output += "\n\n--- Scores par appareil (device) ---"
        output += "\n" + str(device_table)

    output += "\n\nMatrice de confusion (lignes = vérité terrain, colonnes = prédiction LOOCV) :"
    output += "\n" + str(cm_df)

    out_dir.mkdir(exist_ok=True, parents=True)

    with open(out_dir / f"summary.log", "w", encoding='utf-8') as f:
        f.write(output)

    print("\n" + output + "\n")


def write_summary_csv(out_dir: Path, level: str, tag: str, group_table: pd.DataFrame,
                       device_table: pd.DataFrame | None, cm_df: pd.DataFrame) -> None:
    """Écrit les CSV de résumé : un fichier par type de tableau (schémas de
    colonnes différents -- éviter de les mélanger dans un seul CSV).

    `tag` (ex: "espece_S1") suffixe les noms de fichiers pour que deux runs
    successifs (niveau ou appareil différent) n'écrasent pas leurs résultats ;
    `level` (ex: "espece") sert uniquement à l'affichage.
    """
    out_dir.mkdir(exist_ok=True, parents=True)

    group_path = out_dir / f"summary_by_{tag}.csv"
    group_table.to_csv(group_path)
    print(f"Résumé par {level} -> {group_path}")

    if device_table is not None:
        device_path = out_dir / f"summary_by_device_{tag}.csv"
        device_table.to_csv(device_path)
        print(f"Résumé par appareil -> {device_path}")

    cm_path = out_dir / f"confusion_matrix_{tag}.csv"
    cm_df.to_csv(cm_path)
    print(f"Matrice de confusion -> {cm_path}")
