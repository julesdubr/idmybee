"""classification_report.py
Figures et tableaux d'un run de classification (train ou predict) : matrice
de confusion, tableau par espèce, tableau par appareil, alignement GPA,
projection LDA. Lit les sorties déjà écrites par train.py/predict.py
(predictions.csv, params.json) et recharge le modèle pour projeter les
spécimens dans son espace -- ne réajuste rien.

La dispersion de forme (shape_variance) reste exclusivement dans
analysis/variance_report.py.

Usage :
    python -m analysis.classification_report species_train_P1-S1 --step train
    python -m analysis.classification_report species_train_P1-S1 --step predict --eval-tag test
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report as sk_classification_report
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

from utils.dataset import load_dataset, target_groupe
from utils.gpa import align_to_reference, two_d_array
from utils.model_io import load_model
from utils.run_io import FAMILY_LDA, read_params, result_path, run_path, setup_console_logging, write_params

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda v: f"{v:.4f}")


def confusion_matrix_df(df: pd.DataFrame, level: str) -> pd.DataFrame:
    truth, predicted = df[f"true_{level}"], df[f"predicted_{level}"]
    labels = sorted(truth.unique())
    cm = confusion_matrix(truth, predicted, labels=labels)
    return pd.DataFrame(cm, index=labels, columns=labels)


def species_report_df(df: pd.DataFrame, level: str) -> pd.DataFrame:
    """Precision/recall/f1 par classe (espèce ou espèce_caste)."""
    truth, predicted = df[f"true_{level}"], df[f"predicted_{level}"]
    labels = sorted(truth.unique())
    report = sk_classification_report(truth, predicted, labels=labels, output_dict=True, zero_division=0)
    table = pd.DataFrame(report).T.loc[labels, ["precision", "recall", "f1-score"]]
    table.columns = ["precision", "recall", "f1"]
    table.insert(0, "n", truth.value_counts().reindex(labels).astype(int))
    table.index.name = level
    return table


def device_report_df(merged: pd.DataFrame, level: str) -> pd.DataFrame | None:
    """Accuracy par appareil. None si la colonne 'device' est absente ou
    n'a qu'une seule valeur (rien à comparer)."""
    if "device" not in merged.columns or merged["device"].dropna().nunique() <= 1:
        return None
    known = merged["device"].notna()
    sub = merged[known]
    correct = (sub[f"predicted_{level}"] == sub[f"true_{level}"]).values
    rows = []
    for d in sorted(sub["device"].dropna().unique()):
        mask = (sub["device"] == d).values
        rows.append({"device": d, "n": int(mask.sum()), "accuracy": float(correct[mask].mean())})
    return pd.DataFrame(rows).set_index("device")


def plot_confusion_matrix(cm_df: pd.DataFrame, out_path: Path, title: str) -> None:
    n = len(cm_df)
    fig, ax = plt.subplots(figsize=(max(6.0, 0.6 * n), max(5.0, 0.55 * n)))
    im = ax.imshow(cm_df.values, cmap="Blues")

    ax.set_xticks(range(len(cm_df.columns)))
    ax.set_xticklabels(cm_df.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(cm_df.index)))
    ax.set_yticklabels(cm_df.index)
    ax.set_xlabel("Prédiction")
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
    print(f"Matrice de confusion -> {out_path}")


def plot_gpa_alignment(aligned: np.ndarray, groupe: pd.Series, out_path: Path, title: str) -> None:
    """Affiche les formes (déjà alignées sur la référence du modèle) en 2D, coloriées par groupe."""
    if aligned.ndim != 3 or aligned.shape[2] != 2:
        raise ValueError("aligned doit être un tableau de forme (n_specimens, n_points, 2).")

    unique_groups = sorted(groupe.unique())
    cmap = plt.get_cmap("tab20")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    for i, group in enumerate(unique_groups):
        mask = (groupe == group).values
        specimens_coords = aligned[mask]
        if specimens_coords.size == 0:
            continue
        color = cmap(i % 20)

        group_coords = np.vstack(specimens_coords)
        ax1.scatter(group_coords[:, 0], group_coords[:, 1], color=color, alpha=0.05, label=group, s=10)
        centroids = specimens_coords.mean(axis=0)
        ax2.scatter(centroids[:, 0], centroids[:, 1], color=color, s=20, label=group, marker="x")

    for ax, subtitle in ((ax1, "all"), (ax2, "centroids")):
        ax.set_xlabel("Coordonnée X")
        ax.set_ylabel("Coordonnée Y")
        ax.set_title(f"{title} ({subtitle})")
        ax.legend(title="Groupe", loc="best", fontsize="small")
        ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Alignement GPA -> {out_path}")


def plot_lda(lda_scores: np.ndarray, groupe: pd.Series, out_path: Path, title: str) -> None:
    if lda_scores.shape[1] == 1:
        x, y = lda_scores[:, 0], np.zeros(lda_scores.shape[0])
    else:
        x, y = lda_scores[:, 0], lda_scores[:, 1]

    plt.figure(figsize=(10, 8))
    unique_groups = sorted(groupe.unique())
    if len(unique_groups) > 20:
        cmap_combined = np.vstack([plt.cm.tab20c(np.linspace(0, 1, 20)), plt.cm.tab20b(np.linspace(0, 1, 20))])
        colors = cmap_combined[: len(unique_groups)]
    else:
        colors = [plt.cm.tab20(i) for i in range(len(unique_groups))]

    for i, group in enumerate(unique_groups):
        mask = (groupe == group).values
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
    parser = argparse.ArgumentParser(
        description="Figures et tableaux d'un run de classification (voir docstring du module)"
    )
    parser.add_argument("run_id", type=str, help="Identifiant de run (voir utils.run_io.build_run_id)")
    parser.add_argument("--step", type=str, choices=["train", "predict"], default="train",
                         help="Quelles prédictions analyser : celles du LOOCV (train) ou d'un batch predict.py (predict).")
    parser.add_argument("--eval-tag", type=str, default=None,
                         help="Requis si --step predict -- voir data/models/<family>/<run_id>/predict/ pour la liste.")
    parser.add_argument("--family", type=str, default=FAMILY_LDA)
    return parser


def main(argv: list[str] | None = None) -> None:
    setup_console_logging()
    args = build_arg_parser().parse_args(argv)

    if args.step == "predict" and not args.eval_tag:
        available = result_path(args.family, args.run_id, "predict")
        options = sorted(p.name for p in available.iterdir()) if available.exists() else []
        raise SystemExit(f"--eval-tag requis avec --step predict. Disponibles pour {args.run_id!r} : {options}")

    step_path = result_path(args.family, args.run_id, "predict", args.eval_tag) if args.step == "predict" \
        else result_path(args.family, args.run_id, "train")
    if not step_path.exists():
        raise SystemExit(
            f"{step_path} introuvable -- lancer classifiers.train ou classifiers.predict batch d'abord "
            f"pour run_id={args.run_id!r}."
        )
    params = read_params(step_path)

    pred_csv = step_path / ("loocv_predictions.csv" if args.step == "train" else "predictions.csv")
    df = pd.read_csv(pred_csv)
    true_cols = [c for c in df.columns if c.startswith("true_")]
    if not true_cols:
        raise SystemExit(
            f"{pred_csv} n'a pas de colonne 'true_<level>' -- pas de vérité connue, pas de rapport possible "
            "(ex: sorties de predict.py single, qui n'a pas de vérité terrain)."
        )
    level = true_cols[0].removeprefix("true_")

    model_path = result_path(args.family, args.run_id, "train") / "model.joblib"
    if not model_path.exists():
        raise SystemExit(f"Modèle introuvable : {model_path} (train.py a-t-il été lancé avec --no-save-model ?)")
    model = load_model(model_path)

    dataset_root = Path(params["dataset"])
    specimens, meta_df = load_dataset(
        dataset_root,
        split=params.get("resolved_split") or "all",
        devices=params.get("devices"),
        species=params.get("species"),
        castes=params.get("castes"),
        exclude_outliers=bool(params.get("exclude_outliers", False)),
        strict=not bool(params.get("non_strict", False)),
        landmarks_tps=params.get("landmarks_tps"),
        landmarks_status_csv=params.get("landmarks_status_csv"),
        labeled_only=True,
    )
    meta_df = meta_df.copy()
    meta_df["tps_id"] = [sp.tps_id for sp in specimens]

    run_label = f"{args.run_id}/predict/{args.eval_tag}" if args.step == "predict" else f"{args.run_id}/train"
    out_dir = run_path(args.family, args.run_id, args.step, *([args.eval_tag] if args.step == "predict" else []), "report")

    cm_df = confusion_matrix_df(df, level)
    cm_df.to_csv(out_dir / "confusion_matrix.csv")
    plot_confusion_matrix(cm_df, out_dir / "confusion_matrix.png", title=f"Matrice de confusion -- {run_label}")
    print("\nMatrice de confusion :\n" + str(cm_df))

    species_df = species_report_df(df, level)
    species_df.to_csv(out_dir / "species_report.csv")
    print(f"\nTableau par {level} -> {out_dir / 'species_report.csv'}\n" + str(species_df))

    merged = df.merge(meta_df[["tps_id", "device", "device_tag"]], on="tps_id", how="left")
    device_df = device_report_df(merged, level)
    if device_df is not None:
        device_df.to_csv(out_dir / "device_report.csv")
        print(f"\nTableau par appareil -> {out_dir / 'device_report.csv'}\n" + str(device_df))

    pred_tps_ids = set(df["tps_id"])
    keep_mask = [sp.tps_id in pred_tps_ids for sp in specimens]
    plot_specimens = [sp for sp, k in zip(specimens, keep_mask) if k]
    plot_meta = meta_df[keep_mask].reset_index(drop=True)
    if plot_specimens:
        groupe = target_groupe(plot_meta, level)
        aligned = np.stack([align_to_reference(sp.landmarks, model.mean_shape) for sp in plot_specimens])
        plot_gpa_alignment(aligned, groupe, out_dir / "gpa_alignment.png", title=f"Alignement GPA -- {run_label}")

        scores = model.pca.transform(two_d_array(aligned))
        lda_scores = model.lda.transform(scores)
        plot_lda(lda_scores, groupe, out_dir / "lda_projection.png", title=f"Projection LDA -- {run_label}")

    write_params(out_dir, args, extra={"run_id": args.run_id, "eval_tag": args.eval_tag, "family": args.family, "step": args.step})
    print(f"\nRun -> {out_dir}")


if __name__ == "__main__":
    main()
