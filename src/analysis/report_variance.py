"""report_variance.py
Décompose la variance de forme des ailes (landmarks GPA-alignés) en signal
espèce, variation biologique intra-espèce, et variance méthodologique liée
à l'appareil photo -- voir analysis.variance pour la théorie (ANOVA de
Procrustes, tests par permutation).

Phase indépendante de lda.py (pas de classifieur ici, juste la forme et les
métadonnées) : consommable avant ou après l'entraînement du modèle. Utile
en aval de lda.py pour interpréter ses erreurs -- si la variance
méthodologique (appareil) reste petite devant la variation biologique
elle-même plus petite que la séparation entre espèces, les confusions du
modèle s'expliquent par des espèces morphologiquement proches, pas par un
défaut de méthode.

Deux décompositions, sur des sous-ensembles différents (voir docstrings des
tables imprimées) :
  (a) espèce seule, sur tous les spécimens labellisés ;
  (b) appareil emboîté dans l'espèce, sur le sous-ensemble où l'appareil
      (device_type, une info par photo -- voir images.csv) est connu.

Usage:
    python -m analysis.report_variance data/annotations/landmarks_numbered_labeled.tps \\
        data/manifest/specimens.csv data/manifest/images.csv \\
        --exclude-ids data/manifest/landmarks_numbered.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parent))
from analysis.variance import (
    between_within_ss,
    flatten,
    nested_decomposition,
    permutation_p_between,
    permutation_p_nested,
    variance_summary,
)
from utils.dataset import apply_filters, load_labeled_dataset
from utils.gpa import gpagen
from utils.variance import within_group_variance

import matplotlib.pyplot as plt

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda v: f"{v:.6f}")


def drop_small_species(meta_df: pd.DataFrame, min_n: int) -> pd.DataFrame:
    """Une espèce avec très peu de spécimens rend sa propre variance
    intra-groupe non fiable (voir la même logique dans utils.outliers pour
    le flag d'outliers post-GPA) -- exclue de l'ANOVA, pas de la biologie."""
    counts = meta_df["species"].value_counts()
    too_small = counts[counts < min_n].index.tolist()
    if too_small:
        print(f"Espèce(s) < {min_n} spécimen(s), exclue(s) de l'analyse de variance : {too_small}")
    return meta_df[~meta_df["species"].isin(too_small)]


def per_group_table(aligned: np.ndarray, groups: pd.Series) -> pd.DataFrame:
    """n + variance de forme intra-groupe, pour inspection détaillée
    (complète les magnitudes agrégées de l'ANOVA)."""
    variance = within_group_variance(aligned, groups)
    n = groups.value_counts()
    return pd.DataFrame({"n": n, "shape_variance_intra": variance}).reindex(sorted(groups.unique()))


def species_anova(X: np.ndarray, species: pd.Series, n_perm: int, rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    """Table (a) : espèce seule, sur tout `X`."""
    codes, names = pd.factorize(species, sort=True)
    n_groups = len(names)
    ss_species, ss_within = between_within_ss(X, codes, n_groups)
    df_species, df_within = n_groups - 1, len(X) - n_groups
    p_species = permutation_p_between(X, codes, n_groups, ss_species, n_perm, rng)

    ms_species = variance_summary(ss_species, df_species)
    ms_within = variance_summary(ss_within, df_within)
    table = pd.DataFrame([
        {"source": "Espèce (inter)", "SS": ss_species, "df": df_species, "MS": ms_species,
         "F": ms_species / ms_within, "p (permutation)": p_species},
        {"source": "Intra-espèce (résiduel)", "SS": ss_within, "df": df_within, "MS": ms_within,
         "F": np.nan, "p (permutation)": np.nan},
    ]).set_index("source")

    headline = {"n": len(X), "n_species": n_groups, "ms_species": ms_species, "ms_within": ms_within}
    return table, headline


def device_nested_anova(
    X: np.ndarray, species: pd.Series, device: pd.Series, n_perm: int, rng: np.random.Generator
) -> tuple[pd.DataFrame, dict]:
    """Table (b) : appareil emboîté dans l'espèce, sur `X` restreint aux
    lignes où l'appareil est connu (voir appelant)."""
    species_codes, species_names = pd.factorize(species, sort=True)
    device_codes, device_names = pd.factorize(device, sort=True)
    n_species, n_devices = len(species_names), len(device_names)

    nd = nested_decomposition(X, species_codes, n_species, device_codes, n_devices)
    p_species = permutation_p_between(X, species_codes, n_species, nd.ss_a, n_perm, rng)
    p_device = permutation_p_nested(X, species_codes, n_species, device_codes, n_devices, nd.ss_b_nested, n_perm, rng)

    ms_species = variance_summary(nd.ss_a, nd.df_a)
    ms_device = variance_summary(nd.ss_b_nested, nd.df_b_nested)
    ms_error = variance_summary(nd.ss_error, nd.df_error)
    table = pd.DataFrame([
        {"source": "Espèce (inter)", "SS": nd.ss_a, "df": nd.df_a, "MS": ms_species,
         "F": ms_species / ms_error, "p (permutation)": p_species},
        {"source": "Appareil | espèce (inter, emboîté)", "SS": nd.ss_b_nested, "df": nd.df_b_nested,
         "MS": ms_device, "F": ms_device / ms_error, "p (permutation)": p_device},
        {"source": "Résiduel (intra cellule espèce x appareil)", "SS": nd.ss_error, "df": nd.df_error,
         "MS": ms_error, "F": np.nan, "p (permutation)": np.nan},
    ]).set_index("source")

    headline = {
        "n": nd.n, "n_species": n_species, "n_devices": n_devices, "n_cells": nd.n_cells,
        "ms_device_nested": ms_device,
    }
    return table, headline


def plot_variance_comparison(v_inter_species: float, v_intra_species: float, v_inter_device: float, out_path: Path) -> None:
    labels = ["Inter-espèce", "Intra-espèce", "Inter-appareil\n(au sein de l'espèce)"]
    values = [v_inter_species, v_intra_species, v_inter_device]
    colors = ["#2b6cb0", "#38a169", "#d69e2e"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(labels, values, color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("Mean square (distance de Procrustes² / ddl, échelle log)")
    ax.set_title("Décomposition de la variance de forme")
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.2e}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Graphique de comparaison -> {out_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ANOVA de Procrustes espèce / appareil (emboîté) sur un TPS labellisé"
    )
    parser.add_argument("tps_path", type=Path, help="TPS labellisé (ex: landmarks_numbered_labeled.tps)")
    parser.add_argument("specimens_csv", type=Path, help="data/manifest/specimens.csv")
    parser.add_argument("images_csv", type=Path,
                         help="data/manifest/images.csv -- nécessaire pour device_type (info par photo)")
    parser.add_argument("--non-strict", action="store_true", help="Tolérer les blocs TPS malformés")
    parser.add_argument("--exclude-ids", type=Path, nargs="+", default=None,
                         help="Un ou plusieurs CSV 'tps_id'+'status' des spécimens à exclure "
                              "(ex: data/manifest/landmarks_numbered.csv)")
    parser.add_argument("--exclude-species", type=str, nargs="+", default=None)
    parser.add_argument("--dataset", type=str, default=None,
                         help="Restreindre à un jeu de données d'origine (organized/terrain/vrac/basile_m1)")
    parser.add_argument("--min-species-n", type=int, default=5,
                         help="Espèces avec moins de spécimens que ce seuil exclues de l'ANOVA (défaut: 5)")
    parser.add_argument("--n-perm", type=int, default=499,
                         help="Nombre de permutations pour les tests de significativité (défaut: 499)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("out/variance"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    specimens, meta_df = load_labeled_dataset(
        args.tps_path, args.specimens_csv, images_csv=args.images_csv, strict=not args.non_strict
    )
    specimens, meta_df = apply_filters(
        specimens, meta_df, args.exclude_ids, dataset=args.dataset, exclude_species=args.exclude_species
    )
    meta_df = drop_small_species(meta_df, args.min_species_n)
    keep_mask = meta_df.index
    specimens = [specimens[i] for i in keep_mask]
    meta_df = meta_df.reset_index(drop=True)

    gpa_result = gpagen([sp.landmarks for sp in specimens])
    X = flatten(gpa_result.aligned)
    species = meta_df["species"]

    # --- (a) Espèce seule, population complète -------------------------------
    print(f"\n=== (a) Variance inter/intra-espèce -- {len(X)} spécimen(s), {species.nunique()} espèce(s) ===")
    table_a, head_a = species_anova(X, species, args.n_perm, rng)
    print(table_a)
    table_a.to_csv(args.out_dir / "anova_species.csv")

    species_table = per_group_table(gpa_result.aligned, species)
    species_table.to_csv(args.out_dir / "variance_by_species.csv")
    print(f"\nVariance intra-espèce, par espèce ({args.out_dir / 'variance_by_species.csv'}) :")
    print(species_table.sort_values("shape_variance_intra", ascending=False))

    # --- (b) Appareil emboîté dans l'espèce, sous-ensemble device connu ------
    if "device" not in meta_df.columns or meta_df["device"].notna().sum() < 2 * args.min_species_n:
        print("\n'device' indisponible ou trop peu de spécimens avec appareil connu -- "
              "décomposition (b) sautée. Vérifier que images_csv contient bien device_type.")
        table_b, head_b = None, None
    else:
        dev_mask = meta_df["device"].notna()
        meta_dev = meta_df[dev_mask]
        X_dev = X[dev_mask.values]
        print(f"\n=== (b) Variance appareil, au sein de l'espèce -- {len(X_dev)} spécimen(s) avec appareil connu ===")
        table_b, head_b = device_nested_anova(X_dev, meta_dev["species"], meta_dev["device"], args.n_perm, rng)
        print(table_b)
        table_b.to_csv(args.out_dir / "anova_device_nested.csv")

        device_table = per_group_table(gpa_result.aligned[dev_mask.values], meta_dev["device"])
        device_table.to_csv(args.out_dir / "variance_by_device.csv")
        print(f"\nVariance intra-appareil, par appareil ({args.out_dir / 'variance_by_device.csv'}) :")
        print(device_table)

    # --- Comparaison des magnitudes -------------------------------------------
    if table_b is not None:
        v_inter_species, v_intra_species = head_a["ms_species"], head_a["ms_within"]
        v_inter_device = head_b["ms_device_nested"]
        print("\n=== Comparaison des magnitudes (mean squares, mêmes unités) ===")
        print(f"  V inter-espèce               = {v_inter_species:.6e}")
        print(f"  V intra-espèce                = {v_intra_species:.6e}")
        print(f"  V inter-appareil (| espèce)  = {v_inter_device:.6e}")
        hyp_ok = v_inter_species > v_intra_species > v_inter_device
        print(
            f"Hypothèse V_inter-espèce > V_intra-espèce > V_inter-appareil : "
            f"{'CONFIRMÉE' if hyp_ok else 'NON confirmée'}"
        )
        if not hyp_ok and v_inter_device >= v_intra_species:
            print(
                "  -> la variance liée à l'appareil rivalise avec (ou dépasse) la variation "
                "biologique intra-espèce : une partie des erreurs de classification pourrait "
                "venir de la méthode (photo), pas seulement de la biologie."
            )
        plot_variance_comparison(v_inter_species, v_intra_species, v_inter_device,
                                  args.out_dir / "variance_comparison.png")

    print(f"\nRésultats -> {args.out_dir}")


if __name__ == "__main__":
    main()