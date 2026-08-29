"""variance_report.py
CLI pour analysis.variance.nested_anova : décompose la variance de forme
(Procrustes²) entre facteurs emboîtés, dans l'ordre donné par --levels.

--levels attend des colonnes de meta_df (voir utils.dataset.load_dataset) :
species, caste, specimen_id, device -- du plus large au plus fin.

Exemples :
    # 1) plancher biologique : espèce ⊃ caste ⊃ individu, sur une seule
    #    photo par individu (P1) pour ne pas mélanger l'effet appareil
    #    dans le résidu individuel :
    python -m src.analysis.variance_report data/Bombus --split train --devices P1 \\
        --levels species caste specimen_id

    # 2) effet méthodologique : individu ⊃ appareil, sur les mêmes
    #    individus, une photo par appareil (P1 et S1) :
    python -m src.analysis.variance_report data/Bombus --split train --levels specimen_id device

Comparer les deux : le MS "specimen_id" du run (1) est le plancher
biologique (V_individu), le MS "device" du run (2) est l'effet
méthodologique (V_appareil) -- même unité (Procrustes² par ddl), lisibles
côte à côte dans les deux summary.csv produits.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parent))
from utils.dataset import load_dataset
from utils.gpa import gpagen
from analysis.variance import flatten, nested_anova

LEVEL_CHOICES = ("species", "caste", "specimen_id", "device")


def plot_ms_by_level(table, out_path: Path, title: str) -> None:
    """Barres des mean squares par niveau (échelle log -- les écarts entre
    espèce/individu/appareil sont typiquement de plusieurs ordres de
    grandeur)."""
    fig, ax = plt.subplots(figsize=(max(6.0, 1.2 * len(table)), 5.0))
    ax.bar(table.index, table["MS"], color="steelblue")
    ax.set_yscale("log")
    ax.set_ylabel("Mean square (Procrustes² / ddl, log)")
    ax.set_title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Graphe MS par niveau -> {out_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ANOVA emboîtée sur la forme (Procrustes) -- voir docstring du module")
    parser.add_argument("dataset", type=Path, help="Dossier racine (ex: data/Bombus)")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--devices", type=str, nargs="+", default=None,
                         help="Ne garder que ces photos (ex: --devices P1, ou --devices P1 S1).")
    parser.add_argument("--species", type=str, nargs="+", default=None, help="Ne garder que ces espèces.")
    parser.add_argument("--castes", type=str, nargs="+", default=None, help="Ne garder que ces castes.")
    parser.add_argument("--exclude-outliers", action="store_true")
    parser.add_argument("--levels", type=str, nargs="+", required=True, choices=LEVEL_CHOICES,
                         help="Niveaux emboîtés, du plus large au plus fin, ex: species caste specimen_id")
    parser.add_argument("--min-top-level-n", type=int, default=5,
                         help="Groupes du 1er niveau (--levels[0]) avec moins de spécimens sont écartés "
                              "avant l'analyse -- estimation trop instable sinon (défaut: 5).")
    parser.add_argument("--n-perm", type=int, default=0,
                         help="Nombre de permutations pour les p-values (0 = désactivé, juste les magnitudes).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("out/variance"))
    parser.add_argument("--non-strict", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_arg_parser().parse_args(argv)

    specimens, meta_df = load_dataset(
        args.dataset, split=args.split, devices=args.devices, species=args.species,
        castes=args.castes, exclude_outliers=args.exclude_outliers, strict=not args.non_strict,
    )

    # lignes avec une valeur manquante sur un des niveaux demandés (ex: caste
    # non renseignée) -- nested_anova refuse les NaN, donc on les écarte ici
    # avec un avertissement plutôt que de planter plus loin.
    valid = meta_df[list(args.levels)].notna().all(axis=1).tolist()
    n_dropped = valid.count(False) if hasattr(valid, "count") else sum(not v for v in valid)
    if n_dropped:
        print(f"{n_dropped} photo(s) écartée(s) : valeur manquante sur un des niveaux {args.levels}")
    specimens = [sp for sp, keep in zip(specimens, valid) if keep]
    meta_df = meta_df[valid].reset_index(drop=True)

    top_level = args.levels[0]
    counts = meta_df[top_level].value_counts()
    small_groups = counts[counts < args.min_top_level_n].index.tolist()
    if small_groups:
        print(f"{top_level} écarté(s) (< {args.min_top_level_n} spécimens) : {small_groups}")
        mask = (~meta_df[top_level].isin(small_groups)).tolist()
        specimens = [sp for sp, keep in zip(specimens, mask) if keep]
        meta_df = meta_df[mask].reset_index(drop=True)

    if len(specimens) < 3:
        raise SystemExit(f"Seulement {len(specimens)} spécimen(s) après filtrage -- pas assez pour une ANOVA.")

    gpa_result = gpagen([sp.landmarks for sp in specimens])
    X = flatten(gpa_result.aligned)

    rng = np.random.default_rng(args.seed) if args.n_perm else None
    levels = [(name, meta_df[name]) for name in args.levels]
    table = nested_anova(X, levels, n_perm=args.n_perm, rng=rng)

    tag = "_".join(args.levels) + f"_{args.split}" + (f"_{'-'.join(args.devices)}" if args.devices else "")
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    header = f"ANOVA emboîtée : {' ⊃ '.join(args.levels)}  |  split={args.split}  devices={args.devices or 'tous'}  n={len(specimens)}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    print(table)

    csv_path = out_dir / f"anova_{tag}.csv"
    table.to_csv(csv_path)
    print(f"\nTable -> {csv_path}")

    plot_ms_by_level(table, out_dir / f"anova_{tag}.png", header)


if __name__ == "__main__":
    main()