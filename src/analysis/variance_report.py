"""variance_report.py
CLI pour analysis.variance.nested_anova : décompose la variance de forme
(Procrustes²) entre facteurs emboîtés, dans l'ordre donné par --levels.
Seul endroit du pipeline qui calcule une dispersion de forme -- absent de
analysis/classification_report.py.

Indépendant de train.py/predict.py : ne charge ni n'ajuste de modèle,
seulement une GPA sur les specimens filtrés. Écrit dans
data/analysis/variance/<variance_id>/ (namespace séparé des run_id de
modèle) -- plusieurs analyses (devices différents, ordre de niveaux
différent) y cohabitent, comparables en ouvrant leurs anova.csv côte à côte.

--levels attend des colonnes de meta_df (species, caste, specimen_id,
device, device_tag), du plus large au plus fin. Comme train.py/predict.py,
les photos SUSPECT/FAILED sont exclues par défaut (--include-outliers pour
les inclure) -- particulièrement important ici puisque ce script mesure une
dispersion : des erreurs de registration gonfleraient artificiellement la
variance justement mesurée.

Comparer effet biologique et effet méthodologique : species, caste,
specimen_id et device forment une seule chaîne d'emboîtement valide (chaque
photo appartient à un individu, chaque individu à une caste, chaque caste à
une espèce). Un run brut à 4 niveaux SANS restriction est biaisé : chaque
photo pèse pareil à chaque niveau, donc un individu avec plus de photos (ou
une couverture device différente) pèse plus lourd dans la moyenne de son
groupe biologique, et la "moyenne individu" mélange P et S dans des
proportions différentes d'un individu à l'autre -- ce qui contamine à la
fois les niveaux biologiques et l'estimation de l'effet appareil. Utiliser
--balanced-devices : ne garde que les spécimens ayant TOUS les device_tag
demandés (ex: P1, P2, S1, S2), donc chacun contribue le même nombre de
photos, réparties de façon identique entre appareils :

    python -m analysis.variance_report data/Bombus --split train \\
        --devices P1 P2 S1 S2 --balanced-devices \\
        --levels species caste specimen_id device --n-perm 999

    MS(specimen_id) = plancher biologique (variance entre individus,
                       espèce et caste déjà retirées)
    MS(device)       = effet méthodologique (variance entre appareils,
                       POUR UN MÊME individu -- l'identité de l'individu
                       est déjà retirée)
    MS(device) > MS(specimen_id) -> le bruit de mesure dépasse la variation
    biologique réelle entre individus, ce qui questionne la capacité du
    pipeline à discriminer en dessous de ce seuil.

`device` regroupe par device_type (P/S) ; `device_tag` (P1/P2/S1/...)
descend au niveau de la prise individuelle -- utiliser `device_tag` en
dernier niveau pousse aussi le bruit de reprise au sein d'un même appareil
dans MS(device_tag) plutôt que dans le résiduel (adapter alors la liste
--balanced-devices/--devices en conséquence, ex: P1 P2 P3 S1 S2 S3 si trois
prises par appareil).

Chaque niveau du tableau produit teste sa MS contre celle du résiduel
(colonne F), mais la significativité fiable est la colonne "p (permutation)"
(mélange stratifié à l'intérieur de chaque niveau parent, voir
analysis/variance.py) -- n_perm=0 (défaut) saute ce calcul, mettre
--n-perm (ex: 999) pour l'obtenir.

Usage additionnel : isoler la dispersion biologique seule (sans la
dimension appareil), par ex. pour calibrer une attente de difficulté de
classification indépendamment du protocole photo -- restreindre alors à un
seul device_type (--devices P1) :

    python -m analysis.variance_report data/Bombus --split train --devices P1 \\
        --levels species caste specimen_id

Accepte --tps comme train.py/predict.py, pour analyser une autre source de landmarks.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parent))
from utils.cli import add_dataset_args, dataset_kwargs
from utils.dataset import load_dataset, restrict_to_complete_devices
from utils.gpa import gpagen, two_d_array
from utils.run_io import ANALYSIS_ROOT, build_variance_id, run_path, setup_console_logging, write_params, write_run_log
from analysis.variance import nested_anova

LEVEL_CHOICES = ("species", "caste", "specimen_id", "device", "device_tag")


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
    add_dataset_args(parser, default_split="train")
    parser.add_argument("--levels", type=str, nargs="+", required=True, choices=LEVEL_CHOICES,
                         help="Niveaux emboîtés, du plus large au plus fin, ex: species caste specimen_id")
    parser.add_argument("--min-top-level-n", type=int, default=5,
                         help="Groupes du 1er niveau (--levels[0]) avec moins de spécimens sont écartés "
                              "avant l'analyse -- estimation trop instable sinon (défaut: 5).")
    parser.add_argument("--n-perm", type=int, default=0,
                         help="Nombre de permutations pour les p-values (0 = désactivé, juste les magnitudes).")
    parser.add_argument(
        "--balanced-devices", action="store_true",
        help="Ne garder que les spécimens ayant TOUS les device_tag de --devices (requiert --devices). "
             "Élimine le biais de pondération inégale entre individus dans une ANOVA à plusieurs niveaux "
             "biologiques + appareil -- voir utils.dataset.restrict_to_complete_devices.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> None:
    setup_console_logging()
    args = build_arg_parser().parse_args(argv)

    ds_kwargs = dataset_kwargs(args, default_split="train")
    specimens, meta_df = load_dataset(args.dataset, labeled_only=True, **ds_kwargs)

    if args.balanced_devices:
        if not args.devices:
            raise SystemExit("--balanced-devices nécessite --devices (ex: --devices P1 P2 S1 S2).")
        specimens, meta_df = restrict_to_complete_devices(specimens, meta_df, args.devices)

    # lignes avec une valeur manquante sur un des niveaux demandés (ex: caste
    # non renseignée) -- nested_anova refuse les NaN, donc on les écarte ici
    # avec un avertissement plutôt que de planter plus loin.
    valid = meta_df[list(args.levels)].notna().all(axis=1).tolist()
    n_dropped = sum(not v for v in valid)
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
    X = two_d_array(gpa_result.aligned)

    rng = np.random.default_rng(args.seed) if args.n_perm else None
    levels = [(name, meta_df[name]) for name in args.levels]
    table = nested_anova(X, levels, n_perm=args.n_perm, rng=rng)

    variance_id = build_variance_id(args.levels, ds_kwargs["split"], args.devices, args.landmarks_tps, args.run_label)
    if args.balanced_devices:
        variance_id += "_balanced"
    out_dir = run_path("variance", variance_id, root=ANALYSIS_ROOT)

    header = (
        f"ANOVA emboîtée : {' ⊃ '.join(args.levels)}  |  split={ds_kwargs['split']}  "
        f"devices={args.devices or 'tous'}{' (équilibré)' if args.balanced_devices else ''}  n={len(specimens)}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    print(table)

    csv_path = out_dir / "anova.csv"
    table.to_csv(csv_path)
    plot_ms_by_level(table, out_dir / "anova.png", header)

    write_params(out_dir, args, extra={"variance_id": variance_id, "resolved_split": ds_kwargs["split"]})
    write_run_log(out_dir, header + "\n" + table.to_string() + f"\n\nTable -> {csv_path}\n")
    print(f"\nRun -> {out_dir}")


if __name__ == "__main__":
    main()
