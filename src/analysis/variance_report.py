"""variance_report.py
CLI for analysis.variance.nested_anova: decomposes shape variance
(Procrustes²) across nested factors, in the order given by --levels. The
only place in the pipeline that computes shape dispersion -- absent from
analysis/classification_report.py.

Independent of train.py/predict.py: loads or fits no model, just a GPA on
the filtered specimens. Writes to data/analysis/variance/<variance_id>/
(namespace separate from model run_ids) -- several analyses (different
devices, different level order) coexist there, comparable by opening their
anova.csv side by side.

--levels expects meta_df columns (species, caste, specimen_id, device,
device_tag), from broadest to finest. Like train.py/predict.py,
SUSPECT/FAILED photos are excluded by default (--include-outliers to
include them) -- particularly important here since this script measures
dispersion: registration errors would artificially inflate the very
variance being measured.

Comparing biological effect and methodological effect: species, caste,
specimen_id and device form a single valid nesting chain (each photo
belongs to an individual, each individual to a caste, each caste to a
species). A raw 4-level run with NO restriction is biased: every photo
weighs the same at every level, so an individual with more photos (or a
different device coverage) weighs more heavily in its biological group's
average, and the "individual mean" mixes P and S in proportions that differ
from one individual to the next -- which contaminates both the biological
levels and the device effect estimate. Use --balanced-devices: keeps only
specimens that have ALL the requested device_tags (e.g. P1, P2, S1, S2), so
each contributes the same number of photos, split identically across
devices:

    python -m analysis.variance_report data/Bombus --split train \\
        --devices P1 P2 S1 S2 --balanced-devices \\
        --levels species caste specimen_id device --n-perm 999

    MS(specimen_id) = biological floor (variance between individuals,
                       species and caste already removed)
    MS(device)       = methodological effect (variance between devices,
                       FOR THE SAME individual -- individual identity is
                       already removed)
    MS(device) > MS(specimen_id) -> measurement noise exceeds real
    biological variation between individuals, which questions the
    pipeline's ability to discriminate below that threshold.

`device` groups by device_type (P/S); `device_tag` (P1/P2/S1/...) goes down
to the individual shot level -- using `device_tag` as the last level also
pushes within-device retake noise into MS(device_tag) rather than into the
residual (adjust the --balanced-devices/--devices list accordingly, e.g.
P1 P2 P3 S1 S2 S3 for three shots per device).

Every level of the produced table tests its MS against the residual's
(column F), but the reliable significance is the "p (permutation)" column
(stratified shuffle within each parent level, see analysis/variance.py) --
n_perm=0 (default) skips this computation, pass --n-perm (e.g. 999) to
get it.

Additional use: isolating biological dispersion alone (without the device
dimension), e.g. to calibrate an expectation of classification difficulty
independently of the photo protocol -- restrict to a single device_type
(--devices P1):

    python -m analysis.variance_report data/Bombus --split train --devices P1 \\
        --levels species caste specimen_id

Accepts --tps like train.py/predict.py, to analyze a different landmark source.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.variance import nested_anova
from utils.cli import add_dataset_args, add_dataset_positional, add_logging_args, dataset_kwargs, log_level_from_args
from utils.dataset import load_dataset, restrict_to_complete_devices
from core.gpa import gpagen, two_d_array
from utils.run_io import ANALYSIS_ROOT, build_variance_id, run_path, setup_console_logging, write_params, write_run_log

logger = logging.getLogger(__name__)

LEVEL_CHOICES = ("species", "caste", "specimen_id", "device", "device_tag")


def plot_ms_by_level(table, out_path: Path, title: str) -> None:
    """Bar chart of mean squares per level (log scale -- gaps between
    species/individual/device are typically several orders of magnitude)."""
    fig, ax = plt.subplots(figsize=(max(6.0, 1.2 * len(table)), 5.0))
    ax.bar(table.index, table["MS"], color="steelblue")
    ax.set_yscale("log")
    ax.set_ylabel("Mean square (Procrustes² / df, log)")
    ax.set_title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"MS-by-level plot -> {out_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Nested ANOVA on shape (Procrustes) -- see module docstring")
    add_dataset_positional(parser, help="Root folder (e.g. data/Bombus)")
    add_dataset_args(parser, default_split="train")
    parser.add_argument("--levels", type=str, nargs="+", required=True, choices=LEVEL_CHOICES,
                         help="Nested levels, from broadest to finest, e.g. species caste specimen_id")
    parser.add_argument("--min-top-level-n", type=int, default=5,
                         help="1st-level (--levels[0]) groups with fewer specimens are dropped "
                              "before the analysis -- too unstable an estimate otherwise (default: 5).")
    parser.add_argument("--n-perm", type=int, default=0,
                         help="Number of permutations for p-values (0 = disabled, magnitudes only).")
    parser.add_argument(
        "--balanced-devices", action="store_true",
        help="Keep only specimens that have ALL of --devices' device_tags (requires --devices). "
             "Removes the unequal-weighting bias between individuals in a multi-level biological + "
             "device ANOVA -- see utils.dataset.restrict_to_complete_devices.",
    )
    parser.add_argument("--seed", type=int, default=0)
    add_logging_args(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    setup_console_logging(log_level_from_args(args))

    ds_kwargs = dataset_kwargs(args, default_split="train")
    specimens, meta_df = load_dataset(args.dataset, labeled_only=True, **ds_kwargs)

    if args.balanced_devices:
        if not args.devices:
            raise SystemExit("--balanced-devices requires --devices (e.g. --devices P1 P2 S1 S2).")
        specimens, meta_df = restrict_to_complete_devices(specimens, meta_df, args.devices)

    # rows with a missing value on one of the requested levels (e.g. caste
    # not filled in) -- nested_anova rejects NaN, so they're dropped here
    # with a warning rather than failing further down.
    valid = meta_df[list(args.levels)].notna().all(axis=1).tolist()
    n_dropped = sum(not v for v in valid)
    if n_dropped:
        logger.info("%d photo(s) dropped: missing value on one of the levels %s", n_dropped, args.levels)
    specimens = [sp for sp, keep in zip(specimens, valid) if keep]
    meta_df = meta_df[valid].reset_index(drop=True)

    top_level = args.levels[0]
    counts = meta_df[top_level].value_counts()
    small_groups = counts[counts < args.min_top_level_n].index.tolist()
    if small_groups:
        logger.info("%s dropped (< %d specimens): %s", top_level, args.min_top_level_n, small_groups)
        mask = (~meta_df[top_level].isin(small_groups)).tolist()
        specimens = [sp for sp, keep in zip(specimens, mask) if keep]
        meta_df = meta_df[mask].reset_index(drop=True)

    if len(specimens) < 3:
        raise SystemExit(f"Only {len(specimens)} specimen(s) after filtering -- not enough for an ANOVA.")

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
        f"Nested ANOVA: {' ⊃ '.join(args.levels)}  |  split={ds_kwargs['split']}  "
        f"devices={args.devices or 'all'}{' (balanced)' if args.balanced_devices else ''}  n={len(specimens)}"
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
