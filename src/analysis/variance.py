"""variance.py
Shape variance decomposition (squared Procrustes distance on GPA-aligned
landmarks) into nested factors (e.g. species ⊃ caste ⊃ individual, or
individual ⊃ device), with permutation-based significance testing
(PERMANOVA -- no theoretical F distribution: multivariate normality is too
strong an assumption in high dimension, the test's validity comes from
permuting labels, same logic as geomorph::procD.lm).

Everything revolves around nested_anova(): an N-level nested ANOVA (the
first level = the broadest, e.g. species; the last = the finest, e.g.
individual). A single level = a classic one-way ANOVA. Generalizes the old
hardcoded 2-level version (species/device) -- same function for
species⊃caste⊃individual (3 levels) and individual⊃device (2 levels), see
analysis/variance_report.py.

within_group_variance() is a separate function: a DESCRIPTIVE table (one
row per group, e.g. per species) to spot an abnormally dispersed group. It
does NOT replace nested_anova()'s pooled residual MS (which divides by
N - n_groups over the whole population) -- see its docstring.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def flatten(aligned: np.ndarray) -> np.ndarray:
    """(n_specimens, n_landmarks, 2) -> (n_specimens, n_landmarks*2)."""
    return aligned.reshape(len(aligned), -1)


def _total_ss(X: np.ndarray) -> float:
    grand_mean = X.mean(axis=0)
    return float(((X - grand_mean) ** 2).sum())


def between_within_ss(X: np.ndarray, codes: np.ndarray, n_groups: int) -> tuple[float, float]:
    """(SS_between, SS_within) for a single factor: SS_total = SS_between + SS_within
    (standard ANOVA identity, valid regardless of per-group sample sizes).

    `codes` must be dense (0..n_groups-1) -- groups never observed count as
    0 (an actual count via `counts`, not a made-up average)."""
    grand_mean = X.mean(axis=0)
    total_ss = _total_ss(X)

    group_sums = np.zeros((n_groups, X.shape[1]))
    np.add.at(group_sums, codes, X)
    counts = np.bincount(codes, minlength=n_groups).astype(float)
    safe_counts = np.where(counts == 0, 1.0, counts)
    group_means = group_sums / safe_counts[:, None]

    ss_between = float((counts * ((group_means - grand_mean) ** 2).sum(axis=1)).sum())
    return ss_between, total_ss - ss_between


def combine_codes(codes_a: np.ndarray, n_a: int, codes_b: np.ndarray, n_b: int) -> np.ndarray:
    """Combined integer code (a, b) -> a single code in [0, n_a*n_b)."""
    return codes_a.astype(np.int64) * n_b + codes_b.astype(np.int64)


def variance_summary(ss: float, df: int) -> float:
    """Mean square (SS/df) -- comparable magnitude across levels."""
    return ss / df if df > 0 else float("nan")


def within_group_variance(aligned: np.ndarray, groupe: pd.Series) -> pd.Series:
    """DESCRIPTIVE Procrustes variance, group by group (one row per
    species/device/etc.): mean squared distance to the group's centroid,
    bias-corrected (n-1, not n -- the centroid is itself estimated from the
    same points, so n degrees of freedom would overstate the precision).

    WARNING: this is NOT the same quantity as nested_anova()'s pooled
    residual MS (which divides by N - n_groups over the WHOLE population at
    once). The two coexist: this one is for spotting an abnormally
    dispersed group (detail table), nested_anova() is for comparing
    magnitudes across levels (species vs individual vs device)."""
    X = flatten(aligned)
    out = {}
    for g in groupe.unique():
        mask = (groupe == g).values
        n = int(mask.sum())
        centroid = X[mask].mean(axis=0)
        ss = float(((X[mask] - centroid) ** 2).sum())
        out[g] = ss / (n - 1) if n > 1 else float("nan")
    return pd.Series(out, name="shape_variance")


def _cumulative_cells(level_codes: list[np.ndarray], level_sizes: list[int]) -> list[tuple[np.ndarray, int]]:
    """For each level i, the dense code of the CUMULATIVE partition (levels
    1..i combined) + the number of cells present. E.g. levels
    [species, caste] -> level-1 cells = species, level-2 cells =
    (species, caste) -- a "worker" caste from two different species falls
    into two distinct cells automatically, no need to pre-build a
    "species_caste" label."""
    cumulative: list[tuple[np.ndarray, int]] = []
    dense, n_dense = None, 1
    for codes, n in zip(level_codes, level_sizes):
        combined = codes if dense is None else combine_codes(dense, n_dense, codes, n)
        _, dense = np.unique(combined, return_inverse=True)
        n_dense = int(dense.max()) + 1
        cumulative.append((dense, n_dense))
    return cumulative


def _permutation_p(
    X: np.ndarray, codes: np.ndarray, n_groups: int, strata: np.ndarray | None,
    observed_ss: float, ss_parent: float, n_perm: int, rng: np.random.Generator,
) -> float:
    """Permutation p-value for ONE level. `strata=None` (broadest level,
    e.g. species) -> unstratified shuffle over the whole population.
    `strata` provided (nested level, e.g. caste within species) -> STRATIFIED
    shuffle: this level's labels are only shuffled WITHIN each parent-level
    cell (the parent partition, hence ss_parent, stays identical at every
    permutation) -- otherwise an unstratified shuffle would confound this
    level's effect with that of the levels above it."""
    count = 1  # the observed value counts as its own permutation -- avoids p=0
    if strata is None:
        for _ in range(n_perm):
            perm_codes = rng.permutation(codes)
            ss_perm, _ = between_within_ss(X, perm_codes, n_groups)
            if ss_perm >= observed_ss:
                count += 1
        return count / (n_perm + 1)

    strata_groups = [np.where(strata == g)[0] for g in np.unique(strata)]
    n_parent = int(strata.max()) + 1
    codes = np.asarray(codes)
    for _ in range(n_perm):
        perm = codes.copy()
        for idx in strata_groups:
            perm[idx] = rng.permutation(perm[idx])
        combined = combine_codes(strata, n_parent, perm, n_groups)
        _, dense = np.unique(combined, return_inverse=True)
        n_present = int(dense.max()) + 1
        ss_cells, _ = between_within_ss(X, dense, n_present)
        ss_perm = ss_cells - ss_parent
        if ss_perm >= observed_ss:
            count += 1
    return count / (n_perm + 1)


def nested_anova(
    X: np.ndarray, levels: list[tuple[str, pd.Series]],
    n_perm: int = 0, rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """N-level nested ANOVA. `levels` = [(name, labels), ...] from the
    broadest (e.g. species) to the finest (e.g. individual) -- each `labels`
    is a pd.Series the same length as X, row-aligned. Returns an
    SS/df/MS/F/p table, one row per level + one "Residual" row (noise not
    explained by any level, e.g. repeat photos of the same individual).

    Principle (generalization of the identity SS_total = SS_between +
    SS_within to N nested factors):
        SS(level i | levels above) =
            SS_between(cells 1..i) - SS_between(cells 1..i-1)
    Stacking one more level adds exactly the variance it explains on top of
    the levels above -- valid in sum-of-squares terms regardless of
    (often unbalanced) per-cell sample sizes. p by stratified permutation
    (see _permutation_p): n_perm=0 (default) skips significance testing
    (faster, magnitudes only)."""
    if n_perm and rng is None:
        raise ValueError("n_perm > 0 requires providing rng (np.random.default_rng(seed))")

    names = [name for name, _ in levels]
    level_codes, level_sizes = [], []
    for name, series in levels:
        if series.isna().any():
            raise ValueError(f"Level {name!r}: missing values -- exclude these rows before calling.")
        codes, uniques = pd.factorize(series, sort=True)
        level_codes.append(codes)
        level_sizes.append(len(uniques))

    cells = _cumulative_cells(level_codes, level_sizes)
    total_ss = _total_ss(X)

    rows = []
    ss_prev, n_prev = 0.0, 1  # "level 0" = a single cell (the grand mean)
    for i, (name, (dense, n_present)) in enumerate(zip(names, cells)):
        ss_cells, _ = between_within_ss(X, dense, n_present)
        ss_level = ss_cells - ss_prev
        df_level = n_present - n_prev

        p_value = float("nan")
        if n_perm:
            strata = None if i == 0 else cells[i - 1][0]
            p_value = _permutation_p(
                X, level_codes[i], level_sizes[i], strata, ss_level, ss_prev, n_perm, rng
            )
        rows.append({"source": name, "SS": ss_level, "df": df_level, "p (permutation)": p_value})
        ss_prev, n_prev = ss_cells, n_present

    rows.append({
        "source": "Residual", "SS": total_ss - ss_prev, "df": len(X) - n_prev,
        "p (permutation)": float("nan"),
    })

    table = pd.DataFrame(rows).set_index("source")
    table["MS"] = [variance_summary(r.SS, r.df) for r in table.itertuples()]
    ms_residual = table.loc["Residual", "MS"]
    table["F"] = table["MS"] / ms_residual
    table.loc["Residual", "F"] = float("nan")
    return table[["SS", "df", "MS", "F", "p (permutation)"]]
