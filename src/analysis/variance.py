"""variance.py
Décomposition de variance de forme (distance de Procrustes² sur landmarks
GPA-alignés) en facteurs emboîtés (ex : espèce ⊃ caste ⊃ individu, ou
individu ⊃ appareil), avec test de signification par permutation (PERMANOVA
-- pas de distribution F théorique : la normalité multivariée est une
hypothèse trop forte en haute dimension, la validité du test vient de la
permutation des labels, même logique que geomorph::procD.lm).

Tout tourne autour de nested_anova() : ANOVA emboîtée à N niveaux (le
premier niveau = le plus large, ex: espèce ; le dernier = le plus fin, ex:
individu). Un seul niveau = ANOVA à un facteur classique. Généralise
l'ancienne version à 2 niveaux (espèce/appareil) codée en dur -- même
fonction pour espèce⊃caste⊃individu (3 niveaux) et individu⊃appareil
(2 niveaux), voir analysis/variance_report.py.

within_group_variance() est une fonction à part : une table DESCRIPTIVE
(une ligne par groupe, ex: par espèce) pour repérer un groupe anormalement
dispersé. Elle ne remplace PAS le MS résiduel poolé de nested_anova() (qui
lui divise par N - n_groupes sur toute la population) -- voir sa docstring.
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
    """(SS_between, SS_within) pour un facteur seul -- SS_total = SS_between +
    SS_within (identité d'ANOVA standard, valable quels que soient les
    effectifs par groupe). `codes` doit être dense (0..n_groups-1) -- les
    groupes jamais observés comptent pour 0 (comptage réel via `counts`,
    pas une moyenne inventée)."""
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
    """Code entier combiné (a, b) -> code unique dans [0, n_a*n_b)."""
    return codes_a.astype(np.int64) * n_b + codes_b.astype(np.int64)


def variance_summary(ss: float, df: int) -> float:
    """Mean square (SS/df) -- magnitude comparable entre niveaux."""
    return ss / df if df > 0 else float("nan")


def within_group_variance(aligned: np.ndarray, groupe: pd.Series) -> pd.Series:
    """Variance de Procrustes DESCRIPTIVE, groupe par groupe (une ligne par
    espèce/appareil/etc.) : distance quadratique moyenne au centroïde du
    groupe, corrigée du biais d'estimation (n-1, pas n -- le centroïde est
    lui-même estimé à partir des mêmes points, donc n degrés de liberté
    surestimeraient la précision).

    ATTENTION : ce n'est PAS la même quantité que le MS résiduel poolé de
    nested_anova() (qui divise par N - n_groupes sur TOUTE la population en
    une fois). Les deux coexistent : celle-ci sert à repérer un groupe
    anormalement dispersé (table de détail), nested_anova() sert à comparer
    les magnitudes entre niveaux (espèce vs individu vs appareil)."""
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
    """Pour chaque niveau i, le code dense de la partition CUMULÉE (niveaux
    1..i combinés) + le nombre de cellules présentes. Ex : niveaux
    [espèce, caste] -> cellules niveau 1 = espèces, cellules niveau 2 =
    (espèce, caste) -- une caste "worker" de deux espèces différentes tombe
    dans deux cellules distinctes automatiquement, pas besoin de
    pré-construire un label "espèce_caste"."""
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
    """p-value par permutation pour UN niveau. `strata=None` (niveau le plus
    large, ex: espèce) -> mélange non stratifié sur toute la population.
    `strata` fourni (niveau emboîté, ex: caste dans espèce) -> mélange
    STRATIFIÉ : les labels de ce niveau ne sont mélangés qu'AU SEIN de
    chaque cellule du niveau parent (la partition parente, donc ss_parent,
    reste identique à chaque permutation) -- sinon un mélange non stratifié
    confondrait l'effet de ce niveau avec celui des niveaux au-dessus."""
    count = 1  # l'observé compte comme sa propre permutation -- évite p=0
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
    """ANOVA emboîtée à N niveaux. `levels` = [(nom, labels), ...] du plus
    large (ex: espèce) au plus fin (ex: individu) -- chaque `labels` est une
    pd.Series de même longueur que X, alignée ligne à ligne. Retourne une
    table SS/df/MS/F/p, une ligne par niveau + une ligne "Résiduel" (bruit
    non expliqué par aucun niveau, ex: reprises photo du même individu).

    Principe (généralisation de l'identité SS_total = SS_between +
    SS_within à N facteurs emboîtés) :
        SS(niveau i | niveaux au-dessus) =
            SS_between(cellules 1..i) - SS_between(cellules 1..i-1)
    Empiler un niveau supplémentaire ajoute exactement la variance qu'il
    explique en plus des niveaux au-dessus -- valable en somme des carrés
    quels que soient les effectifs (souvent déséquilibrés) par cellule. p par
    permutation stratifiée (voir _permutation_p) : n_perm=0 (défaut) saute
    les tests de significativité (plus rapide, juste les magnitudes)."""
    if n_perm and rng is None:
        raise ValueError("n_perm > 0 nécessite de fournir rng (np.random.default_rng(seed))")

    names = [name for name, _ in levels]
    level_codes, level_sizes = [], []
    for name, series in levels:
        if series.isna().any():
            raise ValueError(f"Niveau {name!r} : valeurs manquantes -- exclure ces lignes avant l'appel.")
        codes, uniques = pd.factorize(series, sort=True)
        level_codes.append(codes)
        level_sizes.append(len(uniques))

    cells = _cumulative_cells(level_codes, level_sizes)
    total_ss = _total_ss(X)

    rows = []
    ss_prev, n_prev = 0.0, 1  # "niveau 0" = une seule cellule (la moyenne globale)
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
        "source": "Résiduel", "SS": total_ss - ss_prev, "df": len(X) - n_prev,
        "p (permutation)": float("nan"),
    })

    table = pd.DataFrame(rows).set_index("source")
    table["MS"] = [variance_summary(r.SS, r.df) for r in table.itertuples()]
    ms_residual = table.loc["Résiduel", "MS"]
    table["F"] = table["MS"] / ms_residual
    table.loc["Résiduel", "F"] = float("nan")
    return table[["SS", "df", "MS", "F", "p (permutation)"]]