"""variance.py
Décomposition de variance de Procrustes (somme des carrés, ANOVA à un ou
deux facteurs emboîtés) sur des coordonnées GPA-alignées, et test de
signification par permutation -- pas de distribution F paramétrique : la
normalité multivariée est une hypothèse trop forte sur des landmarks en
haute dimension. Même logique que geomorph::procD.lm / PERMANOVA : la
validité du test vient de la permutation des labels, pas d'une loi
théorique sur la statistique.

Toutes les fonctions travaillent sur X (n, d) -- coordonnées alignées déjà
aplaties (n_specimens, n_landmarks*2), voir utils.gpa.two_d_array -- et des
codes de groupe entiers denses 0..n_groupes-1 (voir pd.factorize), pas des
labels texte : reste indépendant de pandas et rapide à permuter.

Consommé par analysis/report_variance.py pour décomposer la variance de
forme des ailes en :
  - V_inter-espèce    : les espèces se ressemblent-elles ou non (signal
    biologique attendu -- le plus gros si la classification a un sens) ;
  - V_intra-espèce    : dispersion des individus d'une même espèce
    (variation biologique normale -- caste, sexe, population, etc.) ;
  - V_inter-appareil, AU SEIN de chaque espèce (emboîté) : à quel point le
    choix de l'appareil photo (device_type, voir images.csv) déforme la
    forme mesurée, une fois l'effet espèce retiré (variance méthodologique).
On espère V_inter-espèce > V_intra-espèce > V_inter-appareil : si la
variance méthodologique reste petite devant la variation biologique
elle-même plus petite que la séparation entre espèces, les erreurs de
lda.py s'expliquent par la biologie (espèces morphologiquement proches),
pas par un défaut de méthode.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def flatten(aligned: np.ndarray) -> np.ndarray:
    return aligned.reshape(len(aligned), -1)


def between_within_ss(X: np.ndarray, codes: np.ndarray, n_groups: int) -> tuple[float, float]:
    """(SS_between, SS_within) pour un facteur -- décomposition exacte,
    SS_total = SS_between + SS_within (identité d'ANOVA standard dans
    l'espace euclidien des coordonnées de forme aplaties, valable quels que
    soient les effectifs par groupe). `codes` doit être dense (0..n_groups-1)
    -- les groupes jamais observés comptent pour 0 dans la somme (comptage
    réel via `counts`, pas une moyenne inventée)."""
    grand_mean = X.mean(axis=0)
    total_ss = float(((X - grand_mean) ** 2).sum())

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
    """Mean square (SS/df) -- la magnitude comparable entre facteurs
    (V_inter-espèce, V_intra-espèce, V_inter-appareil)."""
    return ss / df if df > 0 else float("nan")


@dataclass
class NestedDecomposition:
    """Décomposition emboîtée : facteur B (appareil) au sein du facteur A
    (espèce). Les 3 composantes sont calculées sur le MÊME X (le
    sous-ensemble où B est connu), donc somment exactement à son SS_total."""
    n: int
    ss_a: float           # SS espèce (between), sur ce sous-ensemble
    ss_b_nested: float    # SS appareil au sein de l'espèce
    ss_error: float        # résidu (within cellule espèce x appareil)
    df_a: int
    df_b_nested: int
    df_error: int
    n_cells: int


def nested_decomposition(
    X: np.ndarray, a_codes: np.ndarray, n_a: int, b_codes: np.ndarray, n_b: int
) -> NestedDecomposition:
    """SS_total(X) = SS_a (between espèce) + SS_b_nested (between appareil,
    au sein de chaque espèce) + SS_error (résidu par cellule espèce x
    appareil). Repose sur l'identité SS_between(cellules espèce x appareil)
    = SS_between(espèce) + SS_appareil-au-sein-de-l'espèce : passer d'un
    modèle à moyennes par espèce à un modèle à moyennes par cellule
    espèce x appareil ajoute exactement la variance expliquée par
    l'appareil -- valable en somme des carrés quels que soient les
    effectifs (souvent déséquilibrés) par cellule. Une espèce présente avec
    un seul appareil contribue 0 à ss_b_nested (sa cellule unique a la même
    moyenne que l'espèce) : pas de traitement spécial nécessaire."""
    ss_a, _ = between_within_ss(X, a_codes, n_a)
    cell_codes = combine_codes(a_codes, n_a, b_codes, n_b)
    _, dense_codes = np.unique(cell_codes, return_inverse=True)
    n_cells = int(dense_codes.max()) + 1
    ss_cells, ss_error = between_within_ss(X, dense_codes, n_cells)
    ss_b_nested = ss_cells - ss_a

    n_a_present = len(np.unique(a_codes))
    return NestedDecomposition(
        n=len(X), ss_a=ss_a, ss_b_nested=ss_b_nested, ss_error=ss_error,
        df_a=n_a_present - 1, df_b_nested=n_cells - n_a_present, df_error=len(X) - n_cells,
        n_cells=n_cells,
    )


def permutation_p_between(
    X: np.ndarray, codes: np.ndarray, n_groups: int, observed_ss: float,
    n_perm: int, rng: np.random.Generator,
) -> float:
    """p-value par permutation (non stratifiée) pour un effet 'between' seul
    (ex: espèce) : mélange les labels de groupe sur toute la population,
    recalcule SS_between, compare à l'observé. Test unilatéral (une SS ne
    peut être que positive, seules les grandes valeurs sont surprenantes
    sous H0 -- pas d'effet réel du facteur)."""
    count = 1  # l'observé compte comme sa propre permutation -- évite p=0
    for _ in range(n_perm):
        perm_codes = rng.permutation(codes)
        ss_perm, _ = between_within_ss(X, perm_codes, n_groups)
        if ss_perm >= observed_ss:
            count += 1
    return count / (n_perm + 1)


def permutation_p_nested(
    X: np.ndarray, a_codes: np.ndarray, n_a: int, b_codes: np.ndarray, n_b: int,
    observed_ss_b_nested: float, n_perm: int, rng: np.random.Generator,
) -> float:
    """p-value par permutation STRATIFIÉE pour l'effet emboîté (ex: appareil
    au sein de l'espèce) : les labels d'appareil ne sont mélangés qu'AU SEIN
    de chaque espèce (la partition en espèces, donc ss_a, reste identique à
    chaque permutation) -- teste bien "l'appareil a-t-il un effet au-delà de
    l'espèce", sans quoi un mélange non-stratifié confondrait à nouveau
    effet espèce et effet appareil."""
    ss_a_fixed, _ = between_within_ss(X, a_codes, n_a)
    strata = [np.where(a_codes == g)[0] for g in range(n_a) if (a_codes == g).any()]

    count = 1
    b_codes = np.asarray(b_codes)
    for _ in range(n_perm):
        perm = b_codes.copy()
        for idx in strata:
            perm[idx] = rng.permutation(perm[idx])
        cell_codes = combine_codes(a_codes, n_a, perm, n_b)
        _, dense_codes = np.unique(cell_codes, return_inverse=True)
        n_cells = int(dense_codes.max()) + 1
        ss_cells, _ = between_within_ss(X, dense_codes, n_cells)
        ss_b_perm = ss_cells - ss_a_fixed
        if ss_b_perm >= observed_ss_b_nested:
            count += 1
    return count / (n_perm + 1)