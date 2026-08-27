"""hungarian_umeyama.py
Numérotation par assignation hongroise + Umeyama (similarité sans réflexion),
alternées jusqu'à convergence, avec multi-départs (rotation grossière x
miroir) pour éviter les optima locaux.

Anciennement numbering/hungarian_umeyama.py (register.py avant ça). Déplacé
sous landmarks/methods/ pour cohabiter avec d'autres méthodes (ex:
graph_matching.py, testée puis retirée -- moins bonne sur ce jeu de
données) sous un nom qui décrit l'approche plutôt que le rôle générique ;
implémente le contrat landmarks.methods.base.

La rotation+échelle est déléguée à utils.alignment.kabsch_umeyama (le même
coeur SVD que la GPA), pour ne plus réimplémenter cette algèbre séparément.

Principe en une phrase : on ne sait pas à l'avance quel landmark détecté
correspond à quelle zone de la référence (le UNet produit un nuage de
points non-ordonné) -- on alterne donc "étant donné l'orientation actuelle,
quelle est la meilleure assignation point<->zone (Hongrois)" et "étant
donné cette assignation, quelle est la meilleure rotation/échelle
(Umeyama)" jusqu'à ce que l'assignation ne change plus plus. Comme cette
alternance peut se bloquer sur un optimum local (ex: l'aile numérotée à
l'envers, un mauvais minimum mais stable), on la relance depuis plusieurs
orientations de départ et on garde la meilleure.
"""
# NB: l'ancienne détection d'ambiguïté d'orientation par-spécimen
# (comparaison meilleur/deuxième-meilleur départ, paramètre
# `ambiguity_ratio`) a été retirée -- voir la docstring de `numerate()`.
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from landmarks.methods.base import NumberingResult
from utils.alignment import kabsch_umeyama


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Similarité complète (R, scale, t), sans réflexion : dst_hat = scale*(src@R.T)+t."""
    mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
    R, scale = kabsch_umeyama(src - mu_src, dst - mu_dst, estimate_scale=True)
    t = mu_dst - scale * (R @ mu_src)
    return R, scale, t


def apply_transform(pts: np.ndarray, R: np.ndarray, scale: float, t: np.ndarray) -> np.ndarray:
    return scale * (pts @ R.T) + t


def _register_from_start(pts0: np.ndarray, cur_init: np.ndarray, zones: np.ndarray, n_iter: int):
    """Assignation+Umeyama alternés jusqu'à convergence, depuis UN départ.

    `pts0` : nuage centré original -- le transform est TOUJOURS recalculé
    depuis celui-ci, jamais depuis `cur` de l'itération précédente (sinon les
    rotations/miroirs initiaux se composeraient au lieu de ne servir qu'à
    orienter la toute première assignation).
    `cur_init` : pts0 après rotation/miroir initial, sert uniquement à ce
    premier calcul de coût, pour éviter un optimum local de l'assignation.

    Boucle : (1) coût = distance^2 entre chaque point courant et chaque
    zone -> Hongrois donne l'assignation qui minimise la somme des coûts ;
    (2) Umeyama réaligne pts0 entier sur les zones dans cet ordre ;
    (3) on recommence avec la nouvelle position -- jusqu'à ce que
    l'assignation elle-même ne change plus (convergence), ou n_iter atteint.
    """
    cur = cur_init
    assign_prev = None
    transform = None
    assign = None
    for _ in range(n_iter):
        cost = ((cur[:, None, :] - zones[None, :, :]) ** 2).sum(axis=2)
        row_ind, col_ind = linear_sum_assignment(cost)
        assign = col_ind[np.argsort(row_ind)]
        if assign_prev is not None and np.array_equal(assign, assign_prev):
            break
        assign_prev = assign
        transform = umeyama(pts0, zones[assign])
        cur = apply_transform(pts0, *transform)
    final_cost = float(((cur - zones[assign]) ** 2).sum(axis=1).mean())
    return assign, final_cost, transform


def numerate(
    landmarks: np.ndarray,
    reference: np.ndarray,
    n_iter: int = 15,
    angle_inits: tuple[float, ...] = (0, 90, 180, 270),
    mirror_options: tuple[bool, ...] = (False, True),
) -> NumberingResult:
    """Implémente le contrat landmarks.methods.base : renumérote `landmarks`
    (k, 2) selon l'ordre de `reference` (n_zones, 2).

    Ne gère pour l'instant que le cas k == n_zones (nombre de landmarks
    détectés = nombre attendu) ; le cas k != n_zones (points en trop/en
    moins) reste un FAILED explicite plutôt qu'une assignation rectangulaire
    non testée -- à traiter séparément si le détecteur en a besoin un jour
    (voir aussi landmarks/renumber.py, qui court-circuite déjà ce cas avant
    d'appeler numerate()).

    Retourne toujours le meilleur des 8 départs (4 rotations x 2 miroirs :
    on ne sait a priori ni dans quel sens ni avec quelle chiralité l'aile a
    été photographiée), avec son coût de registration dans `score`. Ce
    module ne juge plus lui-même de la fiabilité de ce coût (ancien
    `ambiguity_ratio`, comparant meilleur et deuxième-meilleur départ) : sur
    les données réelles, ce critère par-spécimen s'est avéré beaucoup trop
    agressif (il marquait SUSPECT la quasi-totalité des spécimens, y
    compris des registrations visiblement correctes) car deux départs
    peuvent légitimement converger vers des coûts proches sans que
    l'assignation soit fausse pour autant.

    Voir landmarks.methods.base : la détection d'un coût anormal est une
    décision de population, pas d'un spécimen isolé -- landmarks/renumber.py
    s'en charge maintenant via une méthode plus robuste (comparaison
    post-GPA à la position médiane du landmark, PAR ESPÈCE, voir
    utils.outliers), qui distingue mieux une vraie erreur de registration
    d'une simple variation de forme.
    """
    if landmarks.shape[0] != reference.shape[0]:
        return NumberingResult(
            numbered=landmarks,
            status="FAILED",
            score=float("inf"),
            reason=(
                f"{landmarks.shape[0]} landmarks détectés, "
                f"{reference.shape[0]} attendus dans le template"
            ),
        )

    pts0 = landmarks - landmarks.mean(axis=0)
    best_assign, best_cost = None, float("inf")
    for angle in angle_inits:
        th = np.deg2rad(angle)
        rot0 = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        for mirror in mirror_options:
            mir = np.array([[-1, 0], [0, 1]]) if mirror else np.eye(2)
            cur_init = pts0 @ (rot0 @ mir).T
            assign, cost, _ = _register_from_start(pts0, cur_init, zones=reference, n_iter=n_iter)
            if cost < best_cost:
                best_assign, best_cost = assign, cost

    # `best_assign[k]` = indice de la zone attribuée au k-ième landmark
    # détecté. On veut l'inverse : pour chaque zone j (dans l'ordre de la
    # référence), quel landmark détecté lui correspond -- `inv` est donc la
    # permutation inverse de `best_assign`.
    inv = np.empty(len(reference), dtype=int)
    inv[best_assign] = np.arange(len(best_assign))  # slot j (zone j) <- point assigné à j
    numbered = landmarks[inv]

    return NumberingResult(numbered=numbered, status="OK", score=best_cost)
