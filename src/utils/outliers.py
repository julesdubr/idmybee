"""outliers.py
Diagnostic post-GPA : distingue le bruit de détection ponctuel (1-2
landmarks décalés) des échecs de registration complets (la quasi-totalité
des landmarks du spécimen décalés -- specimen mal aligné dans son
ensemble, souvent une forme d'aile hors gabarit, ou une mauvaise
renumérotation).

Anciennement tools/flag_outlier_specimens.py (script CLI autonome). Extrait
ici en pur module (plus de __main__/argparse) pour être appelé directement
par numbering.reconstruct_tps.py juste après la renumérotation : la
population de spécimens et leur numérotation viennent d'être calculées au
même endroit, pas besoin de réécrire un TPS intermédiaire puis de le
reparser dans un second script pour ce diagnostic.

Le seuil (médiane + `mad_factor`*MAD de la distance à la position médiane,
par landmark) est calculé PAR ESPÈCE, pas sur l'ensemble du jeu de données :
les espèces ont des formes d'aile différentes par nature (c'est la base
même de la classification), donc un seuil global confondrait "aile
différente parce que d'une autre espèce" avec "aile mal alignée" --
gonflant artificiellement le taux d'outliers des espèces les moins
représentées ou aux ailes les plus atypiques (ex: B. rupestris). Un groupe
avec moins de `min_group_size` spécimens n'a pas de médiane/MAD fiable : il
est laissé de côté (considéré OK) plutôt que d'inventer un seuil.
"""
from __future__ import annotations

import numpy as np

from utils.gpa import gpagen
from utils.tps_io import ImageLandmarks

MIN_GROUP_SIZE = 10   # en dessous, une médiane/MAD par landmark n'est pas fiable
MAD_FACTOR = 6.0       # médiane + MAD_FACTOR*MAD -> seuil "landmark outlier"
HEAVY_LANDMARK_FRAC = 0.55  # au-delà de cette fraction de landmarks outliers -> specimen entier suspect


def outlier_matrix(aligned: np.ndarray, mad_factor: float = MAD_FACTOR) -> np.ndarray:
    """(n_specimens, n_landmarks) bool : True si le point est loin de la
    position médiane de son landmark (médiane + mad_factor*MAD)."""
    med = np.median(aligned, axis=0)
    dist = np.linalg.norm(aligned - med[None, :, :], axis=2)
    mad = np.median(np.abs(dist - np.median(dist, axis=0)), axis=0)
    thresh = np.median(dist, axis=0) + mad_factor * mad
    return dist > thresh[None, :]


def flag_by_species(
    specimens: list[ImageLandmarks],
    species: np.ndarray,
    heavy_frac: float = HEAVY_LANDMARK_FRAC,
    min_group_size: int = MIN_GROUP_SIZE,
    mad_factor: float = MAD_FACTOR,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """GPA + seuil MAD calculés séparément pour chaque espèce (voir docstring
    du module). Retourne (n_outlier, heavy), alignés sur `specimens` :
    n_outlier = nombre de landmarks outliers du spécimen, heavy = True si
    >= heavy_frac des landmarks du spécimen sont outliers (specimen entier
    suspect, pas juste un point bruité)."""
    if not specimens:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=bool)

    n_landmarks = specimens[0].n_points
    n_outlier = np.zeros(len(specimens), dtype=int)
    heavy = np.zeros(len(specimens), dtype=bool)

    for sp_name in sorted(set(species)):
        idx = np.where(species == sp_name)[0]
        if len(idx) < min_group_size:
            if verbose:
                print(f"  {sp_name}: {len(idx)} spécimen(s), < {min_group_size} -- non évalué (considéré OK)")
            continue
        group = [specimens[i] for i in idx]
        result = gpagen([s.landmarks for s in group])
        group_outlier = outlier_matrix(result.aligned, mad_factor=mad_factor)
        group_n_outlier = group_outlier.sum(axis=1)
        n_outlier[idx] = group_n_outlier
        heavy[idx] = group_n_outlier >= heavy_frac * n_landmarks

    return n_outlier, heavy