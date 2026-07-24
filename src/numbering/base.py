"""base.py
Contrat commun aux méthodes de numérotation (Phase 3) : associer un nuage de
landmarks non-ordonné (sortie brute d'un détecteur) aux positions numérotées
d'un template de référence.

Toute nouvelle méthode (Hungarian+Umeyama, graph matching, ...) doit exposer
une fonction module-level :

    def numerate(landmarks: np.ndarray, reference: np.ndarray) -> NumberingResult: ...

pour rester interchangeable dans reconstruct_tps.py (pas besoin d'une classe
abstraite pour ça).

Le `status` retourné ici ne couvre que les échecs intrinsèques à la méthode
(ex: nombre de landmarks incompatible avec le template). Le classement en
SUSPECT pour un coût anormalement élevé par rapport au reste de la population
est une décision de population, pas d'un spécimen isolé : elle est prise par
l'appelant (reconstruct_tps.py) une fois tous les scores connus, pas ici.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NumberingResult:
    numbered: np.ndarray   # (n_zones, 2) -- landmarks réordonnés selon le template
    status: str            # "OK" | "SUSPECT" | "FAILED"
    score: float           # coût de registration (plus bas = meilleur), comparable
                           # seulement entre spécimens numérotés par la MÊME méthode
    reason: str = ""       # motif explicite si status != "OK"