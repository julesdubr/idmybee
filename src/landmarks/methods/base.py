"""base.py
Contrat commun aux méthodes de numérotation (Phase 3) : associer un nuage de
landmarks non-ordonné (sortie brute d'un détecteur) aux positions numérotées
d'une forme de référence (voir landmarks.build_reference).

Toute nouvelle méthode (Hungarian+Umeyama, graph matching, ...) doit exposer
une fonction module-level :

    def numerate(landmarks: np.ndarray, reference: np.ndarray) -> NumberingResult: ...

pour rester interchangeable dans landmarks/renumber.py (pas besoin d'une
classe abstraite pour ça -- voir le registre `METHODS` de renumber.py).

Le `status` retourné ici ne couvre que les échecs intrinsèques à la méthode
(ex: nombre de landmarks incompatible avec la référence). Le classement en
SUSPECT pour un coût anormalement élevé par rapport au reste de la
population est une décision de population, pas d'un spécimen isolé : elle
est prise par l'appelant (renumber.py) une fois tous les scores connus, pas
ici.

Anciennement numbering/base.py -- déplacé sous landmarks/methods/ : la
numérotation n'est pas une étape indépendante de la prédiction des
landmarks, c'est la suite directe du même travail (positionner les
landmarks d'une image), donc les deux vivent maintenant dans le même module
`landmarks`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NumberingResult:
    numbered: np.ndarray   # (n_zones, 2) -- landmarks réordonnés selon la référence
    status: str            # "OK" | "SUSPECT" | "FAILED"
    score: float           # coût de registration (plus bas = meilleur), comparable
                           # seulement entre spécimens numérotés par la MÊME méthode
    reason: str = ""       # motif explicite si status != "OK"
