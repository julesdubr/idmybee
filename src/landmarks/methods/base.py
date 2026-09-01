"""base.py
Common contract for numbering methods (Phase 3): matching an unordered
landmark cloud (a detector's raw output) to the numbered positions of a
reference shape (see landmarks.build_reference).

Any new method (Hungarian+Umeyama, graph matching, ...) must expose a
module-level function:

    def numerate(landmarks: np.ndarray, reference: np.ndarray) -> NumberingResult: ...

to stay interchangeable in landmarks/renumber.py (no need for an abstract
class for that -- see renumber.py's `METHODS` registry).

The `status` returned here only covers failures intrinsic to the method
(e.g. landmark count incompatible with the reference). Classifying as
SUSPECT for an abnormally high cost relative to the rest of the population
is a population-level decision, not a single specimen's: it's made by the
caller (renumber.py) once every score is known, not here.

Formerly numbering/base.py -- moved under landmarks/methods/: numbering
isn't a step independent from landmark prediction, it's the direct
continuation of the same work (placing an image's landmarks), so both now
live in the same `landmarks` module.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NumberingResult:
    numbered: np.ndarray   # (n_zones, 2) -- landmarks reordered to match the reference
    status: str            # "OK" | "SUSPECT" | "FAILED"
    score: float           # registration cost (lower = better), comparable
                           # only between specimens numbered by the SAME method
    reason: str = ""       # explicit reason if status != "OK"
