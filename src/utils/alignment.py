"""alignment.py
Cœur SVD partagé pour tout alignement rigide 2D sans réflexion (Kabsch /
Umeyama) : rotation optimale, avec échelle optionnelle.

Utilisé par :
- utils/gpa.py (rotation seule -- les formes sont déjà centrées/mises à
  l'échelle en amont par la standardisation GPA, donc scale=1 toujours).
- numbering/hungarian_umeyama.py (similarité complète -- rotation+échelle,
  les landmarks bruts d'un détecteur ne sont ni centrés ni à l'échelle du
  template de référence).

Avant cette extraction, gpa.py et register.py réimplémentaient chacun leur
version de ce calcul (même algèbre, formulée différemment -- covariance
croisée transposée selon le fichier), avec le risque qu'un bug ou un
changement de convention dans l'un ne soit pas reporté dans l'autre.
"""
from __future__ import annotations

import numpy as np


def _kabsch_svd(source_c: np.ndarray, target_c: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """SVD de la covariance croisée + correction anti-réflexion.

    `source_c`/`target_c` : (n_points, 2), déjà centrés (moyenne nulle).
    Retourne (R, D, d) :
    - R : rotation 2x2 optimale (det(R) = +1), alignant source sur target
      via `source_c @ R.T`.
    - D : valeurs singulières de la covariance croisée (utiles pour le calcul
      de l'échelle optimale, voir kabsch_umeyama).
    - d : signe de correction anti-réflexion appliqué à l'axe de plus faible
      variance (+1 ou -1), aussi nécessaire pour l'échelle.
    """
    H = source_c.T @ target_c
    U, D, Vt = np.linalg.svd(H)
    d = float(np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0)
    R = Vt.T @ np.diag([1.0, d]) @ U.T
    return R, D, d


def kabsch_umeyama(
    source_c: np.ndarray, target_c: np.ndarray, estimate_scale: bool = False
) -> tuple[np.ndarray, float]:
    """Rotation (et échelle isotrope optionnelle) minimisant
    ||target_c - scale * source_c @ R.T||^2, sans réflexion (det(R)=+1).

    `source_c`/`target_c` doivent déjà être centrés (moyenne nulle) -- la
    translation, si nécessaire, reste à la charge de l'appelant (voir
    numbering/hungarian_umeyama.umeyama pour la version avec translation).

    `estimate_scale=False` (cas GPA : les formes sont déjà normalisées à
    centroid size 1, donc l'échelle optimale vaut toujours 1) retourne
    scale=1.0 sans le calculer. `estimate_scale=True` (cas registration :
    les landmarks bruts d'un détecteur ne sont pas à l'échelle du template)
    calcule l'échelle optimale au sens des moindres carrés.

    BUG CORRIGÉ (voir session Phase 3 graph_matching) : `_kabsch_svd` calcule
    `D` à partir de `H = source_c.T @ target_c`, PAS normalisé par n. Pour
    que la formule d'échelle Umeyama (`scale = trace(D)/var_source`) soit
    correcte, `D` doit provenir de la covariance croisée normalisée par n --
    cohérent avec `var_source` ci-dessous, qui LUI est déjà divisé par
    `len(source_c)`. Sans cette division, l'échelle calculée est surestimée
    d'un facteur exactement égal à n (vérifié empiriquement : n=18 -> scale
    ×18 trop grand, sur le cas trivial "vérité terrain de Tancrède contre
    son propre consensus GPA leave-one-out", où l'assignation identité
    devrait donner un coût quasi nul). La rotation R n'est PAS affectée
    (direction de U/Vt, indépendante de l'échelle de H) -- donc la GPA
    (estimate_scale=False, qui n'utilise jamais D) n'a jamais été impactée.
    """
    R, D, d = _kabsch_svd(source_c, target_c)
    if not estimate_scale:
        return R, 1.0
    n = len(source_c)
    var_source = float((source_c**2).sum() / n)
    if var_source == 0:
        raise ValueError("Spécimen dégénéré : landmarks confondus (variance nulle)")
    scale = (D[0] + d * D[1]) / n / var_source
    return R, scale