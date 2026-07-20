"""Alignement rigide+échelle d'un nuage de points non-ordonné sur des
'zones' de référence étiquetées, via Hungarian assignment + Umeyama
(similarité, sans réflexion), alternés jusqu'à convergence.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def umeyama(src, dst):
    """Transform (R, scale, t), det(R)=+1, tel que dst_hat = scale*(src@R.T)+t."""
    mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - mu_src, dst - mu_dst
    var_src = (src_c ** 2).sum() / len(src)
    Sigma = (dst_c.T @ src_c) / len(src)
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    scale = np.trace(np.diag(D) @ S) / var_src
    t = mu_dst - scale * (R @ mu_src)
    return R, scale, t


def apply_transform(pts, R, scale, t):
    return scale * (pts @ R.T) + t


def register_unlabeled(points, zones, n_iter=15, angle_inits=(0, 90, 180, 270),
                        mirror_options=(False, True)):
    """
    points: (k,2) nuage non-ordonné (ex: prédictions brutes du UNet)
    zones:  (k,2) template de référence étiqueté
    Multi-départs (rotation grossière x miroir) pour éviter les optima locaux.
    Retourne (assignment, cost, (R, scale, t)).
    """
    pts0 = points - points.mean(axis=0)
    best = None
    for angle in angle_inits:
        th = np.deg2rad(angle)
        rot0 = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        for mirror in mirror_options:
            mir = np.array([[-1, 0], [0, 1]]) if mirror else np.eye(2)
            cur = pts0 @ (rot0 @ mir).T
            assign_prev = None
            for _ in range(n_iter):
                cost = ((cur[:, None, :] - zones[None, :, :]) ** 2).sum(axis=2)
                row_ind, col_ind = linear_sum_assignment(cost)
                assign = col_ind[np.argsort(row_ind)]
                if assign_prev is not None and np.array_equal(assign, assign_prev):
                    break
                assign_prev = assign
                R, scale, t = umeyama(pts0, zones[assign])
                cur = apply_transform(pts0, R, scale, t)
            final_cost = ((cur - zones[assign]) ** 2).sum(axis=1).mean()
            if best is None or final_cost < best[1]:
                best = (assign.copy(), final_cost, (R, scale, t))
    return best
