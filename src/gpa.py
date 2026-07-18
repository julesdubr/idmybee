"""Generalized Procrustes Analysis (no reflection allowed)."""
import numpy as np


def centroid_size(shape):
    c = shape - shape.mean(axis=0)
    return np.sqrt((c ** 2).sum())


def preshape(shapes):
    """Center each shape at centroid and scale to unit centroid size."""
    shapes = shapes - shapes.mean(axis=1, keepdims=True)
    sizes = np.sqrt((shapes ** 2).sum(axis=(1, 2), keepdims=True))
    return shapes / sizes


def rotate_onto(source, target):
    """Rotation-only (no reflection) optimal rotation of `source` (k,2) onto `target` (k,2)."""
    M = source.T @ target  # (2,2)
    U, S, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U = U.copy()
        U[:, -1] *= -1
        R = U @ Vt
    return source @ R, R


def gpa(shapes, max_iter=100, tol=1e-9):
    """
    shapes: (n, k, 2) array, landmarks already in homologous order.
    Returns aligned (n, k, 2), consensus (k, 2).
    """
    shapes = preshape(shapes.astype(float))
    ref = shapes[0].copy()
    aligned = shapes.copy()
    for it in range(max_iter):
        for i in range(len(shapes)):
            aligned[i], _ = rotate_onto(shapes[i], ref)
        new_ref = aligned.mean(axis=0)
        new_ref = new_ref - new_ref.mean(axis=0)
        new_ref = new_ref / np.sqrt((new_ref ** 2).sum())
        shift = np.sqrt(((new_ref - ref) ** 2).sum())
        ref = new_ref
        if shift < tol:
            break
    return aligned, ref, it + 1