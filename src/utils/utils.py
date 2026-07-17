"""Fonctions utilitaires partagées : heatmaps gaussiennes, redimensionnement
avec conservation du ratio (letterbox), extraction de coordonnées depuis les
heatmaps prédites, et remapping vers les coordonnées de l'image d'origine.
"""

from __future__ import annotations

import numpy as np
import cv2


def letterbox_resize(image: np.ndarray, landmarks: np.ndarray, target_size: int):
    """Redimensionne `image` pour que son plus grand côté fasse `target_size`,
    puis rembourre (padding blanc) pour obtenir un carré target_size x target_size.
    Retourne l'image transformée, les landmarks transformés, et un dict
    permettant l'opération inverse.
    """
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_h, pad_w = target_size - new_h, target_size - new_w
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )

    new_landmarks = landmarks.copy().astype(float)
    new_landmarks[:, 0] = landmarks[:, 0] * scale + left
    new_landmarks[:, 1] = landmarks[:, 1] * scale + top

    transform = {
        "scale": scale,
        "pad_left": left,
        "pad_top": top,
        "orig_h": h,
        "orig_w": w,
        "target_size": target_size,
    }
    return padded, new_landmarks, transform


def crop_around_landmarks(
    image: np.ndarray, landmarks: np.ndarray, margin_ratio: float = 0.25
):
    """Recadre l'image autour de la bounding box des landmarks, avec une marge.
    Utile en Phase 1 (avant d'avoir un détecteur d'aile automatique).
    Retourne l'image recadrée, les landmarks translatés, et l'offset (x0, y0)
    pour repasser aux coordonnées de l'image d'origine.

    Lève ValueError si les landmarks ne recoupent pas du tout l'image (signe
    que les coordonnées ne correspondent pas à ce fichier image -- décalage
    entre l'annotation TPS et l'image sur disque, ex. recadrage effectué
    après l'annotation). Cette vérification est une sécurité supplémentaire :
    `load_samples` dans train.py est censé filtrer ces cas en amont.
    """
    h, w = image.shape[:2]
    x_min, y_min = landmarks.min(axis=0)
    x_max, y_max = landmarks.max(axis=0)

    if x_max < 0 or x_min > w or y_max < 0 or y_min > h:
        raise ValueError(
            f"Landmarks hors de l'image : bbox=({x_min:.0f},{y_min:.0f},{x_max:.0f},{y_max:.0f}) "
            f"vs image {w}x{h} -- coordonnées probablement désalignées avec ce fichier."
        )

    box_w, box_h = x_max - x_min, y_max - y_min
    mx, my = box_w * margin_ratio, box_h * margin_ratio

    x0 = int(np.clip(x_min - mx, 0, w - 1))
    y0 = int(np.clip(y_min - my, 0, h - 1))
    x1 = int(np.clip(x_max + mx, x0 + 1, w))
    y1 = int(np.clip(y_max + my, y0 + 1, h))

    cropped = image[y0:y1, x0:x1]
    new_landmarks = landmarks.copy().astype(float)
    new_landmarks[:, 0] -= x0
    new_landmarks[:, 1] -= y0

    return cropped, new_landmarks, (x0, y0)


def unletterbox_coords(coords: np.ndarray, transform: dict) -> np.ndarray:
    """Inverse de letterbox_resize : ramène des coordonnées dans l'espace
    target_size x target_size vers l'espace de l'image d'origine (avant
    letterbox, mais après un éventuel crop préalable — combiner avec l'offset
    du crop si besoin)."""
    out = coords.copy().astype(float)
    out[:, 0] = (coords[:, 0] - transform["pad_left"]) / transform["scale"]
    out[:, 1] = (coords[:, 1] - transform["pad_top"]) / transform["scale"]
    return out


def generate_heatmaps(
    landmarks: np.ndarray, image_size: int, heatmap_size: int, sigma: float = 1.5
) -> np.ndarray:
    """Génère un tenseur (n_points, heatmap_size, heatmap_size) de gaussiennes
    centrées sur chaque landmark (coordonnées exprimées dans l'espace image_size).
    """
    n_points = landmarks.shape[0]
    heatmaps = np.zeros((n_points, heatmap_size, heatmap_size), dtype=np.float32)
    stride = image_size / heatmap_size

    grid_y, grid_x = np.mgrid[0:heatmap_size, 0:heatmap_size]

    for k in range(n_points):
        cx = landmarks[k, 0] / stride
        cy = landmarks[k, 1] / stride
        if cx < 0 or cy < 0 or cx >= heatmap_size or cy >= heatmap_size:
            continue  # point hors cadre après crop/resize : heatmap vide
        heatmaps[k] = np.exp(
            -((grid_x - cx) ** 2 + (grid_y - cy) ** 2) / (2 * sigma**2)
        )

    return heatmaps


def heatmaps_to_coords(heatmaps: np.ndarray, image_size: int) -> np.ndarray:
    """Extrait les coordonnées (n_points, 2) dans l'espace image_size à partir
    des heatmaps prédites (n_points, H, W), avec raffinement sous-pixel par
    interpolation quadratique locale autour du maximum.
    """
    n_points, hm_h, hm_w = heatmaps.shape
    stride = image_size / hm_h
    coords = np.zeros((n_points, 2), dtype=float)

    for k in range(n_points):
        hm = heatmaps[k]
        idx = np.unravel_index(np.argmax(hm), hm.shape)
        y, x = idx

        # raffinement sous-pixel simple (décalage vers le voisin le plus fort)
        px, py = float(x), float(y)
        if 0 < x < hm_w - 1:
            px += 0.25 * np.sign(hm[y, x + 1] - hm[y, x - 1])
        if 0 < y < hm_h - 1:
            py += 0.25 * np.sign(hm[y + 1, x] - hm[y - 1, x])

        coords[k, 0] = px * stride
        coords[k, 1] = py * stride

    return coords


def normalized_mean_error(
    pred: np.ndarray, gt: np.ndarray, ref_idx_a: int, ref_idx_b: int
) -> float:
    """Erreur euclidienne moyenne sur tous les points, normalisée par la
    distance entre deux landmarks de référence (ex. base et bout de l'aile),
    pour rendre l'erreur comparable entre images d'échelle différente."""
    ref_len = np.linalg.norm(gt[ref_idx_a] - gt[ref_idx_b])
    if ref_len < 1e-6:
        return float("nan")
    dists = np.linalg.norm(pred - gt, axis=1)
    return float(dists.mean() / ref_len)
