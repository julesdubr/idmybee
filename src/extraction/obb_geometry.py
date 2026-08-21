"""
obb_geometry.py
Géométrie d'extraction d'une aile à partir d'une OBB prédite.

Responsabilité unique :
    transformer une OBB en crop horizontal puis en image 512x256
    en conservant exactement la logique de normalisation du pipeline lourd.
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def read_image(path: Path):
    """Charge une image en BGR uint8, avec fallback Pillow."""
    image = cv2.imread(str(path))
    if image is not None and image.size > 0:
        return image

    try:
        pil_image = Image.open(path).convert("RGB")
        return cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def oriented_crop(
    image: np.ndarray,
    obb_points: np.ndarray,
    bg_color=(255, 255, 255),
    pad: float = 0.1,
):
    """
    Reproduit la géométrie du pipeline lourd à partir des 4 coins OBB.

    L'OBB est redressée suivant son grand axe, puis recadrée avec un padding
    symétrique. Le fond n'est pas masqué, comme dans le pipeline lourd par défaut.

    Retourne
        crop
        aspect_ratio
        obb_normalized
    """
    points = np.asarray(obb_points, dtype=np.float32).reshape(4, 2)

    rect = cv2.minAreaRect(points)
    (_, _), (w_rect, h_rect), _ = rect

    if w_rect < 1 or h_rect < 1:
        return None, None, None

    box = cv2.boxPoints(rect).astype(np.float32)

    edge1 = box[1] - box[0]
    edge2 = box[2] - box[1]
    long_edge = edge1 if np.linalg.norm(edge1) >= np.linalg.norm(edge2) else edge2

    raw_angle = np.degrees(np.arctan2(long_edge[1], long_edge[0]))
    theta_deg = ((raw_angle + 90.0) % 180.0) - 90.0

    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)

    matrix = cv2.getRotationMatrix2D(center, theta_deg, 1.0)

    cos_value = abs(matrix[0, 0])
    sin_value = abs(matrix[0, 1])

    new_width = int(height * sin_value + width * cos_value)
    new_height = int(height * cos_value + width * sin_value)

    matrix[0, 2] += new_width / 2.0 - center[0]
    matrix[1, 2] += new_height / 2.0 - center[1]

    rotated = cv2.warpAffine(
        image,
        matrix,
        (new_width, new_height),
        borderValue=bg_color,
    )

    corners_h = np.hstack([box, np.ones((4, 1), dtype=np.float32)])
    rotated_corners = (matrix @ corners_h.T).T

    x_min, y_min = rotated_corners.min(axis=0)
    x_max, y_max = rotated_corners.max(axis=0)

    x_min = max(0, int(round(x_min)))
    y_min = max(0, int(round(y_min)))
    x_max = min(new_width, int(round(x_max)))
    y_max = min(new_height, int(round(y_max)))

    if x_max <= x_min or y_max <= y_min:
        return None, None, None

    pad_x = int((x_max - x_min) * pad / 2.0)
    pad_y = int((y_max - y_min) * pad / 2.0)

    crop = rotated[
        max(0, y_min - pad_y):min(new_height, y_max + pad_y),
        max(0, x_min - pad_x):min(new_width, x_max + pad_x),
    ]

    if crop.size == 0:
        return None, None, None

    long_side = max(w_rect, h_rect)
    short_side = max(min(w_rect, h_rect), 1e-6)
    aspect_ratio = float(long_side / short_side)

    normalized_box = box.copy()
    normalized_box[:, 0] /= float(width)
    normalized_box[:, 1] /= float(height)

    return crop, aspect_ratio, normalized_box.reshape(-1).tolist()


def letterbox(
    crop: np.ndarray,
    out_width: int = 512,
    out_height: int = 256,
    bg_color=(255, 255, 255),
):
    """Redimensionne sans déformation puis complète avec le fond."""
    height, width = crop.shape[:2]

    if height == 0 or width == 0:
        return None

    scale = min(out_width / width, out_height / height)

    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))

    resized = cv2.resize(
        crop,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.full(
        (out_height, out_width, 3),
        bg_color,
        dtype=np.uint8,
    )

    x_offset = (out_width - new_width) // 2
    y_offset = (out_height - new_height) // 2

    canvas[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width,
    ] = resized

    return canvas
