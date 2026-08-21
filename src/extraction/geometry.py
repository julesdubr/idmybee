"""
Lecture image tolérante au format + découpe orientée + letterbox.

Aucune dépendance à YOLOE/CLIP/au manifest : ce module ne sait faire que de
la géométrie et de l'I/O image, rien d'autre.
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Aligné sur IMAGE_EXTS du manifest (src/manifest/build_manifest.py) --
# gardé ici pour qui veut valider une extension sans dépendre du manifest.
IMG_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".tif", ".tiff", ".bmp"}


def read_image(path):
    """Charge une image en BGR uint8, quel que soit le format (dont HEIC/HEIF).

    cv2.imread renvoie un tableau VIDE (pas None) sur un format qu'il ne sait
    pas décoder -- un simple `if img is None` ne suffit donc pas, il faut
    aussi vérifier sa taille. En cas d'échec on retente via PIL (qui gère
    HEIC/HEIF si `pillow-heif` est installé, cf. son enregistrement dans
    extraction/crop_wings.py). Retourne None si les deux méthodes échouent,
    plutôt que de laisser passer un tableau vide en aval.
    """
    img = cv2.imread(str(path))
    if img is not None and img.size > 0:
        return img
    try:
        pil_img = Image.open(path).convert("RGB")
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def oriented_crop(image: np.ndarray, points: np.ndarray, bg_color=(255, 255, 255),
                   mask_background: bool = False, pad: float = 0.1):
    """Redresse le grand axe de l'aile à l'horizontale et recadre.

    `points` : contour du masque en coordonnées pixel de l'image originale
    (ex. r.masks.xy[i]).

    Résolution de l'angle -- SANS CLIP, SANS ambiguïté 180° :
    cv2.minAreaRect ne donne l'angle du grand axe qu'à 180° près (une droite
    n'a pas de sens). On calcule le vecteur du plus grand côté de la boîte
    (cv2.boxPoints), on en tire son angle brut dans (-180°, 180°], puis on le
    ramène par modulo dans (-90°, 90°]. C'est suffisant pour retrouver le bon
    sens de rotation SI l'aile est toujours photographiée "à l'endroit"
    (jamais tête en bas), hypothèse validée empiriquement sur données
    synthétiques (forme asymétrique, balayage complet de -89° à 90°, aucun
    flip détecté -- voir idmybee_probe_orientation.py).
    Avant cette étape, la désambiguïsation nécessitait une comparaison CLIP à
    des références bien orientées ; ce n'est plus nécessaire ici.

    Si `mask_background` est True, tout ce qui n'est pas l'aile (carton,
    doigts, résidus dans la boîte de recadrage) est remplacé par `bg_color`.

    `pad` est une fraction de la taille de la boîte, répartie pour moitié de
    chaque côté sur CHAQUE axe (x et y), de façon symétrique.

    Retourne (crop, aspect_ratio, (x, y, w, h, theta_deg)) où x/y/w/h/theta
    décrivent la boîte orientée dans l'image d'origine (theta = angle de
    redressement appliqué, en degrés).
    Retourne (None, None, None) si dégénéré.
    """
    if points is None or len(points) < 3:
        return None, None, None

    rect = cv2.minAreaRect(points.astype(np.float32))
    (x_rect, y_rect), (w_rect, h_rect), _ = rect
    if w_rect < 1 or h_rect < 1:
        return None, None, None

    box = cv2.boxPoints(rect)
    edge1, edge2 = box[1] - box[0], box[2] - box[1]
    long_edge = edge1 if np.linalg.norm(edge1) >= np.linalg.norm(edge2) else edge2
    raw_angle = np.degrees(np.arctan2(long_edge[1], long_edge[0]))  # (-180, 180]
    theta_deg = ((raw_angle + 90) % 180) - 90  # ramené dans (-90, 90]

    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, theta_deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += new_w / 2 - center[0]
    M[1, 2] += new_h / 2 - center[1]

    rotated = cv2.warpAffine(image, M, (new_w, new_h), borderValue=bg_color)

    if mask_background:
        mask_full = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask_full, [points.astype(np.int32)], 255)
        rotated_mask = cv2.warpAffine(mask_full, M, (new_w, new_h), borderValue=0)
        rotated[rotated_mask == 0] = bg_color

    corners_h = np.hstack([box, np.ones((4, 1))])
    rotated_corners = (M @ corners_h.T).T
    x_min, y_min = rotated_corners.min(axis=0)
    x_max, y_max = rotated_corners.max(axis=0)
    x_min, y_min = max(0, int(round(x_min))), max(0, int(round(y_min)))
    x_max, y_max = min(new_w, int(round(x_max))), min(new_h, int(round(y_max)))
    if x_max <= x_min or y_max <= y_min:
        return None, None, None

    # padding symétrique sur les deux axes (moitié de `pad` de chaque côté)
    px = int((x_max - x_min) * pad / 2)
    py = int((y_max - y_min) * pad / 2)
    crop = rotated[y_min - py:y_max + py, x_min - px:x_max + px]

    box = np.array(box)
    box[:, 0] /= w
    box[:, 1] /= h

    long_side, short_side = max(w_rect, h_rect), max(min(w_rect, h_rect), 1e-6)
    aspect = long_side / short_side
    return crop, aspect, box.reshape(-1).tolist()


def letterbox(crop: np.ndarray, out_w=512, out_h=256, bg_color=(255, 255, 255)):
    """Redimensionne en conservant les proportions puis pad -- jamais d'étirement."""
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return None
    scale = min(out_w / w, out_h / h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((out_h, out_w, 3), bg_color, dtype=np.uint8)
    x_off, y_off = (out_w - new_w) // 2, (out_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas