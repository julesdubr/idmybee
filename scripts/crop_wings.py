"""
Extraction batch des ailes antérieures par YOLOE (visual prompting cross-image).

Pipeline par image :
  1. Détection + segmentation YOLOE. Le VPE (embedding des exemplaires de
     référence) est calculé une seule fois au démarrage puis réutilisé pour
     toutes les images (~100x plus rapide que le recalculer à chaque image).
     `ref.json` accepte une ou plusieurs références ; leurs VPE sont moyennés
     avant injection.
  2. Redressement du grand axe à l'horizontale (cv2.minAreaRect) -> donne
     aussi la boîte orientée (OBB) de l'aile dans l'image d'origine, loguée
     dans le CSV pour servir de données d'entraînement à un futur régresseur
     OBB dédié.
  3. En option (--mask_background) : tout ce qui n'est pas l'aile (carton,
     doigts, résidus) est remplacé par une couleur de fond unie. Désactivé
     par défaut car un masque de segmentation imparfait peut effacer des
     parties internes de l'aile (zones membraneuses très fines/claires
     confondues avec le fond).
  4. Letterbox vers 512x256 (padding, jamais d'étirement).
  5. Désambiguïsation de l'orientation à 180° par similarité CLIP à des crops
     de référence déjà bien orientés (les heuristiques géométriques testées
     avant -- forme de l'aile, position dans le cadre -- se sont révélées peu
     fiables sur données réelles). Sert aussi de second signal qualité, en
     plus de l'aspect ratio.

Chaque image cible est traitée individuellement (pas de liste) pour éviter un
bug de hang connu d'ultralytics avec visual_prompts sur du traitement par lot
(issue #20566).

Usage :
    python crop_wings.py --ref ref.json \
        --input_root data/images/orga_widecrop \
        --output_root data/images/wing_crops \
        --ref_crops data/references/crops

ref.json accepte 1 ou plusieurs références :
    [{"image": "img1.jpg", "boxes": [[x1,y1,x2,y2]]},
     {"image": "img2.jpg", "boxes": [[x1,y1,x2,y2]]}, ...]
"""

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLOE
from ultralytics.cfg import get_cfg
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

IMG_EXTS = {".jpg", ".jpeg", ".png"}
BG_COLORS = {"white": (255, 255, 255), "black": (0, 0, 0)}


# ---------------------------------------------------------------------------
# Détection YOLOE : VPE multi-référence (une ou plusieurs images d'exemple)
# ---------------------------------------------------------------------------

def load_references(ref_json_path: str):
    """Retourne une liste de (image_path, boxes) -- 1 ou plusieurs entrées."""
    with open(ref_json_path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    return [(entry["image"], np.array(entry["boxes"], dtype=np.float32)) for entry in data]


def compute_vpe(model: YOLOE, image_path: str, boxes: np.ndarray,
                 predictor_cls=YOLOEVPSegPredictor, imgsz: int = 1024, device=None):
    """VPE (visual prompt embedding) d'UNE image de référence.

    `refer_image=` du predict() haut niveau ne prend qu'une image par appel
    (assertion dans get_vpe), donc pour moyenner plusieurs références on
    reproduit ici ce que YOLOE.predict() fait en interne pour une seule.
    Le predicteur doit être construit avec les mêmes overrides (task/batch/
    device/imgsz) que le code source, sinon le VPE résultant est dégradé.
    """
    visual_prompts = dict(bboxes=boxes, cls=np.zeros(len(boxes), dtype=int))
    if type(model.predictor) is not predictor_cls:
        args = get_cfg(overrides={**model.overrides, "imgsz": imgsz, "device": device})
        model.predictor = predictor_cls(
            overrides={
                "task": model.model.task, "mode": "predict", "save": False, "verbose": False,
                "batch": 1, "device": args.device, "quantize": args.quantize, "imgsz": args.imgsz,
            },
            _callbacks=model.callbacks,
        )
    model.model.model[-1].nc = 1
    model.model.names = ["object0"]
    model.predictor.set_prompts(visual_prompts.copy())
    model.predictor.setup_model(model=model.model, verbose=False)
    return model.predictor.get_vpe(image_path)


def bake_references(model: YOLOE, references, predictor_cls=YOLOEVPSegPredictor,
                     imgsz: int = 1024, device=None):
    """Calcule le VPE de chaque référence, moyenne, injecte dans le modèle.
    Après appel, model.predict(image, ...) fonctionne directement (VPE déjà
    "baked in", plus besoin de refer_image/visual_prompts).

    Note : set_classes doit être appelé sur le nn.Module sous-jacent
    (model.model.set_classes), pas sur le wrapper YOLOE.set_classes -- ce
    dernier est pensé pour du texte et ne met pas à jour le bon état interne.
    """
    vpes = [compute_vpe(model, img, boxes, predictor_cls, imgsz, device) for img, boxes in references]
    avg_vpe = torch.mean(torch.stack(vpes, dim=0), dim=0)
    avg_vpe = avg_vpe / avg_vpe.norm(dim=-1, keepdim=True)  # les VPE individuels sont unitaires
    model.model.set_classes(["object0"], avg_vpe)
    model.model.names = ["object0"]
    model.predictor = None  # un predicteur standard sera recréé au 1er predict() réel


# ---------------------------------------------------------------------------
# Découpe orientée : redressement, masquage optionnel, letterbox
# ---------------------------------------------------------------------------

def find_images(root: Path):
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS)


def oriented_crop(image: np.ndarray, points: np.ndarray, bg_color=(255, 255, 255),
                   mask_background: bool = False):
    """Redresse le grand axe de l'aile à l'horizontale et recadre.

    `points` : contour du masque en coordonnées pixel de l'image originale
    (ex. r.masks.xy[i]).

    Si `mask_background` est True, tout ce qui n'est pas l'aile (carton,
    doigts, résidus dans la boîte de recadrage) est remplacé par `bg_color`.

    Ne résout PAS l'ambiguïté d'orientation à 180° (indéterminable depuis ce
    seul point de vue géométrique) -> voir `resolve_orientation`.

    Retourne (crop, aspect_ratio, obb_corners) où obb_corners sont les 4
    coins de la boîte orientée dans l'image d'origine.
    Retourne (None, None, None) si dégénéré.
    """
    if points is None or len(points) < 3:
        return None, None, None

    rect = cv2.minAreaRect(points.astype(np.float32))
    (x_rect, y_rect), (w_rect, h_rect), theta = rect
    if w_rect < 1 or h_rect < 1:
        return None, None, None

    obb_corners = cv2.boxPoints(rect)  # coins dans l'image d'ORIGINE, avant rotation

    edge1 = obb_corners[1] - obb_corners[0]
    edge2 = obb_corners[2] - obb_corners[1]
    # long_edge = edge1 if np.linalg.norm(edge1) >= np.linalg.norm(edge2) else edge2
    # angle_deg = np.degrees(np.arctan2(long_edge[1], long_edge[0]))

    angle_deg = theta if np.linalg.norm(edge2) >= np.linalg.norm(edge1) else 90 + theta

    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
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

    corners_h = np.hstack([obb_corners, np.ones((4, 1))])
    rotated_corners = (M @ corners_h.T).T
    x_min, y_min = rotated_corners.min(axis=0)
    x_max, y_max = rotated_corners.max(axis=0)
    x_min, y_min = max(0, int(round(x_min))), max(0, int(round(y_min)))
    x_max, y_max = min(new_w, int(round(x_max))), min(new_h, int(round(y_max)))
    if x_max <= x_min or y_max <= y_min:
        return None, None, None
    crop = rotated[y_min:y_max, x_min:x_max]

    long_side, short_side = max(w_rect, h_rect), max(min(w_rect, h_rect), 1e-6)
    aspect = long_side / short_side
    return crop, aspect, (x_rect, y_rect, w_rect, h_rect, theta)


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


def obb_to_json(obb_corners) -> str:
    """4 coins -> JSON compact [[x,y],[x,y],[x,y],[x,y]], coords image d'origine."""
    return json.dumps([[round(float(x), 1), round(float(y), 1)] for x, y in obb_corners])


# ---------------------------------------------------------------------------
# Orientation / QA : similarité CLIP à des crops de référence déjà bien orientés
# ---------------------------------------------------------------------------

def load_clip(device):
    """Même modèle que celui déjà utilisé pour le QA du pipeline classique
    (open_clip ViT-B-16-plus-240 / laion400m_e32)."""
    import open_clip  # dépendance optionnelle, chargée seulement ici
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16-plus-240", pretrained="laion400m_e32"
    )
    model.eval().to(device)
    return model, preprocess


def embed(bgr_img, model, preprocess, device):
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    x = preprocess(Image.fromarray(rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode_image(x)
    return feat / feat.norm(dim=-1, keepdim=True)


def load_ref_embeddings(ref_crop_paths, model, preprocess, device):
    embs = []
    for p in ref_crop_paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"ATTENTION: référence introuvable/illisible: {p}")
            continue
        embs.append(embed(img, model, preprocess, device))
    if not embs:
        raise ValueError("Aucune image de référence valide dans --ref_crops")
    return embs


def resolve_orientation(crop_512x256, ref_embs, model, preprocess, device):
    """Compare le crop et sa version à 180° aux références connues, garde la
    meilleure. Retourne (crop_corrigé, similarité_max) -- la similarité sert
    aussi de signal qualité (un corps/doigt ne ressemblera à aucune
    référence, dans aucune des deux orientations)."""
    emb0 = embed(crop_512x256, model, preprocess, device)
    sim0 = max((emb0 @ r.T).item() for r in ref_embs)
    return (crop_512x256, sim0)


def cuda_available():
    return torch.cuda.is_available()


# ---------------------------------------------------------------------------
# Programme principal
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Extraction batch des ailes antérieures via YOLOE")
    parser.add_argument("--ref", required=True, help="JSON de référence pour la détection YOLOE : {'image':..., 'boxes':[[x1,y1,x2,y2]]}.")
    parser.add_argument("--ref_crops", required=True, help="Chemins vers un dossier contenant 1+ crops déjà correctement orientés.")
    parser.add_argument("--input_root", required=True, help="Racine du dataset (ex: images/orga_widecrop)")
    parser.add_argument("--output_root", required=True, help="Racine de sortie (arborescence miroir)")
    parser.add_argument("--log", default="wing_extraction_log.csv")
    parser.add_argument("--model", default="yoloe-11s-seg.pt")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default=None, help="'cpu', '0', etc. Vide = auto-détection (YOLOE et CLIP)")
    parser.add_argument("--out_w", type=int, default=512)
    parser.add_argument("--out_h", type=int, default=256)
    parser.add_argument("--mask_background", action="store_true", help="Remplace tout ce qui n'est pas l'aile par --bg_color. Désactivé par défaut.")
    parser.add_argument("--bg_color", choices=["white", "black"], default="white")
    parser.add_argument("--min_aspect_ok", type=float, default=1.3, help="Aspect (long/court côté) en dessous duquel on flague SUSPECT. A calibrer sur votre dataset.")
    parser.add_argument("--min_similarity", type=float, default=0.0, help="Similarité CLIP min. à la meilleure référence pour ne pas flaguer SUSPECT.")
    parser.add_argument("--overwrite", action="store_true", help="Retraiter même si le crop de sortie existe déjà.")
    parser.add_argument("--log_every", type=int, default=50)
    return parser.parse_args()


def process_image(img_path, model, ref_embs, clip_model, clip_preprocess, clip_device, args, bg_color):
    """Traite une image : détection -> découpe orientée -> orientation -> retourne une ligne CSV."""
    results = model.predict(str(img_path), conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)
    r = results[0]
    n_det = len(r.boxes) if r.boxes is not None else 0

    if n_det == 0 or r.masks is None:
        return ["FAILED", n_det, "", "", "", "", ""]

    confs = r.boxes.conf.cpu().numpy()
    best = int(confs.argmax())  # meilleure confiance = aile antérieure ciblée
    points = r.masks.xy[best]

    crop, aspect, obb_corners = oriented_crop(r.orig_img, points, bg_color=bg_color,
                                               mask_background=args.mask_background)
    if crop is None:
        return ["FAILED", n_det, f"{confs[best]:.3f}", "", "", "", ""]

    final = letterbox(crop, args.out_w, args.out_h, bg_color=bg_color)
    if final is None:
        return ["FAILED", n_det, f"{confs[best]:.3f}", obb_to_json(obb_corners), "", "", ""]

    final, similarity = resolve_orientation(final, ref_embs, clip_model, clip_preprocess, clip_device)

    status = "SUSPECT" if (similarity < args.min_similarity or aspect < args.min_aspect_ok) else "OK"
    out_path = Path(args.output_root) / img_path.relative_to(args.input_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), final)

    return [status, n_det, f"{confs[best]:.3f}", obb_to_json(obb_corners), f"{aspect:.2f}", f"{similarity:.3f}", str(out_path)]


def main():
    args = parse_args()
    bg_color = BG_COLORS[args.bg_color]

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    images = find_images(input_root)
    print(f"{len(images)} images trouvées sous {input_root}")

    references = load_references(args.ref)
    print(f"{len(references)} référence(s) YOLOE pour la détection (ref.json)")
    model = YOLOE(args.model)
    print("Calcul du VPE (moyenné si plusieurs références)...")
    bake_references(model, references, imgsz=args.imgsz, device=args.device)

    print("Chargement de CLIP pour estimer la meilleure orientation...")
    clip_device = args.device or ("cuda" if cuda_available() else "cpu")
    clip_model, clip_preprocess = load_clip(clip_device)
    ref_crop_paths = list(Path(args.ref_crops).iterdir())
    ref_embs = load_ref_embeddings(ref_crop_paths, clip_model, clip_preprocess, clip_device)
    print(f"{len(ref_embs)} référence(s) CLIP chargée(s)")

    log_path = Path(args.log)
    write_header = not log_path.exists()
    log_file = open(log_path, "a", newline="")
    writer = csv.writer(log_file)
    if write_header:
        writer.writerow(["name", "image", "status", "n_detections", "confidence", "obb", "aspect", "similarity", "output"])

    counts = {"OK": 0, "SUSPECT": 0, "FAILED": 0, "SKIPPED": 0}
    t0 = time.time()

    for i, img_path in enumerate(images):
        rel = img_path.relative_to(input_root)
        out_path = output_root / rel
        if out_path.exists() and not args.overwrite:
            counts["SKIPPED"] += 1
            continue

        try:
            row = process_image(img_path, model, ref_embs, clip_model, clip_preprocess, clip_device, args, bg_color)
        except Exception as e:
            row = ["FAILED", "", "", "", "", "", f"error: {e}"]

        counts[row[0]] += 1
        writer.writerow([str(Path(rel).stem), str(rel), *row])

        if (i + 1) % args.log_every == 0:
            log_file.flush()
            elapsed = time.time() - t0
            print(
                f"[{i + 1}/{len(images)}] OK={counts['OK']} SUSPECT={counts['SUSPECT']} "
                f"FAILED={counts['FAILED']} SKIPPED={counts['SKIPPED']}  ({elapsed:.0f}s écoulées)"
            )

    log_file.close()
    print("\nTerminé.")
    print(counts)
    print(f"Log détaillé -> {log_path}")


if __name__ == "__main__":
    main()