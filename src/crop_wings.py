"""
Extraction batch des ailes antérieures par YOLOE (visual prompting cross-image).

Pipeline par image :
  1. Détection + segmentation YOLOE. Le VPE (embedding des exemplaires de
     référence) est calculé une seule fois au démarrage puis réutilisé pour
     toutes les images (~100x plus rapide que le recalculer à chaque image).
     `ref.json` accepte une ou plusieurs références ; leurs VPE sont moyennés
     avant injection.
  2. Redressement du grand axe à l'horizontale (cv2.minAreaRect) -> donne
     aussi la boîte orientée (OBB, x/y/w/h/theta) de l'aile dans l'image
     d'origine, loguée dans le CSV comme donnée d'entraînement pour un futur
     régresseur OBB dédié. L'angle est résolu directement à partir de la
     géométrie (pas de 180° indéterminé), sous l'hypothèse que l'aile est
     toujours photographiée "à l'endroit" (entre -89° et 90° par rapport à
     l'horizontale) -- voir le commentaire dans `oriented_crop`.
  3. En option (--mask_background) : tout ce qui n'est pas l'aile (carton,
     doigts, résidus) est remplacé par une couleur de fond unie. Désactivé
     par défaut car un masque de segmentation imparfait peut effacer des
     parties internes de l'aile (zones membraneuses très fines/claires
     confondues avec le fond).
  4. Letterbox vers 512x256 (padding, jamais d'étirement).
  5. QA / désambiguïsation des faux positifs par similarité CLIP à des crops
     de référence déjà bien orientés. Avec --topk_candidates > 1, les N
     détections YOLOE les plus confiantes sont chacune recadrées et comparées
     par CLIP ; celle la plus proche des références est retenue (utile
     quand la détection la plus confiante n'est pas la bonne aile). Avec
     --topk_candidates 1 (défaut), seule la détection la plus confiante est
     utilisée, et CLIP ne sert qu'à calculer la similarité loguée en QA.

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
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png"}
BG_COLORS = {"white": (255, 255, 255), "black": (0, 0, 0)}

CSV_FIELDS = [
    "idx", "image", "status",
    "x", "y", "w", "h", "theta",
    "confidence", "similarity", "aspect_ratio", "n_detections",
    "input_path", "output_path",
]


# ---------------------------------------------------------------------------
# Détection YOLOE : VPE multi-référence (une ou plusieurs images d'exemple)
# ---------------------------------------------------------------------------

def load_references(ref_json_path: str):
    """Retourne une liste de (image_path, boxes) -- 1 ou plusieurs entrées."""
    import json
    with open(ref_json_path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    return [(Path(str(entry["image"])).absolute(), np.array(entry["boxes"], dtype=np.float32)) for entry in data]


def compute_vpe(model, image_path: str, boxes: np.ndarray, predictor_cls, imgsz: int = 1024, device=None):
    """VPE (visual prompt embedding) d'UNE image de référence.

    `refer_image=` du predict() haut niveau ne prend qu'une image par appel
    (assertion dans get_vpe), donc pour moyenner plusieurs références on
    reproduit ici ce que YOLOE.predict() fait en interne pour une seule.
    Le predicteur doit être construit avec les mêmes overrides (task/batch/
    device/imgsz) que le code source, sinon le VPE résultant est dégradé.
    """
    from ultralytics.cfg import get_cfg

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


def bake_references(model, references, predictor_cls, imgsz: int = 1024, device=None):
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
# Découpe orientée : redressement (sans ambiguïté 180°), masquage optionnel,
# letterbox
# ---------------------------------------------------------------------------

def find_images(root: Path):
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS)


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
    
    px, py = int((x_max-x_min)*pad/2), int((y_max-y_min)*pad)
    crop = rotated[y_min-py:y_max+py, x_min-px:x_max+px]

    long_side, short_side = max(w_rect, h_rect), max(min(w_rect, h_rect), 1e-6)
    aspect = long_side / short_side
    return crop, aspect, (float(x_rect), float(y_rect), float(w_rect), float(h_rect), float(theta_deg))


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


# ---------------------------------------------------------------------------
# QA / sélection de candidat : similarité CLIP à des crops de référence
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


def clip_similarity(crop_512x256, ref_embs, model, preprocess, device):
    """Similarité max du crop aux références connues. Sert de signal qualité
    (un corps/doigt mal détecté ne ressemblera à aucune référence) et, si
    plusieurs candidats sont comparés, de critère de sélection."""
    emb = embed(crop_512x256, model, preprocess, device)
    return max((emb @ r.T).item() for r in ref_embs)


def cuda_available():
    return torch.cuda.is_available()


# ---------------------------------------------------------------------------
# Programme principal
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Extraction batch des ailes antérieures via YOLOE")
    parser.add_argument("--ref", required=True, help="JSON de référence pour la détection YOLOE : {'image':..., 'boxes':[[x1,y1,x2,y2]]}.")
    parser.add_argument("--ref_crops", required=True, help="Dossier contenant 1+ crops déjà correctement orientés (référence CLIP).")
    parser.add_argument("--input_root", required=True, help="Racine du dataset (ex: images/orga_widecrop)")
    parser.add_argument("--output_root", required=True, help="Racine de sortie (arborescence miroir)")
    parser.add_argument("--log", default="out/wing_extraction/")
    parser.add_argument("--model", default="yoloe-11s-seg.pt")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default=None, help="'cpu', '0', etc. Vide = auto-détection (YOLOE et CLIP)")
    parser.add_argument("--out_w", type=int, default=512)
    parser.add_argument("--out_h", type=int, default=256)
    parser.add_argument("--mask_background", action="store_true", help="Remplace tout ce qui n'est pas l'aile par --bg_color. Désactivé par défaut.")
    parser.add_argument("--padding", type=float, default=0.)
    parser.add_argument("--bg_color", choices=["white", "black"], default="white")
    parser.add_argument("--topk_candidates", type=int, default=1,
                         help="Nombre de détections YOLOE (par confiance décroissante) comparées via CLIP pour "
                              "départager les faux positifs. 1 (défaut) = désactivé, garde la meilleure confiance.")
    parser.add_argument("--min_aspect_ok", type=float, default=1.3, help="Aspect (long/court côté) en dessous duquel on flague SUSPECT. A calibrer sur votre dataset.")
    parser.add_argument("--min_similarity", type=float, default=0.0, help="Similarité CLIP min. à la meilleure référence pour ne pas flaguer SUSPECT.")
    parser.add_argument("--overwrite", action="store_true", help="Retraiter même si le crop de sortie existe déjà.")
    parser.add_argument("--log_every", type=int, default=50)
    return parser.parse_args()


def process_image(img_path, model, ref_embs, clip_model, clip_preprocess, clip_device, args, bg_color):
    """Traite une image : détection -> découpe orientée -> QA CLIP -> résultat (dict, un futur row CSV)."""
    results = model.predict(str(img_path), conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)
    r = results[0]
    n_det = len(r.boxes) if r.boxes is not None else 0

    empty = dict(status="FAILED", x=None, y=None, w=None, h=None, theta=None,
                 confidence=None, similarity=None, aspect_ratio=None, n_detections=n_det, output_path=None)
    if n_det == 0 or r.masks is None:
        return empty

    confs = r.boxes.conf.cpu().numpy()
    order = np.argsort(-confs)  # indices triés par confiance décroissante
    topk = order[:max(1, args.topk_candidates)]

    candidates = []
    for i in topk:
        crop, aspect, obb = oriented_crop(r.orig_img, r.masks.xy[i], bg_color=bg_color,
                                           mask_background=args.mask_background, pad=args.padding)
        if crop is None:
            continue
        final = letterbox(crop, args.out_w, args.out_h, bg_color=bg_color)
        if final is None:
            continue
        similarity = clip_similarity(final, ref_embs, clip_model, clip_preprocess, clip_device)
        candidates.append(dict(conf=float(confs[i]), aspect=aspect, obb=obb, crop=final, similarity=similarity))

    if not candidates:
        return empty

    best = max(candidates, key=lambda c: c["conf"] + c["similarity"])  # sans effet si un seul candidat

    status = "SUSPECT" if (best["similarity"] < args.min_similarity or best["aspect"] < args.min_aspect_ok) else "OK"
    out_path = Path(args.output_root) / img_path.relative_to(args.input_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), best["crop"])

    x, y, w, h, theta = best["obb"]
    return dict(status=status, x=round(x), y=round(y), w=round(w), h=round(h), theta=round(theta, 3),
                confidence=round(best["conf"], 3), similarity=round(best["similarity"], 3),
                aspect_ratio=round(best["aspect"], 2), n_detections=n_det, output_path=str(out_path))


def main():
    args = parse_args()
    bg_color = BG_COLORS[args.bg_color]

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    images = find_images(input_root)
    print(f"{len(images)} images trouvées sous {input_root}")

    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

    references = load_references(args.ref)
    print(f"{len(references)} référence(s) YOLOE pour la détection (ref.json)")
    model = YOLOE(args.model)
    print("Calcul du VPE (moyenné si plusieurs références)...")
    bake_references(model, references, YOLOEVPSegPredictor, imgsz=args.imgsz, device=args.device)

    print("Chargement de CLIP (QA + sélection de candidats)...")
    clip_device = args.device or ("cuda" if cuda_available() else "cpu")
    clip_model, clip_preprocess = load_clip(clip_device)
    ref_crop_paths = list(Path(args.ref_crops).iterdir())
    ref_embs = load_ref_embeddings(ref_crop_paths, clip_model, clip_preprocess, clip_device)
    print(f"{len(ref_embs)} référence(s) CLIP chargée(s)")
    if args.topk_candidates > 1:
        print(f"Sélection parmi les {args.topk_candidates} détections les plus confiantes (via CLIP) activée.")

    log_path = Path(args.log / f"{input_root.stem}_wings_log.csv")
    write_header = not log_path.exists()
    log_file = open(log_path, "a", newline="")
    writer = csv.DictWriter(log_file, fieldnames=CSV_FIELDS)
    if write_header:
        writer.writeheader()

    counts = {"OK": 0, "SUSPECT": 0, "FAILED": 0, "SKIPPED": 0}
    t0 = time.time()
    idx = 0

    for img_path in images:
        rel = img_path.relative_to(input_root)
        out_path = output_root / rel
        if out_path.exists() and not args.overwrite:
            counts["SKIPPED"] += 1
            continue

        try:
            row = process_image(img_path, model, ref_embs, clip_model, clip_preprocess, clip_device, args, bg_color)
        except Exception as e:
            row = dict(status="FAILED", x=None, y=None, w=None, h=None, theta=None, confidence=None,
                       similarity=None, aspect_ratio=None, n_detections=None, output_path=f"error: {e}")

        counts[row["status"]] += 1
        writer.writerow({"idx": idx, "image": Path(rel).stem, "input_path": str(img_path), **row})
        idx += 1

        if idx % args.log_every == 0:
            log_file.flush()
            elapsed = time.time() - t0
            print(
                f"[{idx}/{len(images)}] OK={counts['OK']} SUSPECT={counts['SUSPECT']} "
                f"FAILED={counts['FAILED']} SKIPPED={counts['SKIPPED']}  ({elapsed:.0f}s écoulées)"
            )

    log_file.close()
    print("\nTerminé.")
    print(counts)
    print(f"Log détaillé -> {log_path}")


if __name__ == "__main__":
    main()