"""
crop_wings.py -- Étape 1 : extraction batch des ailes antérieures (YOLOE).

Phase 1 du pipeline manifest : consomme `images.csv` (et, en option,
`specimens.csv`) produits par la Phase 0 (`manifest/build_manifest.py`), et
écrit un unique `crops.csv` consolidé dans le même dossier manifest, indexé
par `image_id` (le hash de contenu calculé en Phase 0 -- stable, indépendant
du chemin du fichier). Plus de scan de dossier ici : "quelles images
existent" est la responsabilité de la Phase 0, "qu'est-ce qu'on en fait"
celle de ce script.

Pipeline par image (détail dans detection.py / geometry.py / qa_clip.py) :
  1. Détection + segmentation YOLOE (VPE multi-référence, calculé une fois).
  2. Redressement du grand axe à l'horizontale -> crop + OBB (loggés comme
     données d'entraînement pour un futur régresseur OBB dédié).
  3. Masquage de fond optionnel (désactivé par défaut, cf. geometry.py).
  4. Letterbox vers 512x256.
  5. QA / désambiguïsation par similarité CLIP à des crops de référence.

Chaque image cible est traitée individuellement (pas de liste) pour éviter un
bug de hang connu d'ultralytics avec visual_prompts en batch (issue #20566).

Résumabilité : les images déjà loguées OK/SUSPECT (crop toujours présent sur
disque) sont sautées ; celles loguées FAILED sont sautées par défaut
(--retry_failed pour les retenter) ; un crops.csv à un schéma différent
(version antérieure du script) bloque l'exécution plutôt que d'être
silencieusement corrompu en mode append.

Attention : crops.csv est un journal append-only, pas une table mise à jour
en place. Un --retry_failed réussi ajoute une NOUVELLE ligne OK pour le même
image_id plutôt que de réécrire l'ancienne ligne FAILED -- un consommateur
de crops.csv doit garder la DERNIÈRE ligne par image_id (ex:
`df.groupby("image_id").last()` en pandas), pas la première.

Usage :
    python crop_wings.py --ref ref.json --ref_crops data/references/crops \
        --manifest_dir data/manifest --output_root data/images/crop

    # ne traiter qu'un sous-ensemble, pour valider avant de tout lancer :
    python crop_wings.py --ref ref.json --ref_crops data/references/crops \
        --manifest_dir data/manifest --output_root data/images/crop \
        --dataset organized --only_labeled

ref.json accepte 1 ou plusieurs références :
    [{"image": "img1.jpg", "boxes": [[x1,y1,x2,y2]]},
     {"image": "img2.jpg", "boxes": [[x1,y1,x2,y2]]}, ...]
"""

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

# --- bootstrap des imports locaux -------------------------------------
# Pas de paquet installé : on ajoute explicitement au sys.path le dossier de
# ce fichier (pour geometry/detection/qa_clip, meme dossier) et son parent
# src/ (pour manifest.io, paquet frere) -- fonctionne quel que soit le
# repertoire courant ou la facon dont le script est invoque.
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_THIS_DIR.parent))

import detection
import geometry
import qa_clip
from manifest import io as manifest_io

# HEIC/HEIF (courant sur les photos terrain prises au smartphone) : cv2 ne
# sait pas les décoder. pillow-heif, s'il est installé, permet à PIL de les
# ouvrir (utilisé par geometry.read_image) ; sinon on continue -- les
# formats non supportés seront loggés en FAILED avec une raison explicite,
# jamais silencieusement absents du résultat.
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    print(
        "ATTENTION: pillow-heif non installé -- les fichiers .heic/.heif "
        "seront loggés en FAILED (image_illisible_ou_format_non_supporte). "
        "pip install pillow-heif pour les lire.",
        file=sys.stderr,
    )

BG_COLORS = {"white": (255, 255, 255), "black": (0, 0, 0)}

CROPS_FIELDS = [
    "image_id", "specimen_id", "dataset", "status", "error_reason",
    "x", "y", "w", "h", "theta",
    "confidence", "similarity", "aspect_ratio", "n_detections",
    "output_path", "processed_at",
]


# ---------------------------------------------------------------------------
# Sélection des images à traiter (Phase 0 -> Phase 1)
# ---------------------------------------------------------------------------

def load_target_images(manifest_dir: Path, dataset_filter, only_labeled: bool):
    """Lit images.csv (+ specimens.csv si --only_labeled) et retourne la
    liste des lignes à traiter : status_ingest='parsed_ok', pas une copie de
    contenu dupliquée (cf. Phase 0), et les filtres optionnels demandés."""
    images_path = manifest_dir / "images.csv"
    images = manifest_io.read_table(images_path)
    if not images:
        raise SystemExit(
            f"{images_path} est vide ou absent -- as-tu lancé "
            f"manifest/build_manifest.py (Phase 0) avant cette étape ?"
        )

    labeled_ids = None
    if only_labeled:
        specimens_path = manifest_dir / "specimens.csv"
        specimens = manifest_io.read_table(specimens_path)
        if not specimens:
            raise SystemExit(f"--only_labeled demandé mais {specimens_path} est vide ou absent.")
        labeled_ids = {
            row["specimen_id"] for row in specimens if manifest_io.parse_bool(row.get("is_labeled"))
        }

    targets = []
    for row in images:
        if row.get("status_ingest") != "parsed_ok":
            continue  # non-parsées : restent dans unparsed.csv, pas notre role ici
        if manifest_io.parse_bool(row.get("is_duplicate_content")):
            continue  # meme contenu deja traite via sa premiere occurrence
        if dataset_filter and row.get("dataset") != dataset_filter:
            continue
        if labeled_ids is not None and row.get("specimen_id") not in labeled_ids:
            continue
        targets.append(row)
    return targets


def resolve_raw_path(row: dict, base_dir: Path) -> Path:
    p = Path(row["raw_path"])
    return p if p.is_absolute() else base_dir / p


def crop_output_path(output_root: Path, row: dict) -> Path:
    """Nom de sortie dérivé du manifest, pas du chemin d'entrée (les images
    d'une même étape peuvent venir de racines physiques différentes,
    cf. Phase 0 / disque externe). Lisible (dataset/specimen) + garanti
    unique grâce au suffixe image_id."""
    device = row.get("device_type") or row.get("collector") or "x"
    shot = row.get("shot_index") or "0"
    filename = f"{row['dataset']}_{row['specimen_id']}_{device}{shot}_{row['image_id']}.jpg"
    return output_root / row["dataset"] / filename
    # return output_root / row["dataset"] / row["specimen_id"] / filename


# ---------------------------------------------------------------------------
# Traitement d'une image
# ---------------------------------------------------------------------------

def process_image(raw_path: Path, out_path: Path, model, ref_embs, clip_model, clip_preprocess,
                   clip_device, args, bg_color):
    """Lit `raw_path`, détecte + découpe + QA, écrit le meilleur candidat
    dans `out_path` si succès. Retourne un dict prêt à devenir une ligne de
    crops.csv (sans image_id/specimen_id/dataset, ajoutés par l'appelant qui
    les connaît déjà via la ligne images.csv)."""

    def failed(reason, n_det=0):
        return dict(status="FAILED", error_reason=reason, x=None, y=None, w=None, h=None, theta=None,
                    confidence=None, similarity=None, aspect_ratio=None, n_detections=n_det, output_path=None)

    img = geometry.read_image(raw_path)
    if img is None:
        return failed("image_illisible_ou_format_non_supporte")

    results = model.predict(img, conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)
    r = results[0]
    n_det = len(r.boxes) if r.boxes is not None else 0

    if n_det == 0:
        return failed("aucune_detection")
    if r.masks is None:
        return failed("pas_de_masque", n_det)

    confs = r.boxes.conf.cpu().numpy()
    order = np.argsort(-confs)  # indices triés par confiance décroissante
    topk = order[:max(1, args.topk_candidates)]

    candidates = []
    for i in topk:
        crop, aspect, obb = geometry.oriented_crop(r.orig_img, r.masks.xy[i], bg_color=bg_color,
                                                     mask_background=args.mask_background, pad=args.padding)
        if crop is None:
            continue
        final = geometry.letterbox(crop, args.out_w, args.out_h, bg_color=bg_color)
        if final is None:
            continue
        similarity = qa_clip.clip_similarity(final, ref_embs, clip_model, clip_preprocess, clip_device)
        candidates.append(dict(conf=float(confs[i]), aspect=aspect, obb=obb, crop=final, similarity=similarity))

    if not candidates:
        return failed("crop_degenere", n_det)

    # priorité à la similarité CLIP (cf. docstring module) : sans effet si
    # un seul candidat (topk_candidates=1, le défaut).
    best = max(candidates, key=lambda c: c["similarity"])

    status = "SUSPECT" if (best["similarity"] < args.min_similarity or best["aspect"] < args.min_aspect_ok) else "OK"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), best["crop"])

    x, y, w, h, theta = best["obb"]
    return dict(status=status, error_reason=None, x=round(x), y=round(y), w=round(w), h=round(h), theta=round(theta, 3),
                confidence=round(best["conf"], 3), similarity=round(best["similarity"], 3),
                aspect_ratio=round(best["aspect"], 2), n_detections=n_det, output_path=str(out_path))


# ---------------------------------------------------------------------------
# Programme principal
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Extraction batch des ailes antérieures via YOLOE (Phase 1)")
    parser.add_argument("--ref", required=True, help="JSON de référence pour la détection YOLOE.")
    parser.add_argument("--ref_crops", required=True, help="Dossier de crops déjà correctement orientés (référence CLIP).")
    parser.add_argument("--manifest_dir", default="data/manifest", help="Dossier contenant images.csv/specimens.csv (Phase 0) et où écrire crops.csv.")
    parser.add_argument("--base_dir", default=".", help="Racine pour résoudre les raw_path relatifs de images.csv.")
    parser.add_argument("--output_root", required=True, help="Racine où écrire les fichiers de crop.")
    parser.add_argument("--dataset", default=None, help="Ne traiter qu'un dataset (organized/terrain/vrac/...). Vide = tous.")
    parser.add_argument("--only_labeled", action="store_true", help="Ne traiter que les images de spécimens identifiés (species/caste connus dans specimens.csv).")
    parser.add_argument("--model", default="yoloe-11s-seg.pt")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default=None, help="'cpu', '0', etc. Vide = auto-détection (YOLOE et CLIP)")
    parser.add_argument("--out_w", type=int, default=512)
    parser.add_argument("--out_h", type=int, default=256)
    parser.add_argument("--mask_background", action="store_true", help="Remplace tout ce qui n'est pas l'aile par --bg_color. Désactivé par défaut.")
    parser.add_argument("--padding", type=float, default=0.1)
    parser.add_argument("--bg_color", choices=["white", "black"], default="white")
    parser.add_argument("--topk_candidates", type=int, default=1,
                         help="Nombre de détections YOLOE comparées via CLIP pour départager les faux positifs. 1 (défaut) = désactivé.")
    parser.add_argument("--min_aspect_ok", type=float, default=1.3, help="Aspect (long/court côté) en dessous duquel on flague SUSPECT. A calibrer.")
    parser.add_argument("--min_similarity", type=float, default=0.0, help="Similarité CLIP min. pour ne pas flaguer SUSPECT. A calibrer (rarement négative, 0.0 est souvent trop permissif).")
    parser.add_argument("--overwrite", action="store_true", help="Retraiter même si le crop de sortie existe déjà (OK/SUSPECT).")
    parser.add_argument("--retry_failed", action="store_true", help="Retenter les images loguées FAILED lors d'un run précédent.")
    parser.add_argument("--log_every", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    bg_color = BG_COLORS[args.bg_color]
    manifest_dir = Path(args.manifest_dir)
    base_dir = Path(args.base_dir)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    targets = load_target_images(manifest_dir, args.dataset, args.only_labeled)
    print(f"{len(targets)} images à considérer (Phase 0: {manifest_dir}/images.csv, filtres appliqués)")

    crops_path = manifest_dir / "crops.csv"
    manifest_io.check_schema(crops_path, CROPS_FIELDS)
    already = manifest_io.load_existing_by_key(crops_path, "image_id")
    if already:
        print(f"{len(already)} entrées déjà présentes dans {crops_path} (reprise du run précédent).")

    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

    references = detection.load_references(args.ref)
    print(f"{len(references)} référence(s) YOLOE pour la détection (ref.json)")
    model = YOLOE(args.model)
    print("Calcul du VPE (moyenné si plusieurs références)...")
    detection.bake_references(model, references, YOLOEVPSegPredictor, imgsz=args.imgsz, device=args.device)

    print("Chargement de CLIP (QA + sélection de candidats)...")
    clip_device = args.device or ("cuda" if qa_clip.cuda_available() else "cpu")
    clip_model, clip_preprocess = qa_clip.load_clip(clip_device)
    ref_crop_paths = list(Path(args.ref_crops).iterdir())
    ref_embs = qa_clip.load_ref_embeddings(ref_crop_paths, clip_model, clip_preprocess, clip_device)
    print(f"{len(ref_embs)} référence(s) CLIP chargée(s)")
    if args.topk_candidates > 1:
        print(f"Sélection parmi les {args.topk_candidates} détections les plus confiantes (via CLIP) activée.")

    write_header = not crops_path.exists()
    crops_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(crops_path, "a", newline="")
    import csv
    writer = csv.DictWriter(log_file, fieldnames=CROPS_FIELDS)
    if write_header:
        writer.writeheader()

    counts = {"OK": 0, "SUSPECT": 0, "FAILED": 0, "SKIPPED": 0}
    t0 = time.time()
    n_done = 0

    for row in targets:
        image_id = row["image_id"]

        if manifest_io.should_skip(already.get(image_id), args.overwrite, args.retry_failed):
            counts["SKIPPED"] += 1
            continue

        raw_path = resolve_raw_path(row, base_dir)
        out_path = crop_output_path(output_root, row)

        try:
            result = process_image(raw_path, out_path, model, ref_embs, clip_model, clip_preprocess,
                                    clip_device, args, bg_color)
        except Exception as e:
            result = dict(status="FAILED", error_reason=f"exception: {e}", x=None, y=None, w=None, h=None,
                          theta=None, confidence=None, similarity=None, aspect_ratio=None,
                          n_detections=None, output_path=None)

        counts[result["status"]] += 1
        writer.writerow({
            "image_id": image_id,
            "specimen_id": row.get("specimen_id"),
            "dataset": row.get("dataset"),
            "processed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **result,
        })
        n_done += 1

        if n_done % args.log_every == 0:
            log_file.flush()
            elapsed = time.time() - t0
            print(
                f"[{n_done}/{len(targets)}] OK={counts['OK']} SUSPECT={counts['SUSPECT']} "
                f"FAILED={counts['FAILED']} SKIPPED={counts['SKIPPED']}  ({elapsed:.0f}s écoulées)"
            )

    log_file.close()
    print("\nTerminé.")
    print(counts)
    print(f"Log détaillé -> {crops_path}")


if __name__ == "__main__":
    main()