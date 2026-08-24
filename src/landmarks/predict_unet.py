"""
predict_unet.py -- Étape 2 : positionnement automatique des landmarks (UNet).

Consomme `crops.csv` (Phase 1). Ne lit ni images.csv ni specimens.csv : tout
ce qu'il faut (specimen_id, dataset, chemin du crop, statut d'extraction)
est déjà dans crops.csv -- l'espèce/caste/collecteur ne sont utiles qu'aux
étapes suivantes (classification), pas ici.

Sorties :
  - `<manifest_dir>/landmarks.csv` : une ligne par image traitée (statut,
    raison d'échec, nombre de landmarks trouvés, modèle utilisé).
  - `--tps_out` : les coordonnées, via `utils.tps_io` (format standard du
    domaine, compatible avec le script R d'Adrien et avec le reste du
    pipeline Python).

Par rapport au notebook d'origine, deux corrections de fond :

  1. `local_maxima()` peut marquer PLUSIEURS pixels adjacents pour un même
     pic (plateau à valeur quasi identique). Les traiter comme des points
     séparés puis garder "les N valeurs les plus hautes" peut faire
     disparaître un vrai landmark distinct au profit de doublons du même
     pic. On regroupe maintenant chaque pic par composante connexe
     (scipy.ndimage.label) avant de classer/tronquer.
  2. Si le modèle ne produit pas assez de pics distincts, l'ancien code
     tronquait silencieusement (`maximas[-18:,:]` renvoie tout ce qu'il y a
     si moins de 18 lignes -- pas d'erreur). Ici, un résultat avec moins de
     landmarks que prévu est explicitement marqué SUSPECT avec la raison,
     jamais juste écrit tel quel sans avertissement.

`utils.tps_io.ImageLandmarks.tps_id` doit être un entier, et `write_tps`
réécrit le fichier entier (pas d'append). `image_id` (le hash de contenu de
la Phase 0) n'est donc pas directement utilisable comme `tps_id` -- et
`specimen_id` ne convient pas non plus : un même specimen a souvent
plusieurs photos (donc plusieurs lignes de crops.csv), qui doivent rester
des entrées TPS distinctes. `ImageLandmarks.from_image()` calcule
`tps_id = image_id_to_sid(image_id)` (unique par image, déterministe) ET
persiste image_id/specimen_id dans un COMMENT= du TPS : les étapes
suivantes (lda.py, flag_outlier_specimens.py) n'ont alors plus besoin de
rejoindre images.csv pour retrouver le specimen_id de chaque entrée.

Comme write_tps réécrit tout, ce script fonctionne par "checkpoint" : au
démarrage, le TPS existant est reparsé (`parse_tps`, erreurs de blocs
explicitement affichées, jamais juste avalées) pour connaître ce qui est
déjà là ; chaque image traitée MET À JOUR (ou retire, si elle échoue
maintenant après avoir réussi avant) l'entrée correspondante dans un
dictionnaire en mémoire ; le TPS et landmarks.csv sont réécrits en entier
tous les `--log_every` (et une dernière fois à la fin). Un retry/overwrite
remplace donc proprement l'ancienne entrée au lieu d'ajouter un doublon.

Usage :
    python predict_unet.py --model_path data/models/best_model/UNet_150_epoch_lr=0.001_seed=58_func=pow_param=30.pth \
        --manifest_dir data/manifest --tps_out data/annotations/landmarks_unet.tps

    # ne traiter qu'un sous-ensemble, en incluant les crops SUSPECT :
    python predict_unet.py --model_path ... --dataset terrain --include_suspect
"""

import argparse
import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import ndimage
from skimage.morphology import local_maxima

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parent))
from manifest import io as manifest_io
from utils.tps_io import ImageLandmarks, image_id_to_sid, parse_tps, write_tps

LANDMARKS_FIELDS = [
    "image_id", "specimen_id", "dataset", "status", "error_reason",
    "n_landmarks_found", "model_name", "processed_at",
]


# ---------------------------------------------------------------------------
# Extraction des landmarks depuis le heatmap prédit par le UNet
# ---------------------------------------------------------------------------

def extract_top_landmarks(heatmap: np.ndarray, n_landmarks: int):
    """Retourne (points_yx, n_trouvés) : jusqu'à `n_landmarks` points (y, x),
    un par pic distinct du heatmap.

    FIX : `local_maxima` marque parfois plusieurs pixels adjacents pour un
    même pic (plateau) -- on regroupe par composante connexe (un pic = une
    composante) et on garde le pixel de valeur maximale de chaque groupe,
    plutôt que de traiter chaque pixel du plateau comme un landmark à part
    entière (ce qui pouvait faire gagner un doublon sur un vrai landmark
    distinct de valeur légèrement plus faible).

    n_trouvé peut être < n_landmarks si le modèle n'a pas produit assez de
    pics distincts sur cette image -- à l'appelant de décider quoi en faire
    (ici : SUSPECT plutôt qu'un landmark set tronqué sans avertissement).
    """
    mask = local_maxima(heatmap)
    labeled, n_components = ndimage.label(mask)
    if n_components == 0:
        return np.empty((0, 2), dtype=float), 0

    peaks = []
    for label_id in range(1, n_components + 1):
        ys, xs = np.where(labeled == label_id)
        values = heatmap[ys, xs]
        best = np.argmax(values)
        peaks.append((ys[best], xs[best], values[best]))

    peaks.sort(key=lambda p: p[2], reverse=True)
    top = peaks[:n_landmarks]
    coords_yx = np.array([(y, x) for y, x, _ in top], dtype=float)
    return coords_yx, len(top)


def predict_landmarks_for_crop(crop_path: Path, model, device, n_landmarks: int):
    """Retourne (status, error_reason, coords_xy_ou_None, n_trouvés).
    coords_xy est déjà en ordre (x, y), prêt pour tps_io.ImageLandmarks."""
    img = cv2.imread(str(crop_path))
    if img is None or img.size == 0:
        return "FAILED", "crop_illisible", None, 0
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_t = torch.tensor(img.transpose(2, 0, 1) / 255.0, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_t).cpu().squeeze(0).numpy().transpose(1, 2, 0).squeeze(axis=2)

    coords_yx, n_found = extract_top_landmarks(output, n_landmarks)
    if n_found == 0:
        return "FAILED", "aucun_maximum_local", None, 0

    coords_xy = coords_yx[:, [1, 0]]
    if n_found < n_landmarks:
        return "SUSPECT", f"seulement_{n_found}_maxima_sur_{n_landmarks}_attendus", coords_xy, n_found
    return "OK", None, coords_xy, n_found


# ---------------------------------------------------------------------------
# Sélection des crops à traiter (Phase 1 -> Phase 2)
# ---------------------------------------------------------------------------

def load_target_crops(manifest_dir: Path, dataset_filter, include_suspect: bool):
    """Lit crops.csv et ne garde que la DERNIÈRE occurrence de chaque
    image_id (crops.csv est append-only, cf. Phase 1 -- un retry/overwrite
    peut avoir ajouté une ligne plus récente, y compris avec un statut
    différent, pour le même image_id). Le filtre de statut/dataset
    s'applique ENSUITE, sur ce dernier état connu."""
    crops_path = manifest_dir / "crops.csv"
    crops = manifest_io.read_table(crops_path)
    if not crops:
        raise SystemExit(
            f"{crops_path} est vide ou absent -- as-tu lancé crop_wings.py "
            f"(Phase 1) avant cette étape ?"
        )

    latest_by_id = {}
    for row in crops:
        latest_by_id[row["image_id"]] = row

    accepted = {"OK", "SUSPECT"} if include_suspect else {"OK"}
    return [
        row for row in latest_by_id.values()
        if row.get("status") in accepted and (not dataset_filter or row.get("dataset") == dataset_filter)
    ]


def resolve_crop_path(row: dict, base_dir: Path) -> Path:
    p = Path(row["output_path"])
    return p if p.is_absolute() else base_dir / p


# ---------------------------------------------------------------------------
# État TPS + landmarks.csv en mémoire, avec checkpoints réguliers
# ---------------------------------------------------------------------------

def load_working_tps(tps_path: Path) -> dict:
    """Reparse le TPS existant (s'il y en a un) en dict {tps_id: ImageLandmarks}.
    Les blocs malformés sont explicitement signalés (jamais avalés en
    silence), le reste est repris tel quel."""
    if not tps_path.exists():
        return {}
    specimens, errors = parse_tps(tps_path, strict=False)
    if errors:
        print(f"ATTENTION: {len(errors)} bloc(s) illisible(s) dans {tps_path} (ignorés, pas perdus) :")
        for e in errors[:10]:
            print(f"  spécimen #{e.specimen_index}, ligne {e.line_no} : {e.message}")
        if len(errors) > 10:
            print(f"  ... et {len(errors) - 10} de plus.")
    return {sp.tps_id: sp for sp in specimens}


def checkpoint(tps_path: Path, working_tps: dict, landmarks_path: Path, landmarks_status: dict):
    tps_path.parent.mkdir(parents=True, exist_ok=True)
    write_tps(tps_path, sorted(working_tps.values(), key=lambda s: s.tps_id))

    landmarks_path.parent.mkdir(parents=True, exist_ok=True)
    with open(landmarks_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LANDMARKS_FIELDS)
        writer.writeheader()
        writer.writerows(landmarks_status.values())


# ---------------------------------------------------------------------------
# Programme principal
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Positionnement automatique des landmarks (UNet, Phase 2)")
    parser.add_argument("--manifest_dir", default="data/manifest")
    parser.add_argument("--base_dir", default=".", help="Racine pour résoudre les output_path relatifs de crops.csv.")
    parser.add_argument("--model_path", required=True, help="Chemin du .pth du modèle UNet.")
    parser.add_argument("--tps_out", required=True, help="TPS de sortie (réécrit entièrement à chaque checkpoint).")
    parser.add_argument("--n_landmarks", type=int, default=18)
    parser.add_argument("--dataset", default=None, help="Ne traiter qu'un dataset. Vide = tous.")
    parser.add_argument("--include_suspect", action="store_true", help="Traiter aussi les crops SUSPECT (par défaut, seuls les OK sont traités).")
    parser.add_argument("--device", default=None, help="'cpu', 'cuda', etc. Vide = auto-détection.")
    parser.add_argument("--overwrite", action="store_true", help="Retraiter même si déjà logué OK/SUSPECT.")
    parser.add_argument("--retry_failed", action="store_true", help="Retenter les images loguées FAILED lors d'un run précédent.")
    parser.add_argument("--log_every", type=int, default=50, help="Fréquence des checkpoints (réécriture TPS + landmarks.csv) et des messages de progression.")
    return parser.parse_args()


def main():
    args = parse_args()
    manifest_dir = Path(args.manifest_dir)
    base_dir = Path(args.base_dir)
    tps_path = Path(args.tps_out)

    targets = load_target_crops(manifest_dir, args.dataset, args.include_suspect)
    print(f"{len(targets)} crops à considérer (Phase 1: {manifest_dir}/crops.csv, filtres appliqués)")

    landmarks_path = manifest_dir / "landmarks.csv"
    manifest_io.check_schema(landmarks_path, LANDMARKS_FIELDS)
    landmarks_status = manifest_io.load_existing_by_key(landmarks_path, "image_id")
    if landmarks_status:
        print(f"{len(landmarks_status)} entrées déjà présentes dans {landmarks_path} (reprise du run précédent).")

    working_tps = load_working_tps(tps_path)
    if working_tps:
        print(f"{len(working_tps)} spécimen(s) déjà présents dans {tps_path} (repris tels quels si sautés ce run).")

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using", device)
    model = torch.load(args.model_path, weights_only=False, map_location=device)
    model.to(device)
    model.eval()
    model_name = Path(args.model_path).stem

    counts = {"OK": 0, "SUSPECT": 0, "FAILED": 0, "SKIPPED": 0}
    t0 = time.time()
    n_done = 0

    for row in targets:
        image_id = row["image_id"]
        tps_id = image_id_to_sid(image_id)

        if manifest_io.should_skip(landmarks_status.get(image_id), args.overwrite, args.retry_failed):
            counts["SKIPPED"] += 1
            continue

        crop_path = resolve_crop_path(row, base_dir)

        try:
            status, error_reason, coords_xy, n_found = predict_landmarks_for_crop(
                crop_path, model, device, args.n_landmarks
            )
        except Exception as e:
            status, error_reason, coords_xy, n_found = "FAILED", f"exception: {e}", None, 0

        if coords_xy is not None:
            working_tps[tps_id] = ImageLandmarks.from_image(
                n_points=len(coords_xy), landmarks=coords_xy, image_path=str(crop_path),
                image_id=image_id, specimen_id=row.get("specimen_id"),
            )
        elif tps_id in working_tps:
            # succès lors d'un run précédent, échec cette fois (--overwrite) :
            # on retire l'entrée périmée plutôt que de laisser un TPS qui ne
            # correspond plus au statut logué.
            del working_tps[tps_id]

        landmarks_status[image_id] = dict(
            image_id=image_id, specimen_id=row.get("specimen_id"), dataset=row.get("dataset"),
            status=status, error_reason=error_reason or "", n_landmarks_found=n_found,
            model_name=model_name, processed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        counts[status] += 1
        n_done += 1

        if n_done % args.log_every == 0:
            checkpoint(tps_path, working_tps, landmarks_path, landmarks_status)
            elapsed = time.time() - t0
            print(
                f"[{n_done}/{len(targets)}] OK={counts['OK']} SUSPECT={counts['SUSPECT']} "
                f"FAILED={counts['FAILED']} SKIPPED={counts['SKIPPED']}  ({elapsed:.0f}s écoulées, checkpoint écrit)"
            )

    checkpoint(tps_path, working_tps, landmarks_path, landmarks_status)
    print("\nTerminé.")
    print(counts)
    print(f"Statuts     -> {landmarks_path}")
    print(f"Coordonnées -> {tps_path}")


if __name__ == "__main__":
    main()