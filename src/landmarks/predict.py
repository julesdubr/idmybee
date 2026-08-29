"""predict.py
Phase 2 : positionnement automatique des landmarks (UNet) sur les crops
normalisés (sortie de extraction/normalize_crop.py).

Deux usages, même logique de prédiction :
- `predict_landmarks(image, model, device, n_landmarks)` : un crop déjà
  chargé (BGR) -> landmarks (x, y) + statut. Ne fait aucune I/O -- utile
  pour un pipeline "une photo à la fois" (terrain). Voir aussi
  landmarks/renumber.py::numerate_one pour enchaîner jusqu'à la
  numérotation canonique sur cette même image.
- CLI (`python -m landmarks.predict --dataset ... --mode ...`) : traite tout
  un dataset à partir de `extraction/{mode}/crops.csv`, écrit
  `landmarks/<tps>` + `landmarks/landmarks.csv`.

Entrée (CLI) : extraction/{mode}/crops.csv (Phase 1).
Sortie (CLI) : landmarks/<tps>, landmarks/landmarks.csv, <dataset>/pipeline_stats.csv.

Un crop SKIPPED en Phase 1 (fichier déjà présent sur disque, pas une
détection ratée) est un crop valide comme un autre : traité ici comme un
OK, pas ignoré. La version précédente de ce script ne gardait que
status=="OK", perdant silencieusement les crops SKIPPED d'un run Phase 1
précédent (5 images sur ce jeu de données).

Statuts (voir aussi utils.pipeline_io.RunCounter) :
  - OK      : les `n_landmarks` points attendus ont été trouvés.
  - SUSPECT : moins de points que prévu (mais au moins un) -- la
    renumérotation (Phase 3, landmarks/renumber.py) échouera
    automatiquement pour ces spécimens, leur nombre de points ne
    correspondant plus à la référence. SUSPECT ici ne veut donc PAS dire
    "à surveiller", mais "deviendra FAILED en Phase 3, et voici pourquoi".
  - FAILED  : aucun pic trouvé, ou crop illisible.

Par rapport au notebook d'origine, deux corrections de fond :

  1. `local_maxima()` peut marquer PLUSIEURS pixels adjacents pour un même
     pic (plateau à valeur quasi identique). Les traiter comme des points
     séparés puis garder "les N valeurs les plus hautes" peut faire
     disparaître un vrai landmark distinct au profit de doublons du même
     pic. On regroupe chaque pic par composante connexe (scipy.ndimage.label)
     avant de classer/tronquer.
  2. Si le modèle ne produit pas assez de pics distincts, l'ancien code
     tronquait silencieusement (`maximas[-18:,:]` renvoie tout ce qu'il y a
     si moins de 18 lignes -- pas d'erreur). Ici, un résultat avec moins de
     landmarks que prévu est explicitement marqué SUSPECT avec la raison,
     jamais juste écrit tel quel sans avertissement.

`utils.tps_io.ImageLandmarks.tps_id` doit être un entier, et `write_tps`
réécrit le fichier entier (pas d'append) : `image_id` (le hash de contenu de
la Phase 0) n'est donc pas directement utilisable comme `tps_id`, et
`specimen_id` ne convient pas non plus (un même specimen a souvent plusieurs
photos, qui doivent rester des entrées TPS distinctes). `ImageLandmarks.from_image()`
calcule `tps_id` à partir de `image_id` et le persiste dans le TPS
(COMMENT=), avec `specimen_id` : les étapes suivantes n'ont plus besoin de
rejoindre crops.csv/manifest.csv pour retrouver ces identifiants.

Comme write_tps réécrit tout, ce script fonctionne par "checkpoint" : au
démarrage, le TPS existant est reparsé pour connaître ce qui est déjà là ;
chaque image traitée met à jour (ou retire, si elle échoue maintenant après
avoir réussi avant) l'entrée correspondante dans un dictionnaire en
mémoire ; le TPS et landmarks.csv sont réécrits en entier tous les
`--log-every` (et une dernière fois à la fin).

Usage :
    python -m landmarks.predict --dataset data/Bombus --mode light \\
        --model data/models/unet_landmarks/<run_id>/weights.pth

    # ne retenter que les échecs d'un run précédent :
    python -m landmarks.predict --dataset data/Bombus --mode light --model ... --retry-failed
"""
from __future__ import annotations

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
from utils.pipeline_io import RunCounter, format_duration, read_csv_rows, resolve_path, update_pipeline_stats
from utils.tps_io import ImageLandmarks, image_id_to_sid, parse_tps, write_tps
from landmarks_trainer.model import load_weights

LANDMARKS_FIELDS = [
    "image_id", "specimen_id", "split", "status", "error_reason",
    "n_landmarks_found", "model_name", "processing_time_s", "processed_at",
]


# ---------------------------------------------------------------------------
# Extraction des landmarks depuis le heatmap prédit par le UNet
# ---------------------------------------------------------------------------

def extract_top_landmarks(heatmap: np.ndarray, n_landmarks: int) -> tuple[np.ndarray, int]:
    """Retourne (points_yx, n_trouvés) : jusqu'à `n_landmarks` points (y, x),
    un par pic distinct du heatmap.

    Regroupe par composante connexe (un pic = une composante, voir point 1
    de la docstring du module) et garde le pixel de valeur maximale de
    chaque groupe, plutôt que de traiter chaque pixel d'un plateau comme un
    landmark à part entière.

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


def predict_landmarks(
    image_bgr: np.ndarray, model, device, n_landmarks: int,
) -> tuple[str, str, np.ndarray | None, int]:
    """Prédit les landmarks d'UN crop déjà chargé (BGR). Ne fait aucune I/O
    (voir `predict_landmarks_from_path` pour la version fichier, utilisée
    par le CLI). Retourne (status, error_reason, coords_xy_ou_None,
    n_trouvés) ; coords_xy est en ordre (x, y), prêt pour ImageLandmarks."""
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_t = torch.tensor(img.transpose(2, 0, 1) / 255.0, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_t).cpu().squeeze(0).numpy().transpose(1, 2, 0).squeeze(axis=2)

    coords_yx, n_found = extract_top_landmarks(output, n_landmarks)
    if n_found == 0:
        return "FAILED", "aucun_maximum_local", None, 0

    coords_xy = coords_yx[:, [1, 0]]
    if n_found < n_landmarks:
        return "SUSPECT", f"seulement_{n_found}_maxima_sur_{n_landmarks}_attendus", coords_xy, n_found
    return "OK", "", coords_xy, n_found


def predict_landmarks_from_path(
    crop_path: Path, model, device, n_landmarks: int,
) -> tuple[str, str, np.ndarray | None, int]:
    """Version fichier de `predict_landmarks`, pour le CLI (lit le crop)."""
    img = cv2.imread(str(crop_path))
    if img is None or img.size == 0:
        return "FAILED", "crop_illisible", None, 0
    return predict_landmarks(img, model, device, n_landmarks)


# ---------------------------------------------------------------------------
# Sélection des crops à traiter (Phase 1 -> Phase 2)
# ---------------------------------------------------------------------------

def load_target_crops(crops_path: Path, split_filter: str | None) -> list[dict]:
    """Lit crops.csv et ne garde que la DERNIÈRE occurrence de chaque
    image_id (crops.csv est append-only -- un retry/overwrite Phase 1 peut
    avoir ajouté une ligne plus récente, statut différent, pour le même
    image_id). Le filtre de statut/split s'applique ENSUITE, sur ce dernier
    état connu.

    Un crop SKIPPED (fichier déjà présent sur disque en Phase 1, pas une
    détection ratée) est traité comme un OK : le fichier est valide, seule
    la façon dont il a été produit diffère. FAILED (aucun fichier écrit)
    reste seul exclu."""
    if not crops_path.exists():
        raise SystemExit(
            f"{crops_path} introuvable -- lancer d'abord detect_wing.py "
            f"et normalize_crop.py (Phase 1) pour ce mode."
        )
    crops = read_csv_rows(crops_path)
    if not crops:
        raise SystemExit(f"{crops_path} est vide -- rien à traiter.")

    latest_by_id = {row["image_id"]: row for row in crops}
    return [
        row for row in latest_by_id.values()
        if row.get("status") in ("OK", "SKIPPED") and (not split_filter or row.get("split") == split_filter)
    ]


# ---------------------------------------------------------------------------
# État TPS + landmarks.csv en mémoire, avec checkpoints réguliers
# ---------------------------------------------------------------------------

def load_previous_status(landmarks_path: Path) -> dict[str, dict]:
    """Recharge landmarks.csv d'un run précédent (reprise), le cas échéant.
    Un fichier avec un schéma différent (ancienne version du script) est
    signalé explicitement plutôt que mélangé avec le format actuel."""
    if not landmarks_path.exists():
        return {}
    rows = read_csv_rows(landmarks_path)
    if rows and set(rows[0].keys()) != set(LANDMARKS_FIELDS):
        raise SystemExit(
            f"{landmarks_path} existe avec un schéma différent -- le déplacer ou le "
            f"supprimer avant de relancer (colonnes attendues : {LANDMARKS_FIELDS})."
        )
    return {row["image_id"]: row for row in rows}


def should_skip(previous: dict | None, overwrite: bool, retry_failed: bool) -> bool:
    """Une image déjà loguée est resautée sauf si `--overwrite` (retraiter
    tout) ou `--retry-failed` (retraiter seulement les FAILED précédents)."""
    if previous is None or overwrite:
        return False
    if previous.get("status") == "FAILED" and retry_failed:
        return False
    return True


def load_working_tps(tps_path: Path) -> dict[int, ImageLandmarks]:
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


def checkpoint(tps_path: Path, working_tps: dict, landmarks_path: Path, landmarks_status: dict) -> None:
    tps_path.parent.mkdir(parents=True, exist_ok=True)
    write_tps(tps_path, sorted(working_tps.values(), key=lambda s: s.tps_id))

    landmarks_path.parent.mkdir(parents=True, exist_ok=True)
    with landmarks_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LANDMARKS_FIELDS)
        writer.writeheader()
        writer.writerows(landmarks_status.values())


# ---------------------------------------------------------------------------
# Programme principal
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Positionnement automatique des landmarks (UNet, Phase 2).")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--mode", default="light", choices=["heavy", "light"],
                         help="Backend Phase 1 dont les crops sont lus (extraction/{mode}/crops.csv).")
    parser.add_argument("--model", required=True, help="Chemin du .pth du modèle UNet.")
    parser.add_argument("--base-dir", default=None, help="Racine pour résoudre les output_path relatifs de crops.csv.")
    parser.add_argument("--tps", default="landmarks.tps", help="Nom du TPS de sortie (dans <dataset>/landmarks/).")
    parser.add_argument("--n-landmarks", type=int, default=19,
                         help="19 = Tancrede's full blueprint (LM3 included, current default). "
                              "Pass 18 to run an older/legacy model that doesn't predict LM3.")
    parser.add_argument("--split", default=None, help="Ne traiter qu'un split (train/test/...). Vide = tous.")
    parser.add_argument("--device", default=None, help="'cpu', 'cuda', etc. Vide = auto-détection.")
    parser.add_argument("--overwrite", action="store_true", help="Retraiter même si déjà logué.")
    parser.add_argument("--retry-failed", action="store_true", help="Retenter les images loguées FAILED lors d'un run précédent.")
    parser.add_argument("--log-every", type=int, default=50, help="Fréquence des checkpoints et des messages de progression.")
    return parser.parse_args()


def main():
    args = parse_args()
    crops_path = args.dataset / "extraction" / args.mode / "crops.csv"
    tps_path = args.dataset / "landmarks" / args.tps
    landmarks_path = args.dataset / "landmarks" / "landmarks.csv"
    stats_path = args.dataset / "pipeline_stats.csv"
    base_dir = Path(args.base_dir) if args.base_dir else None

    targets = load_target_crops(crops_path, args.split)
    print(f"Mode : {args.mode}")
    print(f"{len(targets)} crop(s) à considérer ({crops_path})")
    if not targets:
        print("Aucun crop sélectionné, rien à faire.")
        return

    landmarks_status = load_previous_status(landmarks_path)
    if landmarks_status:
        print(f"{len(landmarks_status)} entrée(s) déjà présentes dans {landmarks_path} (reprise du run précédent).")

    working_tps = load_working_tps(tps_path)
    if working_tps:
        print(f"{len(working_tps)} spécimen(s) déjà présents dans {tps_path} (repris tels quels si sautés ce run).")

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device :", device)
    model = load_weights(args.model, device=device)  # state_dict, not the old full-pickle format -- see landmarks_trainer/migrate_legacy_weights.py for existing .pth files
    model.eval()
    model_name = Path(args.model).stem

    pipeline_start = time.perf_counter()
    counter = RunCounter()
    n_done = 0
    n_resumed = 0

    for row in targets:
        image_id = row["image_id"]
        tps_id = image_id_to_sid(image_id)

        if should_skip(landmarks_status.get(image_id), args.overwrite, args.retry_failed):
            n_resumed += 1
            continue

        crop_path = resolve_path(row["output_path"], base_dir)
        item_start = time.perf_counter()
        try:
            status, error_reason, coords_xy, n_found = predict_landmarks_from_path(
                crop_path, model, device, args.n_landmarks,
            )
        except Exception as e:
            status, error_reason, coords_xy, n_found = "FAILED", f"exception: {e}", None, 0
        processing_time_s = time.perf_counter() - item_start

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
            image_id=image_id, specimen_id=row.get("specimen_id"), split=row.get("split"),
            status=status, error_reason=error_reason or "", n_landmarks_found=n_found,
            model_name=model_name, processing_time_s=f"{processing_time_s:.4f}",
            processed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        counter.add(status)
        n_done += 1

        if n_done % args.log_every == 0:
            checkpoint(tps_path, working_tps, landmarks_path, landmarks_status)
            elapsed = time.perf_counter() - pipeline_start
            print(
                f"[{n_done}/{len(targets) - n_resumed}] "
                f"temps écoulé : {format_duration(elapsed)} — "
                f"moyenne : {elapsed / n_done:.3f} s/image — {counter}"
                + (f"  ({n_resumed} repris(es))" if n_resumed else "")
            )

    checkpoint(tps_path, working_tps, landmarks_path, landmarks_status)

    total_time_s = time.perf_counter() - pipeline_start
    update_pipeline_stats(stats_path, "landmarks", model_name, counter.as_dict(), total_time_s)

    print("\nTerminé.")
    print(counter, f" (+ {n_resumed} repris(es) d'un run précédent)" if n_resumed else "")
    print(f"Statuts     -> {landmarks_path}")
    print(f"Coordonnées -> {tps_path}")
    print(f"Stats       -> {stats_path}")


if __name__ == "__main__":
    main()
