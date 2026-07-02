"""Entraînement du modèle de détection de landmarks.

Usage:
    python train.py --tps chemin/vers/annotations.tps --images-root chemin/vers/photos

Pensé pour une RTX 2070 (8 Go) : image_size=256, batch_size=16, mixed
precision activée automatiquement si CUDA est disponible. Sur le MacBook M2
(pas de CUDA), le script tourne aussi via MPS ou CPU, utile pour du debug sur
un petit sous-échantillon avant de lancer le vrai entraînement sur le PC fixe.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from tqdm import tqdm

from tps_parser import parse_tps_file, resolve_image_path, build_image_index
from dataset import WingKeypointDataset
from model import ResNetUNet
from utils import heatmaps_to_coords, normalized_mean_error

# Indices (0-based) de deux landmarks utilisés comme référence d'échelle pour
# la métrique NME. À adapter : d'après l'exemple annoté, les points 1 et 9
# (indices 0 et 8) correspondent approximativement à la base et au bout de
# l'aile -- à vérifier sur vos données.
REF_LANDMARK_A = 0
REF_LANDMARK_B = 8


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_samples(tps_path: str, images_root: str) -> list[dict]:
    """Parse le .tps et valide CHAQUE image (résolution du chemin +
    lisibilité par OpenCV) avant de retourner les échantillons. Fait exprès
    avant le début de l'entraînement plutôt que lazily dans le DataLoader :
    une image manquante/corrompue découverte à l'itération 400 fait perdre
    tout le calcul déjà fait, alors qu'un contrôle amont ne coûte que
    quelques secondes et donne un rapport groupé exploitable.
    """
    specimens = parse_tps_file(tps_path)
    image_index = build_image_index(images_root)

    samples, missing, unreadable, misaligned = [], [], [], []
    for spec in specimens:
        try:
            img_path = resolve_image_path(spec, images_root, image_index=image_index)
        except FileNotFoundError:
            missing.append(spec.image_path)
            continue

        test_img = cv2.imread(str(img_path))
        if test_img is None:
            unreadable.append(str(img_path))
            continue

        # Vérifie que les landmarks recoupent bien l'image réelle (détecte un
        # désalignement annotation/image, ex. si les images ont été recadrées
        # après l'annotation TPS -- suspect vu le nom de dossier "orga_cropping").
        h_img, w_img = test_img.shape[:2]
        x_min, y_min = spec.landmarks.min(axis=0)
        x_max, y_max = spec.landmarks.max(axis=0)
        no_overlap = x_max < 0 or x_min > w_img or y_max < 0 or y_min > h_img
        if no_overlap:
            misaligned.append(
                (str(img_path), (x_min, y_min, x_max, y_max), (w_img, h_img))
            )
            continue

        samples.append(
            {
                "image_path": str(img_path),
                "landmarks": spec.landmarks,
                "specimen_id": spec.specimen_id,
                "orig_num": spec.orig_num,  # utile plus tard pour grouper P1/P2/S1/S2/S3 d'un même individu
            }
        )

    if missing:
        print(
            f"[avertissement] {len(missing)} image(s) introuvable(s) sur disque, ex: {missing[:5]}"
        )
    if unreadable:
        print(
            f"[avertissement] {len(unreadable)} image(s) trouvée(s) mais illisible(s) par OpenCV "
            f"(fichier corrompu/tronqué probable), ex: {unreadable[:5]}"
        )
    if misaligned:
        print(
            f"[avertissement] {len(misaligned)} spécimen(s) avec des landmarks totalement hors de "
            "l'image (désalignement annotation/image -- vérifier si les images ont été recadrées "
            "après l'annotation TPS). Exemples (chemin, bbox landmarks, taille image) :"
        )
        for path, bbox, size in misaligned[:5]:
            print(f"    {path}: bbox={tuple(round(v) for v in bbox)} vs image {size}")

    print(f"{len(samples)}/{len(specimens)} échantillons valides et chargés.")
    return samples


def run_epoch(model, loader, device, criterion, optimizer=None, scaler=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, total_nme, n_batches = 0.0, 0.0, 0
    for batch in tqdm(loader, leave=False):
        images = batch["image"].to(device)
        heatmaps_gt = batch["heatmaps"].to(device)

        with torch.set_grad_enabled(is_train):
            if scaler is not None:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    heatmaps_pred = model(images)
                    loss = criterion(heatmaps_pred, heatmaps_gt)
            else:
                heatmaps_pred = model(images)
                loss = criterion(heatmaps_pred, heatmaps_gt)

            if is_train:
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        # métrique NME (sur CPU, à partir des heatmaps prédites)
        image_size = images.shape[-1]
        preds_np = heatmaps_pred.detach().cpu().numpy()
        gts_np = batch["landmarks"].numpy()
        batch_nme = []
        for k in range(preds_np.shape[0]):
            coords = heatmaps_to_coords(preds_np[k], image_size)
            nme = normalized_mean_error(
                coords, gts_np[k], REF_LANDMARK_A, REF_LANDMARK_B
            )
            if not np.isnan(nme):
                batch_nme.append(nme)

        total_loss += loss.item()
        total_nme += np.mean(batch_nme) if batch_nme else 0.0
        n_batches += 1

    return total_loss / n_batches, total_nme / n_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tps", required=True)
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--heatmap-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument(
        "--fold", type=int, default=0, help="Indice du fold utilisé comme validation"
    )
    parser.add_argument("--out-dir", default="checkpoints")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    samples = load_samples(args.tps, args.images_root)
    print(f"{len(samples)} échantillons chargés.")
    n_keypoints = samples[0]["landmarks"].shape[0]

    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    splits = list(kf.split(samples))
    train_idx, val_idx = splits[args.fold]
    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    print(f"Fold {args.fold}: {len(train_samples)} train / {len(val_samples)} val")

    train_ds = WingKeypointDataset(
        train_samples, args.image_size, args.heatmap_size, train=True
    )
    val_ds = WingKeypointDataset(
        val_samples, args.image_size, args.heatmap_size, train=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = ResNetUNet(n_keypoints=n_keypoints).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = torch.nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_val_nme = float("inf")
    best_state = None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        train_loss, train_nme = run_epoch(
            model, train_loader, device, criterion, optimizer, scaler
        )
        val_loss, val_nme = run_epoch(model, val_loader, device, criterion)
        scheduler.step()

        print(
            f"[{epoch+1:03d}/{args.epochs}] "
            f"train_loss={train_loss:.5f} train_nme={train_nme:.4f} | "
            f"val_loss={val_loss:.5f} val_nme={val_nme:.4f}"
        )

        if val_nme < best_val_nme:
            best_val_nme = val_nme
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, out_dir / "best_model.pt")

    print(
        f"Meilleur val_nme : {best_val_nme:.4f} (checkpoint sauvegardé dans {out_dir/'best_model.pt'})"
    )


if __name__ == "__main__":
    main()
