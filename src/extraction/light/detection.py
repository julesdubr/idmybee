"""Détection des ailes avec la méthode light (YOLO-OBB spécialisée).

Ce module expose les deux mêmes points d'entrée que `heavy/detection.py` :

- `load_model(args)` : charge le modèle YOLO-OBB.
- `process_one(ctx, image, row, args)` : détecte l'aile sur une image.
"""

from __future__ import annotations

import numpy as np


def add_arguments(parser) -> None:
    """Déclare les arguments CLI spécifiques au backend light."""
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-det", type=int, default=10)


def load_model(args):
    """Charge le modèle YOLO-OBB léger.

    Retourne un contexte (dict) réutilisé par `process_one` pour chaque image.
    """
    from ultralytics import YOLO

    model = YOLO(args.model)
    return {"model": model}


def process_one(ctx: dict, image: np.ndarray, row: dict, args) -> tuple[dict, np.ndarray | None]:
    """Détecte l'aile sur `image` et complète `row` (status/box/confidence).

    `image` a déjà été chargée par l'appelant. `row` contient déjà les champs
    communs (image_id, specimen_id, dataset, ...) initialisés à leurs valeurs
    par défaut ; cette fonction ne fait que les mettre à jour.

    Contrairement au backend heavy, la détection light ne normalise jamais de
    crop elle-même : elle retourne systématiquement `None`, laissant
    `extract_wings.py` faire la normalisation à partir de la box choisie.
    """
    model = ctx["model"]

    try:
        results = model.predict(
            image,
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            max_det=args.max_det,
            verbose=False,
        )

        result = results[0]
        count = len(result.obb) if result.obb is not None else 0
        row["n_detections"] = str(count)

        if result.obb is None or count == 0:
            row["error_reason"] = "aucune_detection"
            return row, None

        confs = result.obb.conf.detach().cpu().numpy()
        best = int(np.argmax(confs))

        points = result.obb.xyxyxyxy[best].detach().cpu().numpy().reshape(4, 2)
        h, w = image.shape[:2]
        points[:, 0] /= w
        points[:, 1] /= h

        row["status"] = "OK"
        row["confidence"] = f"{float(confs[best]):.4f}"
        for i, (x, y) in enumerate(points, start=1):
            row[f"x{i}"] = f"{x:.8f}"
            row[f"y{i}"] = f"{y:.8f}"

    except Exception as exc:
        row["error_reason"] = f"exception: {exc}"
        return row, None

    return row, None