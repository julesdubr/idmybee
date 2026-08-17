"""flag_outlier_specimens.py
Diagnostic post-GPA : distingue le bruit de détection ponctuel (1-2
landmarks décalés) des échecs de registration complets (la quasi-totalité
des landmarks du spécimen décalés -- specimen mal aligné dans son
ensemble, souvent une forme d'aile hors gabarit).

Le seuil (médiane + 6*MAD de la distance à la position médiane, par
landmark) est calculé PAR ESPÈCE, pas sur l'ensemble du jeu de données :
les espèces ont des formes d'aile différentes par nature (c'est la base
même de la classification), donc un seuil global confondrait "aile
différente parce que d'une autre espèce" avec "aile mal alignée" --
gonflant artificiellement le taux d'outliers des espèces les moins
représentées ou aux ailes les plus atypiques (ex: B. rupestris). Un groupe
avec moins de --min-group-size spécimens n'a pas de médiane/MAD fiable :
il est laissé de côté (considéré OK) plutôt que d'inventer un seuil.

Le chargement/la jointure TPS<->specimens.csv sont délégués à
utils.dataset.load_labeled_dataset (mêmes règles que lda.py : specimen_id
résolu via COMMENT= si présent, sinon repli via --images-csv).

Usage:
    python tools/flag_outlier_specimens.py data/annotations/landmarks_numbered.tps data/manifest/specimens.csv
    python tools/flag_outlier_specimens.py ... --images-csv data/manifest/images.csv --out out/outliers.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/, quel que soit le cwd
from utils.dataset import load_labeled_dataset
from utils.gpa import gpagen

MIN_GROUP_SIZE = 10  # en dessous, une médiane/MAD par landmark n'est pas fiable


def outlier_matrix(aligned: np.ndarray, k: float = 6.0) -> np.ndarray:
    """(n_specimens, n_landmarks) bool : True si le point est loin de la
    position médiane de son landmark (médiane + k*MAD)."""
    med = np.median(aligned, axis=0)
    dist = np.linalg.norm(aligned - med[None, :, :], axis=2)
    mad = np.median(np.abs(dist - np.median(dist, axis=0)), axis=0)
    thresh = np.median(dist, axis=0) + k * mad
    return dist > thresh[None, :]


def flag_by_species(specimens, species: np.ndarray, heavy_frac: float, min_group_size: int):
    """GPA + seuil MAD calculés séparément pour chaque espèce (voir docstring
    du module). Retourne (n_outlier, heavy) alignés sur `specimens`."""
    n_landmarks = specimens[0].n_points
    n_outlier = np.zeros(len(specimens), dtype=int)
    heavy = np.zeros(len(specimens), dtype=bool)

    for sp_name in sorted(set(species)):
        idx = np.where(species == sp_name)[0]
        if len(idx) < min_group_size:
            print(f"  {sp_name}: {len(idx)} spécimen(s), < {min_group_size} -- non évalué (considéré OK)")
            continue
        group = [specimens[i] for i in idx]
        result = gpagen([s.landmarks for s in group])
        group_outlier = outlier_matrix(result.aligned)
        group_n_outlier = group_outlier.sum(axis=1)
        n_outlier[idx] = group_n_outlier
        heavy[idx] = group_n_outlier >= heavy_frac * n_landmarks

    return n_outlier, heavy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tps_path")
    ap.add_argument("specimens_csv", help="data/manifest/specimens.csv")
    ap.add_argument("--images-csv", default=None,
                     help="data/manifest/images.csv -- nécessaire seulement si le TPS n'a pas de "
                          "COMMENT=specimen_id (TPS écrit avant la mise à jour de predict_unet.py)")
    ap.add_argument("--manifest-dir", type=Path, default=Path("data/manifest"),
                     help="Dossier manifest par défaut pour --out (défaut: data/manifest)")
    ap.add_argument("--out", type=Path, default=None,
                     help="CSV de sortie (défaut: <manifest-dir>/outlier_specimens.csv)")
    ap.add_argument("--min-group-size", type=int, default=MIN_GROUP_SIZE,
                     help=f"Taille minimale d'un groupe espèce pour évaluer un seuil (défaut: {MIN_GROUP_SIZE})")
    args = ap.parse_args()

    heavy_frac = 10 / 18  # au-delà de ~55% des landmarks flagués -> specimen entier suspect

    specimens, meta_df = load_labeled_dataset(
        args.tps_path, args.specimens_csv, images_csv=args.images_csv, strict=False
    )

    print(f"Seuils calculés par espèce (min {args.min_group_size} spécimen(s)/groupe) :")
    n_outlier, heavy = flag_by_species(specimens, meta_df["species"].values, heavy_frac, args.min_group_size)

    print(f"\n{heavy.sum()}/{len(specimens)} spécimens avec >= {heavy_frac:.0%} de landmarks outliers\n")

    df = pd.DataFrame({"species": meta_df["species"], "heavy": heavy})
    rate = df.groupby("species")["heavy"].agg(["sum", "count"])
    rate["taux%"] = (100 * rate["sum"] / rate["count"]).round(1)
    print(rate.sort_values("taux%", ascending=False))

    out_path = args.out or (args.manifest_dir / "outlier_specimens.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({
        "tps_id": [s.tps_id for s in specimens],
        "specimen_id": [s.specimen_id for s in specimens],
        "n_outlier_landmarks": n_outlier,
        "status": np.where(heavy, "SUSPECT", "OK"),
    })
    out.to_csv(out_path, index=False)
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()