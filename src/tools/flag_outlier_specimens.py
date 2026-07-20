"""Diagnostic post-GPA : distingue le bruit de détection ponctuel (1-2
landmarks décalés) des échecs de registration complets (la quasi-totalité
des landmarks du spécimen décalés -- specimen mal aligné dans son
ensemble, souvent une forme d'aile hors gabarit).

Pour chaque landmark, seuil robuste = médiane + 6*MAD de la distance à la
position médiane (par landmark). Un spécimen est "à problème" si une
grande partie de ses landmarks dépassent ce seuil.

Usage:
    python tools/flag_outlier_specimens.py data/annotations/gabriel_reordered.tps data/annotations/gabriel.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/, quel que soit le cwd
from utils.gpa import gpagen
from utils.tps_io import parse_tps

OUT_DIR = Path(__file__).resolve().parents[1] / "out"


def outlier_matrix(aligned: np.ndarray, k: float = 6.0) -> np.ndarray:
    """(n_specimens, n_landmarks) bool : True si le point est loin de la
    position médiane de son landmark (médiane + k*MAD)."""
    med = np.median(aligned, axis=0)
    dist = np.linalg.norm(aligned - med[None, :, :], axis=2)
    mad = np.median(np.abs(dist - np.median(dist, axis=0)), axis=0)
    thresh = np.median(dist, axis=0) + k * mad
    return dist > thresh[None, :]


def main():
    tps_path, csv_path = sys.argv[1], sys.argv[2]
    heavy_frac = 10 / 18  # au-delà de ~55% des landmarks flagués -> specimen entier suspect

    specimens, _ = parse_tps(tps_path, strict=False)
    result = gpagen([s.landmarks for s in specimens])
    is_outlier = outlier_matrix(result.aligned)
    n_outlier = is_outlier.sum(axis=1)

    meta = pd.read_csv(csv_path).set_index("id")
    sids = np.array([s.sid for s in specimens])
    matched = np.isin(sids, meta.index)
    if not matched.all():
        unmatched = sids[~matched]
        print(
            f"{len(unmatched)} spécimen(s) du TPS sans entrée CSV correspondante, ignoré(s) pour la "
            f"répartition par espèce (id manquants: {unmatched[:10].tolist()}"
            f"{', ...' if len(unmatched) > 10 else ''})"
        )
    espece = meta.loc[sids[matched], "espece"].values

    heavy = n_outlier >= heavy_frac * is_outlier.shape[1]
    print(f"{heavy.sum()}/{len(specimens)} spécimens avec >= {heavy_frac:.0%} de landmarks outliers\n")

    df = pd.DataFrame({"espece": espece, "heavy": heavy[matched]})
    rate = df.groupby("espece")["heavy"].agg(["sum", "count"])
    rate["taux%"] = (100 * rate["sum"] / rate["count"]).round(1)
    print(rate.sort_values("taux%", ascending=False))

    OUT_DIR.mkdir(exist_ok=True)
    out = pd.DataFrame({"id": sids, "n_outlier_landmarks": n_outlier, "heavy": heavy})
    out.to_csv(OUT_DIR / "outlier_specimens.csv", index=False)
    print(f"\n-> {OUT_DIR / 'outlier_specimens.csv'}")


if __name__ == "__main__":
    main()
