"""Modifie les IDs des spécimens dans un fichier .tps pour qu'ils soient consécutifs
et commencent à 0.

Usage:
    python3 scripts/clean_tps.py --tps annotations.tps --out cleaned.tps
"""

import argparse
from pathlib import Path

from utils.tps_io import parse_tps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tps", required=True)
    parser.add_argument("--out", default="cleaned.tps")
    args = parser.parse_args()

    specimens = parse_tps(args.tps)
    print(f"{len(specimens)} spécimens trouvés dans le fichier .tps")

    with open(args.out, "wb") as f:
        for idx, spec in enumerate(specimens):
            f.write(f"LM={spec.landmarks.shape[0]}\n".encode())
            for x, y in spec.landmarks:
                f.write(f"{x:.5f} {y:.5f}\n".encode())
            f.write(f"IMAGE={spec.image_path}\n".encode())
            f.write(f"ID={idx}\n".encode())
        f.close()

if __name__ == "__main__":
    main()
