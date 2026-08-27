"""build_reference.py
Fige UNE FOIS la forme de référence (consensus GPA du gabarit de vérité
terrain, ex: Tancrède) utilisée par landmarks/renumber.py pour numéroter --
en mode batch comme en mode single-image (terrain).

Pourquoi un artefact séparé plutôt que recalculer le consensus à chaque run
de renumérotation (ancien comportement) : la référence ne doit jamais
bouger silencieusement. Si le TPS de vérité terrain est complété plus tard
(nouveaux spécimens digitalisés, par exemple), un recalcul à la volée
changerait légèrement le consensus -- et donc TOUTE future numérotation --
sans que rien ne le signale, et sans garantir que le mode batch et un futur
mode terrain (une image à la fois) utilisent exactement la même référence.
Un artefact figé et versionné rend ce choix explicite : il faut relancer ce
script à la main pour changer de référence, jamais un simple effet de bord
d'un autre run.

Entrée : un TPS de référence (landmarks ORDONNÉS, vérité terrain, ex:
Tancrède -- 19 landmarks, dont un absent du schéma du détecteur UNet, voir
--drop).
Sortie : un .npz contenant `zones` (n_zones, 2) -- la forme de référence
prête à être passée à `landmarks.methods.*.numerate()` -- et ses métadonnées
(source, nombre de spécimens utilisés, landmark éventuellement retiré, date).

Usage :
    python -m landmarks.build_reference --ref data/references/ref-landmarks.tps \\
        --drop 3 --out data/references/reference_shape.npz
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parent))
from utils.gpa import gpagen
from utils.tps_io import parse_tps


def build_reference_shape(ref_specimens: list, expected_lm: int) -> np.ndarray:
    """Consensus GPA du template de référence, sur les spécimens à
    `expected_lm` landmarks uniquement (un TPS de vérité terrain peut
    contenir quelques spécimens incomplets)."""
    matching = [s for s in ref_specimens if s.n_points == expected_lm]
    if not matching:
        raise SystemExit(
            f"Aucun spécimen à {expected_lm} landmarks dans le TPS de référence "
            f"(essayer --n-landmarks pour forcer un autre nombre)."
        )
    return gpagen([s.landmarks for s in matching]).mean_shape


def load_reference(path: Path) -> tuple[np.ndarray, dict]:
    """Charge une forme de référence figée par ce script. Utilisée par
    landmarks/renumber.py, en mode batch comme en mode single-image."""
    if not path.exists():
        raise SystemExit(
            f"{path} introuvable -- lancer d'abord :\n"
            f"  python -m landmarks.build_reference --ref <tps> --drop <n> --out {path}"
        )
    data = np.load(path, allow_pickle=True)
    meta = {}
    for key in data.files:
        if key == "zones":
            continue
        value = data[key]
        # scalaires (drop, dates, ...) enregistrés comme tableaux 0-d par
        # np.savez : .item() les reconvertit en int/str Python. Un vrai
        # tableau (zone_orig_idx) doit rester tel quel.
        meta[key] = value.item() if value.shape == () else value
    return data["zones"], meta


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ref", type=Path, required=True, help="TPS de référence (vérité terrain).")
    parser.add_argument("--drop", type=int, default=None,
                         help="Landmark(s) de --ref sans équivalent chez le détecteur (ex: 3).")
    parser.add_argument("--n-landmarks", type=int, default=None,
                         help="Nombre de landmarks attendu dans --ref (défaut : le plus fréquent trouvé).")
    parser.add_argument("--out", type=Path, required=True, help="Fichier .npz de sortie.")
    return parser.parse_args()


def main():
    args = parse_args()

    ref_specimens, ref_errors = parse_tps(args.ref, strict=False)
    if not ref_specimens:
        raise SystemExit(
            f"Aucun spécimen valide dans {args.ref} ({len(ref_errors)} erreur(s) de "
            f"parsing). Vérifier que ce fichier est bien un .tps, pas un CSV de métadonnées."
        )

    if args.n_landmarks is not None:
        expected_lm = args.n_landmarks
    else:
        counts = np.bincount([s.n_points for s in ref_specimens])
        expected_lm = int(np.argmax(counts))

    consensus = build_reference_shape(ref_specimens, expected_lm)
    n_used = sum(1 for s in ref_specimens if s.n_points == expected_lm)

    if args.drop is not None:
        zones = np.delete(consensus, args.drop, axis=0)
        zone_orig_idx = [i for i in range(len(consensus)) if i != args.drop]
    else:
        zones = consensus
        zone_orig_idx = list(range(len(consensus)))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        zones=zones,
        source_ref=str(args.ref),
        drop=args.drop if args.drop is not None else -1,
        zone_orig_idx=np.array(zone_orig_idx),
        n_specimens_used=n_used,
        n_landmarks_source=expected_lm,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )

    print(f"Référence : {n_used} spécimen(s) à {expected_lm} landmarks dans {args.ref}")
    print(f"Forme figée : {len(zones)} zones" + (f" (LM{args.drop} exclu)" if args.drop is not None else ""))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
