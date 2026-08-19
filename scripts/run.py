#!/usr/bin/env python3
"""
run.py -- Enchaîne les phases 0->4 du pipeline idmybee en un seul appel.
Equivalent cross-OS (Windows/macOS/Linux) de l'ancien scripts/run.sh.

A lancer DEPUIS LA RACINE DU REPO (idmybee/), environnement conda/venv
déjà activé :

    python scripts/run.py
    # ou, si le fichier est exécutable (macOS/Linux) :
    ./scripts/run.py

Chaque étape peut aussi être relancée seule en appelant directement le
script correspondant (voir sa docstring pour les options de reprise :
--overwrite, --retry_failed pour predict_unet.py, etc.).

Ce script suppose que chaque module src/.../xxx.py expose une fonction
`main(argv=None)` (ou équivalent) plutôt que de tout exécuter dans
`if __name__ == "__main__":` -- comme prévu suite au refactor argparse.
Si ce n'est pas encore le cas pour un module donné, ce script l'appelle
en subprocess à la place (voir USE_SUBPROCESS ci-dessous), donc il
fonctionne aussi pendant la transition.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import config

# --- Bascule migration progressive -----------------------------------------
# Une fois qu'un module expose main(argv), passe son entrée à False pour
# l'appeler in-process (plus rapide, erreurs Python natives) plutôt qu'en
# subprocess. Tant que c'est True, on lance `python module.py <args>`.
USE_SUBPROCESS = {
    "build_manifest": True,
    "crop_wings": True,
    "predict_unet": True,
    "reconstruct_tps": True,
    "flag_outlier_specimens": True,
    "lda": True,
}


def _run_subprocess(script_path: Path, args: list[str]) -> None:
    cmd = [sys.executable, str(script_path), *args]
    print(f"    $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _run_inprocess(module_path: str, args: list[str]) -> None:
    """Importe le module et appelle sa fonction main(argv) directement."""
    import importlib

    mod = importlib.import_module(module_path)
    mod.main(args)


def run_step(name: str, script_path: Path, module_path: str, args: list[str]) -> None:
    if USE_SUBPROCESS.get(name, True):
        _run_subprocess(script_path, args)
    else:
        _run_inprocess(module_path, args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Enchaîne les phases 0->4 du pipeline idmybee.",
    )
    p.add_argument(
        "--level",
        default=config.LEVEL,
        choices=["species", "caste"],
        help="Niveau de classification pour la Phase 4 (défaut: %(default)s).",
    )
    p.add_argument(
        "--drop-landmark",
        type=int,
        default=config.DROP_LANDMARK,
        help="Landmark de référence à exclure en Phase 3 (défaut: %(default)s).",
    )
    p.add_argument(
        "--ref-json",
        type=Path,
        default=config.REF_JSON,
        help="Fichier de référence YOLOE pour la Phase 1 (défaut: %(default)s).",
    )
    p.add_argument(
        "--skip",
        nargs="+",
        default=[],
        choices=["0", "1", "2", "3", "3.5", "4"],
        help="Phases à sauter (utile pour reprendre après une phase déjà faite).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    repo_root = config.REPO_ROOT
    src = repo_root / "src"

    def skip(phase: str) -> bool:
        if phase in args.skip:
            print(f"\n=== {phase} : SKIPPED ===")
            return True
        return False

    # --- Phase 0 : indexation manifest ---------------------------------
    if not skip("0"):
        print("\n=== Phase 0 : indexation manifest ===")
        run_step(
            "build_manifest",
            src / "manifest" / "build_manifest.py",
            "manifest.build_manifest",
            [
                "--roots", str(config.ROOTS_JSON),
                "--species-csv", str(config.SPECIES_CSV),
            ],
        )

    # --- Phase 1 : extraction des crops ---------------------------------
    if not skip("1"):
        print("\n=== Phase 1 : extraction des crops ===")
        run_step(
            "crop_wings",
            src / "extraction" / "crop_wings.py",
            "extraction.crop_wings",
            [
                "--ref", str(args.ref_json),
                "--ref_crops", str(config.REF_CROPS_DIR),
                "--model", str(config.YOLO_MODEL),
                "--manifest_dir", str(config.MANIFEST_DIR),
                "--output_root", str(config.CROPS_OUTPUT_ROOT),
            ],
        )

    # --- Phase 2 : prédiction des landmarks (UNet) ------------------------
    if not skip("2"):
        print("\n=== Phase 2 : prédiction des landmarks (UNet) ===")
        run_step(
            "predict_unet",
            src / "landmarks" / "predict_unet.py",
            "landmarks.predict_unet",
            [
                "--manifest_dir", str(config.MANIFEST_DIR),
                "--model_path", str(config.UNET_MODEL),
                "--tps_out", str(config.UNET_TPS),
            ],
        )

    # --- Phase 3 : renumérotation (Hungarian + Umeyama) --------------------
    if not skip("3"):
        print("\n=== Phase 3 : renumérotation (Hungarian + Umeyama) ===")
        # --ambiguity-ratio : NON calibré
        run_step(
            "reconstruct_tps",
            src / "numbering" / "reconstruct_tps.py",
            "numbering.reconstruct_tps",
            [
                str(config.REFERENCE_TPS),
                str(config.UNET_TPS),
                str(config.NUMBERED_TPS),
                "--drop", str(args.drop_landmark),
                "--manifest-dir", str(config.MANIFEST_DIR),
            ],
        )

    # --- Phase 3.5 : détection d'outliers post-GPA (par espèce) -----------
    if not skip("3.5"):
        print("\n=== Phase 3.5 : détection d'outliers post-GPA (par espèce) ===")
        run_step(
            "flag_outlier_specimens",
            src / "tools" / "flag_outlier_specimens.py",
            "tools.flag_outlier_specimens",
            [
                str(config.NUMBERED_TPS),
                str(config.SPECIMENS_CSV),
                "--manifest-dir", str(config.MANIFEST_DIR),
            ],
        )

    # --- Phase 4 : classification GPA -> PCA -> LDA (LOOCV) ----------------
    if not skip("4"):
        print("\n=== Phase 4 : classification GPA -> PCA -> LDA (LOOCV) ===")
        model_out = config.model_out_path(args.level)
        model_out.parent.mkdir(parents=True, exist_ok=True)
        # --exclude-ids ne prend QUE outlier_specimens.csv pour l'instant (voir
        # remarque Phase 3 : landmarks_numbered.csv sur-marque en SUSPECT tant
        # que le seuil d'ambiguïté n'est pas recalibré).
        run_step(
            "lda",
            src / "classifiers" / "lda.py",
            "classifiers.lda",
            [
                str(config.NUMBERED_TPS),
                str(config.SPECIMENS_CSV),
                "--level", args.level,
                "--exclude-ids", str(config.OUTLIER_CSV),
                "--save-model", str(model_out),
            ],
        )
        print(f"\n=== Terminé. Modèle -> {model_out} ===")
    else:
        print("\n=== Terminé (Phase 4 sautée). ===")


if __name__ == "__main__":
    main()