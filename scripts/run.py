#!/usr/bin/env python3
"""
run.py -- Enchaîne les phases 0->4.6 du pipeline idmybee en un seul appel.
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
# reconstruct_tps/lda/predict/report_variance exposent déjà main(argv) --
# build_manifest/crop_wings/predict_unet (Phases 0-2) pas encore.
USE_SUBPROCESS = {
    "build_manifest": True,
    "crop_wings": True,
    "predict_unet": True,
    "reconstruct_tps": False,
    "lda": False,
    "predict": False,
    "report_variance": False,
}


def _run_subprocess(script_path: Path, args: list[str]) -> None:
    cmd = [sys.executable, str(script_path), *args]
    print(f"    $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _run_inprocess(module_path: str, args: list[str]) -> None:
    """Importe le module et appelle sa fonction main(argv) directement.
    src/ doit être sur sys.path -- fait une fois ici plutôt qu'en important
    config à un endroit inattendu, pour rester local à ce mode d'exécution."""
    import importlib

    src_dir = str(config.REPO_ROOT / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    mod = importlib.import_module(module_path)
    mod.main(args)


def run_step(name: str, script_path: Path, module_path: str, args: list[str]) -> None:
    if USE_SUBPROCESS.get(name, True):
        _run_subprocess(script_path, args)
    else:
        _run_inprocess(module_path, args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Enchaîne les phases 0->4.6 du pipeline idmybee.",
    )
    p.add_argument(
        "--level",
        default=config.LEVEL,
        choices=["species", "caste"],
        help="Niveau de classification pour les Phases 4/4.6 (défaut: %(default)s).",
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
        "--dataset",
        default=None,
        help="Restreint l'entraînement (Phase 4) et l'analyse de variance (Phase 4.5) à un jeu de "
             "données d'origine (ex: organized -- voir specimens.csv 'datasets_present'). Non "
             "appliqué par défaut (tous les jeux de données confondus).",
    )
    p.add_argument(
        "--n-perm",
        type=int,
        default=199,
        help="Nombre de permutations pour les tests de significativité de la Phase 4.5 (défaut: "
             "%(default)s -- augmenter pour un rapport final, voir analysis.report_variance --n-perm "
             "directement pour un run indépendant).",
    )
    p.add_argument(
        "--skip",
        nargs="+",
        default=[],
        choices=["0", "1", "2", "3", "4", "5", "6"],
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
            "extract_wings_yoloe-sam",
            src / "extraction" / "extract_wings_yoloe-sam.py",
            "extraction.extract_wings_yoloe-sam",
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
                # "--dataset", "organized"
            ],
        )

    # --- Phase 3 : renumérotation + split labellisé/non-labellisé + -------
    # --- flag SUSPECT post-GPA par espèce (remplace l'ancienne Phase 3.5) --
    if not skip("3"):
        print("\n=== Phase 3 : renumérotation, split labellisé/non-labellisé, outliers post-GPA ===")
        run_step(
            "reconstruct_tps",
            src / "numbering" / "reconstruct_tps.py",
            "numbering.reconstruct_tps",
            [
                str(config.REFERENCE_TPS),
                str(config.UNET_TPS),
                str(config.SPECIMENS_CSV),
                str(config.NUMBERED_TPS),
                "--drop", str(args.drop_landmark),
                "--manifest-dir", str(config.MANIFEST_DIR),
            ],
        )

    # --- Phase 4 : classification GPA -> PCA -> LDA (LOOCV) ----------------
    if not skip("4"):
        print("\n=== Phase 4 : classification GPA -> PCA -> LDA (LOOCV) ===")
        model_out = config.model_out_path(args.level)
        model_out.parent.mkdir(parents=True, exist_ok=True)
        lda_args = [
            str(config.NUMBERED_TPS_LABELED),
            str(config.SPECIMENS_CSV),
            "--images-csv", str(config.IMAGES_CSV),
            "--level", args.level,
            "--exclude-ids", str(config.NUMBERED_LOG),
            "--save-model", str(model_out),
        ]
        if args.dataset:
            lda_args += ["--dataset", args.dataset]
        run_step("lda", src / "classifiers" / "lda.py", "classifiers.lda", lda_args)

    # --- Phase 5 : analyse de variance (ANOVA espèce / appareil) --------
    if not skip("5"):
        print("\n=== Phase 5 : analyse de variance (ANOVA espèce / appareil) ===")
        variance_args = [
            str(config.NUMBERED_TPS_LABELED),
            str(config.SPECIMENS_CSV),
            str(config.IMAGES_CSV),
            "--exclude-ids", str(config.NUMBERED_LOG),
            "--n-perm", str(args.n_perm),
        ]
        if args.dataset:
            variance_args += ["--dataset", args.dataset]
        run_step("report_variance", src / "analysis" / "report_variance.py",
                  "analysis.report_variance", variance_args)

    # --- Phase 6 : classification du pool non labellisé ------------------
    if not skip("6"):
        model_out = config.model_out_path(args.level)
        if not model_out.exists():
            print(f"\n=== Phase 6 : SAUTÉE ({model_out} introuvable -- lancer la Phase 4 d'abord) ===")
        else:
            print("\n=== Phase 6 : classification du pool non labellisé ===")
            predictions_out = config.predictions_out_path(args.level)
            run_step(
                "predict",
                src / "classifiers" / "predict.py",
                "classifiers.predict",
                [
                    str(model_out),
                    str(config.NUMBERED_TPS_UNLABELED),
                    "--out", str(predictions_out),
                ],
            )

    print("\n=== Terminé. ===")


if __name__ == "__main__":
    main()