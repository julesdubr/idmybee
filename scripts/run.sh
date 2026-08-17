#!/usr/bin/env bash
# run.sh -- Enchaîne les phases 0->4 du pipeline idmybee en un seul appel.
#
# A lancer DEPUIS LA RACINE DU REPO (idmybee/), environnement conda/venv déjà activé :
#   bash scripts/run.sh
#
# Chaque étape peut aussi être relancée seule (voir docstring de chaque script pour
# les options de reprise : --overwrite, --retry_failed pour predict_unet.py, etc.).
#
# ATTENTION : les commandes des Phases 0 et 1 (build_manifest.py, crop_wings.py) sont
# reconstruites à partir de leur docstring/description, PAS vérifiées ligne à ligne --
# revérifier les noms d'options exacts (marqués TODO ci-dessous) avant le premier run.
set -euo pipefail

# --- Configuration (à adapter) ------------------------------------------------
MANIFEST_DIR="data/manifest"
MODEL_PATH="data/models/best_model/UNet_150_epoch_lr=0.001_seed=58_func=pow_param=30.pth"
REFERENCE_TPS="data/annotations/tancrede.tps"        # TPS de référence (Tancrède, 19 landmarks)
UNET_TPS="data/annotations/landmarks_unet.tps"       # sortie Phase 2 (non renuméré)
NUMBERED_TPS="data/annotations/landmarks_numbered.tps"  # sortie Phase 3 (renuméré, prêt pour la GPA)
LEVEL="species"                                       # "species" ou "caste"
DROP_LANDMARK=3                                       # LM3 de Tancrède, sans équivalent UNet

echo "=== Phase 0 : indexation manifest ==="
# TODO: vérifier les options exactes (racines locales/externes via config/*.json ?)
python src/manifest/build_manifest.py \
    --manifest_dir "$MANIFEST_DIR"

echo "=== Phase 1 : extraction des crops ==="
# TODO: vérifier les options exactes (--dataset, --only_labeled mentionnés dans le
# résumé d'architecture mais jamais collés dans cette conversation)
python src/extraction/crop_wings.py \
    --manifest_dir "$MANIFEST_DIR"

echo "=== Phase 2 : prédiction des landmarks (UNet) ==="
python src/landmarks/predict_unet.py \
    --manifest_dir "$MANIFEST_DIR" \
    --model_path "$MODEL_PATH" \
    --tps_out "$UNET_TPS"

echo "=== Phase 3 : renumérotation (Hungarian + Umeyama) ==="
python src/numbering/reconstruct_tps.py \
    "$REFERENCE_TPS" "$UNET_TPS" "$NUMBERED_TPS" \
    --drop "$DROP_LANDMARK" \
    --manifest-dir "$MANIFEST_DIR"
    # --ambiguity-ratio : NON calibré, voir résumé de session -- ne pas se fier
    # aveuglément aux SUSPECT "orientation ambiguë" de landmarks_numbered.csv pour l'instant.

echo "=== Phase 3.5 : détection d'outliers post-GPA (par espèce) ==="
python src/tools/flag_outlier_specimens.py \
    "$NUMBERED_TPS" "$MANIFEST_DIR/specimens.csv" \
    --manifest-dir "$MANIFEST_DIR"

echo "=== Phase 4 : classification GPA -> PCA -> LDA (LOOCV) ==="
python src/classifiers/lda.py \
    "$NUMBERED_TPS" "$MANIFEST_DIR/specimens.csv" \
    --level "$LEVEL" \
    --exclude-ids "$MANIFEST_DIR/outlier_specimens.csv" \
    --save-model "out/model_${LEVEL}.joblib"
    # --exclude-ids ne prend QUE outlier_specimens.csv pour l'instant (voir remarque
    # Phase 3 : landmarks_numbered.csv sur-marque en SUSPECT tant que le seuil
    # d'ambiguïté n'est pas recalibré).

echo "=== Terminé. Modèle -> out/model_${LEVEL}.joblib ==="