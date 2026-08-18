#!/usr/bin/env bash
# run.sh -- Enchaîne les phases 0->4 du pipeline idmybee en un seul appel.
#
# A lancer DEPUIS LA RACINE DU REPO (idmybee/), environnement conda/venv déjà activé :
#   bash scripts/run.sh
#
# Chaque étape peut aussi être relancée seule (voir docstring de chaque script pour
# les options de reprise : --overwrite, --retry_failed pour predict_unet.py, etc.).
#
set -euo pipefail

# --- Configuration (à adapter) ------------------------------------------------
CONFIG_DIR="config"
DATA_DIR="data"

MANIFEST_DIR="$DATA_DIR/manifest"
REFERENCES_DIR="$DATA_DIR/references"
MODELS_DIR="$DATA_DIR/models"

YOLO_MODEL="$MODELS_DIR/yoloe-11s-seg.pt"
UNET_MODEL="$MODELS_DIR/best_model/UNet_150_epoch_lr=0.001_seed=58_func=pow_param=30.pth"

REFERENCE_TPS="$DATA_DIR/annotations/tancrede.tps"           # TPS de référence (Tancrède, 19 landmarks)
UNET_TPS="$DATA_DIR/annotations/landmarks_unet.tps"          # sortie Phase 2 (non renuméré)
NUMBERED_TPS="$DATA_DIR/annotations/landmarks_numbered.tps"  # sortie Phase 3 (renuméré, prêt pour la GPA)

LEVEL="species"  # "species" ou "caste"
DROP_LANDMARK=3  # LM3 de Tancrède, sans équivalent UNet

echo "\n=== Phase 0 : indexation manifest ==="
python3 src/manifest/build_manifest.py \
    --roots "$CONFIG_DIR/roots_mac.json" \
    --external-roots "$CONFIG_DIR/external_roots.json" \
    --species-csv "$DATA_DIR/csv/identification.csv"

echo "\n=== Phase 1 : extraction des crops ==="
python3 src/extraction/crop_wings.py \
    --ref "$REFERENCES_DIR/ref_mac.json" \
    --ref_crops "$REFERENCES_DIR/crops" \
    --model "$YOLO_MODEL" \
    --manifest_dir "$MANIFEST_DIR" \
    --output_root "$DATA_DIR/crops" \
    --dataset organized

echo "\n=== Phase 2 : prédiction des landmarks (UNet) ==="
python3 src/landmarks/predict_unet.py \
    --manifest_dir "$MANIFEST_DIR" \
    --model_path "$UNET_MODEL" \
    --tps_out "$UNET_TPS" \
    --dataset organized

echo "\n=== Phase 3 : renumérotation (Hungarian + Umeyama) ==="
python3 src/numbering/reconstruct_tps.py \
    "$REFERENCE_TPS" "$UNET_TPS" "$NUMBERED_TPS" \
    --drop "$DROP_LANDMARK" \
    --manifest-dir "$MANIFEST_DIR"
    # --ambiguity-ratio : NON calibré, voir résumé de session -- ne pas se fier
    # aveuglément aux SUSPECT "orientation ambiguë" de landmarks_numbered.csv pour l'instant.

echo "\n=== Phase 3.5 : détection d'outliers post-GPA (par espèce) ==="
python3 src/tools/flag_outlier_specimens.py \
    "$NUMBERED_TPS" "$MANIFEST_DIR/specimens.csv" \
    --manifest-dir "$MANIFEST_DIR"

echo "\n=== Phase 4 : classification GPA -> PCA -> LDA (LOOCV) ==="
python3 src/classifiers/lda.py \
    "$NUMBERED_TPS" "$MANIFEST_DIR/specimens.csv" \
    --level "$LEVEL" \
    --exclude-ids "$MANIFEST_DIR/outlier_specimens.csv" \
    --save-model "out/model_${LEVEL}.joblib"
    # --exclude-ids ne prend QUE outlier_specimens.csv pour l'instant (voir remarque
    # Phase 3 : landmarks_numbered.csv sur-marque en SUSPECT tant que le seuil
    # d'ambiguïté n'est pas recalibré).

echo "\n=== Terminé. Modèle -> out/model_${LEVEL}.joblib ==="