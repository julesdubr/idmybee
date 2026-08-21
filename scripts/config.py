"""
config.py -- Variables globales du pipeline idmybee.

Centralise tous les chemins et paramètres utilisés par run.py (et,
potentiellement, par les scripts individuels s'ils veulent l'importer
plutôt que redéfinir leurs propres défauts argparse).

Tous les chemins sont des objets pathlib.Path, résolus relativement à
la racine du repo (REPO_ROOT), donc indépendants de l'OS et du cwd
depuis lequel run.py est lancé.
"""

from pathlib import Path

# --- Racine du repo -----------------------------------------------------
# run.py est dans scripts/, donc la racine est son parent.
REPO_ROOT = Path(__file__).resolve().parent.parent

# --- Répertoires principaux ----------------------------------------------
CONFIG_DIR = REPO_ROOT / "config"
DATA_DIR = REPO_ROOT / "data"
OUT_DIR = REPO_ROOT / "out"

MANIFEST_DIR = DATA_DIR / "manifest"
REFERENCES_DIR = DATA_DIR / "references"
MODELS_DIR = DATA_DIR / "models"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
CSV_DIR = DATA_DIR / "csv"

# --- Modèles ---------------------------------------------------------------
YOLO_MODEL = MODELS_DIR / "yoloe-11s-seg.pt"
UNET_MODEL = (
    MODELS_DIR / "best_model"
    / "UNet_150_epoch_lr=0.001_seed=58_func=pow_param=30.pth"
)

# --- Fichiers d'entrée / sortie par phase -----------------------------------
ROOTS_JSON = CONFIG_DIR / "roots.json"
SPECIES_CSV = CSV_DIR / "identification.csv"

REFERENCE_TPS = ANNOTATIONS_DIR / "landmarks_refs.tps"    # référence Tancrède (19 lm)
UNET_TPS = ANNOTATIONS_DIR / "landmarks_unet.tps"         # sortie Phase 2 (non renuméré)

# Sortie Phase 3 (renuméré) -- reconstruct_tps.py dérive automatiquement les
# variantes _labeled/_unlabeled de NUMBERED_TPS (même dossier, même suffixe),
# donc un seul chemin de base à tenir à jour ici.
NUMBERED_TPS = ANNOTATIONS_DIR / "landmarks_numbered.tps"
NUMBERED_TPS_LABELED = ANNOTATIONS_DIR / "landmarks_numbered_labeled.tps"
NUMBERED_TPS_UNLABELED = ANNOTATIONS_DIR / "landmarks_numbered_unlabeled.tps"

# Statut OK/SUSPECT/FAILED par spécimen (numérotation + outliers post-GPA
# par espèce) -- une seule source désormais, utilisable tel quel en
# --exclude-ids de lda.py/analysis.report_variance.
NUMBERED_LOG = MANIFEST_DIR / "landmarks_numbered.csv"

SPECIMENS_CSV = MANIFEST_DIR / "specimens.csv"
IMAGES_CSV = MANIFEST_DIR / "images.csv"

# --- Paramètres ---------------------------------------------------------
LEVEL = "species"   # "species" ou "caste"
DROP_LANDMARK = 3   # LM3 de Tancrède, sans équivalent UNet

# Chemin du modèle sauvegardé, dépend de LEVEL -> calculé en fonction,
# pas en constante, pour rester cohérent si LEVEL est surchargé en CLI.
def model_out_path(level: str = LEVEL) -> Path:
    return OUT_DIR / f"model_{level}.joblib"


# Prédictions du pool non labellisé (Phase 4.6, predict.py) -- dépend de
# LEVEL comme le modèle, même raison.
def predictions_out_path(level: str = LEVEL) -> Path:
    return OUT_DIR / f"predictions_unlabeled_{level}.csv"


# Références utilisées par Phase 1 (extraction des crops).
# NB: l'ancien run.sh pointait vers ref_mac.json, spécifique à macOS.
# Si tu as / crées un équivalent multi-OS, mets à jour cette valeur
# (ou passe --ref explicitement à run.py).
REF_JSON = REFERENCES_DIR / "references.json"
REF_CROPS_DIR = REFERENCES_DIR / "crops"

CROPS_OUTPUT_ROOT = DATA_DIR / "crops"