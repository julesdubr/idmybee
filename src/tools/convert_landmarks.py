"""
Nettoyage et conversion des landmarks.

Entrées
-------
manifest.csv
landmarks_numbered.tps
landmarks_numbered.csv
specimens.csv

manifest.csv fournit notamment
    image_id
    specimen_id
    split
    device_type
    shot_index
    raw_path

landmarks_numbered.csv fournit
    image_id
    status

specimens.csv fournit
    specimen_id
    species
    caste

Sorties
-------
landmarks_<split>.tps
biological_data_<split>.csv

Filtrage
--------
--split train
    uniquement les images de collection

--split test
    uniquement les images de terrain

--split all
    collection + terrain

Dans tous les cas
-----------------
- seuls les landmarks avec status == OK sont conservés
- les IDs originaux sont remplacés par un ID numérique séquentiel
- le même ID est utilisé dans le TPS et le CSV
- COMMENT= est supprimé du TPS
- l'ordre du TPS et du CSV est strictement identique
"""

from pathlib import Path
import argparse
import re

import pandas as pd


# =====================================================================
# Arguments
# =====================================================================

parser = argparse.ArgumentParser(
    description="Nettoyage et conversion des landmarks."
)

parser.add_argument(
    "--split",
    choices=["train", "test", "all"],
    default="all",
    help="Split à conserver : train, test ou all.",
)

parser.add_argument(
    "--input-dir",
    type=Path,
    required=True,
    help="Dossier contenant les fichiers d'entrée.",
)

parser.add_argument(
    "--output-dir",
    type=Path,
    required=True,
    help="Dossier dans lequel écrire les fichiers de sortie.",
)

args = parser.parse_args()

INPUT_DIR = args.input_dir
OUTPUT_DIR = args.output_dir
SPLIT = args.split

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================================
# Fichiers d'entrée
# =====================================================================

MANIFEST_CSV  = INPUT_DIR / "manifest.csv"
SPECIMENS_CSV = INPUT_DIR / "specimens.csv"
LANDMARKS_CSV = INPUT_DIR / "landmarks" / "landmarks_numbered.csv"
LANDMARKS_TPS = INPUT_DIR / "landmarks" / "landmarks_numbered_18.tps"


for path in [
    MANIFEST_CSV,
    LANDMARKS_CSV,
    LANDMARKS_TPS,
    SPECIMENS_CSV,
]:
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}"
        )


# =====================================================================
# Fichiers de sortie
# =====================================================================

TPS_OUT = OUTPUT_DIR / f"landmarks_{SPLIT}_18.tps"
BIO_OUT = OUTPUT_DIR / f"biological_data_{SPLIT}.csv"

# =====================================================================
# Lecture des CSV
# =====================================================================

manifest = pd.read_csv(
    MANIFEST_CSV,
    dtype=str,
)

landmarks = pd.read_csv(
    LANDMARKS_CSV,
    dtype=str,
)

specimens = pd.read_csv(
    SPECIMENS_CSV,
    dtype=str,
)


# =====================================================================
# Vérification des colonnes
# =====================================================================

def check_columns(df, required, filename):

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Colonnes manquantes dans {filename} : {missing}"
        )


check_columns(
    manifest,
    [
        "image_id",
        "specimen_id",
        "split",
        "device_type",
        "shot_index",
    ],
    MANIFEST_CSV.name,
)

check_columns(
    landmarks,
    [
        "image_id",
        "status",
    ],
    LANDMARKS_CSV.name,
)

check_columns(
    specimens,
    [
        "specimen_id",
        "species",
        "caste",
    ],
    SPECIMENS_CSV.name,
)


# =====================================================================
# Préparation du manifest
# =====================================================================

# Une image_id doit correspondre à une seule ligne du manifest.
if manifest["image_id"].duplicated().any():

    duplicates = manifest[
        manifest["image_id"].duplicated(
            keep=False
        )
    ]

    raise ValueError(
        "manifest.csv contient plusieurs lignes "
        "pour certains image_id.\n"
        f"Exemples :\n{duplicates.head(10)}"
    )


manifest = manifest.set_index(
    "image_id"
)


# =====================================================================
# Filtrage des landmarks
# =====================================================================

# On garde uniquement les landmarks dont le statut est OK.
ok = landmarks[
    landmarks["status"]
    .fillna("")
    .str.upper()
    .eq("OK")
].copy()


# =====================================================================
# Association avec manifest.csv
# =====================================================================

# On récupère uniquement les informations nécessaires.
#
# Important :
# on ne fait PAS de merge afin d'éviter les collisions de colonnes
# comme specimen_id_x / specimen_id_y.

manifest_columns = [
    "specimen_id",
    "split",
    "device_type",
    "shot_index",
    "raw_path",
]

missing_image_ids = (
    ok.loc[
        ~ok["image_id"].isin(manifest.index),
        "image_id",
    ]
    .tolist()
)

if missing_image_ids:

    print(
        f"WARNING : {len(missing_image_ids)} "
        "images OK absentes de manifest.csv."
    )

    ok = ok[
        ok["image_id"].isin(manifest.index)
    ].copy()


# Ajouter les informations du manifest en conservant
# exactement l'ordre actuel de landmarks_numbered.csv.
manifest_data = manifest.loc[
    ok["image_id"],
    manifest_columns,
].copy()

manifest_data.index = ok.index

for column in manifest_columns:

    ok[column] = manifest_data[column]


# =====================================================================
# Filtrage du split
# =====================================================================

if SPLIT != "all":

    ok = ok[
        ok["split"]
        .fillna("")
        .str.lower()
        .eq(SPLIT)
    ].copy()


# =====================================================================
# Préparation des informations biologiques
# =====================================================================

bio = specimens[
    [
        "specimen_id",
        "species",
        "caste",
    ]
].copy()


# Un specimen_id doit idéalement être unique.
if bio["specimen_id"].duplicated().any():

    duplicates = bio[
        bio["specimen_id"].duplicated(
            keep=False
        )
    ]

    print(
        "WARNING : plusieurs lignes pour certains "
        "specimen_id dans specimens.csv."
    )

    print(
        duplicates.head(10)
    )


bio = bio.drop_duplicates(
    subset="specimen_id",
    keep="first",
)

bio = bio.set_index(
    "specimen_id"
)


# =====================================================================
# Lecture du TPS
# =====================================================================

tps_text = LANDMARKS_TPS.read_text(
    encoding="utf-8"
)

blocks = re.split(
    r"(?=^LM=\d+\s*$)",
    tps_text,
    flags=re.MULTILINE,
)


tps_by_image = {}


for block in blocks:

    block = block.strip()

    if not block:
        continue

    lines = [
        line.strip()
        for line in block.splitlines()
        if line.strip()
    ]

    if not lines:
        continue

    lm_match = re.fullmatch(
        r"LM=(\d+)",
        lines[0],
    )

    if not lm_match:
        continue

    landmark_count = int(
        lm_match.group(1)
    )

    coords = []
    meta = {}

    for line in lines[1:]:

        if "=" in line:

            key, value = line.split(
                "=",
                1,
            )

            meta[key.strip()] = value.strip()

        else:

            parts = line.split()

            if len(parts) >= 2:

                coords.append(
                    (
                        parts[0],
                        parts[1],
                    )
                )


    # ---------------------------------------------------------------
    # Récupération de image_id
    # ---------------------------------------------------------------

    image_id = None

    # Dans le TPS actuel, image_id est normalement dans COMMENT.
    comment = meta.get(
        "COMMENT",
        "",
    )

    match = re.search(
        r"(?:^|;)image_id=([^;]+)",
        comment,
    )

    if match:

        image_id = match.group(1)

    elif "IMAGE_ID" in meta:

        image_id = meta["IMAGE_ID"]


    if image_id is None:
        continue


    tps_by_image[image_id] = {
        "landmark_count": landmark_count,
        "coords": coords,
        "image": meta.get(
            "IMAGE",
            "",
        ),
    }


# =====================================================================
# Conversion
# =====================================================================

tps_output = []
csv_rows = []

missing_tps = []
missing_bio = []

new_id = 1


for _, row in ok.iterrows():

    image_id = row["image_id"]


    # ---------------------------------------------------------------
    # Correspondance TPS
    # ---------------------------------------------------------------

    if image_id not in tps_by_image:

        missing_tps.append(
            image_id
        )

        continue


    tps_record = tps_by_image[
        image_id
    ]


    # ---------------------------------------------------------------
    # ID original du specimen
    # ---------------------------------------------------------------

    original_specimen_id = (
        row["specimen_id"]
    )


    # ---------------------------------------------------------------
    # Informations biologiques
    # ---------------------------------------------------------------

    if original_specimen_id in bio.index:

        species = bio.loc[
            original_specimen_id,
            "species",
        ]

        caste = bio.loc[
            original_specimen_id,
            "caste",
        ]

        if pd.isna(species):
            species = ""

        if pd.isna(caste):
            caste = ""

    else:

        species = ""
        caste = ""

        missing_bio.append(
            original_specimen_id
        )


    # ---------------------------------------------------------------
    # Device
    # ---------------------------------------------------------------

    device = row["device_type"]

    if pd.isna(device):
        device = ""

    device = str(
        device
    ).strip().upper()


    # ---------------------------------------------------------------
    # Shot
    # ---------------------------------------------------------------

    shot = row["shot_index"]

    if pd.isna(shot):
        shot = ""

    shot = str(
        shot
    ).strip()


    # ---------------------------------------------------------------
    # Tag
    # ---------------------------------------------------------------

    if device and shot:

        tag = f"{device}{shot}"

    else:

        tag = ""


    # ---------------------------------------------------------------
    # Nouvel ID
    # ---------------------------------------------------------------

    specimen_id = str(
        new_id
    )


    # ---------------------------------------------------------------
    # TPS
    # ---------------------------------------------------------------

    tps_output.append(
        f"LM={len(tps_record['coords'])}"
    )


    for x, y in tps_record["coords"]:

        tps_output.append(
            f"{float(x):.4f} {float(y):.4f}"
        )


    # IMAGE conservé si présent.
    #
    # COMMENT n'est jamais réécrit.

    if tps_record["image"]:

        tps_output.append(
            f"IMAGE={tps_record['image']}"
        )


    # ID = nouvel identifiant numérique.
    tps_output.append(
        f"ID={specimen_id}"
    )

    # tps_output.append("")


    # ---------------------------------------------------------------
    # CSV biologique
    # ---------------------------------------------------------------

    csv_rows.append({

        "specimen": specimen_id,

        "espece": species,

        "caste": caste,

        "tag": tag,

        "device": device,

        "shot": shot,

    })


    new_id += 1


# =====================================================================
# Écriture du TPS
# =====================================================================

TPS_OUT.write_text(
    "\n".join(tps_output),
    encoding="utf-8",
)


# =====================================================================
# Écriture du CSV
# =====================================================================

bio_output = pd.DataFrame(
    csv_rows,
    columns=[
        "specimen",
        "espece",
        "caste",
        "tag",
        "device",
        "shot",
    ],
)

# bio_output.to_csv(
#     BIO_OUT,
#     index=False,
#     encoding="utf-8",
# )


# ---------------------------------------------------------------------
# Résumé
# ---------------------------------------------------------------------

print()
print("=" * 60)
print("Conversion terminée")
print("=" * 60)
print(f"Split                  : {SPLIT}")
print(f"Images OK retenues     : {len(csv_rows)}")
print(f"Images sans TPS        : {len(missing_tps)}")
print(f"Spécimens sans bio     : {len(set(missing_bio))}")
print()
print(f"TPS                    : {TPS_OUT}")
print(f"CSV                    : {BIO_OUT}")
print("=" * 60)