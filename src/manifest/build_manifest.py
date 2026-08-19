"""
build_manifest.py — Phase 0 : indexation en lecture seule des images brutes.

Ne deplace, ne renomme, ne modifie AUCUN fichier. Se contente de lire
l'arborescence + le CSV d'identification et d'ecrire des tables CSV
consolidees dans --out-dir (par defaut data/manifest/) :

  images.csv       une ligne par photo brute trouvee
  specimens.csv     une ligne par numero d'inventaire (jointe au CSV d'identif)
  unparsed.csv       fichiers dont le nom ne correspond a aucun schema connu
  duplicates.csv    groupes de fichiers ayant un contenu identique (meme hash)

Usage typique :

    python build_manifest.py \
        --roots roots.json \
        --external-roots external_roots.json \
        --species-csv data/csv/identification.csv \
        --id-col num_inv --species-col species --caste-col caste \
        --out-dir data/manifest

roots.json (organized/terrain/vrac locaux) et external_roots.json (disque
externe, optionnel — le script ignore proprement les racines absentes,
p.ex. si le disque n'est pas monte) suivent le meme format :

    [
      {"path": "data/images/wide/organized", "dataset": "organized", "collector_subfolder": false},
      {"path": "data/images/wide/terrain",   "dataset": "terrain",   "collector_subfolder": true},
      {"path": "data/images/wide/vrac",      "dataset": "vrac",      "collector_subfolder": false}
    ]

collector_subfolder=true veut dire : le sous-dossier immediat sous `path`
est le nom du collecteur (terrain/adrien/..., terrain/basile/...), et le
nom de fichier suit le schema terrain (num_inv_<n>, pas de lettre S/P).
collector_subfolder=false veut dire schema organized/vrac (num_inv_[S|P]<n>).

Convention de nommage (confirmee le 22/07/2026) :
  organized / vrac : num_inv_[S|P]<n>   -> meme appareil (S=smartphone,
                      P=appareil photo), <n> = n-ieme photo prise avec CET
                      appareil pour ce specimen (PAS un identifiant d'appareil).
  terrain           : num_inv_<n>       -> pas de lettre. Le smartphone est
                      constant pour un num_inv donne mais peut varier d'un
                      num_inv a l'autre (un collecteur = potentiellement un
                      smartphone different). Le collecteur est deduit du
                      sous-dossier, pas du nom de fichier.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".tif", ".tiff", ".bmp"}

# suffixe organized/vrac : ex. "S1", "P12"
_ORGANIZED_SUFFIX = re.compile(r"^([SP])(\d+)$", re.IGNORECASE)
# suffixe terrain : ex. "3"
_TERRAIN_SUFFIX = re.compile(r"^(\d+)$")


def _parse_organized_underscore(stem: str):
    """num_inv_[S|P]<n>  (lettre+numero colles, separateur = underscore)"""
    if "_" not in stem:
        return None
    specimen_id, suffix = stem.rsplit("_", 1)
    if not specimen_id:
        return None
    m = _ORGANIZED_SUFFIX.match(suffix)
    if not m:
        return None
    return specimen_id, m.group(1).upper(), int(m.group(2))


def _parse_terrain_underscore(stem: str):
    """num_inv_<n>  (pas de lettre d'appareil, separateur = underscore)"""
    if "_" not in stem:
        return None
    specimen_id, suffix = stem.rsplit("_", 1)
    if not specimen_id:
        return None
    m = _TERRAIN_SUFFIX.match(suffix)
    if not m:
        return None
    return specimen_id, None, int(m.group(1))


def _parse_organized_hyphen(stem: str):
    """num_inv-[S|P]-<n>  (lettre et numero separes par un tiret, PAS colles
    comme dans le schema underscore -> convention vue sur le disque externe)."""
    parts = stem.split("-")
    if len(parts) < 3:
        return None
    *specimen_parts, device, shot = parts
    if device.upper() not in ("S", "P") or not shot.isdigit():
        return None
    specimen_id = "-".join(specimen_parts)
    if not specimen_id:
        return None
    return specimen_id, device.upper(), int(shot)


# registre des conventions de nommage connues -> ajouter une nouvelle
# convention = une fonction de plus ici, rien d'autre a toucher.
NAMING_PARSERS = {
    "organized_underscore": _parse_organized_underscore,
    "terrain_underscore": _parse_terrain_underscore,
    "organized_hyphen": _parse_organized_hyphen,
}


def resolve_naming(root_cfg: dict) -> str:
    """Convention de nommage a utiliser pour cette racine : explicite via
    root_cfg['naming'] si presente, sinon deduite de collector_subfolder
    pour rester compatible avec les roots.json ecrits avant l'ajout du
    schema a tirets."""
    if "naming" in root_cfg:
        return root_cfg["naming"]
    return "terrain_underscore" if root_cfg.get("collector_subfolder") else "organized_underscore"


@dataclass
class ImageRecord:
    image_id: str  # = content_hash tronque : stable meme si le fichier est deplace/copie
    specimen_id: Optional[str]
    dataset: str
    collector: Optional[str]
    device_type: Optional[str]   # "S" / "P" / None (terrain -> None, voir collector)
    shot_index: Optional[int]
    raw_path: str
    source_root: str
    naming: str
    ext: str
    file_size_bytes: int
    content_hash: str
    status_ingest: str  # parsed_ok | unparsed_name | unreadable


def compute_hash(path: Path, chunk_size: int = 1 << 20) -> str:
    """Hash du CONTENU brut (pas de decodage image) -> marche aussi sur HEIC,
    et sert de cle de dedup si la meme photo existe en double (disque externe
    + copie locale, par ex.)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()[:16]


def parse_filename(stem: str, naming: str):
    """Retourne (specimen_id, device_type, shot_index, ok) en appliquant la
    convention de nommage `naming` (voir NAMING_PARSERS). On ne suppose rien
    sur le format du numero d'inventaire lui-meme (peut contenir des
    underscores/tirets) : chaque parseur ne consomme que le suffixe qui lui
    est propre et laisse le reste comme specimen_id."""
    parser = NAMING_PARSERS.get(naming)
    if parser is None:
        raise ValueError(
            f"convention de nommage inconnue: {naming!r} "
            f"(connues: {list(NAMING_PARSERS)})"
        )
    result = parser(stem)
    if result is None:
        return None, None, None, False
    specimen_id, device_type, shot_index = result
    return specimen_id, device_type, shot_index, True


def scan_root(root_cfg: dict, base_dir: Path) -> list[ImageRecord]:
    root_path = base_dir / Path(root_cfg["path"])
    dataset = root_cfg["dataset"]
    collector_subfolder = bool(root_cfg.get("collector_subfolder", False))
    naming = resolve_naming(root_cfg)

    if not root_path.exists():
        print(f"[avertissement] racine introuvable, ignoree : {root_path}", file=sys.stderr)
        return []

    records: list[ImageRecord] = []
    for path in root_path.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue

        collector = None
        if collector_subfolder:
            try:
                rel = path.relative_to(root_path)
                collector = rel.parts[0] if len(rel.parts) > 1 else None
            except ValueError:
                collector = None

        specimen_id, device_type, shot_index, ok = parse_filename(
            path.stem, naming=naming
        )

        try:
            content_hash = compute_hash(path)
            size = path.stat().st_size
            status = "parsed_ok" if ok else "unparsed_name"
        except OSError as e:
            content_hash = ""
            size = -1
            status = "unreadable"
            print(f"[avertissement] illisible : {path} ({e})", file=sys.stderr)

        image_id = content_hash if content_hash else f"unreadable:{path}"

        records.append(
            ImageRecord(
                image_id=image_id,
                specimen_id=specimen_id,
                dataset=dataset,
                collector=collector,
                device_type=device_type,
                shot_index=shot_index,
                raw_path=str(path),
                source_root=str(root_path),
                naming=naming,
                ext=path.suffix.lower(),
                file_size_bytes=size,
                content_hash=content_hash,
                status_ingest=status,
            )
        )
    return records


def load_roots_config(path: Optional[str]) -> list[dict]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        print(f"[avertissement] fichier de racines introuvable, ignore : {p}", file=sys.stderr)
        return []
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def build_images_table(all_records: list[ImageRecord]) -> tuple[list[dict], list[dict], list[dict]]:
    """Separe les enregistrements en (images_ok, unparsed, duplicate_groups)."""
    by_hash: dict[str, list[ImageRecord]] = defaultdict(list)
    for r in all_records:
        if r.content_hash:
            by_hash[r.content_hash].append(r)

    seen_hash: set[str] = set()
    images_rows: list[dict] = []
    unparsed_rows: list[dict] = []

    for r in all_records:
        row = asdict(r)
        if r.status_ingest != "parsed_ok":
            unparsed_rows.append(row)
            continue
        # marque les doublons de contenu (memes octets, chemins differents)
        # sans les exclure : on garde tout, on ajoute juste un indicateur.
        row["is_duplicate_content"] = r.content_hash in seen_hash
        seen_hash.add(r.content_hash)
        images_rows.append(row)

    duplicate_rows = []
    for content_hash, recs in by_hash.items():
        if len(recs) > 1:
            duplicate_rows.append(
                {
                    "content_hash": content_hash,
                    "n_copies": len(recs),
                    "paths": " | ".join(r.raw_path for r in recs),
                }
            )

    return images_rows, unparsed_rows, duplicate_rows


def build_specimens_table(
    images_rows: list[dict],
    species_csv: Optional[str],
    id_col: str,
    species_col: str,
    caste_col: Optional[str],
) -> list[dict]:
    # specimens vus dans les images (source de verite pour "quels num_inv existent")
    by_specimen: dict[str, dict] = defaultdict(lambda: {"n_images": 0, "datasets": set()})
    for row in images_rows:
        sid = row.get("specimen_id")
        if not sid:
            continue
        by_specimen[sid]["n_images"] += 1
        by_specimen[sid]["datasets"].add(row["dataset"])

    identif: dict[str, dict] = {}
    if species_csv:
        p = Path(species_csv)
        if not p.exists():
            print(f"[avertissement] CSV d'identification introuvable, ignore : {p}", file=sys.stderr)
        else:
            with open(p, encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                if id_col not in (reader.fieldnames or []):
                    print(
                        f"[avertissement] colonne id '{id_col}' absente de {p} "
                        f"(colonnes trouvees : {reader.fieldnames})",
                        file=sys.stderr,
                    )
                else:
                    for row in reader:
                        identif[row[id_col]] = row

    all_ids = set(by_specimen.keys()) | set(identif.keys())
    out = []
    for sid in sorted(all_ids):
        info = by_specimen.get(sid, {"n_images": 0, "datasets": set()})
        ident = identif.get(sid, {})
        species = ident.get(species_col) if species_col else None
        caste = ident.get(caste_col) if caste_col else None

        out.append(
            {
                "specimen_id": sid,
                "species": species or "",
                "caste": caste or "",
                "is_labeled": bool(species),
                "n_images": info["n_images"],
                "datasets_present": ",".join(sorted(info["datasets"])),
                "in_identification_csv": sid in identif,
                "in_images": sid in by_specimen,
            }
        )
    return out


def write_csv(rows: list[dict], out_path: Path, fieldnames: Optional[list[str]] = None):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        # ecrit quand meme un fichier vide avec en-tete si on la connait, pour
        # que les etapes suivantes n'aient pas a gerer un fichier absent.
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            if fieldnames:
                csv.writer(f).writerow(fieldnames)
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", required=True, help="JSON des racines locales (organized/terrain/vrac)")
    ap.add_argument("--species-csv", default=None, help="CSV d'identification espece/caste par num_inv")
    ap.add_argument("--id-col", default="num_inv")
    ap.add_argument("--species-col", default="species")
    ap.add_argument("--caste-col", default="caste")
    ap.add_argument("--out-dir", default="data/manifest")
    args = ap.parse_args()

    roots = load_roots_config(args.roots)
    if not roots:
        print("Aucune racine valide a scanner.", file=sys.stderr)
        sys.exit(1)

    base_dir = Path(roots["base_roots"][sys.platform])

    all_records: list[ImageRecord] = []
    for root_cfg in roots["datasets"]:
        recs = scan_root(root_cfg, base_dir)
        print(f"  {root_cfg['path']:60s} [{root_cfg['dataset']:10s}] -> {len(recs)} images")
        all_records.extend(recs)

    images_rows, unparsed_rows, duplicate_rows = build_images_table(all_records)
    specimens_rows = build_specimens_table(
        images_rows, args.species_csv, args.id_col, args.species_col, args.caste_col
    )

    out_dir = Path(args.out_dir)
    image_fields = [
        "image_id", "specimen_id", "dataset", "collector", "device_type", "shot_index",
        "raw_path", "source_root", "naming", "ext", "file_size_bytes", "content_hash",
        "status_ingest", "is_duplicate_content",
    ]
    write_csv(images_rows, out_dir / "images.csv", image_fields)
    write_csv(unparsed_rows, out_dir / "unparsed.csv", image_fields[:-1])
    write_csv(duplicate_rows, out_dir / "duplicates.csv", ["content_hash", "n_copies", "paths"])
    write_csv(
        specimens_rows,
        out_dir / "specimens.csv",
        ["specimen_id", "species", "caste", "is_labeled", "n_images",
         "datasets_present", "in_identification_csv", "in_images"],
    )

    n_unlabeled = sum(1 for r in specimens_rows if not r["is_labeled"])
    print("\n--- Resume ---")
    print(f"images.csv     : {len(images_rows)} lignes")
    print(f"unparsed.csv   : {len(unparsed_rows)} lignes (noms non reconnus -> a revoir a la main)")
    print(f"duplicates.csv : {len(duplicate_rows)} groupes de contenu identique")
    print(f"specimens.csv  : {len(specimens_rows)} specimens ({n_unlabeled} sans espece -> pool de prediction)")
    
    print(f"\nEcrit dans : {out_dir.resolve()}")


if __name__ == "__main__":
    main()