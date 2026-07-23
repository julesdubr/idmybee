"""
Utilitaires I/O partagés pour les tables CSV consolidées du manifest
(images.csv, specimens.csv, crops.csv, futures landmarks.csv, ...).

Chaque étape du pipeline (extraction, numérotation, classification) lit et
écrit ses tables via ces mêmes fonctions, pour ne pas réimplémenter à chaque
fois la même logique de reprise/compatibilité -- et pour qu'un changement de
comportement (ex: comment on détecte un schéma incompatible) se fasse à un
seul endroit.
"""

import csv
from pathlib import Path


def parse_bool(value) -> bool:
    """Un booléen Python écrit par csv.DictWriter redevient la chaîne
    'True'/'False' à la lecture -- il faut le réinterpréter explicitement."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def read_table(path) -> list[dict]:
    """Lit une table CSV en liste de dicts. Table absente -> liste vide,
    pour qu'une étape avale en aval sans planter si l'étape amont n'a pas
    encore tourné (elle traitera juste 0 ligne, avec un message clair à
    la charge de l'appelant)."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def check_schema(path, expected_fields: list[str]):
    """Refuse de continuer si une table existante a un schéma de colonnes
    différent de celui attendu par le code actuel (ex: écrite par une
    version antérieure du script) -- mieux vaut s'arrêter net que corrompre
    la table en mode append avec des colonnes mal alignées."""
    path = Path(path)
    if not path.exists():
        return
    with open(path, newline="", encoding="utf-8") as f:
        header = next(csv.reader(f), [])
    if header and header != expected_fields:
        raise SystemExit(
            f"{path} a un schéma différent de celui attendu par le code actuel.\n"
            f"  attendu : {expected_fields}\n"
            f"  trouvé  : {header}\n"
            f"Renomme/déplace l'ancien fichier avant de relancer (rien n'a été modifié)."
        )


def load_existing_by_key(path, key_field: str) -> dict:
    """Relit une table existante (déjà vérifiée compatible via check_schema)
    indexée par `key_field`, pour reprendre un run sans dupliquer ni reperdre
    les lignes déjà calculées."""
    return {row[key_field]: row for row in read_table(path)}


def should_skip(prev_row: dict | None, overwrite: bool, retry_failed: bool) -> bool:
    """Décision de reprise générique, réutilisable par toute étape qui
    produit un résultat par image/specimen avec un statut OK/SUSPECT/FAILED
    et (parfois) un fichier de sortie.

    OK/SUSPECT : sautée si sa sortie existe encore (sinon on la reproduit --
    auto-guérison si le dossier de sortie a été partiellement supprimé) et
    que --overwrite n'est pas demandé.
    FAILED : sautée par défaut, sinon un run répété rejoue et reloggue les
    mêmes échecs indéfiniment ; --retry_failed pour les retenter explicitement.
    """
    if prev_row is None:
        return False
    status = prev_row.get("status")
    if status in ("OK", "SUSPECT"):
        out = prev_row.get("output_path") or ""
        return bool(out) and Path(out).exists() and not overwrite
    if status == "FAILED":
        return not retry_failed
    return False