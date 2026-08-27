"""pipeline_io.py
Utilitaires génériques de suivi d'exécution, communs à TOUTES les étapes du
pipeline (détection, normalisation, landmarks, renumérotation) : compteur de
statuts, écriture incrémentale de CSV, résolution de chemin, et
`pipeline_stats.csv` -- un seul fichier à la racine du dataset qui résume
chaque (step, approach) pour comparer les étapes et les approches entre
elles (nombre de sorties par statut, temps moyen/total par image).

Rien ici n'est spécifique à une étape : les champs de CSV propres à une
étape (DETECTION_FIELDS, CROP_FIELDS, LANDMARKS_FIELDS, ...) restent définis
localement dans le module de cette étape (voir extraction/extraction_io.py,
landmarks/predict.py).
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

PIPELINE_STATS_FIELDS = [
    "step", "approach", "total", "ok", "suspect", "skipped", "failed",
    "mean_time_s", "total_time_s", "updated_at",
]


def read_csv_rows(path: Path) -> list[dict]:
    """Charge un CSV quelconque en liste de dicts."""
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def append_rows(path: Path, rows: list[dict], fields: list[str], write_header: bool) -> None:
    """Ajoute des lignes à un CSV, en écrivant l'entête si nécessaire.

    `write_header=True` uniquement lors du tout premier flush d'un run
    (écrase un éventuel fichier précédent) ; `False` pour les flush suivants.
    """
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if write_header else "a"
    with path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def resolve_path(raw_path: str, base_dir: Path | None = None) -> Path:
    """Résout un chemin absolu ou relatif à `base_dir`.

    Normalise aussi les séparateurs `\\` : un CSV produit sous Windows (ex:
    output_path de crops.csv) contient des chemins avec des antislashs,
    illisibles tels quels comme chemins relatifs sous Linux/macOS.
    """
    path = Path(str(raw_path).replace("\\", "/"))
    if path.is_absolute() or base_dir is None:
        return path
    return base_dir / path


def format_duration(seconds: float) -> str:
    """Formate une durée en `1h05m30.0s` / `5m12.3s` / `3.2s`."""
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{int(hours)}h{int(minutes):02d}m{secs:04.1f}s"
    if minutes:
        return f"{int(minutes)}m{secs:04.1f}s"
    return f"{secs:.1f}s"


class RunCounter:
    """Compte les statuts (OK/SUSPECT/SKIPPED/FAILED) au fil d'un run, pour
    affichage + pipeline_stats.csv.

    Générique et ouvert : un statut jamais ajouté reste à 0 plutôt que
    d'être rangé ailleurs par erreur (l'ancienne version, propre à
    l'extraction, comptait tout statut inconnu comme FAILED -- correct tant
    qu'il n'existait que OK/SKIPPED/FAILED, plus vrai depuis SUSPECT).
    """

    def __init__(self):
        self._counts: dict[str, int] = {}

    def add(self, status: str) -> None:
        self._counts[status] = self._counts.get(status, 0) + 1

    def get(self, status: str) -> int:
        return self._counts.get(status, 0)

    @property
    def total(self) -> int:
        return sum(self._counts.values())

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "ok": self.get("OK"),
            "suspect": self.get("SUSPECT"),
            "skipped": self.get("SKIPPED"),
            "failed": self.get("FAILED"),
        }

    def __str__(self) -> str:
        return " ".join(f"{k}={v}" for k, v in sorted(self._counts.items()))


def update_pipeline_stats(
    stats_path: Path, step: str, approach: str, counts: dict, total_time_s: float,
) -> None:
    """Met à jour pipeline_stats.csv : une ligne par (step, approach).

    `counts` vient de RunCounter.as_dict() (total/ok/suspect/skipped/failed).
    La ligne existante pour ce (step, approach) est remplacée ; les autres
    sont conservées -- permet de comparer plusieurs approches d'une même
    étape (ex: light vs heavy, ou une future méthode de numérotation) et
    plusieurs étapes entre elles, dans un seul fichier par dataset.
    """
    rows = read_csv_rows(stats_path) if stats_path.exists() else []
    rows = [r for r in rows if not (r.get("step") == step and r.get("approach") == approach)]

    total = counts["total"]
    mean_time_s = total_time_s / total if total else 0.0
    rows.append({
        "step": step,
        "approach": approach,
        "total": str(total),
        "ok": str(counts["ok"]),
        "suspect": str(counts["suspect"]),
        "skipped": str(counts["skipped"]),
        "failed": str(counts["failed"]),
        "mean_time_s": f"{mean_time_s:.4f}",
        "total_time_s": f"{total_time_s:.4f}",
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PIPELINE_STATS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
