"""run_io.py
Convention de sortie partagée par train.py, predict.py,
analysis/classification_report.py et analysis/variance_report.py :

    data/models/<family>/<run_id>/<step>/...

`family` distingue ce pipeline (classification GPA->PCA->LDA, "lda") des
autres familles du projet (détection OBB, landmarks UNet, ...). `run_id`
identifie une combinaison de données (niveau, split, filtres, source de
landmarks). `step` identifie l'étape (train, predict, classification_report,
variance).

Chaque step écrit params.json (arguments CLI, pour reproductibilité et
rechargement en aval) et run.log (résumé), plus selon le cas metrics.json /
model.joblib / *.csv / *.png.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

FAMILY_LDA = "lda"
MODELS_ROOT = Path("data/models")


def slugify(text: str) -> str:
    """Nom de fichier/dossier sûr : alphanumérique + '-', '_' seulement."""
    text = re.sub(r"[^A-Za-z0-9_-]+", "-", text.strip())
    return re.sub(r"-{2,}", "-", text).strip("-_") or "run"


def tag_from_tps(tps_path: str | Path | None) -> str | None:
    """Étiquette courte dérivée du nom de fichier d'un --tps custom, ex:
    'tancrede_reference_19lm.tps' -> 'tancrede_reference_19lm'. None si pas de --tps."""
    if tps_path is None:
        return None
    return slugify(Path(tps_path).stem)


def build_run_id(
    level: str, split: str, devices: list[str] | None = None,
    landmarks_tps: str | Path | None = None, run_label: str | None = None,
) -> str:
    """Identifiant de run déterministe : level_split[_devices][_source].
    `run_label` prime sur le nom de fichier tps s'il est fourni."""
    parts = [slugify(level), slugify(split)]
    if devices:
        parts.append(slugify("-".join(devices)))
    source = run_label or tag_from_tps(landmarks_tps)
    if source:
        parts.append(slugify(source))
    return "_".join(parts)


def step_dir(run_id: str, step: str, family: str = FAMILY_LDA, root: Path = MODELS_ROOT) -> Path:
    d = root / family / run_id / step
    d.mkdir(parents=True, exist_ok=True)
    return d


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def write_params(out_dir: Path, args: argparse.Namespace, extra: dict | None = None) -> Path:
    payload = {k: v for k, v in vars(args).items() if k != "func"}
    if extra:
        payload.update(extra)
    path = out_dir / "params.json"
    path.write_text(json.dumps(payload, indent=2, default=_json_default, ensure_ascii=False), encoding="utf-8")
    return path


def write_metrics(out_dir: Path, metrics: dict) -> Path:
    path = out_dir / "metrics.json"
    path.write_text(json.dumps(metrics, indent=2, default=_json_default, ensure_ascii=False), encoding="utf-8")
    return path


def read_params(step_dir_path: Path) -> dict:
    return json.loads((step_dir_path / "params.json").read_text(encoding="utf-8"))


def read_metrics(step_dir_path: Path) -> dict:
    return json.loads((step_dir_path / "metrics.json").read_text(encoding="utf-8"))


def write_run_log(out_dir: Path, text: str) -> Path:
    path = out_dir / "run.log"
    path.write_text(text, encoding="utf-8")
    return path


def setup_console_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")
