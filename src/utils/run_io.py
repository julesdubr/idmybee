"""run_io.py
Convention de sortie partagée par train.py, predict.py,
analysis/classification_report.py et analysis/variance_report.py :

    data/models/<family>/<run_id>/train/                        ajustement du modèle (train.py)
    data/models/<family>/<run_id>/predict/<eval_tag>/            évaluation de ce modèle sur d'autres données (predict.py batch)
    data/models/<family>/<run_id>/classification_report/train/           figures/tableaux du train
    data/models/<family>/<run_id>/classification_report/predict/<eval_tag>/   figures/tableaux d'une évaluation
    data/analysis/variance/<variance_id>/                        analyse de variance de forme (indépendante de tout modèle)

`run_id` identifie un modèle entraîné (niveau, split, devices, source de
landmarks -- voir build_run_id), un seul par appel à train.py. `eval_tag`
identifie une évaluation de ce modèle par predict.py (voir build_eval_tag) ;
un même run_id peut avoir plusieurs eval_tag (test, terrain, autre source de
landmarks...). predict.py retrouve le run_id à partir du model.joblib fourni
(voir run_id_from_model_path) plutôt que d'en recalculer un.

`variance_id` (build_variance_id) est indépendant de tout run_id :
analysis/variance_report.py ne charge ni n'ajuste de modèle -- son résultat
vit sous data/analysis/, pas data/models/, précisément pour ça.

Chaque dossier de sortie a params.json (arguments CLI) et run.log (résumé),
plus selon le cas metrics.json / model.joblib / *.csv / *.png.
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
ANALYSIS_ROOT = Path("data/analysis")


def slugify(text: str) -> str:
    """Nom de fichier/dossier sûr : alphanumérique + '-', '_' seulement."""
    text = re.sub(r"[^A-Za-z0-9_-]+", "-", text.strip())
    return re.sub(r"-{2,}", "-", text).strip("-_") or "run"


def tag_from_tps(tps_path: str | Path | None) -> str | None:
    """Étiquette courte dérivée du nom de fichier d'un --tps custom. None si pas de --tps."""
    if tps_path is None:
        return None
    return slugify(Path(tps_path).stem)


def _tag_parts(
    split: str, devices: list[str] | None, landmarks_tps: str | Path | None, run_label: str | None,
) -> list[str]:
    parts = [slugify(split)]
    if devices:
        parts.append(slugify("-".join(devices)))
    source = run_label or tag_from_tps(landmarks_tps)
    if source:
        parts.append(slugify(source))
    return parts


def build_run_id(
    level: str, split: str, devices: list[str] | None = None,
    landmarks_tps: str | Path | None = None, run_label: str | None = None,
) -> str:
    """Identifiant d'un modèle entraîné : level_split[_devices][_source]."""
    return "_".join([slugify(level)] + _tag_parts(split, devices, landmarks_tps, run_label))


def build_eval_tag(
    split: str, devices: list[str] | None = None,
    landmarks_tps: str | Path | None = None, run_label: str | None = None,
) -> str:
    """Identifiant d'une évaluation predict.py batch : split[_devices][_source],
    nichée sous le run_id du modèle évalué (voir run_id_from_model_path)."""
    return "_".join(_tag_parts(split, devices, landmarks_tps, run_label))


def build_variance_id(
    levels: list[str], split: str, devices: list[str] | None = None,
    landmarks_tps: str | Path | None = None, run_label: str | None = None,
) -> str:
    """Identifiant d'une analyse de variance : levels_split[_devices][_source]."""
    return "_".join([slugify("-".join(levels))] + _tag_parts(split, devices, landmarks_tps, run_label))


def run_id_from_model_path(model_path: str | Path) -> tuple[str, str]:
    """Retrouve (family, run_id) à partir de data/models/<family>/<run_id>/train/model.joblib."""
    model_path = Path(model_path).resolve()
    if model_path.name != "model.joblib" or model_path.parent.name != "train":
        raise ValueError(
            f"{model_path} ne suit pas la convention data/models/<family>/<run_id>/train/model.joblib "
            "-- impossible d'en déduire le run_id."
        )
    return model_path.parent.parent.parent.name, model_path.parent.parent.name


def run_path(family: str, *parts: str, root: Path = MODELS_ROOT) -> Path:
    """Construit et crée un dossier de sortie, ex: run_path("lda", run_id, "train")."""
    d = root.joinpath(family, *parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


def result_path(family: str, *parts: str, root: Path = MODELS_ROOT) -> Path:
    """Comme run_path, mais en lecture seule (ne crée rien) -- pour localiser une sortie déjà écrite."""
    return root.joinpath(family, *parts)


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
