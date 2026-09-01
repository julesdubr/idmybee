"""run_io.py
Output convention shared by train.py, predict.py,
analysis/classification_report.py and analysis/variance_report.py:

    data/models/<family>/<run_id>/train/                              model fit (train.py)
    data/models/<family>/<run_id>/predict/<eval_tag>/                 evaluation of that model on other data (predict.py batch)
    data/models/<family>/<run_id>/classification_report/train/            figures/tables for the train run
    data/models/<family>/<run_id>/classification_report/predict/<eval_tag>/ figures/tables for an evaluation
    data/analysis/variance/<variance_id>/                             shape variance analysis (independent of any model)

`run_id` identifies a trained model (level, split, devices, landmarks
source -- see build_run_id), one per call to train.py. `eval_tag`
identifies one evaluation of that model by predict.py (see build_eval_tag);
a single run_id can have several eval_tag (test, field data, another
landmarks source...). predict.py recovers the run_id from the given
model.joblib (see run_id_from_model_path) rather than recomputing one.

`variance_id` (build_variance_id) is independent of any run_id:
analysis/variance_report.py neither loads nor fits a model -- its output
lives under data/analysis/, not data/models/, precisely for that reason.

Every output folder has params.json (CLI arguments) and run.log (summary),
plus metrics.json / model.joblib / *.csv / *.png depending on the case.
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
    """Safe file/folder name: alphanumeric plus '-', '_' only."""
    text = re.sub(r"[^A-Za-z0-9_-]+", "-", text.strip())
    return re.sub(r"-{2,}", "-", text).strip("-_") or "run"


def tag_from_tps(tps_path: str | Path | None) -> str | None:
    """Short label derived from a custom --tps file's name. None if no --tps."""
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
    """Identifier for a trained model: level_split[_devices][_source]."""
    return "_".join([slugify(level)] + _tag_parts(split, devices, landmarks_tps, run_label))


def build_eval_tag(
    split: str, devices: list[str] | None = None,
    landmarks_tps: str | Path | None = None, run_label: str | None = None,
) -> str:
    """Identifier for a predict.py batch evaluation: split[_devices][_source],
    nested under the run_id of the model being evaluated (see run_id_from_model_path)."""
    return "_".join(_tag_parts(split, devices, landmarks_tps, run_label))


def build_variance_id(
    levels: list[str], split: str, devices: list[str] | None = None,
    landmarks_tps: str | Path | None = None, run_label: str | None = None,
) -> str:
    """Identifier for a variance analysis: levels_split[_devices][_source]."""
    return "_".join([slugify("-".join(levels))] + _tag_parts(split, devices, landmarks_tps, run_label))


def run_id_from_model_path(model_path: str | Path) -> tuple[str, str]:
    """Recover (family, run_id) from data/models/<family>/<run_id>/train/model.joblib."""
    model_path = Path(model_path).resolve()
    if model_path.name != "model.joblib" or model_path.parent.name != "train":
        raise ValueError(
            f"{model_path} does not follow the data/models/<family>/<run_id>/train/model.joblib "
            "convention -- cannot infer its run_id."
        )
    return model_path.parent.parent.parent.name, model_path.parent.parent.name


def run_path(family: str, *parts: str, root: Path = MODELS_ROOT) -> Path:
    """Build and create an output folder, e.g. run_path("lda", run_id, "train")."""
    d = root.joinpath(family, *parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


def result_path(family: str, *parts: str, root: Path = MODELS_ROOT) -> Path:
    """Like run_path, but read-only (creates nothing) -- to locate an already-written output."""
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
