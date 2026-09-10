"""run_io.py
Output convention shared by train.py, predict.py,
analysis/classification_report.py and analysis/variance_report.py:

    models/<family>/<run_id>/train/                              model fit (train.py)
    models/<family>/<run_id>/predict/<eval_tag>/                 evaluation of that model on other data (predict.py batch)
    models/<family>/<run_id>/classification_report/train/            figures/tables for the train run
    models/<family>/<run_id>/classification_report/predict/<eval_tag>/ figures/tables for an evaluation
    data/analysis/variance/<variance_id>/                             shape variance analysis (independent of any model)

`run_id` identifies a trained model (level, dataset_label, devices,
landmarks source -- see build_run_id), one per call to train.py.
`dataset_label` is the dataset root's own name (e.g. `Path(dataset).name`,
"collection"/"terrain") -- there is no train/test split within one dataset
anymore, each dataset root is either used to fit a model or to evaluate
one. `eval_tag` identifies one evaluation of that model by predict.py (see
build_eval_tag); a single run_id can have several eval_tag (one per
dataset evaluated, e.g. "terrain", plus devices/landmarks source).
predict.py recovers the run_id from the given model.joblib (see
run_id_from_model_path) rather than recomputing one.

`variance_id` (build_variance_id) is independent of any run_id:
analysis/variance_report.py neither loads nor fits a model -- its output
lives under data/analysis/, not models/, precisely for that reason.

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
MODELS_ROOT = Path("models")
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
    devices: list[str] | None, landmarks_tps: str | Path | None, run_label: str | None,
) -> list[str]:
    parts = []
    if devices:
        parts.append(slugify("-".join(devices)))
    source = run_label or tag_from_tps(landmarks_tps)
    if source:
        parts.append(slugify(source))
    return parts


def build_run_id(
    level: str, dataset_label: str, devices: list[str] | None = None,
    landmarks_tps: str | Path | None = None, run_label: str | None = None,
) -> str:
    """Identifier for a trained model: level_dataset_label[_devices][_source]."""
    return "_".join([slugify(level), slugify(dataset_label)] + _tag_parts(devices, landmarks_tps, run_label))


def build_eval_tag(
    dataset_label: str, devices: list[str] | None = None,
    landmarks_tps: str | Path | None = None, run_label: str | None = None,
) -> str:
    """Identifier for a predict.py batch evaluation: dataset_label[_devices][_source],
    nested under the run_id of the model being evaluated (see run_id_from_model_path)."""
    return "_".join([slugify(dataset_label)] + _tag_parts(devices, landmarks_tps, run_label))


def build_variance_id(
    levels: list[str], dataset_label: str, devices: list[str] | None = None,
    landmarks_tps: str | Path | None = None, run_label: str | None = None,
) -> str:
    """Identifier for a variance analysis: levels_dataset_label[_devices][_source]."""
    return "_".join(
        [slugify("-".join(levels)), slugify(dataset_label)] + _tag_parts(devices, landmarks_tps, run_label)
    )


def resolve_model_slug(family: str, base_name: str, root: Path = MODELS_ROOT) -> str:
    """Folder name for a new training run under root/<family>/, keyed by a
    human-facing `base_name` (--model-name, or its deterministic fallback --
    see classifiers.train) instead of a recomputed run_id: every run gets
    its own, non-clobbering folder now, disambiguated by version rather
    than silently overwriting an earlier run with the same params.

    slugify(base_name) if root/<family>/<slug>/ doesn't exist yet.
    Otherwise, the next free `_v{i+1}` suffix: an existing unversioned
    folder (`<slug>/`) counts as v1, so the first collision becomes
    `<slug>_v2`, the next `<slug>_v3`, and so on."""
    slug = slugify(base_name)
    family_dir = root / family
    if not (family_dir / slug).exists():
        return slug

    version_re = re.compile(rf"^{re.escape(slug)}_v(\d+)$")
    versions = [1]  # the unversioned folder itself counts as v1
    if family_dir.exists():
        for entry in family_dir.iterdir():
            m = version_re.match(entry.name)
            if m:
                versions.append(int(m.group(1)))
    return f"{slug}_v{max(versions) + 1}"


def run_id_from_model_path(model_path: str | Path) -> tuple[str, str]:
    """Recover (family, run_id) from models/<family>/<run_id>/train/model.joblib."""
    model_path = Path(model_path).resolve()
    if model_path.name != "model.joblib" or model_path.parent.name != "train":
        raise ValueError(
            f"{model_path} does not follow the models/<family>/<run_id>/train/model.joblib "
            "convention -- cannot infer its run_id."
        )
    return model_path.parent.parent.parent.name, model_path.parent.parent.name


def model_display_name(model_path: str | Path) -> str:
    """Human-facing name for a model.joblib, for a picker (CLI or UI): the
    model_name recorded in its training run's metrics.json (see
    classifiers.train --model-name / core.model_io.TrainedModel.model_name)
    if present, otherwise the run_id itself -- never unpickles the model
    just to get a label. Falls back to the model's grandparent folder name
    if the path doesn't even follow the run_id convention (e.g. a
    model.joblib moved out of models/)."""
    model_path = Path(model_path)
    try:
        metrics = read_metrics(model_path.parent)
    except FileNotFoundError:
        metrics = {}
    name = metrics.get("model_name")
    if name:
        return name
    try:
        _family, run_id = run_id_from_model_path(model_path)
        return run_id
    except ValueError:
        return model_path.resolve().parent.parent.name


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
