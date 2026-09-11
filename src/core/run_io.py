"""run_io.py
Output convention shared by train.py, predict.py,
analysis/classification_report.py and analysis/variance_report.py.

`models/` holds only the deployable artifact, nothing else:

    models/<family>/<run_id>/model.joblib                        model fit (train.py)

`runs/` holds every performance record for that artifact -- how it did,
on which dataset, never the artifact itself -- so `models/` stays a small,
swappable store while `runs/` is the full ledger a comparison table (see
analysis/compare_runs.py) scans:

    runs/<family>/<run_id>/train/<dataset_name>/                  LOOCV performance of the fit (train.py)
    runs/<family>/<run_id>/predict/<eval_tag>/                    evaluation of that model on other data (predict.py batch)
    runs/<family>/<run_id>/train/<dataset_name>/report/           figures/tables for the train run
    runs/<family>/<run_id>/predict/<eval_tag>/report/             figures/tables for an evaluation
    data/analysis/variance/<variance_id>/                         shape variance analysis (independent of any model)
    data/analysis/compare/<label>/                                cross-run comparison table (independent of any single run)

`run_id` identifies a trained model (level, dataset_label, devices,
landmarks source -- see build_run_id), one per call to train.py, and is the
folder name shared by both trees (models/<family>/<run_id>/ and
runs/<family>/<run_id>/) -- the only link between an artifact and its
performance records. `dataset_label` is the dataset root's human-facing
name (`core.dataset_config.resolve_dataset_name(dataset)`, sanitized to be
folder-safe -- falls back to `Path(dataset).name` if the root has no
dataset_config.json yet) -- there is no train/test split within one
dataset anymore, each dataset root is either used to fit a model or to
evaluate one. train.py's `dataset_name` folder and predict.py's `eval_tag`
(build_eval_tag) both start from this same dataset_label -- eval_tag
additionally appends devices/landmarks-source disambiguation, since a
single run_id can be evaluated several times (different datasets, devices,
or landmark sources), while a run_id has exactly one train() call behind
it (resolve_model_slug versions the whole run_id on a name collision
instead). predict.py reads the training dataset_label straight off the
loaded model (`TrainedModel.dataset_label`) to locate that train run,
rather than searching for it.

`variance_id` (build_variance_id) is independent of any run_id:
analysis/variance_report.py neither loads nor fits a model -- its output
lives under data/analysis/, not models/ or runs/, precisely for that reason.

Every performance folder has params.json (CLI arguments) and run.log
(summary), plus metrics.json / *.csv / *.png depending on the case.
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
RUNS_ROOT = Path("runs")
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
    """Recover (family, run_id) from models/<family>/<run_id>/model.joblib."""
    model_path = Path(model_path).resolve()
    if model_path.name != "model.joblib":
        raise ValueError(
            f"{model_path} does not follow the models/<family>/<run_id>/model.joblib "
            "convention -- cannot infer its run_id."
        )
    return model_path.parent.parent.name, model_path.parent.name


def _train_metrics(family: str, run_id: str, root: Path = RUNS_ROOT) -> dict | None:
    """metrics.json for run_id's train run, nested one level deeper than it
    used to be (runs/<family>/<run_id>/train/<dataset_name>/, see module
    docstring) -- None if there isn't one yet."""
    matches = sorted((root / family / run_id / "train").glob("*/metrics.json"))
    if not matches:
        return None
    try:
        return json.loads(matches[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def find_model_path(model_name: str, family: str = FAMILY_LDA, root: Path = MODELS_ROOT) -> Path:
    """Resolves a human-facing --model-name (classifiers.train's own
    --model-name, or its deterministic fallback) to its model.joblib, for a
    caller (tools/pipeline/predict_dataset.py) that takes a name rather
    than a full path -- the reverse of model_display_name below.

    Tries slugify(model_name) as the run_id first: exact and cheap for
    every name classifiers.train itself produced, since the output folder
    IS slugify(--model-name) (or its own _v2/_v3/... suffix, itself part
    of the slugified name -- see resolve_model_slug/CONVENTIONS.md
    "--model-name"). Falls back to scanning every trained model's
    performance record (_train_metrics) for a literal model_name match, in
    case slugify() isn't idempotent on this particular name (e.g. two
    different names collapsing to the same slug).
    """
    slug_path = root / family / slugify(model_name) / "model.joblib"
    if slug_path.exists():
        return slug_path

    family_dir = root / family
    matches: list[Path] = []
    if RUNS_ROOT.joinpath(family).exists():
        for entry in sorted(RUNS_ROOT.joinpath(family).iterdir()):
            metrics = _train_metrics(family, entry.name, root=RUNS_ROOT)
            if metrics and metrics.get("model_name") == model_name:
                candidate = family_dir / entry.name / "model.joblib"
                if candidate.exists():
                    matches.append(candidate)

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"{model_name!r} matches several models under {family_dir}/: "
            f"{[str(m) for m in matches]} -- pass the exact run_id (folder name) instead."
        )
    available = sorted(e.name for e in family_dir.iterdir()) if family_dir.exists() else []
    raise FileNotFoundError(
        f"No model named {model_name!r} under {family_dir}/ (checked as a run_id, and "
        f"against every model's own model_name in runs/{family}/*/train/*/metrics.json). "
        f"Available: {available}"
    )


def model_display_name(model_path: str | Path) -> str:
    """Human-facing name for a model.joblib, for a picker (CLI or UI): the
    model_name recorded in its training run's performance record (see
    classifiers.train --model-name / core.model_io.TrainedModel.model_name),
    or the run_id itself if there isn't one -- never unpickles the model
    just to get a label. Falls back to the model's own parent folder name
    if the path doesn't even follow the run_id convention (e.g. a
    model.joblib moved out of models/)."""
    model_path = Path(model_path)
    try:
        family, run_id = run_id_from_model_path(model_path)
    except ValueError:
        return model_path.resolve().parent.name
    metrics = _train_metrics(family, run_id, root=RUNS_ROOT) or {}
    return metrics.get("model_name") or run_id


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
