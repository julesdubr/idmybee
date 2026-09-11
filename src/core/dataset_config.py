"""dataset_config.py
Small per-dataset-root config file (dataset_config.json): a human-facing
`dataset_name`, plus n_photos/n_specimens. Written by
tools.ingestion.prepare_dataset / app/setup_dataset.py once a dataset root
has manifest.csv + biological_data.csv. Read back by classifiers.train/
classifiers.predict (resolve_dataset_name) to organize runs under
runs/<family>/<run_id>/{train,predict}/<dataset_name>/ instead of the raw
folder name -- see core.run_io module docstring.

`dataset_name` is sanitized (see sanitize_dataset_name) since it becomes a
folder name: a dataset root's own path may legitimately contain "/" (e.g.
"data/Bombus/collection"), but the run folder it names can't.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

DATASET_CONFIG_FILENAME = "dataset_config.json"


def dataset_config_path(dataset_root: str | Path) -> Path:
    return Path(dataset_root) / DATASET_CONFIG_FILENAME


def sanitize_dataset_name(name: str) -> str:
    """Strips path separators (and surrounding whitespace) so `name` is
    safe to use as a single folder name, e.g. "Bombus/collection" ->
    "Bombus_collection"."""
    return re.sub(r"[\\/]+", "_", name.strip()) or "dataset"


def read_dataset_config(dataset_root: str | Path) -> dict | None:
    """None if this dataset root has no dataset_config.json yet."""
    path = dataset_config_path(dataset_root)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_dataset_name(dataset_root: str | Path) -> str:
    """dataset_config.json's `dataset_name` if this root has one, otherwise
    the folder's own name (dataset roots prepared before this file existed,
    or throwaway roots from utils.uploaded_dataset.write_dataset_root,
    which never get one)."""
    config = read_dataset_config(dataset_root)
    if config and config.get("dataset_name"):
        return config["dataset_name"]
    return Path(dataset_root).name


def write_dataset_config(dataset_root: str | Path, dataset_name: str) -> Path:
    """Writes dataset_root/dataset_config.json: a sanitized `dataset_name`
    plus n_photos/n_specimens read from manifest.csv/biological_data.csv."""
    dataset_root = Path(dataset_root)
    manifest_path = dataset_root / "manifest.csv"
    bio_path = dataset_root / "biological_data.csv"
    config = {
        "dataset_name": sanitize_dataset_name(dataset_name),
        "n_photos": len(pd.read_csv(manifest_path)) if manifest_path.exists() else None,
        "n_specimens": len(pd.read_csv(bio_path)) if bio_path.exists() else None,
    }
    path = dataset_config_path(dataset_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
