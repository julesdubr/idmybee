"""renumber.py
Phase 3: renumbers a detector's unordered landmarks (e.g. Gabriel's UNet,
output of landmarks/predict.py) to the canonical numbering of a FROZEN
reference shape (see landmarks/build_reference.py), via a numbering method
that follows the landmarks.methods.base contract -- then flags suspect
registrations by comparison against the population of the same species.

Two uses, same numbering logic:
- `numerate_one(landmarks, zones)`: an already-loaded point cloud -> a
  NumberingResult. This is exactly what the CLI does for each specimen,
  independently of the others -- nothing prevents calling it on a single
  image in field mode, as long as the SAME frozen reference shape is used
  as in batch mode (the whole point of build_reference.py: without a
  frozen artifact, recomputing the consensus on every run risks slightly
  desyncing batch numbering from field numbering).
- CLI (`python -m landmarks.renumber <dataset> ...`): processes a whole
  dataset's TPS, writes the renumbered TPS + a detailed per-specimen log.

Input (CLI):
  - <dataset>/landmarks/<tps>        (output of landmarks/predict.py)
  - <reference>                      (frozen artifact, see build_reference.py)
  - <dataset>/biological_data.csv    (optional: enables the per-species
    outlier diagnostic -- without it, every numbered specimen stays OK.
    See tools/ingestion/export_clean_dataset.py -- "labeled" here means a non-empty
    `species`, there is no `is_labeled` column in this schema)

Output (CLI):
  - <dataset>/landmarks/<tps stem>_numbered.tps: every successfully
    numbered specimen (OK + SUSPECT).
  - <dataset>/landmarks/landmarks_numbered.csv: one status per specimen
    (OK/SUSPECT/FAILED) + reason, for ALL input specimens -- including
    FAILED ones absent from the TPS: nothing is lost without a trace.
  - <dataset>/pipeline_stats.csv (step="renumbering")

Statuses:
  - FAILED  : landmark count incompatible with the reference (never
    numbered). This is the fate of EVERY SUSPECT specimen coming out of
    predict.py (Phase 2): fewer landmarks than the reference expects, so
    an automatic failure here regardless of registration quality -- not a
    bug, numerate_one() doesn't handle differing point counts (see
    landmarks.methods.hungarian_umeyama).
  - SUSPECT : successfully numbered, but a post-GPA outlier relative to its
    own species (see core.outliers.flag_by_species) -- requires
    --biological-data; without it, or for an unlabeled specimen, stays OK.
  - OK      : numbered, no anomaly signal.

Usage:
    python -m landmarks.build_reference --ref references/ref-landmarks.tps \\
        --drop 3 --out references/shapes/reference_shape.tps          # once

    python -m landmarks.renumber data/clean/collection --tps landmarks.tps \\
        --reference references/shapes/reference_shape.tps \\
        --biological-data data/clean/collection/biological_data.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from landmarks.build_reference import load_reference
from landmarks.methods.base import NumberingResult
from landmarks.methods.hungarian_umeyama import numerate as numerate_hungarian_umeyama
from utils.cli import add_dataset_positional, add_logging_args, log_level_from_args
from core.outliers import HEAVY_LANDMARK_FRAC, MAD_FACTOR, MIN_GROUP_SIZE, flag_by_species
from core.pipeline_io import update_pipeline_stats
from core.run_io import setup_console_logging
from core.tps_io import ImageLandmarks, parse_tps, write_tps

logger = logging.getLogger(__name__)

# Only one method implemented so far (graph_matching.py was tried, then
# dropped -- worse on this dataset). This registry avoids touching the CLI
# the day a second method conforming to the landmarks.methods.base contract
# is added: `--method <name>`.
METHODS = {"hungarian_umeyama": numerate_hungarian_umeyama}


def numerate_one(
    landmarks: np.ndarray, zones: np.ndarray, method: str = "hungarian_umeyama",
) -> NumberingResult:
    """Numbers ONE specimen (batch or field) against an already-loaded
    reference shape (see build_reference.load_reference). The only FAILED
    possible here is intrinsic to the method (incompatible point count) --
    the per-species (population) SUSPECT diagnostic only makes sense in the
    presence of a group, see flag_by_species / the CLI below."""
    if landmarks.shape[0] != zones.shape[0]:
        return NumberingResult(
            numbered=landmarks, status="FAILED", score=float("inf"),
            reason=f"{landmarks.shape[0]} landmarks, {zones.shape[0]} expected",
        )
    return METHODS[method](landmarks, zones)


def load_specimen_labels(biological_data_csv: Path) -> tuple[dict[str, str], set[str]]:
    """Reads biological_data.csv -> (species per labeled inv_id, set of
    labeled inv_id). A specimen absent from the dict/set is treated as
    unlabeled (prediction pool), including if it's absent from
    biological_data_csv.

    "Labeled" = non-empty `species` -- there is no `is_labeled` column in
    this schema (see tools/ingestion/export_clean_dataset.py), matching the same
    convention already used by core.dataset.load_dataset."""
    df = pd.read_csv(biological_data_csv)
    required = {"inv_id", "species"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns in {biological_data_csv}: {missing}")
    labeled = df[df["species"].notna()]
    species_by_id = dict(zip(labeled["inv_id"], labeled["species"]))
    return species_by_id, set(species_by_id)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_dataset_positional(parser)
    parser.add_argument("--tps", type=Path, default=Path("landmarks.tps"),
                         help="TPS to renumber, unordered landmarks (in <dataset>/<landmarks-dir>/).")
    parser.add_argument(
        "--landmarks-dir", default="landmarks",
        help="Subfolder (under <dataset>/) to read --tps from and write the numbered TPS/CSV into "
             "(default: 'landmarks') -- see landmarks.predict's own --landmarks-dir.",
    )
    parser.add_argument("--reference", type=Path, required=True,
                         help="Frozen reference shape (see landmarks.build_reference).")
    parser.add_argument("--biological-data", type=Path, default=None,
                         help="biological_data.csv -- enables the per-species SUSPECT diagnostic. "
                              "Without it, every numbered specimen stays OK.")
    parser.add_argument("--method", default="hungarian_umeyama", choices=sorted(METHODS),
                         help="Numbering method.")
    parser.add_argument("--log", type=Path, default=None,
                         help="Path to the status CSV (default: <dataset>/landmarks/landmarks_numbered.csv)")
    parser.add_argument("--min-group-size", type=int, default=MIN_GROUP_SIZE,
                         help=f"Minimum species group size to evaluate the post-GPA outlier "
                              f"threshold (default: {MIN_GROUP_SIZE})")
    parser.add_argument("--outlier-mad-factor", type=float, default=MAD_FACTOR,
                         help=f"Threshold = median + factor*MAD of the distance to the landmark's "
                              f"median position, per species (default: {MAD_FACTOR})")
    parser.add_argument("--outlier-landmark-frac", type=float, default=HEAVY_LANDMARK_FRAC,
                         help=f"Fraction of outlier landmarks beyond which a whole specimen "
                              f"is marked SUSPECT (default: {HEAVY_LANDMARK_FRAC})")
    add_logging_args(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    setup_console_logging(log_level_from_args(args))
    input_path = args.dataset / args.landmarks_dir / args.tps

    zones = load_reference(args.reference)
    logger.info("Reference: %d zones (%s)", len(zones), args.reference)

    inputs, parse_errors = parse_tps(input_path, strict=False)
    if not inputs:
        raise SystemExit(
            f"No valid specimen in {input_path} ({len(parse_errors)} parsing error(s))."
        )
    if parse_errors:
        logger.warning("%d TPS parsing error(s) ignored in %s", len(parse_errors), input_path)

    # --- Renumbering, specimen by specimen --------------------------------------
    pipeline_start = time.perf_counter()
    results: list[NumberingResult] = []
    processing_times: list[float] = []
    specimen_refs = []  # (photo_id, inv_id, tps_id, image_path), parallel to `results`
    for sp in inputs:
        item_start = time.perf_counter()
        results.append(numerate_one(sp.landmarks, zones, method=args.method))
        processing_times.append(time.perf_counter() - item_start)
        specimen_refs.append((sp.photo_id, sp.inv_id, sp.tps_id, sp.image_path))

    numbered_specimens = [
        replace(sp, landmarks=r.numbered) for sp, r in zip(inputs, results) if r.status != "FAILED"
    ]
    # index in `results`/`specimen_refs` of each numbered_specimens entry, same order
    numbered_indices = [i for i, r in enumerate(results) if r.status != "FAILED"]

    # --- Post-GPA per-species outlier diagnostic (SUSPECT) -----------------------
    # The only source of SUSPECT now: unlike hungarian_umeyama.numerate()'s
    # old `ambiguity_ratio` (per-specimen comparison of the best vs.
    # second-best start, removed -- flagged nearly 100% of specimens as
    # SUSPECT on this dataset), this diagnostic compares each specimen to
    # its species' POPULATION, which makes it far more specific. Requires
    # --biological-data.
    n_outlier_by_index: dict[int, int] = {}
    if args.biological_data is not None:
        species_by_id, labeled_ids = load_specimen_labels(args.biological_data)
        labeled_positions = [
            j for j, sp in enumerate(numbered_specimens) if sp.inv_id in labeled_ids
        ]
        if labeled_positions:
            labeled_specimens = [numbered_specimens[j] for j in labeled_positions]
            species = np.array([species_by_id[s.inv_id] for s in labeled_specimens])
            n_outlier, heavy = flag_by_species(
                labeled_specimens, species,
                heavy_frac=args.outlier_landmark_frac,
                min_group_size=args.min_group_size,
                mad_factor=args.outlier_mad_factor,
            )
            for local_j, is_heavy, n_out in zip(labeled_positions, heavy, n_outlier):
                result_idx = numbered_indices[local_j]
                n_outlier_by_index[result_idx] = int(n_out)
                if is_heavy:
                    results[result_idx].status = "SUSPECT"
    else:
        logger.info("No --biological-data given: per-species outlier diagnostic disabled (all OK).")

    # --- Writing the renumbered TPS ----------------------------------------------
    out_dir = args.dataset / args.landmarks_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.tps.stem
    out_path = out_dir / f"{stem}_numbered.tps"
    write_tps(out_path, numbered_specimens)

    n_failed = len(inputs) - len(numbered_specimens)
    print(f"\nWrote {len(numbered_specimens)}/{len(inputs)} specimen(s) -> {out_path}")
    if n_failed:
        print(f"  ({n_failed} FAILED excluded)")

    n_by_status = {s: sum(r.status == s for r in results) for s in ("OK", "SUSPECT", "FAILED")}
    print(f"Statuses: {n_by_status}")

    # --- Detailed log + stats -----------------------------------------------------
    log_path = args.log or (out_dir / "landmarks_numbered.csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "photo_id", "inv_id", "tps_id", "image_path", "status",
            "registration_cost", "n_outlier_landmarks", "error_reason", "processing_time_s",
        ])
        for i, ((photo_id, inv_id, tps_id, image_path), r, pt) in enumerate(
            zip(specimen_refs, results, processing_times)
        ):
            w.writerow([
                photo_id, inv_id, tps_id, image_path, r.status, r.score,
                n_outlier_by_index.get(i, ""), r.reason, f"{pt:.4f}",
            ])
    print(f"Detailed statuses -> {log_path}")

    total_time_s = time.perf_counter() - pipeline_start
    counter_dict = {
        "total": len(inputs), "ok": n_by_status["OK"], "suspect": n_by_status["SUSPECT"],
        "skipped": 0, "failed": n_by_status["FAILED"],
    }
    stats_path = args.dataset / "pipeline_stats.csv"
    update_pipeline_stats(stats_path, "renumbering", args.method, counter_dict, total_time_s)
    print(f"Stats -> {stats_path}")


if __name__ == "__main__":
    main()
