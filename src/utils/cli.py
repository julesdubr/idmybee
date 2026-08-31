"""cli.py
Arguments de ligne de commande partagés entre classifiers/train.py,
classifiers/predict.py (batch) et analysis/variance_report.py.

add_dataset_args() ajoute : --split, --devices, --species, --castes,
--exclude-outliers, --non-strict, --tps, --landmarks-status-csv, --run-label.

resolve_split() : --split explicite > "all" si --tps custom est fourni sans
--split > default_split.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def add_dataset_args(parser: argparse.ArgumentParser, default_split: str = "train") -> None:
    parser.add_argument(
        "--split", type=str, default=None,
        help=f"Valeur de la colonne 'split' de manifest.csv à garder, ou 'all'. "
             f"Défaut : {default_split!r} -- sauf si --tps est fourni, auquel cas 'all' "
             f"(voir utils.cli.resolve_split).",
    )
    parser.add_argument("--devices", type=str, nargs="+", default=None,
                         help="Ne garder que ces photos (ex: --devices P1 S1). Voir utils.dataset._device_tag.")
    parser.add_argument("--species", type=str, nargs="+", default=None, help="Ne garder que ces espèces.")
    parser.add_argument("--castes", type=str, nargs="+", default=None, help="Ne garder que ces castes.")
    parser.add_argument(
        "--include-outliers", dest="exclude_outliers", action="store_false", default=True,
        help="Inclut les photos SUSPECT/FAILED (exclues par défaut, voir --landmarks-status-csv).",
    )
    parser.add_argument("--non-strict", action="store_true", help="Tolérer les blocs TPS malformés")
    parser.add_argument(
        "--tps", type=Path, default=None, dest="landmarks_tps",
        help="Fichier .tps à utiliser à la place de root/landmarks/landmarks_numbered.tps "
             "(ex: annotations manuelles de référence). specimens.csv/manifest.csv restent "
             "utilisés tels quels pour la jointure -- voir utils.dataset.load_dataset.",
    )
    parser.add_argument(
        "--landmarks-status-csv", type=Path, default=None,
        help="CSV de statut (tps_id+status) associé à --tps, pour --exclude-outliers. "
             "Optionnel : sans --tps, défaut = landmarks_numbered.csv ; avec --tps, "
             "pas de statut par défaut (exclusion sautée avec avertissement).",
    )
    parser.add_argument(
        "--run-label", type=str, default=None,
        help="Étiquette humaine ajoutée à l'identifiant de run (ex: 'tancrede19lm'). "
             "Par défaut, dérivée du nom du fichier --tps si fourni, sinon absente.",
    )


def resolve_split(args: argparse.Namespace, default_split: str = "train") -> str:
    """Résout la valeur effective de --split : explicite > 'all' si --tps custom > default_split."""
    if args.split is not None:
        return args.split
    if getattr(args, "landmarks_tps", None) is not None:
        return "all"
    return default_split


def dataset_kwargs(args: argparse.Namespace, default_split: str = "train") -> dict:
    """Construit les kwargs communs à load_dataset() à partir d'un Namespace
    peuplé par add_dataset_args()."""
    return dict(
        split=resolve_split(args, default_split=default_split),
        devices=args.devices,
        species=args.species,
        castes=args.castes,
        exclude_outliers=args.exclude_outliers,
        strict=not args.non_strict,
        landmarks_tps=args.landmarks_tps,
        landmarks_status_csv=args.landmarks_status_csv,
    )
