"""
Unified DeepNTuplizer AK8 data preparation pipeline.

Reads DeepNTuplizer ROOT ntuples, applies feature engineering, optionally
scales features, and writes trainable pickle files for LeJEPA / PART training.

Example (smoke test, unscaled pickles for LeJEPA):

python -u scripts/prepare_data.py \
  --background /HEP/export/home/hgao50/jetcharge-work/output/smoke_qcd_QCD_Bin_PT_600to800_..._local_0.root \
  --signal /HEP/export/home/hgao50/jetcharge-work/output/smoke_qcd_QCD_Bin_PT_80to120_..._local_0.root \
  --lowerpt 200 --upperpt 800 \
  --max-jets 500 \
  --no-scale \
  --output-dir data/processed/ak8-smoke/unscaled
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List, Optional, Sequence, Tuple

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import constants as c
from helpers import helpers_main
from preprocess.feature_engineering import modify_df
from preprocess.read_deepntuplizer import read_root_paths
from preprocess.scaling import apply_scalers, find_scalers

config = helpers_main.load_config()

VALID_PDG = ["-11", "11", "-13", "13", "-211", "211"]

# Columns produced by modify_df that should be scaled (when --scale is used).
SCALABLE_FEATURE_COLUMNS = [
    "pt",
    "eta",
    "phi",
    "charge",
    "puppiWeight",
    "d0",
    "d0Err",
    "dz",
    "dzErr",
    "mass",
    "dz/dzErr",
    "d0/d0Err",
    "dR",
    "log_pt",
]


def _existing_scalable_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in SCALABLE_FEATURE_COLUMNS if col in df.columns]


def load_and_engineer(
    paths: Sequence[str],
    label: str,
    lowerpt: Optional[float],
    upperpt: Optional[float],
    max_jets: Optional[int],
    max_files: Optional[int],
    recursive: bool,
) -> pd.DataFrame:
    logging.info("Reading ROOT files for %s from %s", label, paths)
    raw_df = read_root_paths(
        paths=paths,
        lowerpt=lowerpt,
        upperpt=upperpt,
        max_jets=max_jets,
        max_files=max_files,
        recursive=recursive,
    )
    if raw_df.empty:
        raise ValueError(f"No jets passed selection for {label}.")

    logging.info("Loaded %s jets for %s; running feature engineering...", len(raw_df), label)
    engineered = modify_df(raw_df.copy(), VALID_PDG)
    engineered = engineered.dropna()
    logging.info("%s has %s jets after feature engineering.", label, len(engineered))
    if engineered.empty:
        raise ValueError(f"All jets were dropped during feature engineering for {label}.")
    return engineered


def scale_pair(
    background: pd.DataFrame,
    signal: pd.DataFrame,
    label_bg: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cols = _existing_scalable_columns(background)
    logging.info("Scaling columns: %s", cols)
    scaler_dict = find_scalers(background.copy(), label_bg, cols=cols)
    bg_scaled, _, _, _ = apply_scalers(background.copy(), scaler_dict)
    sg_scaled, _, _, _ = apply_scalers(signal.copy(), scaler_dict)
    return bg_scaled, sg_scaled


def save_pickle(df: pd.DataFrame, output_dir: str, filename: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    helpers_main.create_missing_dir(filepath)
    df.to_pickle(filepath)
    logging.info("Saved %s (%s jets)", filepath, len(df))
    return filepath


def main(args: argparse.Namespace) -> None:
    bg_paths = [args.background] if isinstance(args.background, str) else list(args.background)
    sg_paths = (
        [args.signal]
        if args.signal is None
        else ([args.signal] if isinstance(args.signal, str) else list(args.signal))
    )

    background = load_and_engineer(
        paths=bg_paths,
        label=args.label_bg,
        lowerpt=args.lowerpt,
        upperpt=args.upperpt,
        max_jets=args.max_jets,
        max_files=args.max_files,
        recursive=args.recursive,
    )

    if args.signal is not None:
        signal = load_and_engineer(
            paths=sg_paths,
            label=args.label_sg,
            lowerpt=args.signal_lowerpt if args.signal_lowerpt is not None else args.lowerpt,
            upperpt=args.signal_upperpt if args.signal_upperpt is not None else args.upperpt,
            max_jets=args.max_jets,
            max_files=args.max_files,
            recursive=args.recursive,
        )
    else:
        logging.warning(
            "No --signal provided; writing background pickles only. "
            "Use a second sample for evaluation once WJets / H->bb ntuples are ready."
        )
        signal = None

    os.makedirs(args.output_dir, exist_ok=True)

    if args.scale:
        if signal is None:
            cols = _existing_scalable_columns(background)
            scaler_dict = find_scalers(background.copy(), args.label_bg, cols=cols)
            background_out, _, _, _ = apply_scalers(background.copy(), scaler_dict)
            save_pickle(background_out, args.output_dir, f"{args.label_bg}_scaled.pkl")
        else:
            background_out, signal_out = scale_pair(background, signal, args.label_bg)
            save_pickle(background_out, args.output_dir, f"{args.label_bg}_scaled.pkl")
            save_pickle(signal_out, args.output_dir, f"{args.label_sg}_scaled.pkl")
    else:
        save_pickle(background, args.output_dir, f"{args.label_bg}_unscaled.pkl")
        if signal is not None:
            save_pickle(signal, args.output_dir, f"{args.label_sg}_unscaled.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="prepare_data",
        description="Prepare trainable pickle files from DeepNTuplizer AK8 ROOT ntuples.",
    )
    parser.add_argument(
        "--background",
        "--bg",
        "-b",
        required=True,
        help="ROOT file or directory for background (e.g. QCD).",
    )
    parser.add_argument(
        "--signal",
        "--sg",
        "-s",
        default=None,
        help="Optional ROOT file or directory for signal / evaluation sample.",
    )
    parser.add_argument(
        "--label-bg",
        "-B",
        default="QCD",
        help="Output label prefix for background pickles.",
    )
    parser.add_argument(
        "--label-sg",
        "-S",
        default="Signal",
        help="Output label prefix for signal pickles.",
    )
    parser.add_argument(
        "--lowerpt",
        type=float,
        default=None,
        help="Lower bound on AK8 jet pT [GeV].",
    )
    parser.add_argument(
        "--upperpt",
        type=float,
        default=None,
        help="Upper bound on AK8 jet pT [GeV] for background.",
    )
    parser.add_argument(
        "--signal-lowerpt",
        type=float,
        default=None,
        help="Optional lower pT bound for signal. Defaults to --lowerpt.",
    )
    parser.add_argument(
        "--signal-upperpt",
        type=float,
        default=None,
        help="Optional upper pT bound for signal. Defaults to --upperpt.",
    )
    parser.add_argument(
        "--max-jets",
        type=int,
        default=None,
        help="Maximum number of jets to keep per sample (for smoke tests).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of ROOT files to read per sample.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=config["data"].get("processed_data_dir_ak8", "data/processed/ak8/unscaled"),
        help="Directory for output pickle files.",
    )
    parser.add_argument(
        "--scale",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply QCD-based percentile scaling. Default: off (preferred for LeJEPA).",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recursively search directories for ROOT files.",
    )

    cli_args = parser.parse_args()
    helpers_main.log_config(f"logs/prepare_data_{helpers_main.curr_time()}.log")

    if config["dbg"]["measure_perf"]:
        helpers_main.profile_func(
            f"logs/prepare_data_{helpers_main.curr_time()}.prof",
            main,
            cli_args,
        )
    else:
        main(cli_args)
