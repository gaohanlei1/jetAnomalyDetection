"""
Preprocess and visualize raw jet data for anomaly detection.

This script:
- Concatenates jet data into one dataframe per class for two classes, 
    and loads them (e.g. background: QCD, signal: WJets).
- Applies feature engineering including derived variables and filtering.
- Scales all features using percentile-based normalization.
- Visualizes distributions of selected features before and after scaling.
- Saves the cleaned and scaled datasets as new pickle files.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import yaml
import argparse

# Add parent directory to import local project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocess.feature_engineering import modify_df
from preprocess.scaling import find_scalers, apply_scalers
from visualize.plot_property_distributions import plot_property_distribution

# PDG IDs to consider as valid charged particles
VALID_PDG = [-11, 11, -13, 13, -211, 211]

# def arg_setup():
#     # Returns the qcd and wjet arguments (str | None)
#     parser = argparse.ArgumentParser(
#         prog="Processing",
#         description="Processes and scales preprocessed QCD + WJet data for training"
#     )
#     parser.add_argument(
#         "--qcd", type=str, required=False,
#         help="name of the QCD file in the preprocessed data directory to process; if both --qcd and --wjet are none, then this will randomly pair the files present"
#     )
#     parser.add_argument(
#         "--wjet", type=str, required=False,
#         help="name of the WJet file to process; if both --qcd and --wjet are none, then random pairing"
#     )
#     args = parser.parse_args()
#     return args.qcd, args.wjet

def load_config():
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    config = load_config()
    # qcd_file, wjet_file = arg_setup()

    # TODO: separate these sections into functions for the diff stages

    # Path to folder containing preprocessed raw data (used by default for concatenation)
    data_folder_path = config["data"]["preprocessed_data_dir"]
    # note: if you are running on cern resources, use your "eos" path here for the data

    # Dataset labels (used for file naming and plots)
    qcd_label  = config["data"]["qcd_label"]
    wjet_label = config["data"]["wjet_label"]

    # TODO: using a loop to iterate through files in the preprocessed dir, and add to these lists
    # File paths for each dataset (assumes single file for each class)
    qcd_pkl_files  = [data_folder_path + "qcd.pkl"]
    wjet_pkl_files = [data_folder_path + "wjet.pkl"]
    
    # === LOAD DATA ===
    # Load and concatenate all pickled QCD and WJET files
    qcd_raw = pd.concat(
        [pd.read_pickle(f) for f in tqdm(qcd_pkl_files, desc=f"Loading {qcd_label}")],
        ignore_index=True
    )
    wjet_raw = pd.concat(
        [pd.read_pickle(f) for f in tqdm(wjet_pkl_files, desc=f"Loading {wjet_label}")],
        ignore_index=True
    )

    print(f"{qcd_label} data length: {len(qcd_raw)}")
    print(f"{wjet_label} data length: {len(wjet_raw)}")

    # === FEATURE ENGINEERING ===

    # Compute derived features, apply one-hot encoding, and filter PFCands
    qcd = modify_df(qcd_raw.copy(), VALID_PDG)
    wjet = modify_df(wjet_raw.copy(), VALID_PDG)

    # Drop rows with missing or invalid entries
    qcd.dropna(inplace=True)
    wjet.dropna(inplace=True)

    # === SCALING ===

    # Select variables to scale (assumes first 17 are metadata or unscaled base features)
    variables_to_analyze = qcd.columns[17:]

    # Compute robust scaling values using QCD dataset
    scaler_dict = find_scalers(qcd.copy(), qcd_label, cols=variables_to_analyze)

    # Apply scaling to both datasets using QCD-derived scalers
    qcd_scaled, qcd_scaled_vals, qcd_raw_vals, zero1 = apply_scalers(qcd.copy(), scaler_dict)
    wjet_scaled, wjet_scaled_vals, wjet_raw_vals, zero2 = apply_scalers(wjet.copy(), scaler_dict)


    # === VISUALIZATION ===

    # Plot raw and scaled distributions for selected variable(s)
    for prop in ["log_pt"]:  # Modify this list to include other features as needed
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        # Raw, with zeros
        plot_property_distribution(qcd_raw_vals[prop], wjet_raw_vals[prop], prop,
                                qcd_label, wjet_label,
                                ax=axes[0], is_scaled=False, include_zeros=True)

        # Raw, excluding zeros
        plot_property_distribution(qcd_raw_vals[prop], wjet_raw_vals[prop], prop,
                                qcd_label, wjet_label,
                                ax=axes[1], is_scaled=False, include_zeros=False,
                                scaled_zero1=0.0, scaled_zero2=0.0)

        # Scaled, with zeros
        plot_property_distribution(qcd_scaled_vals[prop], wjety_scaled_vals[prop], prop,
                                qcd_label, wjet_label,
                                ax=axes[2], is_scaled=True, include_zeros=True)

        # Scaled, excluding zeros
        plot_property_distribution(qcd_scaled_vals[prop], wjet_scaled_vals[prop], prop,
                                qcd_label, wjet_label,
                                ax=axes[3], is_scaled=True, include_zeros=False,
                                scaled_zero1=zero1[prop], scaled_zero2=zero2[prop])

        plt.tight_layout()
        plt.show()

    qcd_scaled.to_pickle(f"./data/processed/{qcd_label}_scaled.pkl")
    wjet_scaled.to_pickle(f"./data/processed/{wjet_label}_scaled.pkl")

