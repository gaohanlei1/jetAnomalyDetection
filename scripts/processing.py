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
import argparse

# Add parent directory to import local project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from helpers import helpers
config = helpers.load_config()

import logging
helpers.log_config(f"logs/proc_{helpers.curr_time()}.log")

if config["dbg"]["measure_perf"]:
    from cProfile import Profile
    import pstats

from preprocess.feature_engineering import modify_df
from preprocess.scaling import find_scalers, apply_scalers
from visualize.plot_property_distributions import plot_property_distribution

# PDG IDs to consider as valid charged particles
VALID_PDG = [-11, 11, -13, 13, -211, 211]
# NOTE: Modify this list to include other features as needed
PROPS = ["log_pt"]

# Dataset labels (used for file naming and plots)
QCD_LABEL  = config["data"]["qcd_label"]
WJET_LABEL = config["data"]["wjet_label"]

'''
ISSUES:
- oh god that scaled data struct is horrendous
- subfolders!!!
'''

def load_modify_data():
    # Paths to folders containing preprocessed raw data (concatenated together before processing)
    # NOTE: remember that WJet HTs should be around double that of QCD's Pts!
    qcd_folder_path  = os.path.join(config["data"]["preprocessed_data_dir"], "qcd")
    wjet_folder_path = os.path.join(config["data"]["preprocessed_data_dir"], "wjet")
    # note: if you are running on cern resources, use your "eos" path here for the data

    # aligning qcd and wjet infos, since we'll run a for loop on both separately
    df_lists     = qcd_df_list, wjet_df_list = [], []
    folder_paths = qcd_folder_path, wjet_folder_path
    labels       = QCD_LABEL, WJET_LABEL
    combined_dfs = [qcd_combined, wjet_combined] = [None, None]
    # modified_dfs = [qcd_modified, wjet_modified] = [None, None]

    for i in range(len(df_lists)):
        for file in tqdm(os.listdir(folder_paths[i]), desc=f"Loading {labels[i]} files"):
            curr_path = os.path.join(folder_paths[i], file)
            if os.path.isfile(curr_path) and os.path.splitext(file)[1] == ".pkl":
                df_lists[i].append(pd.read_pickle(curr_path))
        
        logging.info(f"Concatenating {labels[i]}:")
        combined_dfs[i] = pd.concat(df_lists[i], ignore_index=True)
        logging.info(f"Combined {labels[i]} data length: {len(combined_dfs[i])}")

        # === FEATURE ENGINEERING ===
        # Compute derived features, apply one-hot encoding, filter PFCands, drop rows w/ invalid or missing entries
        df_cpy = combined_dfs[i].copy()
        logging.info(f"Copied {labels[i]},")
        modified_df = modify_df(df_cpy, VALID_PDG)
        logging.info(f"modified {labels[i]},")
        yield modified_df.dropna()

def scale_data(qcd_modified, wjet_modified):
    logging.info("Now, scaling data with QCD as the base!")

    # Select variables to scale (assumes first 17 are metadata or unscaled base features)
    variables_to_analyze = qcd_modified.columns[17:]

    # Compute robust scaling values using QCD dataset
    scaler_dict = find_scalers(qcd_modified.copy(), QCD_LABEL, cols=variables_to_analyze)

    # Apply scaling to both datasets using QCD-derived scalers
    logging.info("Scalers found, now applying to qcd:")
    qcd_scaled,  qcd_scaled_vals,  qcd_raw_vals,  zero1 = apply_scalers(qcd_modified.copy(), scaler_dict)
    logging.info("And now wjet:")
    wjet_scaled, wjet_scaled_vals, wjet_raw_vals, zero2 = apply_scalers(wjet_modified.copy(), scaler_dict)
    logging.info("Done!")

    return (
        qcd_scaled,  qcd_scaled_vals,  qcd_raw_vals,  zero1,
        wjet_scaled, wjet_scaled_vals, wjet_raw_vals, zero2
    )

def visualize_data(scaled):
    (
        qcd_scaled,  qcd_scaled_vals,  qcd_raw_vals,  zero1,
        wjet_scaled, wjet_scaled_vals, wjet_raw_vals, zero2
    ) = scaled

    # Plot raw and scaled distributions for selected variable(s)
    logging.info("Plotting data:")
    for prop in PROPS:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        # Raw, with zeros
        plot_property_distribution(qcd_raw_vals[prop], wjet_raw_vals[prop], prop,
                                QCD_LABEL, WJET_LABEL,
                                ax=axes[0], is_scaled=False, include_zeros=True)

        # Raw, excluding zeros
        plot_property_distribution(qcd_raw_vals[prop], wjet_raw_vals[prop], prop,
                                QCD_LABEL, WJET_LABEL,
                                ax=axes[1], is_scaled=False, include_zeros=False,
                                scaled_zero1=0.0, scaled_zero2=0.0)

        # Scaled, with zeros
        plot_property_distribution(qcd_scaled_vals[prop], wjet_scaled_vals[prop], prop,
                                QCD_LABEL, WJET_LABEL,
                                ax=axes[2], is_scaled=True, include_zeros=True)

        # Scaled, excluding zeros
        plot_property_distribution(qcd_scaled_vals[prop], wjet_scaled_vals[prop], prop,
                                QCD_LABEL, WJET_LABEL,
                                ax=axes[3], is_scaled=True, include_zeros=False,
                                scaled_zero1=zero1[prop], scaled_zero2=zero2[prop])

        plt.tight_layout()
        # plt.show()
        fig_path = f"plots/proc_distr_{helpers.curr_time()}.png"
        plt.savefig(fig_path)
        logging.info(f"Saved figure into {fig_path}")

def save_data(scaled):
    (
        qcd_scaled,  qcd_scaled_vals,  qcd_raw_vals,  zero1,
        wjet_scaled, wjet_scaled_vals, wjet_raw_vals, zero2
    ) = scaled

    qcd_output_path  = f"{config['data']['processed_data_dir']}{QCD_LABEL}_scaled.pkl"
    wjet_output_path = f"{config['data']['processed_data_dir']}{WJET_LABEL}_scaled.pkl"
    logging.info("Now, saving data!")
    qcd_scaled.to_pickle(qcd_output_path),   logging.info(f"Saved in {qcd_output_path}!")
    wjet_scaled.to_pickle(wjet_output_path), logging.info(f"Saved in {wjet_output_path}!")

def main():
    qcd_modified, wjet_modified = tuple(load_modify_data())

    # === SCALING ===
    scaled = scale_data(qcd_modified, wjet_modified)
    
    # === VISUALIZATION ===
    visualize_data(scaled)

    # === SAVING ===
    save_data(scaled)


if __name__ == "__main__":
    if config["dbg"]["measure_perf"]:
        helpers.profile_func(f"logs/proc_{helpers.curr_time()}.prof", main)
    else:
        main()
