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


def load_modify_data(qcd_folder_path, wjet_folder_path, label_bg, label_sg):
    # NOTE: remember that WJet HTs should be around double that of QCD's Pts!
    # note: if you are running on cern resources, use your "eos" path for the data

    # aligning qcd and wjet infos, since we'll run a for loop on both separately
    df_lists     = qcd_df_list, wjet_df_list = [], []
    folder_paths = qcd_folder_path, wjet_folder_path
    labels       = label_bg, label_sg
    combined_dfs = [qcd_combined, wjet_combined] = [None, None]
    # modified_dfs = [qcd_modified, wjet_modified] = [None, None]

    # now do all the steps for qcd and then wjet
    for i in range(2):
        for file in tqdm(os.listdir(folder_paths[i]), desc=f"Loading {labels[i]} files"):
            curr_path = os.path.join(folder_paths[i], file)
            if os.path.isfile(curr_path) and os.path.splitext(file)[1] == ".pkl":
                df_lists[i].append(pd.read_pickle(curr_path))
        
        helpers.secs_since_last_ping()
        logging.info(f"Joining {labels[i]}:")
        combined_dfs[i] = pd.concat(df_lists[i], ignore_index=True)
        logging.info(f"Combined {labels[i]} data length: {len(combined_dfs[i])} {helpers.time_taken()}")

        # === FEATURE ENGINEERING ===
        # Compute derived features, apply one-hot encoding, filter PFCands, drop rows w/ invalid or missing entries
        df_cpy = combined_dfs[i].copy()
        logging.info(f"Copied {labels[i]} {helpers.time_taken()}")
        modified_df = modify_df(df_cpy, VALID_PDG)
        logging.info(f"Modified {labels[i]} {helpers.time_taken()}")
        yield modified_df.dropna()
        logging.info(f"Dropped missing vals and yielded {helpers.time_taken()}")

def scale_data(qcd_modified, wjet_modified, label_bg, label_sg):
    logging.info("Now, scaling data with QCD as the base!")

    # Select variables to scale (assumes first 17 are metadata or unscaled base features)
    variables_to_analyze = qcd_modified.columns[17:]

    # Compute robust scaling values using QCD dataset
    scaler_dict = find_scalers(qcd_modified.copy(), label_bg, cols=variables_to_analyze)

    # Apply scaling to both datasets using QCD-derived scalers
    logging.info(f"Scalers found {helpers.time_taken()}, now applying to qcd:")
    qcd_scaled,  qcd_scaled_vals,  qcd_raw_vals,  zero1 = apply_scalers(qcd_modified.copy(), scaler_dict)
    logging.info(f"{helpers.time_taken()} Now wjet:")
    wjet_scaled, wjet_scaled_vals, wjet_raw_vals, zero2 = apply_scalers(wjet_modified.copy(), scaler_dict)
    logging.info(f"Done! {helpers.time_taken()}")

    return (
        qcd_scaled,  qcd_scaled_vals,  qcd_raw_vals,  zero1,
        wjet_scaled, wjet_scaled_vals, wjet_raw_vals, zero2
    )

def visualize_data(data, label_bg, label_sg):
    (
        qcd_scaled,  qcd_scaled_vals,  qcd_raw_vals,  zero1,
        wjet_scaled, wjet_scaled_vals, wjet_raw_vals, zero2
    ) = data

    # Plot raw and scaled distributions for selected variable(s)
    logging.info("Plotting data:")
    for prop in PROPS:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        # Raw, with zeros
        plot_property_distribution(qcd_raw_vals[prop], wjet_raw_vals[prop], prop,
                                label_bg, label_sg,
                                ax=axes[0], is_scaled=False, include_zeros=True)

        # Raw, excluding zeros
        plot_property_distribution(qcd_raw_vals[prop], wjet_raw_vals[prop], prop,
                                label_bg, label_sg,
                                ax=axes[1], is_scaled=False, include_zeros=False,
                                scaled_zero1=0.0, scaled_zero2=0.0)

        # Scaled, with zeros
        plot_property_distribution(qcd_scaled_vals[prop], wjet_scaled_vals[prop], prop,
                                label_bg, label_sg,
                                ax=axes[2], is_scaled=True, include_zeros=True)

        # Scaled, excluding zeros
        plot_property_distribution(qcd_scaled_vals[prop], wjet_scaled_vals[prop], prop,
                                label_bg, label_sg,
                                ax=axes[3], is_scaled=True, include_zeros=False,
                                scaled_zero1=zero1[prop], scaled_zero2=zero2[prop])

        plt.tight_layout()
        # plt.show()
        fig_path = f"plots/proc_distr_{helpers.curr_time()}.png"
        plt.savefig(fig_path)
        logging.info(f"Saved figure into {fig_path}")

def save_data(data, path_bg, path_sg, label_bg, label_sg):
    (
        qcd_scaled,  qcd_scaled_vals,  qcd_raw_vals,  zero1,
        wjet_scaled, wjet_scaled_vals, wjet_raw_vals, zero2
    ) = data

    qcd_output_path  = os.path.join(config["data"]["processed_data_dir"], label_bg + "_scaled.pkl")
    wjet_output_path = os.path.join(config["data"]["processed_data_dir"], label_sg + "_scaled.pkl")
    logging.info("Now, saving data!")
    qcd_scaled.to_pickle(qcd_output_path),   logging.info(f"Saved in {qcd_output_path}!")
    wjet_scaled.to_pickle(wjet_output_path), logging.info(f"Saved in {wjet_output_path}!")

def main(path_bg, path_sg, label_bg, label_sg):
    qcd_modified, wjet_modified = tuple(load_modify_data(path_bg, path_sg, label_bg, label_sg))

    # === SCALING ===
    scaled_data = scale_data(qcd_modified, wjet_modified, label_bg, label_sg)
    
    # === VISUALIZATION ===
    visualize_data(scaled_data, label_bg, label_sg)

    # === SAVING ===
    save_data(scaled_data, path_bg, path_sg, label_bg, label_sg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Process",
        description="processes a background + signal pair of preprocessed jets, e.g. QCD and WJet"
    )
    parser.add_argument(
        "--background", "--bg", "-b", type=str, required=False, default=config["data"]["preprocessed_qcd"],
        help="Folder path to the background preprocessed data (QCD). Default: uses the path in configs/config.yaml"
    )
    parser.add_argument(
        "--signal", "--sg", "-s", type=str, required=False, default=config["data"]["preprocessed_wjet"],
        help="Folder path to the background signal data (WJet). Default: uses the path in configs/config.yaml"
    )
    parser.add_argument(
        "--label_bg", "--lb", "-B", type=str, required=False, default="QCD",
        help="Label/name for the background jet, usually with the jet Pt range (e.g. QCD_Pt1400to1800)"
    )
    parser.add_argument(
        "--label_sg", "--ls", "-S", type=str, required=False, default="WJet",
        help="Label/name for the signal jet, usually with the jet HT range (e.g. WJet_HT1400to1800)"
    )
    args = parser.parse_args()

    if config["dbg"]["measure_perf"]:
        helpers.profile_func(
            f"logs/proc_{helpers.curr_time()}.prof", main,
            args.background, args.signal, args.label_bg, args.label_sg
        )
    else:
        main(args.background, args.signal, args.label_bg, args.label_sg)
