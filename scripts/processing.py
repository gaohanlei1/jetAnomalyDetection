"""
Preprocess and visualize raw jet data for anomaly detection.

This script:
- Concatenates jet data into one dataframe per class for two classes, 
    and loads them (e.g. background: QCD, signal: WJets).
- Applies feature engineering including derived variables and filtering.
- Scales all features using percentile-based normalization.
- Visualizes distributions of selected features before and after scaling.
- Saves the cleaned and scaled datasets as new pickle files.

e.g. python3.9 scripts/processing.py -b data/preprocessed/qcd/btvnano/concat_QCD_PT-170to300_TuneCP5_13p6TeV_pythia8_1.pkl -s data/preprocessed/wjet/WJetsToQQ_HT-400to600/concat_nano_mc2017_102.pkl -B QCD_PT-170to300_13p6TeV -S WJetsToQQ_HT-400to600
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
from helpers import helpers_main
config = helpers_main.load_config()

import logging
helpers_main.log_config(f"logs/proc_{helpers_main.curr_time()}.log")

if config["dbg"]["measure_perf"]:
    from cProfile import Profile
    import pstats

from preprocess.feature_engineering import modify_df
from preprocess.scaling import find_scalers, apply_scalers
from visualize.plot_property_distributions import plot_property_distribution


class DataProcessor:
    # PDG IDs to consider as valid charged particles
    VALID_PDG = [-11, 11, -13, 13, -211, 211]
    # NOTE: Modify this list to include other features as needed.
    # possible props: ['mass', 'puppiWeight', 'puppiWeightNoLep', 'trkChi2', 'vtxChi2', 'dz/dzErr', 'd0/d0Err', 'dR', 'within_bounds', 'log_pt']
    PROPS = ["log_pt"]
    # disable when running over SSH or similar
    DISPLAY_PLOT = False

    # NOTE: remember that WJet HTs should be around double that of QCD's Pts!
    def __init__(self, cli_args):
        self.data_bg, self.data_sg = cli_args.background, cli_args.signal
        self.label_bg,  self.label_sg  = cli_args.label_bg, cli_args.label_sg
        self.filter = cli_args.filter

        self.qcd_modified, self.wjet_modified = None, None
        self.qcd_scaled,  self.qcd_scaled_vals,  self.qcd_raw_vals,  self.zero1 = None, None, None, None
        self.wjet_scaled, self.wjet_scaled_vals, self.wjet_raw_vals, self.zero2 = None, None, None, None
    
    def load_concat_jet_data(self, data_path, jet_label):
        '''
        Loads and joins all the preprocessed .pkl files in the given directory.
        '''
        helpers_main.secs_since_last_ping()
        preproc_paths = [data_path] if os.path.isfile(data_path) else [
            os.path.join(data_path, file) for file in os.listdir(data_path)
        ]
        preproc_dfs = [
            pd.read_pickle(file)            #.head(100)
            for file in tqdm(preproc_paths, desc=f"Loading {jet_label} files")
            if os.path.isfile(file)
            and os.path.splitext(file)[1] == ".pkl"
            and (not self.filter or jet_label in file)
        ]
   
        logging.info(f"Concatenating {jet_label}:" + helpers_main.time_taken())
        combined = pd.concat(preproc_dfs, ignore_index=True)
        logging.info(f"Combined {jet_label} data length: {len(combined)} {helpers_main.time_taken()}")
        return combined

    def feature_engineer(self, combined_data, jet_label):
        '''
        === FEATURE ENGINEERING ===
        Compute derived features, apply one-hot encoding, filter PFCands,
        drop rows w/ invalid or missing entries
        '''
        # df_cpy = combined_data.copy()
        # logging.info(f"Copied {jet_label} {helpers_main.time_taken()}")
        # modified_df = modify_df(df_cpy, VALID_PDG)

        modified_data = modify_df(combined_data.copy(), self.VALID_PDG)
        logging.info(f"Modified {jet_label} {helpers_main.time_taken()}")
        modified_data = modified_data.dropna()
        logging.info(f"Dropped missing vals {helpers_main.time_taken()}")
        return modified_data

    def load_modify(self):
        # Loads the preprocessed files and performs feature engineering on them.
        # note: if you are running on cern resources, use your "eos" path for the data
        self.qcd_modified  = self.feature_engineer(
            self.load_concat_jet_data(self.data_bg, self.label_bg), self.label_bg
        )
        self.wjet_modified = self.feature_engineer(
            self.load_concat_jet_data(self.data_sg, self.label_sg), self.label_sg
        )

    def scale_data(self):
        # Scales the data and saves the results
        logging.info("Now, scaling data with QCD as the base!")

        # Select variables to scale (assumes first 17 are metadata or unscaled base features)
        variables_to_analyze = self.qcd_modified.columns[17:]

        # Compute robust scaling values using QCD dataset
        scaler_dict = find_scalers(self.qcd_modified.copy(), self.label_bg, cols=variables_to_analyze)
        # scaler_dict = find_scalers(self.qcd_modified, self.label_bg, cols=variables_to_analyze)

        # Apply scaling to both datasets using QCD-derived scalers
        logging.info(f"Scalers found {helpers_main.time_taken()}, now applying to qcd:")
        (
            self.qcd_scaled,  self.qcd_scaled_vals,  self.qcd_raw_vals,  self.zero1
        ) = apply_scalers(self.qcd_modified.copy(), scaler_dict)
        # .copy()

        logging.info(f"{helpers_main.time_taken()} Now wjet:")
        (
            self.wjet_scaled, self.wjet_scaled_vals, self.wjet_raw_vals, self.zero2
        ) = apply_scalers(self.wjet_modified.copy(), scaler_dict)
        # .copy()
        logging.info(f"Done! {helpers_main.time_taken()}")

        logging.info(f"{type(self.wjet_scaled_vals)=}\n{self.wjet_scaled_vals=}\n{self.qcd_scaled_vals=}")

    def visualize_data(self):
        # Plot raw and scaled distributions for selected variable(s)
        logging.info("Plotting data:")
        for prop in self.PROPS:
            fig, axes = plt.subplots(1, 2, figsize=(20, 5))

            # # Raw, with zeros
            # plot_property_distribution(
            #     self.qcd_raw_vals[prop], self.wjet_raw_vals[prop], prop,
            #     self.label_bg, self.label_sg,
            #     ax=axes[0], is_scaled=False, include_zeros=True
            # )

            # # Raw, excluding zeros
            # plot_property_distribution(
            #     self.qcd_raw_vals[prop], self.wjet_raw_vals[prop], prop,
            #     self.label_bg, self.label_sg,
            #     ax=axes[1], is_scaled=False, include_zeros=False,
            #     scaled_zero1=0.0, scaled_zero2=0.0
            # )

            # Scaled, with zeros
            plot_property_distribution(
                self.qcd_scaled_vals[prop], self.wjet_scaled_vals[prop], prop,
                self.label_bg, self.label_sg,
                ax=axes[0], is_scaled=True, include_zeros=True
            )

            # Scaled, excluding zeros
            plot_property_distribution(
                self.qcd_scaled_vals[prop], self.wjet_scaled_vals[prop], prop,
                self.label_bg, self.label_sg,
                ax=axes[1], is_scaled=True, include_zeros=False,
                scaled_zero1=self.zero1[prop], scaled_zero2=self.zero2[prop]
            )

            plt.tight_layout()
            if self.DISPLAY_PLOT: plt.show()
            fig_path = f"plots/proc_distr_{prop}_{self.label_bg}+{self.label_sg}_{helpers_main.curr_time()}.png"
            plt.savefig(fig_path)
            logging.info(f"Saved figure into {fig_path}")

    def save_data(self):
        # Save data as .pkl!
        output_folder = os.path.join(config["data"]["processed_data_dir"], "scaledby_" + self.label_bg)
        qcd_output_path  = os.path.join(output_folder, self.label_bg + "_scaled.pkl")
        wjet_output_path = os.path.join(output_folder, self.label_sg + "_scaled.pkl")
        logging.info("Now, saving data!")
        self.qcd_scaled.to_pickle(qcd_output_path),   logging.info(f"Saved in {qcd_output_path}!")
        self.wjet_scaled.to_pickle(wjet_output_path), logging.info(f"Saved in {wjet_output_path}!")


def visualize(qcd_scaled_vals, wjet_scaled_vals, label_bg, label_sg, props):
    # Plot raw and scaled distributions for selected variable(s)
    logging.info("Plotting data:")
    for prop in props:
        fig, axes = plt.subplots(1, 1, figsize=(20, 5))
        
        plot_property_distribution(
            # ISSUE: this uses vals, not actual scaled
            qcd_scaled_vals[prop], wjet_scaled_vals[prop], prop,
            label_bg, label_sg,
            ax=axes, is_scaled=True, include_zeros=True
        )

        plt.tight_layout()
        fig_path = f"plots/procd_viz-{prop}_{label_bg}+{label_sg}_{helpers_main.curr_time()}.png"
        plt.savefig(fig_path)
        logging.info(f"Saved figure into {fig_path}")


def main(args):
    proc = DataProcessor(args)

    proc.load_modify()

    proc.scale_data()

    proc.visualize_data()

    proc.save_data()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Process",
        description="processes a background + signal pair of preprocessed jets, e.g. QCD and WJet"
    )
    parser.add_argument(
        "--background", "--bg", "-b", type=str, required=False, default=config["data"]["preprocessed_qcd"],
        help="File/folder path to the background preprocessed data (QCD). Default: uses the path in configs/config.yaml"
    )
    parser.add_argument(
        "--signal", "--sg", "-s", type=str, required=False, default=config["data"]["preprocessed_wjet"],
        help="File/folder path to the background signal data (WJet). Default: uses the path in configs/config.yaml"
    )
    parser.add_argument(
        "--label_bg", "--lb", "-B", type=str, required=False, default="QCD",
        help="Label/name for the background jet, usually with the jet Pt range (e.g. QCD_Pt1400to1800)"
    )
    parser.add_argument(
        "--label_sg", "--ls", "-S", type=str, required=False, default="WJet",
        help="Label/name for the signal jet, usually with the jet HT range (e.g. WJet_HT1400to1800)"
    )
    parser.add_argument(
        "--filter", "-f", required=False, default=False, action=argparse.BooleanOptionalAction,
        help="If provided, use the labels as FILTERS within the data folders. i.e. only files containing the label will be processed (to single out one type of jet)"
    )
    args = parser.parse_args()

    if config["dbg"]["measure_perf"]:
        helpers_main.profile_func(f"logs/proc_{helpers_main.curr_time()}.prof", main, args)
    else:
        main(args)
