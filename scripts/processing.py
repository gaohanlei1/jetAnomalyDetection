"""
Preprocess and visualize raw jet data for anomaly detection.

This script:
- Loads pickled jet data for two classes (e.g. background: QCD, signal: WJets).
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

# Add parent directory to import local project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocess.feature_engineering import modify_df
from preprocess.scaling import find_scalers, apply_scalers
from visualize.plot_property_distributions import plot_property_distribution

# === SETTINGS ===
# Load YAML configuration for data and hyperparameters
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Path to folder containing preprocessed raw data
data_folder_path = config["data"]["preprocessed_data_dir"]
# note: if you are running on cern resources, use your "eos" path here for the data

# Dataset labels (used for file naming and plots)
qcd_label  = 'QCD600to700'
wjet_label = 'WJET600to700'

# File paths for each dataset (assumes single file for each class)
datatype1_pkl_files = [data_folder_path + 'qcd.pkl']    # qcd
datatype2_pkl_files = [data_folder_path + 'wjet.pkl']   # wjet

# PDG IDs to consider as valid charged particles
valid_pdg = [-11, 11, -13, 13, -211, 211]

# === LOAD DATA ===

# Load and concatenate all pickled QCD and WJET files
datatype1_raw = pd.concat(
    [pd.read_pickle(f) for f in tqdm(datatype1_pkl_files, desc=f'Loading {qcd_label}')],
    ignore_index=True
)
datatype2_raw = pd.concat(
    [pd.read_pickle(f) for f in tqdm(datatype2_pkl_files, desc=f'Loading {wjet_label}')],
    ignore_index=True
)

print(f"{qcd_label} data length: {len(datatype1_raw)}")
print(f"{wjet_label} data length: {len(datatype2_raw)}")

# === FEATURE ENGINEERING ===

# Compute derived features, apply one-hot encoding, and filter PFCands
datatype1 = modify_df(datatype1_raw.copy(), valid_pdg)
datatype2 = modify_df(datatype2_raw.copy(), valid_pdg)

# Drop rows with missing or invalid entries
datatype1.dropna(inplace=True)
datatype2.dropna(inplace=True)

# === SCALING ===

# Select variables to scale (assumes first 17 are metadata or unscaled base features)
variables_to_analyze = datatype1.columns[17:]

# Compute robust scaling values using QCD dataset
scaler_dict = find_scalers(datatype1.copy(), qcd_label, cols=variables_to_analyze)

# Apply scaling to both datasets using QCD-derived scalers
datatype1_scaled, data1_scaled_vals, data1_raw_vals, zero1 = apply_scalers(datatype1.copy(), scaler_dict)
datatype2_scaled, data2_scaled_vals, data2_raw_vals, zero2 = apply_scalers(datatype2.copy(), scaler_dict)

# === VISUALIZATION ===

# Plot raw and scaled distributions for selected variable(s)
for prop in ['log_pt']:  # Modify this list to include other features as needed
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Raw, with zeros
    plot_property_distribution(data1_raw_vals[prop], data2_raw_vals[prop], prop,
                               qcd_label, wjet_label,
                               ax=axes[0], is_scaled=False, include_zeros=True)

    # Raw, excluding zeros
    plot_property_distribution(data1_raw_vals[prop], data2_raw_vals[prop], prop,
                               qcd_label, wjet_label,
                               ax=axes[1], is_scaled=False, include_zeros=False,
                               scaled_zero1=0.0, scaled_zero2=0.0)

    # Scaled, with zeros
    plot_property_distribution(data1_scaled_vals[prop], data2_scaled_vals[prop], prop,
                               qcd_label, wjet_label,
                               ax=axes[2], is_scaled=True, include_zeros=True)

    # Scaled, excluding zeros
    plot_property_distribution(data1_scaled_vals[prop], data2_scaled_vals[prop], prop,
                               qcd_label, wjet_label,
                               ax=axes[3], is_scaled=True, include_zeros=False,
                               scaled_zero1=zero1[prop], scaled_zero2=zero2[prop])

    plt.tight_layout()
    plt.show()

datatype1_scaled.to_pickle(f'./data/processed/{qcd_label}_scaled.pkl')
datatype2_scaled.to_pickle(f'./data/processed/{wjet_label}_scaled.pkl')
