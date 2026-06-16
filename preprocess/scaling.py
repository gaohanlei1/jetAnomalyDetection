"""
Feature scaling utilities for jet constituent data.

This module standardizes per-feature arrays using percentile-based scaling (16th to 84th percentile)
to minimize sensitivity to outliers. It returns transformed values along with mappings for 
original and scaled data, useful for analysis and visualization.

Functions:
- find_scalers: Computes 16th and 84th percentile values for each feature.
- apply_scalers: Applies scaling to features using computed percentiles.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from tqdm import tqdm
import logging

import sys
import os

# Add parent directory to import local project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import constants as c
from helpers import helpers_main

def find_scalers(df: pd.DataFrame, df_label: str, cols: List[str]) -> Dict[str, np.ndarray]:
    """
    Compute scaling values (16th and 84th percentiles) for each feature column.

    Args:
        df (pd.DataFrame): DataFrame with array-valued columns (per-jet particle features).
        df_label (str): Label for progress tracking (e.g., "QCD600to700").
        cols (List[str]): List of column names to compute scalers for.

    Returns:
        Dict[str, np.ndarray]: Dictionary mapping feature names to (16th, 84th) percentiles.
                               PDG one-hot columns are skipped and mapped to [-1].
    """
    scaler_dict = {}
    for col in cols:
        logging.info(f"In find_scalers, {col=}")
        if col.startswith("pdg") or col.startswith(c.RAW_FATJET_PROPERTIES_PREFIX) or col.startswith("FatJet_particleNetMD"):   # compatibility w/ Arjun's files
            # Skip scaling for PDG one-hot columns and fatjet metadata
            scaler_dict[col] = [-1]
        else:
            flattened = pd.to_numeric(df[col].explode(), errors="coerce").to_numpy()
            valid_values = np.array([
                item for item in tqdm(
                    flattened,
                    desc=f"Finding scalers for {df_label} - {col}"
                )
                if np.isfinite(item) and item != 0.0
            ])

            if valid_values.size == 0:
                logging.warning(
                    f"Preserving {col} without scaling because the QCD "
                    "reference contains no finite non-zero values."
                )
                scaler_dict[col] = None
            else:
                percentiles = np.percentile(valid_values, [16, 84])
                if np.isclose(percentiles[0], percentiles[1]):
                    logging.warning(
                        f"Preserving {col} without scaling because its QCD "
                        "16th and 84th percentiles are identical."
                    )
                    scaler_dict[col] = None
                else:
                    scaler_dict[col] = percentiles

    return scaler_dict


def apply_scalers(df: pd.DataFrame, scaler_dict: Dict[str, np.ndarray]) -> Tuple[pd.DataFrame, dict, dict, dict]:
    """
    Apply standardization to the dataset using precomputed percentile-based scalers.

    Args:
        df (pd.DataFrame): Input DataFrame with array-valued features.
        scaler_dict (Dict[str, np.ndarray]): Output of `find_scalers`.

    Returns:
        Tuple containing:
            - df (pd.DataFrame): Scaled version of input DataFrame.
            - data_dict (dict): Flattened scaled values for each column.
            - org_data_dict (dict): Flattened original (unscaled) values for each column.
            - scaled_zero (dict): Value of 0.0 after scaling, useful for visualizing zero filtering.
    """
    data_dict = {}       # Flattened scaled values
    org_data_dict = {}   # Flattened original values
    scaled_zero = {}     # Value of 0.0 after scaling

    for col in df.columns:
        if col.startswith(c.RAW_FATJET_PROPERTIES_PREFIX):
            logging.info(f"Preserving {col=}")
        elif col not in scaler_dict:
            logging.info(f"Skipping {col=}, not in scaler")
            continue

        logging.info(f'Standardising {col}')
        flattened_list = df[col].explode()
        indices = [i for i, item in enumerate(flattened_list) if item != 0.0]
        org_data_dict[col] = flattened_list

        if col.startswith("pdg"):
            # For one-hot columns, preserve raw values
            data_dict[col] = [
                item for sublist in df[col]
                for item in np.array(sublist).flatten()
            ]
            scaled_zero[col] = np.nan
        elif col.startswith(c.RAW_FATJET_PROPERTIES_PREFIX):
            data_dict[col] = df[col]
            org_data_dict[col] = df[col]
            scaled_zero[col] = np.nan
        elif scaler_dict[col] is None:
            logging.info(f"Preserving uninformative {col=}")
            data_dict[col] = [
                item for sublist in df[col]
                for item in np.array(sublist).flatten()
            ]
            scaled_zero[col] = 0.0
        else:
            per_minus_36, per_plus_36 = scaler_dict[col]
            denominator = per_plus_36 - per_minus_36 + 1e-6
            df[col] = df.apply(lambda row: (
                (((np.array(row[col]).reshape(-1, 1)) - per_minus_36) / denominator) * 2 - 1
            ).flatten(), axis=1)
            zero = ((0.0 - per_minus_36) / denominator) * 2 - 1
            scaled_zero[col] = zero
            data_dict[col] = [item for sublist in df[col] for item in sublist]            

    # Reattach unscaled fj_ columns
    for col in df.columns:
        if col.startswith("fj_") and col not in data_dict:
            data_dict[col] = df[col]
            org_data_dict[col] = df[col]
            scaled_zero[col] = np.nan

    return df, data_dict, org_data_dict, scaled_zero
