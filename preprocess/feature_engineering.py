"""
Feature engineering utilities for jet constituent data.

This module performs:
- Derived variable computation (e.g., d0/d0Err, dz/dzErr, log(pt), dR).
- Charged particle filtering using `within_bounds`.
- One-hot encoding of PDG IDs.
- Row-wise filtering of DataFrame to keep only valid PFCands.

All transformations are designed for preprocessing LHC jet datasets for ML pipelines.
"""

import numpy as np
import pandas as pd
from typing import List
import logging

import sys
import os

# Add parent directory to import local project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import constants as c
from helpers import helpers_main
helpers_main.log_config(f"logs/proc_feature_{helpers_main.curr_time()}.log")

def calculate_d_over_dErr(row: pd.Series, label: str, valid_pdg: List[str]) -> np.ndarray:
    """
    Compute signed impact parameter significance: d / dErr, clipped and filtered.

    Args:
        row (pd.Series): A row of the DataFrame containing arrays of particle values.
        label (str): Either 'dz', 'dxy', or 'd0'.
        valid_pdg (List[str]): List of PDG IDs that are considered valid (charged particles).

    Returns:
        np.ndarray: Array of d/dErr values, clipped and filtered by PDG and error sign.
    """
    if label not in ['dz', 'dxy', 'd0']:
        raise ValueError(f"Invalid label {label}. Choose from 'dz', 'dxy', 'd0'.")

    arr = np.array(row[label]) / np.array(row[label + 'Err'])
    arr = np.where(np.array(row[label + 'Err']) >= 0.0, arr, 0.0)
    arr = np.where(np.isin(np.array(row['pdgId']), valid_pdg), arr, 0.0)
    arr = np.clip(arr, -5.0, 5.0)
    return arr


def calculate_dR(row: pd.Series) -> np.ndarray:
    """
    Calculate dR = sqrt(eta^2 + phi^2) for each PFCand.

    Only includes particles with puppiWeight == 1.0.

    Args:
        row (pd.Series): A row of the DataFrame.

    Returns:
        np.ndarray: Array of dR values for each particle.
    """
    return np.sqrt(np.array(row['eta'])**2 + np.array(row['phi'])**2)


def within_bounds(row: pd.Series) -> np.ndarray:
    """
    Identify indices of PFCands within acceptable detector bounds.

    Conditions:
    - dR ≤ 0.8
    - puppiWeight == 1.0

    Args:
        row (pd.Series): A row of the DataFrame.

    Returns:
        np.ndarray: Indices of valid particles within bounds.
    """
    return np.intersect1d(
        np.where(np.abs(np.array(row['dR'])) <= 0.8)[0],
        np.where(np.array(row['puppiWeight']) == 1.0)[0]
    )


def filter_row(row: pd.Series, indices: np.ndarray) -> pd.Series:
    """
    Filter all array fields in a DataFrame row to keep only values at given indices.

    Args:
        row (pd.Series): Original row with all particle arrays.
        indices (np.ndarray): Indices to keep.

    Returns:
        pd.Series: Filtered row.
    """
    for col in row.index:
        if not(col == "within_bounds" or col.startswith(c.RAW_FATJET_PROPERTIES_PREFIX)):
            row[col] = np.array(row[col])[indices].flatten()
    return row


def one_hot_encode_pdgId(row: pd.Series, pdg_ids: List[int]) -> pd.Series:
    """
    One-hot encode the PDG ID for each particle in a row.

    Args:
        row (pd.Series): Row with a 'pdgId' field containing an array of PDG codes.
        pdg_ids (List[int]): Full list of unique PDG IDs across dataset.

    Returns:
        pd.Series: Row with additional columns `pdgId_<value>` for each PDG ID.
    """
    # this part takes the most time, since it's run `len(pdg_ids) * len(df)` times! (~6 * 18000)
    pdg_array = np.array(row['pdgId'])
    # logging.debug(f"Entered one-hot, {len(pdg_array)=}")
    for pdg_id in pdg_ids:
        row[f'pdgId_{pdg_id}'] = (pdg_array == pdg_id).astype(int)
    return row


def modify_df(df: pd.DataFrame, pdg: List[str]) -> pd.DataFrame:
    """
    Perform full preprocessing pipeline on raw jet data.

    Steps:
    - Compute dz/dzErr and d0/d0Err significance variables.
    - Calculate dR and identify valid PFCands.
    - Apply log(pt) transformation.
    - One-hot encode PDG IDs.
    - Filter out-of-bound particles.

    Args:
        df (pd.DataFrame): Raw jet data with arrays of particle features per row.
        pdg (List[str]): List of valid PDG IDs to keep (charged particles).

    Returns:
        pd.DataFrame: Fully processed DataFrame ready for graph construction.
    """

    # the data is represented as a list of particles for each event, so len(df.pt) == len(df.pdgIg) == ...
    
    logging.info(f"Entered modify_df! {len(df)=}. Now calculating dz/dzErr...")
    helpers_main.secs_since_last_ping()
    # takes some time
    df['dz/dzErr'] = df.apply(lambda row: calculate_d_over_dErr(row, label='dz', valid_pdg=pdg), axis=1)
    logging.info(f"Found dz/dzErr {helpers_main.time_taken()}")

    df['d0/d0Err'] = df.apply(lambda row: calculate_d_over_dErr(row, label='d0', valid_pdg=pdg), axis=1)
    logging.info(f"Found d0/d0Err {helpers_main.time_taken()}")

    df['dR'] = df.apply(calculate_dR, axis=1)
    logging.info(f"Found dR {helpers_main.time_taken()}")

    df['within_bounds'] = df.apply(within_bounds, axis=1)
    logging.info(f"Found within_bounds {helpers_main.time_taken()}, extracting jet-level metadata...")

    df['log_pt'] = df.apply(lambda row: np.log(np.array(row['pt'])), axis=1)
    logging.info(f"Found log_pt {helpers_main.time_taken()}, calculating one-hot lists...")

    # Compute one-hot encodings using full PDG set
    unique_pdg_ids = sorted(df['pdgId'].explode().unique().tolist())
    # TAKES THE MOST TIME!!!
    df = df.apply(lambda row: one_hot_encode_pdgId(row, unique_pdg_ids), axis=1)
    logging.info(f"Found one-hot lists {helpers_main.time_taken()}, now filtering out-of-bounds particles...")

    # Filter out particles outside bounds
    df = df.apply(lambda row: filter_row(row, row['within_bounds']), axis=1)
    logging.info(f"Done! {helpers_main.time_taken()}")

    return df
