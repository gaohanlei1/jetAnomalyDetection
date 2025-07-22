from coffea.nanoevents import NanoEventsFactory, PFNanoAODSchema
import numpy as np
import awkward as ak
from fast_histogram import histogram2d
import os
import sys
import pandas as pd 
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import uproot
import math

# Add parent directory to import local project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from helpers import helpers_main
config = helpers_main.load_config()

import logging
helpers_main.log_config(f"logs/plotpt_{helpers_main.curr_time()}.log")

from scripts.preprocessing import get_fatjets
from visualize import plot_property_distributions


def plot_distributions(data_list, prop_name, label_list=None, bins=150, include_zeros=True, scaled_zero=0.0):
    '''Plots and saves multiple datasets as overlapping histograms'''
    
    num = len(data_list)
    if len(label_list) != num: label_list = [None] * num
    plt.figure(figsize=(10, 6))

    plt.xlabel(prop_name)
    plt.ylabel("Density")
    plt.title(f"Distribution of {prop_name} for {num} datasets" + ("" if include_zeros else " (without scaled zeroes)"))

    bin_range = (0, 1)
    for i in range(len(data_list)):
        data = [
            item for item in data_list[i]
            if (include_zeros or not math.isclose(item, scaled_zero, rel_tol=1e-4, abs_tol=0.0))
            and not math.isnan(item)
        ]

        if len(data) == 0:
            logging.warning(f"{i}-th dataset ({label_list[i]}) has no data! Length before filtering NaNs and 0s: {len(data_list[i])}")
        else:
            bin_range = (np.min(data), np.max(data))
        
        plt.hist(data, bins=bins, range=bin_range, density=True, label=label_list[i], alpha=0.5)

    plt.legend(loc="upper right")
    plt.savefig(f"plots/distributions/hists_{prop_name}_{len(data_list)}_{helpers_main.curr_time()}.png")
    plt.clf()


def plot_pt(qcd_events, wjet_events):
    fig, axes = plt.subplots(1, 2, figsize=(20, 5))

    qcd_fj, qcd_pfc   = get_fatjets(qcd_events)
    wjet_fj, wjet_pfc = get_fatjets(wjet_events)
    qcd_pt  = get_pt_from_events(qcd_fj)
    wjet_pt = get_pt_from_events(wjet_fj)

    # before any preproc
    plot_property_distribution(
        qcd_pt, wjet_pt, "Pt before scaling",
        "QCD", "WJet",
        ax=axes[0], is_scaled=False, include_zeros=True
    )

    # after preproc
    # plot_property_distribution(
    #     qcd_pt, wjet_pt, "Pt before scaling",
    #     "QCD", "WJet",
    #     ax=axes[0], is_scaled=False, include_zeros=True
    # )

    # after proc



    # and scaled
    
    plt.savefig(f"plots/plotraw_coffea-pt_{len(qcd_events)}_{len(wjet_events)}_{helpers_main.curr_time()}.png")


def get_pt_from_root(filename, branch="FatJet_pt", treename="Events"):
    with uproot.open(filename) as file:
        return ak.flatten(file[treename][branch].array()).to_numpy()


def get_pt_from_events(fatjets):
    logging.info(f"{fatjets=}")
    if isinstance(fatjets, int): return np.array([0])
    # flattens out instances of multiple pts! can plot differences in this later
    return fatjets["pt"].to_numpy().flatten()


def load_events(root_path):
    return NanoEventsFactory.from_root(root_path, schemaclass = PFNanoAODSchema).events()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Plot Pt",
        description="plots the Pts for preprocessing (comparing uproot and coffea!), processing, and/or post-processing"
    )
    parser.add_argument(
        "--path1", "-q", type=str, required=True,
        help="path to a single data file (combine with helpers_main/join_dfs.py if needed), OR a folder of data files"
    )
    parser.add_argument(
        "--path2", "-w", type=str, required=False,
        help="(optional) path to 2nd data file (to plot both together); no folders! If provided, this will be plotted on top of the first file's distributions (TODO)"
    )
    parser.add_argument(
        "--type", "-t", choices=["raw", "preproc", "proc", "all"], default="raw",
        help=f"plot the raw .root distributions (using both uproot and coffea), the pickled distributions after preprocessing, and/or the pickled distributions after processing"
    )

    args = parser.parse_args()

    if args.path2 is not None:
        # working on this

        # plot qcd/wjet pair
        # Coffea loading and plotting
        qcd_ev, wjet_ev = load_events(args.path1), load_events(args.path2)
        plot_pt(qcd_ev, wjet_ev)
    else:
        if args.type in ("raw", "all"):
            files = [args.path1] if os.path.isfile(args.path1) else [
                os.path.join(args.path1, file) for file in os.listdir(args.path1)
                if os.path.isfile(os.path.join(args.path1, file))
            ]

            for file in files:
                logging.info(f"Now plotting {file=}!")

                filename = helpers_main.get_trimmed_name(file)
                root_pt = get_pt_from_root(file)
                evs_pt  = get_pt_from_events(get_fatjets(load_events(file))[0])
                
                plot_distributions([root_pt, evs_pt], "Pt", label_list=[filename + "_uproot", filename + "_events"])



