from coffea.nanoevents import NanoEventsFactory, PFNanoAODSchema
import numpy as np
import awkward as ak
from fast_histogram import histogram2d
import os
import sys
import pandas as pd 
from tqdm import tqdm
import argparse
import logging
import matplotlib.pyplot as plt

# Add parent directory to import local project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from helpers import helpers_main
config = helpers_main.load_config()
helpers_main.log_config(f"logs/plotpt_{helpers_main.curr_time()}.log")

from scripts.preprocessing import get_fatjets
from visualize.plot_property_distributions import plot_property_distribution


def plot_pt(qcd_events, wjet_events):
    fig, axes = plt.subplots(1, 2, figsize=(20, 5))

    qcd_fatjets, qcd_pfcands = get_fatjets(qcd_events)
    wjet_fatjets, wjet_pfcands = get_fatjets(wjet_events)
    qcd_pt  = get_pt(qcd_fatjets)
    wjet_pt = get_pt(wjet_fatjets)

    plot_property_distribution(
        qcd_pt, wjet_pt, "Pt before scaling",
        "QCD", "WJet",
        ax=axes[0], is_scaled=False, include_zeros=True)
    
    plt.savefig(f"plots/{helpers_main.curr_time()}.png")


def plot_raw_pt(events):
    fatjet_pts = ak.flatten(events.FatJet).pt.to_numpy()
    counts, bins = np.histogram(fatjet_pts, bins=100)
    plt.stairs(counts, bins)
    plt.show()
    plt.savefig(f"plots/{helpers_main.curr_time()}.png")

def get_pt(fatjets):
    logging.info(f"{fatjets=}")
    if isinstance(fatjets, int): return np.array([0])
    return fatjets["pt"].to_numpy().flatten()


def load_events(qcd_path, wjet_path):
    return (
        NanoEventsFactory.from_root(qcd_path, schemaclass = PFNanoAODSchema).events(),
        NanoEventsFactory.from_root(wjet_path, schemaclass = PFNanoAODSchema).events()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Plot Pt",
        description="plots the Pts for preprocessing, processing, and post-processing"
    )
    parser.add_argument(
        "--qcdpath", type=str, required=True,
        help="path to a single raw .root QCD file (combine with helpers_main/join_dfs.py if needed)"
    )
    parser.add_argument(
        "--wjetpath", type=str, required=True,
        help="path to a single raw .root WJet file (make sure its HT is around double that of QCD's Pt!)"
    )

    args = parser.parse_args()

    qcd, wjet = load_events(args.qcdpath, args.wjetpath)
    plot_pt(qcd, wjet)













