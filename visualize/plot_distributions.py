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
import constants as c
from helpers import helpers_main
config = helpers_main.load_config()

import logging
helpers_main.log_config(f"logs/plotpt_{helpers_main.curr_time()}.log")

from scripts.preprocessing import get_fatjets, process_event_root
from visualize import plot_property_distributions

'''
Usage:
`python3.9 visualize/plot_distributions.py -q data/raw/path/` to plot the raw data distributions of the ROOT files in the path

To-dos:
- support to compare different files, e.g. raw vs preprocessed vs processed data
'''

def plot_distributions(
    data_list, prop_name, long_label=None, filename=None, label_list=None,
    bins=150, include_zeros=True, scaled_zero=0.0, bin_range=None
):
    '''Plots and saves multiple datasets as overlapping histograms'''
    if long_label is None: long_label = prop_name
    if filename is None: filename = ''

    num = len(data_list)
    if len(label_list) != num: label_list = [None] * num
    plt.figure(figsize=(10, 6))

    plt.xlabel(prop_name)
    plt.ylabel("Density")
    plt.title(f"Distribution of '{long_label}' for {num} datasets" + ("" if include_zeros else " (without scaled zeroes)"))

    for i in range(len(data_list)):
        data = [
            item for item in data_list[i]
            if (include_zeros or not math.isclose(item, scaled_zero, rel_tol=1e-4, abs_tol=0.0))
            and not math.isnan(item)
        ]

        logging.info(f"{i}-th dataset {label_list[i]}'s length: {len(data)}! Length before filtering NaNs and 0s: {len(data_list[i])}")

        if len(data) == 0:
            logging.warning("No data for this one!")
            continue
        else:
            if bin_range is not None:
                if bin_range[0] is None: bin_range[0] = np.min(data)
                if bin_range[1] is None: bin_range[1] = np.max(data)
        
        plt.hist(data, bins=bins, range=bin_range, density=False, label=label_list[i], alpha=0.5)

    plt.legend(loc="upper right")
    fig_name = f"plots/distributions/hists_{prop_name}_{filename}_{len(data_list)}_{helpers_main.curr_time()}.png"
    helpers_main.create_missing_dir(fig_name)
    plt.savefig(fig_name)
    logging.info(f"Saved into '{fig_name}'")
    plt.clf()


def plot_pt(qcd_events, wjet_events):
    fig, axes = plt.subplots(1, 2, figsize=(20, 5))

    qcd_fj, qcd_pfc   = get_fatjets(qcd_events)
    wjet_fj, wjet_pfc = get_fatjets(wjet_events)
    qcd_pt  = get_prop_from_fatjets(qcd_fj)
    wjet_pt = get_prop_from_fatjets(wjet_fj)

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


def get_prop_from_root(filename, branch="pt", treename="Events"):
    if branch == "pt": branch = "FatJet_pt"
    with uproot.open(filename) as file:
        return ak.flatten(file[treename][branch].array()).to_numpy()


def get_prop_from_fatjets(fatjets, prop="pt"):
    logging.debug(f"{fatjets=}")
    if isinstance(fatjets, int): return np.array([0])
    # flattens out instances of multiple pts! can plot differences in this later
    return fatjets[prop].to_numpy().flatten()


def load_events(root_path):
    return NanoEventsFactory.from_root(root_path, schemaclass = PFNanoAODSchema).events()

# def preproc(events):
#     data = {}
#     for i in tqdm(range(len(events))):
#         properties, property_names = process_event_root(events[i:i+1])
#         if properties == -1: continue

#         if not data: 
#             data = {property_name: [] for property_name in property_names}
        
#         for j, prop in enumerate(properties): 
#             data[property_names[j]].append(prop)
#     return data["pt"]

def corresponding_preproc(raw_file_name, proc_files):
    '''Gets the preprocessed data file corresponding to this raw data file'''
    for file in proc_files:
        if helpers_main.trim_name(raw_file_name) in file: return file 
    logging.warning(f"{raw_file_name=} not found within processed files - skipping!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Plot Pt",
        description="Plots Pts of different characteristics (FatJets, FatJetPFCands, PFCands) at different processing stages (raw, preproc, proc)"
    )
    parser.add_argument(
        "--path1", "-q", type=str, required=True,
        help="path to a single data file (combine with helpers_main/join_dfs.py if needed), OR a folder of data files"
    )
    parser.add_argument(
        "--path2", "-w", type=str, required=False,
        help="(optional) path to 2nd data file (to plot both together); if provided, no folders for path1 or path2! TODO"
    )
    parser.add_argument(
        "--type", "-t", choices=["raw", "preproc", "proc", "raw_fjlen"], default="raw",
        help=
        f"""'raw': raw pt with uproot vs coffea;
            'preproc': preproc flattened pt (ratio of FatJetPFCands.pt to FatJet.pt);
            (BELOW ARE TO-DOs)
            'proc': same as preproc, but modified and scaled (TODO: add include_zeros!);
            'raw_fjlen': raw pts with uproot/coffea, but also partitioned into the number of jets per event
            (LESS LIKELY TO-DOs)
            'raw_fj_pfcs': raw pts of FatJetPFCands
            'raw_pfcs': raw pts of PFCands
            'raw_pfcslen': raw lengths of PFCands"""
    )
    parser.add_argument(
        "--prop", "-p", type=str, default="pt",
        help="the property to graph"
    )
    parser.add_argument(
        "--lower", "-l", type=float, default=None,
        help="the lower limit for the x-axis"
    )
    parser.add_argument(
        "--upper", "-u", type=float, default=None,
        help="the upper limit for the y-axis"
    )

    args = parser.parse_args()

    if args.path2 is not None:
        pts = []
        for filepath in (args.path1, args.path2):
            ext = helpers_main.get_extension(filepath)
            if ext == ".root":
                pts.append(get_prop_from_root(filepath))
                # pts.append(ak.flatten(load_events(filepath).FatJet).pt.to_numpy())
            elif ext == ".pkl":
                pts.append(pd.read_pickle(filepath)["pt"].explode().to_numpy())
            else:
                raise Exception("Wrong filetype")
        
        plot_distributions(pts, "Pt", label_list=[helpers_main.trim_name(args.path1), helpers_main.trim_name(args.path2)])
    else:
        if args.type in ("raw", "raw_fjlen"):
            files = helpers_main.get_files(args.path1, extension=".root")
            for file in files:
                logging.info(f"Now plotting {file=}!") #, with {preproc_file=}!")

                filename = helpers_main.trim_name(file)
                # root_prop = get_prop_from_root(file, branch="FatJet_" + args.prop)
                ev = load_events(file)

                if args.type == "raw":
                    evs_raw_prop = ak.flatten(ev.FatJet)[args.prop].to_numpy()
                    plot_distributions(
                        # [root_prop, evs_raw_prop], args.prop,
                        # label_list=[filename + "_uproot", filename + "_coffea"],
                        [evs_raw_prop], args.prop, label_list=[filename + "_coffea"],
                        bin_range=[args.lower, args.upper]
                    )
                
                elif args.type == "raw_fjlen":
                    lens = np.array([len(fjs) for fjs in ev.FatJet])
                    len_arrs = []
                    max_fjs = max(lens)
                    for num_fjs in range(max_fjs):
                        len_arrs.append(ev.FatJet[lens == num_fjs].pt)
                        logging.info(f"Events with {num_fjs} sub-fjs: {len(len_arrs[-1])}")
                    
                    # plot the flattened distributions of the diff fjs
                    plot_distributions(
                        [ak.flatten(len_arr).to_numpy() for len_arr in len_arrs],
                        "FatJet-Pt-flattened",
                        label_list=[f"{i} fjs" for i in range(max_fjs)], filename=filename
                    )

                    # plot the distributions of the event fjs, averaged by event
                    plot_distributions(
                        [ak.mean(len_arr, axis=1).to_numpy() for len_arr in len_arrs],
                        "FatJet-Pt-mean",
                        label_list=[f"{i} fjs" for i in range(max_fjs)], filename=filename
                    )

                    # plot the distributions of the event fjs, summed by event
                    plot_distributions(
                        [ak.sum(len_arr, axis=1).to_numpy() for len_arr in len_arrs],
                        "FatJet-Pt-sum",
                        label_list=[f"{i} fjs" for i in range(max_fjs)], filename=filename
                    )
                

        elif args.type in ("preproc", "proc"):
            files = helpers_main.get_files(args.path1, extension=".pkl")
            for file in files:
                logging.info(f"Now plotting {file=}!") #, with {preproc_file=}!")
                filename = helpers_main.trim_name(file)
                pkl = pd.read_pickle(file)
                
                if args.prop not in pkl.columns.tolist():
                    logging.warning(f"Skipping no-{args.prop} {file=}")
                    continue
                props = pkl[args.prop].explode().to_numpy()

                plot_distributions([props], args.prop, label_list=[filename + "_" + args.prop])

