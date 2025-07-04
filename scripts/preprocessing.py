from coffea.nanoevents import NanoEventsFactory, PFNanoAODSchema
import numpy as np
import awkward as ak
from fast_histogram import histogram2d
import os
import sys
import pandas as pd 
from tqdm import tqdm 
# import warnings
import argparse
import logging

# Add parent directory to import local project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import constants as c
from helpers import helpers
config = helpers.load_config()

MEASURE_PERF = config["dbg"]["measure_perf"]
if MEASURE_PERF:
    from cProfile import Profile
    import pstats

# after preprocessing, move the raw data file into a subfolder?
MOVE_AFTERWARDS = True
# number of preprocessed events per saved file; 0 for no divisions
#   this program processes like 5 events/s
EVENTS_PER_FILE = 10000

def get_fatjets(events): 
    fatjets = events.FatJet
    # store unmasked fjs
    store_fj = [fj for fj in fatjets[0]]

    # accept jets that do not have electrons or muons nearby   
    electrons = events.Electron
    electrons = fatjets.nearest(electrons[electrons.pt > c.ELECTRON_PT_LOWER_BOUND])
    muons = events.Muon
    muons = fatjets.nearest(muons[muons.pt > c.MUON_PT_LOWER_BOUND])

    mask = (
        (ak.fill_none(fatjets.delta_r(electrons) > c.ELECTRON_R_LOWER_BOUND, True)) &
        (ak.fill_none(fatjets.delta_r(muons) > c.MUON_R_LOWER_BOUND, True)) & 
        (~ak.is_none(fatjets.matched_gen, axis=1)) & 
        (fatjets.delta_r(fatjets.matched_gen) < c.MATCHED_GEN_R_LOWER_BOUND)& 
        (fatjets.pt > c.FATJET_PT_LOWER_BOUND) &
        (abs(fatjets.eta) < c.FATJET_ETA_BOUNDS) &
        (ak.num(fatjets) > 0) &
        (~ak.is_none(fatjets))
    )

    fatjets = fatjets[mask]
    sort_i = ak.argsort(fatjets.pt, axis=1)

    if len(fatjets) == 0 or len(fatjets[0]) == 0 or (len(fatjets) == 1 and fatjets[0][0] is None) or len(sort_i[0]) == 0:
        # logging.warning(f"Skipped fatjet after masking: {fatjets}")
        return -1, -1
    
    fatjets = ak.firsts(fatjets[sort_i])

    for j, fj in enumerate(store_fj):
        if (
            abs(fj.eta - fatjets.eta[0]) < c.FATJET_DELTA_ETA_BOUND and
            abs(fj.phi - fatjets.phi[0]) < c.FATJET_DELTA_PHI_BOUND and
            abs(fj.pt - fatjets.pt[0]) < c.FATJET_DELTA_PT_BOUND
        ):
            pfcs = [pfcand["pFCandsIdx"] for pfcand in events.FatJetPFCands[0] if pfcand["jetIdx"] == j]

    return fatjets, pfcs

def process_event_root(events):
    fatjets, pfcs = get_fatjets(events)
    
    if isinstance(fatjets, int):
        return -1, -1
    pfcands = events.PFCands

    eta = ak.to_numpy(pfcands["phi"] - fatjets["phi"]).flatten()[pfcs]
    phi = ak.to_numpy(pfcands["eta"] - fatjets["eta"]).flatten()[pfcs]
    pt  = ak.to_numpy(pfcands["pt"]/fatjets["pt"]).flatten()[pfcs]
    # TODO: old ratio based on whether it is qcd or wjet -> this is not model agnostic !!!
      # check that current pt scheme is correct

    properties = [pt, eta, phi]
    property_names = ["pt", "eta", "phi"]

    # add all other fields
    fields = pfcands.fields
    for field in fields: 
        if field not in ("pt", "eta", "phi"):
            properties.append(ak.to_numpy(pfcands[field]).flatten()[pfcs])
            property_names.append(field)

    return properties, property_names

def load_root(filepath):
    data = {}
    logging.info(f"Loading root, {filepath=}")
    
    if os.path.splitext((filepath))[-1] == ".root" and os.path.isfile(filepath):
        events = NanoEventsFactory.from_root(filepath, schemaclass = PFNanoAODSchema).events()

        for i in tqdm(range(len(events))):
            logging.info(f"{i=}")
            # if EVENT_NUM and i >= EVENT_NUM:
            if i >= 100:
                # logging.info(f"{EVENT_NUM} events reached.")
                return pd.DataFrame.from_dict(data)

            properties, property_names = process_event_root(events[i:i+1])
            if properties == -1: continue

            if not data: 
                data = {property_name: [] for property_name in property_names}
            
            for i, prop in enumerate(properties): 
                data[property_names[i]].append(prop)

            logging.info(f"Processed!")

    return pd.DataFrame.from_dict(data)

# def load_h5():
#     raise NotImplementedError

def preprocess_file(data_filename, jet_type):
    logging.info(f"Now preprocessing {data_filename} ({jet_type})")
    data_folder_path = config["data"]["raw_" + jet_type]
    data_file_path = data_folder_path + data_filename
    output_file_path = f"{config['data']['preprocessed_data_dir']}/{jet_type}/{data_filename.replace('/', '').replace('.root','')}.pkl" 

    df = load_root(data_file_path)
    df.to_pickle(output_file_path)
    logging.info(f"Preprocessed {data_filename} ({data_type}) into {output_file_path}")
    
    if MOVE_AFTERWARDS:
        os.renames(data_file_path, data_folder_path + data_filename)

def main(data_filename, data_type):
    # TODO: should we actually care about this warning? 
    logging.warning("Found duplicate branch")

    # if filetype == ".h5":
    #     load_h5()
    # else:
    jet_type = config["data"][data_type]
    
    if data_filename is None:
        data_folder_path = config["data"]["raw_" + jet_type]
        for file in os.listdir(data_folder_path):
            if os.path.isfile(os.path.join(data_folder_path, file)) and os.path.splitext(file)[1] == ".root":
                preprocess_file(file, jet_type)
    else:
        preprocess_file(data_filename, jet_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Preprocess",
        description="preprocesses jet data for anomaly detection"
    )
    parser.add_argument(
        "--filename", type=str, required=False,
        help="name of file to preprocess; if none, preprocesses all files in the data folder (defined in configs/config.yaml)"
    )
    # parser.add_argument("--save_path", type=str, required=False, help="path where the processed data will be saved")
    parser.add_argument("--data_type", choices=["background", "signal"], required=True, help="'background' or 'signal'")
    # parser.add_argument("--file_type", choices=[".root", ".h5"], required=False, default=".root")

    args = parser.parse_args()
    data_filename = args.filename
    # save_path = args.save_path
    data_type = args.data_type
    # file_type = args.file_type

    session_name = f"preproc_{data_filename}_{data_type}_{helpers.curr_time()}"
    logging.basicConfig(
        filename = f"logs/{session_name}.log",
        level=config["dbg"]["logging_level"]
    )

    if MEASURE_PERF:
        with Profile() as prof:
            main(data_filename, data_type)
        
        stats = pstats.Stats(prof)
        stats.sort_stats(pstats.SortKey.TIME)
        stats.dump_stats(filename=f"logs/{session_name}.prof")
    else:
        main(data_filename, data_type)


# example call: 
# python preprocessing.py --data_path "/isilon/data/users/jpfeife2/AutoEncoder-Anomaly-Detection/data/WJET/400to600/nano_mc2018_1-1.root" --save_path "/isilon/data/users/jpfeife2/AutoEncoder-Anomaly-Detection/processed_data" --data_type "signal" --file_type ".root"
    
# background ="/isilon/data/users/jpfeife2/AutoEncoder-Anomaly-Detection/data/QCD/300to500/nano_mc2018_12_a677915bd61e6c9ff968b87c36658d9d_0.root"
# signal = "/isilon/data/users/jpfeife2/AutoEncoder-Anomaly-Detection/data/WJET/400to600/nano_mc2018_1-1.root"
# main(signal, "/isilon/data/users/jpfeife2/AutoEncoder-Anomaly-Detection/processed_data", "signal", ".root")
