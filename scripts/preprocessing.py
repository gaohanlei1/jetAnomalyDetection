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

'''
NOTE!
I'm suppressing like 50 warnings that come up everytime I run the program,
which all complain about duplicate branches in the .root files used to load events.
Coffea only takes the first instance of each duplicate branch
- dunno if this is an issue?
'''
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="coffea")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="coffea")

from multiprocessing import Pool, Manager, cpu_count
# brux keeps freezing if I try to do cpu-1?
CPUS = (cpu_count() * 3)//4

# Add parent directory to import local project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import constants as c
from helpers import helpers_main
from helpers import join_dfs
config = helpers_main.load_config()

'''
Possible improvements: classes

Notes:
- the treenames in the root files: ['tag;1', 'Events;1', 'LuminosityBlocks;1', 'Runs;1', 'MetaData;1', 'ParameterSets;1']
- branchnames in ../helpers/raw_data_info
'''

# class DataProcessor:

#     def __init__(self):
#         args = parser.parse_args()
#         self.path = args.path
#         self.type = args.type
#         self.upperpt, self.lowerpt = args.upperpt, args.lowerpt
#         self.subfolder = args.subfolder




def get_fatjets(events): 
    fatjets = events.FatJet
    # store unmasked fjs
    store_fj = [fj for fj in fatjets[0]]
    logging.debug(f"{fatjets.fields=}")

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
        # (fatjets.pt > c.FATJET_PT_LOWER_BOUND) &
        (fatjets.pt > 200) &
        (fatjets.pt < 300) &
        (abs(fatjets.eta) < c.FATJET_ETA_BOUNDS) &
        (ak.num(fatjets) > 0) &
        (~ak.is_none(fatjets))
    )

    # log after each mask!!

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
    
    eta = ak.flatten(pfcands["phi"] - fatjets["phi"])[pfcs]
    phi = ak.flatten(pfcands["eta"] - fatjets["eta"])[pfcs]
    pt  = ak.flatten(pfcands["pt"]/fatjets["pt"])[pfcs]
    # TODO: old ratio based on whether it is qcd or wjet -> this is not model agnostic !!!
      # check that current pt scheme is correct

    properties = [pt, eta, phi]
    property_names = ["pt", "eta", "phi"]

    # add all other fields
    fields = pfcands.fields
    for field in fields: 
        if field not in ("pt", "eta", "phi"):
            # logging.debug(f"Added {field=} to property_names")
            properties.append(ak.flatten(pfcands[field])[pfcs])
            property_names.append(field)

    return properties, property_names

def load_root(filepath, jet_type, subfolder):
    logging.info(f"Now preprocessing {filepath}")

    # - So, I'll be doing sth very stupid here: loading the same file multiple times
    # - coz pickle can't pickle coffea events for whatever reason
    # if EVENT_LIMIT: events = events[:EVENT_LIMIT]
    num_events = len(NanoEventsFactory.from_root(filepath, schemaclass = PFNanoAODSchema).events())
    logging.info(f"{num_events} events in total, splitting into {CPUS} chunks.")
    start_i = np.linspace(0, num_events, endpoint=False, dtype=int, num=CPUS)
    end_i   = np.append(start_i[1:], num_events)

    with (Pool(CPUS) as pool, Manager() as m):
        logging.info(f"Intialised pool {pool} with {CPUS} CPUs.")

        pid_list = m.list()
        pid_lock = m.Lock()
        shared_datas = [
            {
                "start" : start,
                "end"   : end,
                "jet_type"  : jet_type,
                "filepath" : filepath,
                "pid_list"  : pid_list,
                "pid_lock"  : pid_lock,
                "subfolder" : subfolder
            }
            for start, end in zip(start_i, end_i)
        ]

        # each invocation of the function will receive an ELT of the iterator, not a slice
        # the chunk is just loosely defined as the range over which to pick elts for each cpu
        pool.map(preproc_events_slice, shared_datas)

    if config["data"]["move_to_used"]:
        move_to_used(filepath)

    if subfolder:
        subfolder_path = os.path.join(config["data"]["preprocessed_" + jet_type], helpers_main.get_trimmed_name(filepath))
        join_dfs.concat_pkls(subfolder_path)
        logging.info(f"Concatenated into {subfolder_path}; you can delete the non-concat files!")

def move_to_used(filepath):
    new_path = os.path.join(os.path.dirname(filepath), "used", os.path.basename(filepath))
    os.renames(filepath, new_path)
    logging.info(f"Moved '{filepath}' to '{new_path}'.")

# def load_h5():
#     raise NotImplementedError

def preproc_events_slice(metadata):
    pid = os.getpid()
    with metadata["pid_lock"]:
        # no need to check whether it already exists, since duplicate pids aren't possible?
        metadata["pid_list"].append(pid)
        pid_posn = metadata["pid_list"].index(pid)

    curr_events_num = metadata["end"] - metadata["start"]
    logging.info(f"#{pid_posn} ({pid}): Received range {metadata['start']} to {metadata['end']} ({curr_events_num} events).")
    events = NanoEventsFactory.from_root(
        metadata['filepath'], schemaclass = PFNanoAODSchema
        ).events()[metadata['start'] : metadata['end']]
    data = {}

    logging.debug(f"{events.fields=}")
    
    # iterator = range(metadata['start'], metadata['end'])
    iterator = range(curr_events_num)
    iterator = iterator if config["dbg"]["only_one_progress_bar"] and pid_posn != 0 else tqdm(
        iterator, desc=f"Process {pid_posn}", position=pid_posn, leave=None
    )

    for i in iterator:
        properties, property_names = process_event_root(events[i:i+1])
        if properties == -1: continue

        if not data: 
            data = {property_name: [] for property_name in property_names}
        
        for j, prop in enumerate(properties): 
            data[property_names[j]].append(prop)

        # curr_i = i - metadata['start']
        # if curr_i != 0 and curr_i % 20 == 0: break
    
    logging.info(f"#{pid_posn} ({pid}):  Preprocessed last few events, from {metadata['start']} to {metadata['end']}!")
    save_df(data, metadata)

def save_df(data_dict, metadata):
    logging.info(f"Received new df, saving!")
    basename = helpers_main.get_trimmed_name(metadata["filepath"])
    output_file_path = os.path.join(
        config["data"]["preprocessed_" + metadata["jet_type"]],
        basename if metadata["subfolder"] else "",
        f"{basename}_{os.getpid()}_{helpers_main.curr_time()}.pkl"
    )
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    pd.DataFrame.from_dict(data_dict).to_pickle(output_file_path)
    logging.info(f"Preprocessed into {output_file_path}.")

def main(data_path, data_type, subfolder):
    # if filetype == ".h5":
    #     load_h5()
    # else:
    jet_type = config["data"][data_type]
    folder_path = config["data"]["raw_" + jet_type]
    
    # default, set to folder path defined in config
    if data_path is None:
        data_path = folder_path

    if not os.path.exists(data_path):
        # if it's a filename within default folder, use that
        data_path = os.path.join(folder_path, data_path)
        if not os.path.exists(data_path):
            raise Exception(f"Invalid path: {data_path=}\n{jet_type=}")

    # by now, guaranteed to be full file paths
    files = [data_path] if os.path.isfile(data_path) else [
        os.path.join(data_path, file)
        for file in os.listdir(data_path)
        if os.path.isfile(os.path.join(data_path, file))
    ]

    for file in files:
        logging.info(f"Next file for {data_path=}: '{file}'")
        if os.path.splitext(file)[1] == ".root":
            load_root(file, jet_type, subfolder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Preprocess",
        description="preprocesses jet data for anomaly detection"
    )
    parser.add_argument(
        "--path", "-p", "--filename", type=str, required=False,
        help="Path to single .root file to preprocess, or name of single file within the data folder, or path to folder containing multiple .root files with the same type/label (e.g. QCD170to300). Default: uses the path in configs/config.yaml"
    )
    parser.add_argument(
        "--type", "--data_type", "-t", choices=["background", "signal"], required=True,
        help=f"'background' ({config['data']['background']}) or 'signal' ({config['data']['signal']})"
    )
    parser.add_argument(
        "--subfolder", "-s", required=False, default=False, action=argparse.BooleanOptionalAction,
        help=f"If provided, save to subfolder within preprocessed directory. Also concatenates all .pkl files within that subfolder!\n(Make sure the subfolder name is unique)"
    )
    parser.add_argument(
        "--upperpt", "-B", required=False,
        help=f"upper bound on fatjet Pt?"
    )
    parser.add_argument(
        "--lowerpt", "-b", required=False,
        help=f"lower bound on fatjet Pt?"
    )

    args = parser.parse_args()

    session_name = f"preproc_{args.type}_{helpers_main.curr_time()}"
    helpers_main.log_config(f"logs/{session_name}.log")
    logging.info("Set up logger!")

    if config["dbg"]["measure_perf"]:
        helpers_main.profile_func(f"logs/{session_name}.prof", main, args.path, args.type, args.subfolder)
    else:
        main(args.path, args.type, args.subfolder)
