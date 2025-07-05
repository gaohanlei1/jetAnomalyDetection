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

from multiprocessing import Pool, Manager, cpu_count, Value, Lock
CPUS = cpu_count() - 1

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
EVENTS_PER_FILE = 100
# to limit the number of events; 0 for all events
EVENT_LIMIT = 0

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

def init_proc(value_arg):
    global value
    value = value_arg

def load_root(data_filename, jet_type):
    data_folder_path = config["data"]["raw_" + jet_type]
    data_file_path = os.path.join(data_folder_path, data_filename)
    output_file_basename = f"{os.path.splitext(os.path.basename(data_filename))[0]}"
    logging.info(f"Now preprocessing {data_file_path}")

    # should move this check outside
    if os.path.splitext((data_file_path))[-1] == ".root" and os.path.isfile(data_file_path):
        # - So, I'll be doing sth very stupid here: loading the same file multiple times
        # - Why? coz stupid ass pickle can't pickle coffea events for whatever reason,
        #  and coffea has fuckall documentation, so idk how to get length etc
        # - if this bugs you, you can try using pathos.multiprocessing, which might pickle
        # if EVENT_LIMIT: events = events[:EVENT_LIMIT]
        num_events = len(NanoEventsFactory.from_root(data_file_path, schemaclass = PFNanoAODSchema).events())
        logging.info(f"{num_events} events in total, splitting into {CPUS} chunks.")
        start_i = np.linspace(0, num_events, endpoint=False, dtype=int, num=CPUS)
        end_i   = np.append(start_i[1:], num_events)
        first_pid = Value('i', 0)
        split_events = [
            {
                "start" : start,
                "end"   : end,
                "filename" : data_filename,
                "jet_type" : jet_type,
                "file_path": data_file_path,
                "basename" : output_file_basename
            }
            for start, end in zip(start_i, end_i)
        ]

        with Pool(CPUS, initializer=init_proc, initargs=(first_pid,)) as pool:
            logging.info(f"Intialised pool {pool} with {CPUS} CPUs.")
            # each invocation of the function will receive an ELT of the iterator, not a slice
            # the chunk is just loosely defined as the range over which to pick elts for each cpu
            pool.map(preproc_events_slice, split_events)

# def load_h5():
#     raise NotImplementedError

def preproc_events_slice(event_range):
    with value.get_lock():
        if value.value == 0:
            value.value = os.getpid()

    logging.info(f"{os.getpid()}, parent {os.getppid()}: Received range {event_range['start']} to {event_range['end']} ({event_range['end'] - event_range['start']} events).")
    events = NanoEventsFactory.from_root(event_range['file_path'], schemaclass = PFNanoAODSchema).events()
    data = {}

    iterator = range(event_range['start'], event_range['end'])
    iterator = tqdm(iterator) if value.value == os.getpid() else iterator
    for i in iterator:
        properties, property_names = process_event_root(events[i:i+1])
        if properties == -1: continue

        if not data: 
            data = {property_name: [] for property_name in property_names}
        
        for j, prop in enumerate(properties): 
            data[property_names[j]].append(prop)
        
        if (i - event_range["start"]) != 0 and i % EVENTS_PER_FILE == 0:
            logging.info(f"Preprocessed {i - EVENTS_PER_FILE} to {i} ({EVENTS_PER_FILE} events)")
            save_df(data, event_range)
            data = {}
    
    save_df(data, event_range)

def save_df(data, event_range):
    logging.info(f"Received new df, saving!")
    output_file_path = os.path.join(
        config["data"]["preprocessed_data_dir"],
        f"{event_range['jet_type']}/{event_range['basename']}_{os.getpid()}_{helpers.curr_time()}.pkl"
    )
    pd.DataFrame.from_dict(data).to_pickle(output_file_path)
    logging.info(f"Preprocessed into {output_file_path}.")

def preprocess_file(data_filename, jet_type):
    load_root(data_filename, jet_type)

    if MOVE_AFTERWARDS:
        new_path = os.path.join(data_folder_path, config["data"]["used_raw_data_dir"], data_filename)
        os.renames(data_file_path, new_path)
        logging.info(f"Moved {data_file_path} to {new_path}.")

def main(data_filename, data_type):
    # if filetype == ".h5":
    #     load_h5()
    # else:
    jet_type = config["data"][data_type]
    
    if data_filename is None:
        data_folder_path = config["data"]["raw_" + jet_type]
        for file in os.listdir(data_folder_path):
            logging.info(f"Next file/dir in {data_folder_path}: '{file}'")
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

    session_name = f"preproc_{'all' if data_filename is None else data_filename}_{data_type}_{helpers.curr_time()}"
    logging.basicConfig(
        # filename = f"logs/{session_name}.log",
        handlers = [
            logging.FileHandler(f"logs/{session_name}.log"),
            logging.StreamHandler(sys.stdout)
        ],
        level=config["dbg"]["logging_level"]
    )
    logging.info("Set up logger!")

    if MEASURE_PERF:
        logging.info("Will measure perf.")
        with Profile() as prof:
            main(data_filename, data_type)
        
        stats = pstats.Stats(prof)
        stats.sort_stats(pstats.SortKey.TIME)
        stats.dump_stats(filename=f"logs/{session_name}.prof")
    else:
        main(data_filename, data_type)
