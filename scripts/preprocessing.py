"""
Preprocesses raw ROOT files into pandas DataFrames of jet constituent properties, saved as pickles.
Example command:
python scripts/preprocessing.py -p "data/raw/btv-nano/QCD_PT-170to300_TuneCP5_13p6TeV_pythia8_1.root" -t "background"
Default save dir:
data/preprocessed/
"""

from coffea.nanoevents import NanoEventsFactory, PFNanoAODSchema, BaseSchema
from multiprocessing import Pool, Manager, RLock, cpu_count
import numpy as np
import awkward as ak
from fast_histogram import histogram2d
import os
import sys
import pandas as pd 
from tqdm import tqdm
import argparse
import logging
import uproot

'''
NOTE!
I'm suppressing like 50 warnings that come up everytime I run the program,
which all complain about duplicate branches in the .root files used to load events.
Coffea only takes the first instance of each duplicate branch.
Shouldn't be an issue unless you start using said branches, maybe.
'''
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="coffea.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="coffea.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="coffea.*")

DEFAULT_WORKERS = max(
    4,
    int(os.environ.get("SLURM_CPUS_PER_TASK", cpu_count())),
)

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

class Preprocessor:
    DATA_FILE_EXT = ".root"

    def __init__(self, args):
        self.path = args.path
        self.type = args.type
        self.upperpt, self.lowerpt = args.upperpt, args.lowerpt
        self.savepath = args.savepath
        self.workers = args.workers
        self.move_to_used = args.move_to_used
        self.recursive = args.recursive
        self.max_events = args.max_events
        
    def setup_log(self):
        self.session_name = f"preproc_{self.type}_pt{helpers_main.strnone_to_str(self.lowerpt)}-{helpers_main.strnone_to_str(self.upperpt)}_{helpers_main.curr_time()}"
        helpers_main.log_config(f"logs/{self.session_name}.log")
        logging.info("Set up logger!")
        return self.session_name
    
    def get_files(self):
        self.jet_type    = config["data"][self.type]
        self.folder_path = config["data"]["raw_" + self.jet_type]

        # path default: folder path defined in config
        if self.path is None:
            self.path = self.folder_path

        if not os.path.exists(self.path):
            old = self.path
            # if it's a filename within default folder, use that
            self.path = os.path.join(self.folder_path, self.path)
            if not os.path.exists(self.path):
                raise Exception(f"Invalid path: {old}\n{self.path=}\n{self.jet_type=}")

        # by now, guaranteed to be full file/folder paths
        if os.path.isdir(self.path) and self.recursive:
            files = sorted(
                os.path.join(root, filename)
                for root, _, filenames in os.walk(self.path, followlinks=True)
                for filename in filenames
                if filename.endswith(self.DATA_FILE_EXT)
            )
        else:
            files = helpers_main.get_files(
                self.path, extension=self.DATA_FILE_EXT
            )

        logging.info(f"Loaded {len(files)} file(s)!")
        if not files:
            raise FileNotFoundError(
                f"No {self.DATA_FILE_EXT} files found at {self.path}"
            )
        return files
    
    def load_root(self, filepath):
        logging.info(f"Now preprocessing {filepath}")

        if self.savepath is None: self.savepath = config["data"]["preprocessed_" + self.jet_type]
        logging.info(f"{self.savepath=}")

        with uproot.open(filepath) as f:
            num_events = f["Events"].num_entries
        if self.max_events is not None:
            num_events = min(num_events, self.max_events)
        if num_events == 0:
            logging.warning(f"Skipping empty ROOT file: {filepath}")
            return

        workers = min(self.workers, num_events)
        logging.info(f"{num_events} events in total, splitting into {workers} chunks.")
        start_i = np.linspace(0, num_events, endpoint=False, dtype=int, num=workers)
        end_i   = np.append(start_i[1:], num_events)

        tqdm.set_lock(RLock())  # for cleaner tqdm output across processes
        with (Pool(workers) as pool, Manager() as m):
            logging.info(f"Initialised pool {pool} with {workers} workers.")

            pid_list = m.list()
            pid_lock = m.Lock()
            shared_datas = [
                {
                    "start" : start,
                    "end"   : end,
                    "jet_type"  : self.jet_type,
                    "filepath" : filepath,
                    "pid_list"  : pid_list,
                    "pid_lock"  : pid_lock,
                    "save_path" : self.savepath,
                    "lowerpt"   : self.lowerpt,
                    "upperpt"   : self.upperpt
                }
                for start, end in zip(start_i, end_i)
            ]

            # each invocation of the function will receive an ELT of the iterator, not a slice
            # the chunk is just loosely defined as the range over which to pick elts for each cpu
            results = pool.map(preproc_events_slice, shared_datas)

            logging.info(f"All done! Converting {len(results)=} dicts to dataframes and concatenating...")
            # results = pd.concat([pd.DataFrame.from_dict(data_dict) for data_dict in results])


            logging.info(f"Concatenated {len(results)=} fatjets! Saving...")

            basename = helpers_main.trim_name(filepath)
            for i, data_dict in enumerate(results):
                if not data_dict:
                    logging.warning(
                        f"Chunk {i} contained no jets passing the selection."
                    )
                    continue
                lengths = [len(v) for v in data_dict.values()]
                if len(set(lengths)) != 1:
                    raise ValueError(
                        f"Chunk {i} has inconsistent array lengths: "
                        f"{dict(zip(data_dict.keys(), lengths))}"
                    )
                df = pd.DataFrame.from_dict(data_dict)

                output_file_path = os.path.join(
                    self.savepath,
                    f"{basename}_chunk{i}_Pt{helpers_main.strnone_to_str(self.lowerpt)}-{helpers_main.strnone_to_str(self.upperpt)}_{os.getpid()}_{helpers_main.curr_time()}.pkl"
                )
                helpers_main.create_missing_dir(output_file_path)
                df.to_pickle(output_file_path)
                logging.info(f"Saved chunk {i} to {output_file_path}.")
                
            # output_file_path = os.path.join(
            #     self.savepath,
            #     f"{basename}_Pt{helpers_main.strnone_to_str(self.lowerpt)}-{helpers_main.strnone_to_str(self.upperpt)}_{os.getpid()}_{helpers_main.curr_time()}.pkl"
            # )
            # helpers_main.create_missing_dir(output_file_path)
            # results.to_pickle(output_file_path)
            # logging.info(f"Preprocessed into {output_file_path}.")

        if self.move_to_used:
            move_to_used(filepath)
        
def delta_phi(phi1, phi2):
    return (phi1 - phi2 + np.pi) % (2 * np.pi) - np.pi


def delta_r(eta1, phi1, eta2, phi2):
    return np.sqrt((eta1 - eta2) ** 2 + delta_phi(phi1, phi2) ** 2)


def min_delta_r_to_objects(fj_eta, fj_phi, obj_eta, obj_phi):
    if len(obj_eta) == 0:
        return np.inf
    drs = delta_r(fj_eta, fj_phi, obj_eta, obj_phi)
    return float(ak.min(drs))


def get_event_array(events, branch_name):
    """
    BaseSchema keeps branches as flat names like FatJet_pt, PFCands_eta, etc.
    Since process_event_root passes events[i:i+1], each branch is a length-1 jagged array.
    This returns the per-event content, e.g. events.FatJet_pt[0].
    """
    return getattr(events, branch_name)[0]


def get_fatjets(events, lowerpt=None, upperpt=None):
    """
    BaseSchema-compatible replacement for the old PFNanoAODSchema version.

    Returns:
        selected_fatjet: dict of selected FatJet scalar properties.
        pfcs: numpy array of PFCands indices associated with the selected FatJet.
    """
    lowerpt = lowerpt if lowerpt is not None else float(c.FATJET_PT_LOWER_BOUND)
    upperpt = upperpt if upperpt is not None else float(1e10)

    # FatJet branches
    fj_pt = get_event_array(events, "FatJet_pt")
    fj_eta = get_event_array(events, "FatJet_eta")
    fj_phi = get_event_array(events, "FatJet_phi")

    if len(fj_pt) == 0:
        return -1, -1

    # Lepton branches
    ele_pt = get_event_array(events, "Electron_pt")
    ele_eta = get_event_array(events, "Electron_eta")
    ele_phi = get_event_array(events, "Electron_phi")

    mu_pt = get_event_array(events, "Muon_pt")
    mu_eta = get_event_array(events, "Muon_eta")
    mu_phi = get_event_array(events, "Muon_phi")

    ele_mask = ele_pt > c.ELECTRON_PT_LOWER_BOUND
    mu_mask = mu_pt > c.MUON_PT_LOWER_BOUND

    # GenJetAK8 matching branches
    fj_gen_idx = get_event_array(events, "FatJet_genJetAK8Idx")
    gen_eta = get_event_array(events, "GenJetAK8_eta")
    gen_phi = get_event_array(events, "GenJetAK8_phi")

    good_fatjet_indices = []

    for j in range(len(fj_pt)):
        pt_j = float(fj_pt[j])
        eta_j = float(fj_eta[j])
        phi_j = float(fj_phi[j])

        if not (pt_j > lowerpt and pt_j < upperpt):
            continue

        if abs(eta_j) >= c.FATJET_ETA_BOUNDS:
            continue

        min_dr_ele = min_delta_r_to_objects(
            eta_j,
            phi_j,
            ele_eta[ele_mask],
            ele_phi[ele_mask],
        )
        if min_dr_ele <= c.ELECTRON_R_LOWER_BOUND:
            continue

        min_dr_mu = min_delta_r_to_objects(
            eta_j,
            phi_j,
            mu_eta[mu_mask],
            mu_phi[mu_mask],
        )
        if min_dr_mu <= c.MUON_R_LOWER_BOUND:
            continue

        gen_idx_j = int(fj_gen_idx[j])
        if gen_idx_j < 0 or gen_idx_j >= len(gen_eta):
            continue

        dr_gen = delta_r(
            eta_j,
            phi_j,
            float(gen_eta[gen_idx_j]),
            float(gen_phi[gen_idx_j]),
        )
        if dr_gen >= c.MATCHED_GEN_R_LOWER_BOUND:
            continue

        good_fatjet_indices.append(j)

    if len(good_fatjet_indices) == 0:
        return -1, -1

    # Choose leading-pT fatjet.
    # If you want to reproduce the old ak.argsort(...); ak.firsts(...) behavior exactly,
    # change max(...) to min(...). The old code likely selected the lowest-pT passing jet.
    best_j = max(good_fatjet_indices, key=lambda j: float(fj_pt[j]))

    # Find PFCands associated with this selected FatJet.
    fjpc_jet_idx = get_event_array(events, "FatJetPFCands_jetIdx")
    fjpc_pfc_idx = get_event_array(events, "FatJetPFCands_pFCandsIdx")

    pfcs = ak.to_numpy(fjpc_pfc_idx[fjpc_jet_idx == best_j])

    if len(pfcs) == 0:
        return -1, -1

    selected_fatjet = {
        "index": best_j,
        "pt": float(fj_pt[best_j]),
        "eta": float(fj_eta[best_j]),
        "phi": float(fj_phi[best_j]),
    }

    # Preserve extra raw FatJet properties requested in constants.RAW_FATJET_PROPERTIES.
    for prop in c.RAW_FATJET_PROPERTIES:
        branch = f"FatJet_{prop}"
        if hasattr(events, branch):
            selected_fatjet[prop] = get_event_array(events, branch)[best_j]
        else:
            logging.warning(f"Missing requested FatJet branch: {branch}")

    return selected_fatjet, pfcs


def process_event_root(events, lowerpt=None, upperpt=None):
    """
    BaseSchema-compatible event processor.
    Processes one event slice: events[i:i+1].
    """
    fatjet, pfcs = get_fatjets(events, lowerpt, upperpt)

    if isinstance(fatjet, int):
        return -1, -1

    # PFCands core branches
    pfc_pt = get_event_array(events, "PFCands_pt")
    pfc_eta = get_event_array(events, "PFCands_eta")
    pfc_phi = get_event_array(events, "PFCands_phi")

    # Constituent features relative to selected fatjet
    pt = ak.to_numpy(pfc_pt[pfcs] / fatjet["pt"])
    eta = ak.to_numpy(pfc_eta[pfcs] - fatjet["eta"])
    phi = ak.to_numpy(delta_phi(pfc_phi[pfcs], fatjet["phi"]))

    properties = [pt, eta, phi]
    property_names = ["pt", "eta", "phi"]

    # Add selected FatJet-level metadata
    for new_prop in c.RAW_FATJET_PROPERTIES:
        if new_prop in fatjet:
            properties.append(fatjet[new_prop])
            property_names.append(c.RAW_FATJET_PROPERTIES_PREFIX + new_prop)

    # Add all PFCands branches.
    # BaseSchema field names are flat, e.g. PFCands_d0, PFCands_charge, etc.
    pfcand_fields = [
        field[len("PFCands_"):]
        for field in events.fields
        if field.startswith("PFCands_")
    ]

    for field in pfcand_fields:
        if field in property_names:
            continue

        branch = f"PFCands_{field}"
        values = get_event_array(events, branch)

        try:
            properties.append(ak.to_numpy(values[pfcs]))
            property_names.append(field)
        except Exception as e:
            logging.warning(f"Skipping PFCands field {field} due to error: {e}")

    if len(property_names) != len(set(property_names)):
        raise ValueError(f"Duplicate output column names: {property_names}")

    return properties, property_names

def move_to_used(filepath):
    new_path = os.path.join(os.path.dirname(filepath), "used", os.path.basename(filepath))
    os.renames(filepath, new_path)
    logging.info(f"Moved '{filepath}' to '{new_path}'.")

# def load_h5():
#     raise NotImplementedError

def preproc_events_slice(metadata):
    """Process a slice of events from a ROOT file and return a dictionary of properties for each jet."""
    pid = os.getpid()
    with metadata["pid_lock"]:
        # no need to check whether it already exists, since duplicate pids aren't possible?
        metadata["pid_list"].append(pid)
        pid_posn = metadata["pid_list"].index(pid)

    curr_events_num = metadata["end"] - metadata["start"]
    logging.info(f"#{pid_posn} ({pid}): Received range {metadata['start']} to {metadata['end']} ({curr_events_num} events).")
    events = NanoEventsFactory.from_root(
        {metadata['filepath']: "Events"}, schemaclass = BaseSchema, 
        delayed=False, entry_start=metadata['start'], entry_stop=metadata['end'],
        ).events()
    data = {}

    logging.debug(f"{events.fields=}")
    
    it = tqdm(
        range(len(events)),
        desc=f"Worker {pid_posn} [{metadata['start']}:{metadata['end']}]",
        position=pid_posn,
        leave=False,
        dynamic_ncols=True,
    )

    for i in it:
        properties, property_names = process_event_root(
            events[i:i+1], metadata["lowerpt"], metadata["upperpt"]
        )
        if properties == -1: continue

        if not data: 
            data = {property_name: [] for property_name in property_names}
        
        for j, prop in enumerate(properties): 
            data[property_names[j]].append(prop)

        # curr_i = i - metadata['start']
        # if curr_i != 0 and curr_i % 20 == 0: break
    
    logging.info(f"#{pid_posn} ({pid}):  Preprocessed last few events, from {metadata['start']} to {metadata['end']}!")
    # save_df(data, metadata)
    return data


def main(preproc):
    files = preproc.get_files()

    for file in files:
        preproc.load_root(file)
    

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
        "--savepath", "-s", "--save", type=str, required=False,
        help="Path to preprocess into. Defaults to config"
    )
    parser.add_argument(
        "--type", "--data_type", "-t", choices=["background", "signal"], default="background",
        help=f"'background' ({config['data']['background']}) or 'signal' ({config['data']['signal']})"
    )
    parser.add_argument(
        "--upperpt", "-B", type=float, default=None,
        help=f"upper bound on fatjet Pt?"
    )
    parser.add_argument(
        "--lowerpt", "-b", type=float, default=None,
        help=f"lower bound on fatjet Pt?"
    )
    parser.add_argument(
        "--workers", "-j", type=int, default=DEFAULT_WORKERS,
        help=f"Number of worker processes. Default: {DEFAULT_WORKERS}"
    )
    parser.add_argument(
        "--move-to-used", default=config["data"]["move_to_used"],
        action=argparse.BooleanOptionalAction,
        help="Move each raw ROOT file into a used/ subdirectory after it succeeds. Default: disabled"
    )
    parser.add_argument(
        "--recursive", default=False, action=argparse.BooleanOptionalAction,
        help="Search subdirectories recursively when --path is a directory"
    )
    parser.add_argument(
        "--max-events", type=int, default=None,
        help="Process at most this many events from each ROOT file (for smoke tests)"
    )

    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.max_events is not None and args.max_events < 1:
        parser.error("--max-events must be at least 1")

    preproc = Preprocessor(args)
    preproc.setup_log()

    # time the operation
    tic = helpers_main.curr_time()
    if config["dbg"]["measure_perf"]:
        helpers_main.profile_func(f"logs/{preproc.session_name}.prof", main, preproc)
    else:
        main(preproc)
