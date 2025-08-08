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
Coffea only takes the first instance of each duplicate branch.
Shouldn't be an issue unless you start using said branches, maybe.
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

class Preprocessor:
    DATA_FILE_EXT = ".root"

    def __init__(self, args):
        self.path = args.path
        self.type = args.type
        self.upperpt, self.lowerpt = args.upperpt, args.lowerpt
        self.savepath = args.savepath
        
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
        files = helpers_main.get_files(self.path, extension=self.DATA_FILE_EXT)

        logging.info(f"Loaded {len(files)} file(s)!")
        return files
    
    def load_root(self, filepath):
        logging.info(f"Now preprocessing {filepath}")

        if self.savepath is None: self.savepath = config["data"]["preprocessed_" + self.jet_type]
        logging.info(f"{self.savepath=}")

        # - So, I'll be doing sth very stupid here: loading the same file multiple times
        # - coz pickle can't pickle coffea events for whatever reason
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
                lengths = [len(v) for v in data_dict.values()]
                if len(set(lengths)) != 1:
                    logging.error(f"Chunk {i} has inconsistent array lengths: {dict(zip(data_dict.keys(), lengths))}")
                    continue  # or raise an error if you'd rather fail loudly
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

        if config["data"]["move_to_used"]:
            move_to_used(filepath)
        

def get_fatjets(events, lowerpt=None, upperpt=None): 
    fatjets = events.FatJet
    # store unmasked fjs
    store_fj = [fj for fj in fatjets[0]]
    logging.debug(f"{fatjets.fields=}")

    # accept jets that do not have electrons or muons nearby   
    electrons = events.Electron
    electrons = fatjets.nearest(electrons[electrons.pt > c.ELECTRON_PT_LOWER_BOUND])
    muons = events.Muon
    muons = fatjets.nearest(muons[muons.pt > c.MUON_PT_LOWER_BOUND])

    lowerpt = lowerpt if lowerpt else float(c.FATJET_PT_LOWER_BOUND)
    upperpt = upperpt if upperpt else float(1e10)

    mask = (
        (ak.fill_none(fatjets.delta_r(electrons) > c.ELECTRON_R_LOWER_BOUND, True)) &
        (ak.fill_none(fatjets.delta_r(muons) > c.MUON_R_LOWER_BOUND, True)) & 
        (~ak.is_none(fatjets.matched_gen, axis=1)) & 
        (fatjets.delta_r(fatjets.matched_gen) < c.MATCHED_GEN_R_LOWER_BOUND) & 
        (fatjets.pt > lowerpt) & (fatjets.pt < upperpt) &
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

def process_event_root(events, lowerpt=None, upperpt=None):
    # processes on a per-jet basis, NOT per-event! hence the flattening
    fatjets, pfcs = get_fatjets(events, lowerpt, upperpt)
    
    if isinstance(fatjets, int):
        return -1, -1
    pfcands = events.PFCands

    if len(fatjets["pt"]) != 1:
        logging.warning(f"Fatjets array size isn't 1*events!\n{len(fatjets['pt'][0])=}, {fatjets['pt']=}\n")
        # raise Exception("FatJet wrong shape!")

    eta = ak.to_numpy(ak.flatten(pfcands["phi"] - fatjets["phi"])[pfcs])
    phi = ak.to_numpy(ak.flatten(pfcands["eta"] - fatjets["eta"])[pfcs])
    pt  = ak.to_numpy(ak.flatten(pfcands["pt"]/fatjets["pt"])[pfcs])
    # TODO: old ratio based on whether it is qcd or wjet ->x     this is not model agnostic !!!
      # check that current pt scheme is correct

    properties = [pt, eta, phi]
    property_names = ["pt", "eta", "phi"]

    # add more "raw fields" to preserve through constants.RAW_FATJET_PROPERTIES!
    for new_prop in c.RAW_FATJET_PROPERTIES:
        properties.append(fatjets[new_prop][0])
        property_names.append(c.RAW_FATJET_PROPERTIES_PREFIX + new_prop)
        
    # add all other fields
    for field in pfcands.fields:
        if field not in property_names:
            # logging.debug(f"Added {field=} to property_names")
            properties.append(ak.to_numpy(pfcands[field]).flatten()[pfcs])
            property_names.append(field)

    
    softdrop_mass = fatjets.msoftdrop[0]
    properties.append(np.array([softdrop_mass]))
    property_names.append("fj_msoftdrop")

    # --- ParticleNet Scores ---
    pnet_keys = [
    "particleNetWithMass_QCD",
    "particleNet_XbbVsQCD",
    "particleNet_XccVsQCD",
    "particleNet_XqqVsQCD",
    ]

    for key in pnet_keys:
        if key in fatjets.fields:
            score = fatjets[key][0]
            properties.append(np.array([score]))
            property_names.append(f"fj_" + key)
        else:
            # print(list(fatjets.fields))
            logging.warning(f"FatJet missing expected field: {key}")

    return properties, property_names


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
        ).events()
    data = {}

    logging.debug(f"{events.fields=}")
    
    # iterator = range(metadata['start'], metadata['end'])
    iterator = range(metadata['start'], metadata['end'])
    iterator = iterator if config["dbg"]["only_one_progress_bar"] and pid_posn != 0 else tqdm(
        iterator, desc=f"Process {pid_posn}", position=pid_posn, leave=None
    )

    for i in iterator:
        properties, property_names = process_event_root(events[i:i+1], metadata["lowerpt"], metadata["upperpt"])
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

    preproc = Preprocessor(parser.parse_args())
    preproc.setup_log()

    if config["dbg"]["measure_perf"]:
        helpers_main.profile_func(f"logs/{preproc.session_name}.prof", main, preproc)
    else:
        main(preproc)
