import torch
import yaml
from datetime import datetime as dt
import pandas as pd
import logging
import sys
import os
from time import time

from torch import cuda
from pandas import read_pickle

def load_config():
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)

config = load_config()

def log_config(filename):
    create_missing_dir(filename)
    logging.basicConfig(
        # filename = filename,
        handlers = [
            logging.FileHandler(filename),
            logging.StreamHandler(sys.stdout)
        ],
        level = config["dbg"]["logging_level"]
    )

if config["dbg"]["measure_perf"]:
    from cProfile import Profile
    import pstats

def profile_func(path, func, *args):
    logging.info("Will measure perf.")
    with Profile() as prof:
        func(*args)
    
    stats = pstats.Stats(prof)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats(filename=path)

def curr_time() -> str:
    # Returns current time to use as a timestamp for naming files
    return dt.now().strftime("%j-%H%M-%S")  # %f gives 6 digit microseconds

class LeTimer:
    def __init__(self):
        self.last_ping = time()

    def ping(self):
        # Updates the ping, and returns it
        self.last_ping = time()
        return self.last_ping

    def secs_since_last_ping(self, dps=4) -> float:
        old_last_ping = self.last_ping
        elapsed = self.ping() - old_last_ping
        return round(elapsed, dps)

    def time_taken(self) -> str:
        '''Formats secs_since_last_ping()'''
        return f" (took {self.secs_since_last_ping()} s)"

def to_df(file_path):
    # Returns the given pickled pandas DataFrame.
    assert file_path.endswith(".pkl"), "Only .pkl files, sorry!"
    return pd.read_pickle(file_path)

def df_info(df, printcols):
    print(f"{df.shape=}\n")
    if printcols: print(f"{df.columns=}")

def trim_name(filename):
    return os.path.splitext(os.path.basename(filename))[0].replace("/","").replace("\\","")

def get_extension(filename):
    return os.path.splitext(os.path.basename(filename))[1].replace("/","").replace("\\","")

def strnone_to_str(strnone):
    return '' if strnone is None else str(strnone)

def create_missing_dir(filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

def get_device():
    requested_device = os.environ.get(
        "JET_DEVICE", config["training"]["device"]
    ).lower()
    if requested_device not in {"cpu", "cuda", "mps"}:
        raise ValueError(
            f"Unsupported device '{requested_device}'. Use 'cpu', 'cuda', or 'mps'."
        )

    if requested_device == "cuda" and cuda.is_available():
        device = "cuda"
    elif requested_device == "mps" and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    if requested_device == "cuda" and device == "cpu":
        logging.warning("CUDA was requested but is unavailable; falling back to CPU.")
    if requested_device == "mps" and device == "cpu":
        logging.warning("MPS was requested but is unavailable; falling back to CPU.")
    logging.info(f"Using {device=}")
    return device

def get_files(path, extension=None, filter_name=None, pickled_df=False):
    '''Returns the given path (with the given extension and/or filter phrase) if it's a filepath;
    otherwise, returns all files in that directory'''
    files = [path] if os.path.isfile(path) else [
        os.path.join(path, file) for file in os.listdir(path)
        if os.path.isfile(os.path.join(path, file))
    ]
    
    if extension:   files = [f for f in files if get_extension(f) == extension]
    if filter_name: files = [f for file in files if filter_name in f]
    if pickled_df:  files = [read_pickle(f) for f in files if os.path.getsize(f) > 0]
    
    return files

def intersect_complement(a: list, b: list) -> list:
    # complement of the intersection of a and b
    # the below is just english, python is fun
    return [elt for elt in set(a+b) if elt not in a or elt not in b]
