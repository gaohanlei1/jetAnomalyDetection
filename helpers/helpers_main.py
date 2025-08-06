import yaml
from datetime import datetime as dt
import pandas as pd
import logging
import sys
import os
from time import time
from torch import cuda

def load_config():
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)

config = load_config()
LAST_PING = None

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

def secs_since_last_ping() -> float:
    '''
    Returns seconds (to 4 d.p.) past since the last time this function was called,
    or 0 if first time.
    Saves state! Using globals... :(
    '''
    # yes, this would be better as a class;
    # yes, I shouldn't care coz I'm already over-engineering this
    global LAST_PING
    curr_time = time()
    if LAST_PING is None: LAST_PING = curr_time
    elapsed = curr_time - LAST_PING
    LAST_PING = curr_time
    return round(elapsed, 4)

def time_taken() -> str:
    '''Formats the time since last call to secs_since_last_ping()'''
    return f" (took {secs_since_last_ping()} s)"

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
    device = "cuda" if config["training"]["device"] == "cuda" and cuda.is_available() else "cpu"
    logging.info(f"Using {device=}")

def get_files(path, extension=None, filter_name=None):
    '''Returns the given path if it's a filepath; otherwise, returns all files in that directory'''
    files = [path] if os.path.isfile(path) else [
        os.path.join(path, file) for file in os.listdir(path)
        if os.path.isfile(os.path.join(path, file))
    ]
    if extension: files = [f for f in files if get_extension(f) == extension]
    if filter_name: files = [f for file in files if filter_name in f]
    return files