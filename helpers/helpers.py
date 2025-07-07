import yaml
from datetime import datetime as dt
import pandas as pd
import logging
import sys

def load_config():
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)

config = load_config()

def log_config(filename):
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

def to_df(file_path):
    # Returns the given pickled pandas DataFrame.
    assert file_path.endswith(".pkl"), "Only .pkl files, sorry!"
    return pd.read_pickle(file_path)

def df_info(df, printcols):
    print(f"{df.shape=}\n")
    if printcols: print(f"{df.columns=}")