import yaml
from datetime import datetime as dt
import pandas as pd

def load_config():
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)

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