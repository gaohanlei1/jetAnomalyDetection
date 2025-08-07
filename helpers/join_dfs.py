import pandas as pd
import os
import sys
import argparse
import logging

# Add parent directory to import local project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from helpers import helpers_main
import constants as c
helpers_main.log_config(f"logs/concat_{helpers_main.curr_time()}.log")

def concat_pkls(folder_path, filter_str=None, output_name=None, lowerpt=None, upperpt=None):
    '''
    Joins all the pickled pd.DataFrames in the folder into one, saves this, and returns it.
    Checks whether the columns are consistent across all files.
    '''
    timer = helpers_main.LeTimer()
    logging.info(f"Getting files from {folder_path=}")
    
    dfs = helpers_main.get_files(folder_path, extension=".pkl", filter_name=filter_str, pickled_df=True)
    logging.info(f"Read {len(dfs)=} {timer.time_taken()}, now filtering...")

    # if filter_func: dfs = [df[filter_func(df)] for df in dfs]
    rawfj_pt_col = c.RAW_FATJET_PROPERTIES_PREFIX + "pt"
    if lowerpt: dfs = [df[df[rawfj_pt_col] > lowerpt] for df in dfs]
    if upperpt: dfs = [df[df[rawfj_pt_col] < upperpt] for df in dfs]
    
    dfs = [df for df in dfs if not df.empty]
    if not dfs:
        logging.warning(f"No non-empty pickled dataframes in {folder_path=}, after filtering!")
        return
    logging.info(f"Filtered, new {len(dfs)=} {timer.time_taken()}, now checking columns...")
    
    columns = dfs[0].columns.tolist()
    # sanity check
    for df in dfs:
        next_columns = df.columns.tolist()
        if next_columns != columns:
            raise Exception(f"Columns are different:\n{columns=}\nvs\n{next_columns=}\nmismatched elts: {helpers_main.intersect_complement(columns, next_columns)}")
    
    logging.info("Now concatenating...")
    concatted = pd.concat(dfs)
    logging.info(f"Done, {len(concatted)=}! {timer.time_taken()}")

    if not output_name: output_name = f"concat_{helpers_main.strnone_to_str(filter_str)}_{helpers_main.strnone_to_str(lowerpt)}-{helpers_main.strnone_to_str(upperpt)}_{helpers_main.curr_time()}.pkl"
    if not output_name.endswith(".pkl"): output_name += ".pkl"
    output_path = os.path.join(folder_path, output_name)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    logging.info(f"Now saving...")
    concatted.to_pickle(output_path)
    print(f"Concatenated {len(dfs)} files in {folder_path} with filter '{'' if not filter_str else filter_str}', resulting in {len(concatted)=},\ninto {output_path=}")
    
    return concatted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Concatenator3000",
        description="Concatenates DataFrames (e.g. preprocessed or processed data files) in a folder. Default behaviour: concatenate all .pkl files in current working directory"
    )
    parser.add_argument(
        "--path", "-p", type=str, required=False, default=".",
        help='Path of folder that contains all the pickled DataFrames to concatenate; default: "." (current directory)'
    )
    parser.add_argument(
        "--filter", "-f", type=str, required=False,
        help='If provided, will only join the .pkl files that contain this keyword (e.g. 170to300)'
    )
    parser.add_argument(
        "--name", "-n", type=str, required=False,
        help='Name of output file; by default, named with the first filename and the current time'
    )
    parser.add_argument(
        "--lowerpt", "-b", type=float, required=False,
        help="Lower bound on the rawfj_pt. Defaults to no bound"
    )
    parser.add_argument(
        "--upperpt", "-B", type=float, required=False,
        help="Upper bound on the rawfj_pt. Defaults to no bound"
    )
    args = parser.parse_args()
    
    concat_pkls(args.path, args.filter, args.name, args.lowerpt, args.upperpt)