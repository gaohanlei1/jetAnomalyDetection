import pandas as pd
import os
import sys
import argparse

import helpers

def concat_pkls(folder_path, filter_str, output_name):
    '''
    Joins all the pickled pd.DataFrames in the folder into one, saves this, and returns it.
    Checks whether the columns are consistent across all files.
    '''
    dfs = [
        pd.read_pickle(os.path.join(folder_path, name))
        for name in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, name))
            and os.path.splitext(name)[1] == ".pkl"
            and (filter_str is None or filter_str in name)
    ]
    if not dfs: raise Exception(f"No files in directory {folder_path}!")
    columns = dfs[0].columns.tolist()

    # sanity check
    for df in dfs:
        next_columns = df.columns.tolist()
        if next_columns != columns:
            raise Exception(f"Columns are different:\n{columns=}\nvs\n{next_columns=}")
    
    concatted = pd.concat(dfs)

    os.makedirs(folder_path, exist_ok=True)
    if not output_name: output_name = f"concat_{helpers.curr_time()}.pkl"
    output_path = os.path.join(folder_path, output_name)
    concatted.to_pickle(output_path)
    print(f"Concatenated the files in {folder_path} with filter '{'' if not filter_str else filter_str}',\ninto {output_path=}")
    
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
    args = parser.parse_args()
    
    concat_pkls(args.path, args.filter, args.name)