import pandas as pd
import argparse
import helpers

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Pickled DataFrame Analyser",
        description="outputs info about the pickled DataFrame\ne.g. 'python3.9 helpers/check_pkl_len.py --path data/preprocessed/qcd/data1.pkl'"
    )
    parser.add_argument(
        "--path", type=str, required=True,
        help=".pkl file path"
    )

    file_path = parser.parse_args().path
    helpers.df_info(helpers.to_df(file_path))