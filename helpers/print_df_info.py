import pandas as pd
import argparse
import helpers
import os

def print_info(path, printcols, printall):
    print(f"{path=}")
    df = helpers.to_df(path)
    helpers.df_info(df, printcols)
    if printall: print(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Pickled DataFrame Analyser",
        description="outputs info about the pickled DataFrame\ne.g. 'python3.9 helpers/check_pkl_len.py --path data/preprocessed/qcd/data1.pkl'"
    )
    parser.add_argument(
        "--path", type=str, required=True,
        help=".pkl file or folder path; if folder, loops over all files"
    )
    parser.add_argument(
        "--print", default=False, action=argparse.BooleanOptionalAction,
        help="print all rows"
    )
    parser.add_argument(
        "--printcols", default=False, action=argparse.BooleanOptionalAction,
        help="print columns"
    )

    args = parser.parse_args()
    if os.path.isfile(args.path):
        print_info(args)
    else:
        # print(f"Printing info of dfs in {args.path}")
        for file in os.listdir(args.path):
            curr_file = os.path.join(args.path, file)
            if os.path.isfile(curr_file) and curr_file.endswith(".pkl"):
                print_info(curr_file, args.printcols, args.print)