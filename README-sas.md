# Usage

## First time

- `source setup_venv.sh` to setup the venv and automatically download all requirements. Will take a while!
    - if anything goes wrong here, deactivate and try removing whatever requirement caused the issue?

- `source start_venv.sh` anytime to start the venv, and `deactivate` to terminate it.

- Modify the configs in `configs/config.yaml` to customise parameters like the data saving locations, model hyperparameters, etc.

## Preprocessing

- Collect the backgorund (QCD) `.root` data files into one folder, and the signal (WJet) ones into another.
    - Add the folder paths into `./configs/config.yaml`.

    - `source setup_data_symlinks.sh` to create symlinks to the remote raw data folders into `./data/raw/`, to access the data easier. This will allow you to use `config['data']['move_to_used']: true`, which will move each data file into a `./used/` subdirectory, in case you abort preprocessing halfway, so that the next run will exclude the `used` data files.

- After activating the venv, `python3.9 scripts/preprocessing.py --data_type <background OR signal>` to preprocess the given data type (with the raw data path being specified in `configs/config.yaml`).
    - If you want to process a specific file, add `--filename <filename_in_data_folder>`.

## Processing

- Assuming all the preprocessed data files are in `config['data']['preprocessed_data_dir']/<jet_type>/` (not any lower subdirectories), run `python3.9 scripts/processing.py` to start processing the `qcd` and `wjet` files.
    - This will concatenate all the data files of each jet type into their own combined DataFrame, process that, and output into the folder specified in the config.

## Helpers

- `python3.9 helpers/print_df_info.py --path <file_or_folder>` to inspect the size `(rows * columns)` and column names of a pickled DataFrame (or an entire folder of these), as a sanity check to make sure a data file contains actual data.
    - Also add `--printcols` to print the columns, and `--print` to try straight up printing the DataFrames.

## To-dos
