# Usage

## TL;DR:

`git clone ...`\
`cd jetAnomalyDetection`\
\
`source setup_venv.sh`\
\
(modify `setup_data_symlinks.sh` to add qcd and wjet remote data directories)\
`source setup_data_symlinks.sh`\
\
(modify `configs/config.yaml` if needed)\
`python3.9 scripts/preprocessing.py --data_type background`\
`python3.9 scripts/preprocessing.py --data_type signal`\
\
(`python3.9 helpers/print_df_info.py --path data/preprocessed/qcd/` or `wjet/` to check sizes of DataFrames)\
(`python3.9 helpers/join_dfs.py --filter 170to300 --path data/preprocessed/qcd/` or `wjet/` to concatenate all dataframes containing the given label into one `.pkl` file)\
\
(modify `configs/config.yaml` to point to the preprocessed files)\
`python3.9 scripts/processing.py`\
\
(TODO! sync w/ arjun's changes)
`python3.9 scripts/run_train_autoencoder.py`\

## First time

- `source setup_venv.sh` to setup the venv and automatically download all requirements. Will take a while!
    - if anything goes wrong here, deactivate and try removing whatever requirement caused the issue?

- `source start_venv.sh` anytime to start the venv, and `deactivate` to terminate it.

- Modify the configs in `configs/config.yaml` to customise parameters like the data saving locations, model hyperparameters, etc.

## Preprocessing

- Collect the backgorund (QCD) `.root` data files into one folder, and the signal (WJet) ones into another.
    - Add the folder paths into `./configs/config.yaml`.

    - Modify `setup_data_symlinks.sh` (add `qcd_dir` and `wjet_dir`), and then `source setup_data_symlinks.sh` to create symlinks to the remote raw data folders into `./data/raw/`, to access the data easier. This will allow you to use `config['data']['move_to_used']: true`, which will move each data file into a `./used/` subdirectory, in case you abort preprocessing halfway, so that the next run will exclude the `used` data files.

- After activating the venv, `python3.9 scripts/preprocessing.py --data_type <background OR signal>` to preprocess the given data type (with the raw data path being specified in `configs/config.yaml`).
    - To save into a subfolder within the preprocessed data directory, add `--subfolder <name>`. I'd recommend using the label of the jet you're using, like `QCD_Pt400to600`, since we need this info when processing. (TODO!!!)
    - If you want to process a specific file, add `--filename <filename_in_data_folder>`.

## Processing

- (TODO: implement subfolders/labels)

- Assuming all the preprocessed data files are in `config['data']['preprocessed_data_dir']/<jet_type>/` (not any lower subdirectories), run `python3.9 scripts/processing.py` to start processing the `qcd` and `wjet` files.
    - This will concatenate all the data files of each jet type into their own combined DataFrame, process that, and output into the folder specified in the config.

## Helpers

- `python3.9 helpers/print_df_info.py --path <file_or_folder>` to inspect the size `(rows * columns)` and column names of a pickled DataFrame (or an entire folder of these), as a sanity check to make sure a data file contains actual data.
    - Also add `--printcols` to print the columns, and `--print` to try straight up printing the DataFrames.

- `python3.9 helpers/join_dfs.py --path <file_or_folder> --filter <jet_label>` to join all the `.pkl` files in the specified folder containing the specified label, e.g. to join all `QCD_Pt1800to2400` files for processing.

## To-dos

### Fri - 11.07.25

- Analysing Pt distributions of the new btv-nano data between a matching QCD and WJet pair:
    - from the raw ROOT files (Arjun already did this, showed pretty big separation)
    - after preprocessing and combining into one file each (does my preprocessing squish the separation, somehow?)
    - after processing
    - all of the above, but with log Pt instead; this is what `processing.py` currently does, but it shows very little separation compared to Arjun's plots

- Adding arguments to preprocessing:
    - `--subfolder` option to save preprocessed files in their own subfolders from each `.root`
    - `--concat` option to concatenate saved `.root` files automatically

- Adding `--qcd-name` and `--wjet-name` options to processing to be able to specify the filenames directly, instead of from `config.yaml`

- Updating readme after the above; maybe replacing the main README?

### Later

- Parameter sweeps take a LOOONG while (hours), so we should try to use the GPU on Brux or LXPlus, or speed it up anyhow
    - possibilities: GPU computation, using Jax or other JIT options