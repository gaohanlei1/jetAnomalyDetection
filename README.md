Jet Anomaly Detection is a PyTorch Geometric-based framework for detecting anomalous high-energy jets using graph neural networks. The system supports both unsupervised (autoencoder-based) and supervised (classifier-based) learning, utilizing particle-level features and k-nearest-neighbor graph constructions.

# Features

- Graph-based autoencoder for unsupervised anomaly detection
- Binary classifier for supervised learning tasks
- Hyperparameter sweep module for model tuning
- Visualization tools for loss curves, ROC curves, anomaly scores
- Modular preprocessing pipeline with feature engineering and normalization

# Usage

## TL;DR:

```bash
cd jetAnomalyDetection
source setup_venv.sh

# modify `setup_data_symlinks.sh` to add qcd and wjet remote data directories
source setup_data_symlinks.sh

# modify `configs/config.yaml` if needed
python3.9 scripts/preprocessing.py -t background
python3.9 scripts/preprocessing.py -t signal
# can use paths to singular files instead

# optional
# python3.9 helpers/print_df_info.py --path data/preprocessed/qcd/              # or `wjet/`
# python3.9 helpers/join_dfs.py --filter 170to300 --path data/preprocessed/qcd/ # or `wjet/`

# modify `configs/config.yaml to point to the preprocessed data folders
python3.9 scripts/processing.py

# TODO! sync w/ arjun's changes
...
python3.9 scripts/run_train_autoencoder.py
```

**NOTE:** if you want to run the time-consuming scripts on Brux/LXPlus over SSH, prepend the command with `nohup`!\
With this, you can close the SSH connection and the script will keep running - the terminal output just goes to `nohup.out` in the current dir.\
e.g. `nohup python3.9 scripts/preprocessing.py -t background`\
(Remember to check terminal output often with `cat nohup.out` to check for errors; e.g. because of no virtual environment)

## Overview

- We preprocess `.root` jet data files into intermediate pickled DataFrames (`.pkl`),

- process these intermediate files (scale using background data, feature engineer, etc.),

- and, once we have the proper distributions, train the classifier (old) or autoencoder (currently developing).

- Parameter sweeps are performed via: TODO! Sync w/ Arjun

### First time

- Run `source setup_venv.sh` to setup the venv and download all requirements.
    - if anything goes wrong, `deactivate` and try adding or removing requirements from `reqs-short.txt`

- Run `source start_venv.sh` anytime to start the venv, and `deactivate` to terminate it

- Modify `configs/config.yaml` to customise data locations, hyperparameters, etc.

### Preprocessing

- If you only have a single file to preprocess, run `python3.9 scripts/preprocessing.py -p <filepath> -t [background/signal]`
    - You should also set `move_to_used` to `false` if this is in a directory you can't write to
    - Remember to add `nohup` if you want to leave it running over SSH!

- Otherwise, collect your background (QCD) and/or signal (WJet) `.root` data files into two folders

- Assuming your raw `.root` data is within some remote or local directory:
    - Add those directories as `qcd_dir` and `wjet_dir` into `setup_data_symlinks.sh` (or comment one out)
    - Run `source setup_data_symlinks.sh` to create data symlinks/shortcuts into `./data/raw/`, to access the data easier
    - Now `move_to_used: true` in the config will work, moving `.root` files into a `./used/` subdirectory to keep track of which files have been preprocessed

- Run `python3.9 scripts/preprocessing.py -t [background/signal]` to preprocess using the default raw data directory
    - The default directory defined in `config.yaml` is where `setup_data_symlinks.sh` saves the shortcuts to 
    - If you have multiple `.root` files with the same label, e.g. "170to300_1", "..._2", "..._3",
        - this will save everything into the same folder, assuming your data paths are set up correctly
    - ^ with different labels, e.g. "QCD170to300", "QCD1400to1800",
        - then add `-s` to save each file's outputs to a subfolder
        - NOTE: this will also automatically concatenate all the files in the subfolder into `concat_....pkl`, so you should delete the files in subfolders other than the `concat_....pkl` one

### Processing

- NOTE: This processes a pair of QCD and WJet jets together, scaling using the QCD data
    - The WJet's HT range should be around double the QCD Pt range!
    - You'll likely have a folder of multiple `.pkl` files for each jet; these files will be joined during processing

- Run `python3.9 scripts/processing.py --background <path/to/qcd/preprocessed/> --signal <path/to/wjet/preprocessed>`
    - This will process the preprocessed `.pkl` files in the specified paths (defaults are set in `config.yaml`)
    - You can also add labels for each jet using `--label_bg` and `--label_sg`
    - NOTE: if you add `--filter`, then the program will use those labels to filter out preprocessed files


### Training

- TODO!!

- Run `python3.9 scripts/run_train_autoencoder.py`??? After setting the train and test files?

### Visualisation

- `processing.py` saves plots of the processed data distributions (mainly their `log_pt`) into `plots/proc_distr_....png`
    - If you enable DISPLAY_PLOT, then it'll try `plt.show`-ing these plots

- TODO!!!!!!!

### Helpers

- `python3.9 helpers/print_df_info.py --path <file_or_folder>` to inspect the size `(rows * columns)` of a `.pkl` DataFrame (or a folder of `.pkl` files)
    - Use as a sanity check to make sure a data file contains actual data
    - Add `-c` to print the columns, and `-r` to try printing the entire DataFrame

- `python3.9 helpers/join_dfs.py --path <folder> --filter <jet_label>` to join all the `.pkl` files in the folder containing the specified label
    - e.g. to join all `QCD_Pt1800to2400_*.pkl` files

- `raw_data_info` contains the treenames and branch names for the TTrees in the raw `.root` files, e.g. "Events" or "FatJet"