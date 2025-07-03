# Usage

## First time

- `source setup_venv.sh` to setup the venv and automatically download all requirements. Will take a while!
    - if anything goes wrong here, deactivate and try removing whatever requirement caused the issue?

- `source start_venv.sh` anytime to start the venv, and `deactivate` to terminate it.

- Modify the configs in `configs/config.yaml` to customise parameters like the data saving locations, model hyperparameters, etc.

## Preprocessing

- Collect the backgorund (QCD) `.root` data files into one folder, and the signal (WJet) ones into another.
    - Add the folder paths into `./configs/config.yaml`.

    - `source setup_data_symlinks.sh` to create symlinks to the remote raw data folders into `./data/raw/`, to access the data easier.

- After activating the venv, `source prep_bg_qcd.sh/prep_sg_wjet.pkl <.root data file path>` to preprocess the specified .root file. It's saved in the location specified in `configs/config.yaml`, by default (TODO!!)

(I want to be able to run the preprocessor once and have it preprocess every file in the given folder. However, it's not very realistic to be able to preprop everything in one go.
For now, I'll just make a script that preprocesses one file?)
