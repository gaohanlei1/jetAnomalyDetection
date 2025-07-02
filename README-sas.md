# Usage

## First time

- Run `source setup_venv.sh` to setup the venv and automatically download all requirements. Will take a while!

- Now, you can run `source start_venv.sh` anytime to start the venv, and `deactivate` to terminate it.

- You can modify the configs in `configs/config.yaml` to customise parameters like the data saving locations, model hyperparameters, etc.

## Preprocessing

- After activating the venv, run `source prep_bg_qcd.sh/prep_sg_wjet.pkl <.root data file path>` to preprocess the specified .root file. It's saved in the location specified in `configs/config.yaml`, by default ......

(I want to be able to run the preprocessor once and have it preprocess every file in the given folder. However, it's not very realistic to be able to preprop everything in one go.
For now, I'll just make a script that preprocesses one file?)
