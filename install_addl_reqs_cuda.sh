#!/bin/bash

python3.9 -m pip install --ignore-installed torch-geometric==2.6.1
# --no-deps, coz some of these try to install numpy==1.25.0, which is incompatible w/ other dependencies??
python3.9 -m pip install --ignore-installed torch-scatter --no-deps -f https://data.pyg.org/whl/torch-2.2.1+cu118.html
python3.9 -m pip install --ignore-installed torch-sparse  --no-deps -f https://data.pyg.org/whl/torch-2.2.1+cu118.html
python3.9 -m pip install --ignore-installed torch-cluster --no-deps -f https://data.pyg.org/whl/torch-2.2.1+cu118.html
# vvv not needed? not in original venv
# python3.9 -m pip install --ignore-installed torch-geometric --no-deps -f https://data.pyg.org/whl/torch-2.2.1+cu118.html

# need to enable deps!
# python3.9 -m pip install --ignore-installed torch==2.2.1+cu118
python3.9 -m pip install --ignore-installed torch==2.2.1 --index-url https://download.pytorch.org/whl/cu118

# pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118

# however, these don't work with brux yet, coz no CUDA drivers?
# so i'll now try to switch everything to cpu, or install w/o cuda drivers, or install w/ cpu drivers