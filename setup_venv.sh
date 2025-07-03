#!/bin/bash

# can move these setup files to ./setup/?
python3.9 -m venv .venv
source start_venv.sh
#.venv/bin/python -m pip install --upgrade pip

#pip install -r reqs-short.txt
#pip install torch==2.2.1+cu118 torchvision==0.17.1+cu118 torchaudio==2.2.1 --extra-index-url https://download.pytorch.org/whl/cu118

pip install --ignore-installed torch-scatter -f https://data.pyg.org/whl/torch-2.2.1+cu121.html
pip install --ignore-installed  torch-sparse -f https://data.pyg.org/whl/torch-2.2.1+cu121.html
pip install --ignore-installed  torch-cluster -f https://data.pyg.org/whl/torch-2.2.1+cu121.html
pip install --ignore-installed  torch-geometric -f https://data.pyg.org/whl/torch-2.2.1+cu121.html

pip install -r ../reqs-trans.txt
