#!/bin/bash

# can move these setup files to ./setup/?
python -m venv .venv
source start_venv.sh

pip install -r reqs-short.txt
pip install torch==2.2.1+cu118 torchvision==0.17.1+cu118 torchaudio==2.2.1 --extra-index-url https://download.pytorch.org/whl/cu118

pip install -r reqs-long.txt
