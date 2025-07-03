#!/bin/bash

# can move these setup files to ./setup/?
python3.9 -m venv .venv
source start_venv.sh
#.venv/bin/python -m pip install --upgrade pip

pip install -r ../reqs-trans.txt
