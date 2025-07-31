#!/bin/bash

# can move these setup files to ./setup/?
python3.9 -m venv .venv
source start_venv.sh
#.venv/bin/python -m pip install --upgrade pip

pip install -r reqs.txt
source install_addl_reqs_cuda.sh

# for some reason, it ALWAYS INSTALLS THE WRONG NUMPY VERSION!!!
# and then the correct one REFUSES TO WORK IF YOU UNINSTALL THE WRONG ONE!
# so this! is the only soln i've found!!!
pip uninstall numpy
pip uninstall numpy==1.25.0
pip install numpy==1.25.0