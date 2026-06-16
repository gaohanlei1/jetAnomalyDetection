#!/bin/bash

set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}

"${PYTHON_BIN}" -m venv .venv
source start_venv.sh
python -m pip install --upgrade pip

python -m pip install -r reqs.txt
source install_addl_reqs_cuda.sh

python -m pip install --force-reinstall numpy==1.25.0
