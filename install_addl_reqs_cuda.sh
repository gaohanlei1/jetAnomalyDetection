#!/bin/bash

set -euo pipefail

python -m pip install --force-reinstall \
    torch==2.2.1 \
    --index-url https://download.pytorch.org/whl/cu118

python -m pip install torch-geometric==2.6.1
python -m pip install --force-reinstall --no-deps \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    -f https://data.pyg.org/whl/torch-2.2.1+cu118.html
