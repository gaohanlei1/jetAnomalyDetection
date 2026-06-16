#!/bin/bash
source .venv/bin/activate

# Add a short `py` command inside this environment.
ln -sf "$PWD/.venv/bin/python" "$PWD/.venv/bin/py"
export PATH="$PWD/.venv/bin:$PATH"
