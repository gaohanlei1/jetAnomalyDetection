#!/bin/bash
source .venv/bin/activate

# add py=.venv/bin/python3.9 to PATH to run scripts easier!
ln -sf "$PWD/.venv/bin/python3.9" "$PWD/.venv/bin/py"
export PATH="$PWD/.venv/bin:$PATH"
