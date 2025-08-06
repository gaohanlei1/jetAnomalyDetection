#!/bin/bash
source .venv/bin/activate

# add py=.venv/bin/python3.9 to PATH to run scripts easier!
cd .venv/bin
ln -sf python3.9 py
export PATH=./:$PATH
cd ../..
