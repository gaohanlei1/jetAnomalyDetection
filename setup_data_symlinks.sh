#!/bin/bash
# can move these setup files to ./setup/?

# symlinks all the files in the QCD and WJet data directories (as defined in this file) into ./data/raw/QCD or WJet!
#   this ensures that if you don't put the filename while running the preprocessing script,
#   the symlinked files will be moved to /data/raw/used/    

# NOTE: remember to make sure the save directory exists! this will NOT make the intermediate dirs
# NOTE: add -f to the ln command if you already have some symlinked files in the save dir, to not error

data_dir=/home/mstamenk/jet-anomaly-summer25/btv-nano

for filepath in $data_dir/*.root; do 
    ln -s "$filepath" "$PWD/data/raw/qcd/subfolder/${filepath##*/}"
done
