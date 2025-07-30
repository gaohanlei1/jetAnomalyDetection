#!/bin/bash

raw_dir=/home/mstamenk/jet-anomaly-summer25/btv-nano
jet_path=~/CERN/git/jetAnomalyDetection
data_dir=$jet_path/data/preprocessed/new0707/qcd
concat_dir=$data_dir/concatted
label=btvnano

mkdir $concat_dir
mkdir $data_dir/$label

for filepath in $raw_dir/*.root; do 
    filename="${filepath##*/}"
    trimmed="${filename%.*}"
    echo "bash: now joining with filter '$trimmed'"
    python3.9 $jet_path/helpers/join_dfs.py -p $data_dir -f $trimmed -n $label/concat_$trimmed.pkl
    mv $data_dir/*$trimmed* $concat_dir
done

