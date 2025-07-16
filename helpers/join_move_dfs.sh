#!/bin/bash

wjet_dir=/isilon/export/home/mstamenk/jet-anomaly-summer25/downloads/jekrupa/v2_2/2017/WJetsToQQ/WJetsToQQ_HT-400to600_TuneCP5_13TeV-madgraphMLM-pythia8/WJetsToQQ_HT-400to600/211108_171900/0000
jet_path=~/CERN/git/jetAnomalyDetection
data_dir=$jet_path/data/preprocessed/wjet
concat_dir=$data_dir/concatted
label=WJetsToQQ_HT-400to600

cd $jet_path
mkdir $concat_dir
mkdir $data_dir/$label

for filepath in $wjet_dir/*.root; do
    filename="${filepath##*/}"
    trimmed="${filename%.*}"
    echo "bash: now joining with filter $trimmed"
    python3.9 $jet_path/helpers/join_dfs.py -p $data_dir -f $trimmed -n $label/concat_$trimmed.pkl
    mv $data_dir/*$trimmed* $concat_dir/
done