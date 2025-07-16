#!/bin/bash

dir=/isilon/export/home/mstamenk/jet-anomaly-summer25/downloads/jekrupa/v2_2/2017/WJetsToQQ/WJetsToQQ_HT-400to600_TuneCP5_13TeV-madgraphMLM-pythia8/WJetsToQQ_HT-400to600/211108_171900/0000

for filepath in $dir/*.root; do 
    filename="${filepath##*/}"
    trimmed="${filename%.*}"
    echo "$trimmed"
done
