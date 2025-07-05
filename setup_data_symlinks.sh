#!/bin/bash
# can move these setup files to ./setup/?

# symlinks all the files in the QCD and WJet data directories (as defined in this file) into ./data/raw/QCD or WJet!
#   this ensures that if you don't put the filename while running the preprocessing script,
#   the symlinked files will be moved to /data/raw/used/    

qcd_dir=/isilon/export/home/mstamenk/jet-anomaly-summer25/downloads/jekrupa/v2_2/2017/QCD/QCD_Pt_1000to1400_TuneCP5_13TeV_pythia8/QCD_Pt_1000to1400/211109_133059/0000
wjet_dir=/isilon/export/home/mstamenk/jet-anomaly-summer25/downloads/jekrupa/v2_2/2017/WJetsToQQ/WJetsToQQ_HT-400to600_TuneCP5_13TeV-madgraphMLM-pythia8/WJetsToQQ_HT-400to600/211108_171900/0000

for filepath in $qcd_dir/*.root; do 
    ln -s "$filepath" "$PWD/data/raw/qcd/${filepath##*/}"
done

for filepath in $wjet_dir/*.root; do 
    ln -s "$filepath" "$PWD/data/raw/wjet/${filepath##*/}"
done

# ln -s  ./data/raw/QCD

# ln -s  ./data/raw/WJet
