#!/bin/bash

PROPS=("particleNetWithMass_QCD" "particleNet_XbbVsQCD" "particleNet_XccVsQCD" "particleNet_XqqVsQCD" "particleNet_QCD" "particleNet_massCorr")

for prop in "${PROPS[@]}"; do
    py visualize/plot_distributions.py -q data/raw/qcd/220725/used/ -t raw -p "${prop}"
done