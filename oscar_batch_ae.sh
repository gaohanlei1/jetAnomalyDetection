#!/bin/bash

# Request a GPU partition node, and access to 1 GPUs
#SBATCH -p gpu --gres=gpu:2

# Request 4 CPU cores
#SBATCH -n 4

# Memory
#SBATCH --mem=8G

# Request time; e.g. 01:00:00 hrs
#SBATCH -t 00:30:00

module load cuda

# Run the autoencoder training script with $4 as the bg, $5 as the signal
.venv/bin/python3.9 scripts/run_train_autoencoder.py -b /HEP/export/home/anagaman/jet-anomaly-summer25/jetAnomalyDetection_updated/jetAnomalyDetection/data/processed/scaledby_QCD_PT-80to470/scaledby_QCD/QCD_scaled.pkl -s /HEP/export/home/anagaman/jet-anomaly-summer25/jetAnomalyDetection_updated/jetAnomalyDetection/data/processed/scaledby_QCD_PT-80to470/scaledby_QCD/WJet_scaled.pkl -m hybrid_knn
