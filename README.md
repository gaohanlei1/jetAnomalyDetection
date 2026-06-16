# Jet Anomaly Detection

A PyTorch Geometric project for detecting anomalous jets with graph neural
networks. The current workflow creates QCD and WJet pickle datasets from ROOT
files and trains a graph autoencoder on Brown's Oscar cluster.

## Current workflow

Use the single beginner guide:

[From raw data to a graph-autoencoder job on Oscar](docs/OSCAR_GUIDE.md)

It covers:

1. copying this local checkout to Oscar;
2. copying the selected ROOT source files from Brux to Oscar;
3. installing the Python environment;
4. submitting a CPU job to create `QCD_scaled.pkl` and `WJet_scaled.pkl`;
5. submitting a GPU job to train the graph autoencoder;
6. monitoring jobs and reading the resulting AUC.

Do not run heavy preprocessing or training directly on an Oscar login node.

## Main scripts

- `oscar_batch_prepare_data.sh`: CPU batch job that creates the pickle pair.
- `oscar_batch_ae.sh`: GPU batch job that trains and evaluates the graph
  autoencoder.
- `scripts/preprocessing.py`: converts ROOT events into intermediate pickle
  chunks.
- `scripts/processing.py`: combines, feature-engineers, filters, and scales the
  intermediate data.
- `scripts/run_train_autoencoder.py`: trains and evaluates the model.

## Local repository

The local checkout used by Cursor is:

```text
/Users/gaohanlei1/Documents/Codex/jetAnomalyDetection
```
