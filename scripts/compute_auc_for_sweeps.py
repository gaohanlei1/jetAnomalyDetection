import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Load existing sweep summary
results_path = 'sweeps/autoencoder_param_sweep.csv'
results_df = pd.read_csv(results_path)

# Container for AUC scores
auc_scores = []

for idx in range(len(results_df)):
    try:
        # Load losses for this run
        bkg_loss = np.load(f"sweeps/run_{idx}_background_test_loss.npy")
        sig_loss = np.load(f"sweeps/run_{idx}_signal_loss.npy")

        # Combine and label
        y_true = np.concatenate([np.zeros_like(bkg_loss), np.ones_like(sig_loss)])
        y_scores = np.concatenate([bkg_loss, sig_loss])  # reconstruction loss

        # Compute AUC
        auc = roc_auc_score(y_true, y_scores)
    except Exception as e:
        print(f"[!] Skipping run {idx} due to error: {e}")
        auc = np.nan

    auc_scores.append(auc)

# Add to DataFrame and save
results_df['auc_score'] = auc_scores
results_df.to_csv(results_path, index=False)
print("AUC scores appended to autoencoder_param_sweep.csv")

# Prints top 5 best runs by AUC score
df_sorted = results_df.sort_values(by="auc_score", ascending=False)
print("\nTop 5 runs by AUC score:")
print(df_sorted[['auc_score', 'learning_rate', 'weight_decay', 'nearest_neighbors', 'smallest_dim']].head())

