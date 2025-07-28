import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import yaml

def summarize_jet_features(df):
    return pd.DataFrame({
        'pt_mean': df['pt'].apply(np.mean),
        'pt_std': df['pt'].apply(np.std),
        # 'eta_mean': df['eta'].apply(np.mean),
        'eta_std': df['eta'].apply(np.std),
        # 'phi_mean': df['phi'].apply(np.std),
        'phi_std': df['phi'].apply(np.std),
        'd0sig_mean': df['d0/d0Err'].apply(np.mean),
        # 'd0sig_std': df['d0/d0Err'].apply(np.std),
        'dzsig_mean': df['dz/dzErr'].apply(np.mean),
        # 'dzsig_std': df['dz/dzErr'].apply(np.std)
    })

def plot_corr_heatmap(corr_matrix, title, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# LOAD CONFIGURATIONS FROM YAML FILE
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

qcd_file = config['data']['processed_data_dir'] + config['data']['train_file']
wjet_file = config['data']['processed_data_dir'] + config['data']['test_file']

# LOAD DATA
qcd = pd.read_pickle(qcd_file)
wjet = pd.read_pickle(wjet_file)

# SUMMARIZE FEATURES
qcd = pd.read_pickle(qcd_file)
wjet = pd.read_pickle(wjet_file)

# SUMMARIZE FEATURES
qcd_summary = summarize_jet_features(qcd)
wjet_summary = summarize_jet_features(wjet)

# COMPUTE CORRELATIONS
qcd_corr = qcd_summary.corr()
wjet_corr = wjet_summary.corr()
corr_diff = wjet_corr - qcd_corr

# OUTPUT DIRECTORY
os.makedirs("plots", exist_ok=True)

# PLOT HEATMAPS
plot_corr_heatmap(qcd_corr, "QCD Correlation", "plots/heatmaps/qcd_corr.png")
plot_corr_heatmap(wjet_corr, "WJets Correlation", "plots/heatmaps/wjet_corr.png")
plot_corr_heatmap(corr_diff, "WJets - QCD Correlation Difference", "plots/heatmaps/diff_corr.png")