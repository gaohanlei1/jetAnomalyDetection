import matplotlib.pyplot as plt
import numpy as np
import os
import yaml
import pandas as pd

# Load YAML configuration for data and hyperparameters
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# File paths for training and signal data
train_file = config['data']['processed_data_dir'] + config['data']['train_file']
test_file = config['data']['processed_data_dir'] + config['data']['test_file']

print(train_file)
print(test_file)
# Load datasets from pickle files
datatype1 = pd.read_pickle(train_file) # background
datatype2 = pd.read_pickle(test_file) # signal

# Features to analyze
features = ['pt', 'eta', 'phi', 'd0/d0Err', 'dz/dzErr']

# Create output directory
output_dir = 'plots/test-plots/raw_feature_comparison/'
os.makedirs(output_dir, exist_ok=True)

# Helper: Extract feature values from dataset

def extract_feature_values(dataset, feature):
    cleaned = []
    
    for i, x in enumerate(dataset[feature].values):
        try:
            if isinstance(x, (list, np.ndarray)):
                if hasattr(x, '__len__') and len(x) > 0:
                    cleaned.extend(x)
                else:
                    # This handles numpy scalars (e.g. 0-dim arrays)
                    cleaned.append(x.item() if hasattr(x, "item") else x)
            else:
                # Scalar fallback (pure float/int)
                cleaned.append(x)
        except Exception as e:
            print(f"[!] Skipping index {i}: {e} — value: {x} (type={type(x)})")

    return np.array(cleaned)


# Loop over features and generate plots
f# Loop over features and generate side-by-side plots
for feature in features:
    bg_values = extract_feature_values(datatype1, feature)
    sig_values = extract_feature_values(datatype2, feature)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # Background plot
    axes[0].hist(bg_values, bins=100, density=True, alpha=0.7, color='blue')
    axes[0].set_title(f'Background: {feature}')
    axes[0].set_xlabel(feature)
    axes[0].set_ylabel('Normalized Frequency')
    axes[0].grid(True)

    # Signal plot
    axes[1].hist(sig_values, bins=100, density=True, alpha=0.7, color='red')
    axes[1].set_title(f'Signal: {feature}')
    axes[1].set_xlabel(feature)
    axes[1].grid(True)

    plt.suptitle(f'Raw Distribution Comparison: {feature}')
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for the suptitle

    safe_name = feature.replace('/', '_')
    plt.savefig(f"{output_dir}raw_distribution_{safe_name}_side_by_side.png")
    plt.close()
