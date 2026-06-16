"""
Plot pt distributions for bg and sg data (scaled).

Already implemented in plot_distributions.py, but this is a separate script 
for clarity.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Plot bg distribution and sg distribution of pt
PATH_TO_DATA_BG = "data/processed/PT-200to400/QCD_scaled.pkl"
PATH_TO_DATA_SG = "data/processed/PT-200to400/WJet_scaled.pkl"


data_bg = pd.read_pickle(PATH_TO_DATA_BG)
data_sg = pd.read_pickle(PATH_TO_DATA_SG)


def flatten_and_concat(data, column_name):
    """Aggregate a column of arrays into a single flattened array, ignoring None and NaN values."""
    arrays = []
    for v in data[column_name]:
        if v is None:
            continue
        if isinstance(v, float) and np.isnan(v):
            continue

        arr = np.atleast_1d(np.asarray(v))

        if arr.size > 0:
            arrays.append(arr)

    return np.concatenate(arrays)
    
pt_all_flatten_bg = flatten_and_concat(data_bg, 'pt') # for histogram
pt_all_flatten_sg = flatten_and_concat(data_sg, 'pt') # for histogram

# two subplots, left one use density = False, right one use density = True
fig, ax = plt.subplots(1, 2, figsize=(16, 5))
# Left subplot: raw counts
ax[0].hist(pt_all_flatten_bg, bins=50, alpha=0.7, color='blue', label='Non-Anomalous (QCD)', density=False)
ax[0].hist(pt_all_flatten_sg, bins=50, alpha=0.7, color='red', label='Anomalous (WJet)', density=False)
ax[0].set_xlabel('Scaled pT')
ax[0].set_ylabel('Frequency')
ax[0].set_title('Distribution of Scaled pT for Scaled Background and Signal Data')
ax[0].legend()

# Right subplot: normalized counts
ax[1].hist(pt_all_flatten_bg, bins=50, alpha=0.7, color='blue', label='Non-Anomalous (QCD)', density=True)
ax[1].hist(pt_all_flatten_sg, bins=50, alpha=0.7, color='red', label='Anomalous (WJet)', density=True)
ax[1].set_xlabel('Scaled pT')
ax[1].set_ylabel('Density')
ax[1].set_title('Normalized Distribution of Scaled pT for Scaled Background and Signal Data')
ax[1].legend()

plt.tight_layout()
plt.show()