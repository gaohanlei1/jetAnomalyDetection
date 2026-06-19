"""
Plot all input variables / 
plot 2D distributions of phi vs eta centered around the jet-axis, then on the 
Z-axis (color) plot the value of each variables.

Input: a path to pkl files containing data. Doesn't matter if it's bg data or 
sg data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set the path to your data file and the variables you want to plot!
# PATH_TO_DATA = "data/processed/PT-200to400/QCD_scaled.pkl"
PATH_TO_DATA = "data/processed/PT-200to400/WJet_scaled.pkl"
PLOT_TITLE = f"Variable Distributions for Scaled WJet Data"
Z_VARIABLES = ['pt', 'd0/d0Err', 'dz/dzErr']
IS_SCALED = "_scaled" in PATH_TO_DATA
NUM_SAMPLES = 1  # number of samples to plot, -1 to plot all

data = pd.read_pickle(PATH_TO_DATA)
if NUM_SAMPLES != -1:
    data = data.sample(n=NUM_SAMPLES)
    PLOT_TITLE += f" ({NUM_SAMPLES} Events Subset)"

for var in Z_VARIABLES:
    assert var in data.columns, f"Variable {var} not found in data columns."
    
num_events = data.size
print(f"Loaded {num_events} events from {PATH_TO_DATA}")

def flatten_and_concat(column_name):
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
    
phi_all_flatten = flatten_and_concat('phi') # use as x-axis
eta_all_flatten = flatten_and_concat('eta') # use as y-axis

# use phi and eta to create 2D scatter plots, colored by each variable in Z_VARIABLES
fig, ax = plt.subplots(1, len(Z_VARIABLES), figsize=(6 * len(Z_VARIABLES), 5))
for i, var in enumerate(Z_VARIABLES):
    if var == 'pt':
        z_values = flatten_and_concat('pt')
    elif var == 'd0/d0Err':
        z_values = flatten_and_concat('d0/d0Err')
    elif var == 'dz/dzErr':
        z_values = flatten_and_concat('dz/dzErr')
    else:
        raise ValueError(f"Unknown variable {var}")

    scatter = ax[i].scatter(phi_all_flatten, eta_all_flatten, c=z_values, cmap='viridis', s=4, alpha=0.3)
    ax[i].set_title(f"{var} distribution")
    ax[i].set_xlabel("Phi")
    ax[i].set_ylabel("Eta")
    # fix each axis to be between -0.9 and 0.9
    ax[i].set_xlim(-0.9, 0.9)
    ax[i].set_ylim(-0.9, 0.9)
    # ensure the aspect ratio is equal
    ax[i].set_aspect('equal', adjustable='box')
    fig.colorbar(scatter, ax=ax[i], label=var if not IS_SCALED else f"{var} (scaled)")
plt.suptitle(PLOT_TITLE)
plt.tight_layout()
plt.show()