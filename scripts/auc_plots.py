import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Add the parent directory to Python's path to allow local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helpers import helpers_main
config = helpers_main.load_config()

# Load the sweep results with AUC included
df = pd.read_csv('sweeps/autoencoder_param_sweep.csv')

# --------- HEATMAP (using AUC) ---------
# Choose fixed values for two other parameters to slice the data
fixed_wd = 0.0
fixed_k = 16

# Filter dataframe for those values
heatmap_df = df[(df['weight_decay'] == fixed_wd) & (df['nearest_neighbors'] == fixed_k)]

# Create pivot table: rows = smallest_dim, columns = learning_rate
pivot_auc = heatmap_df.pivot(index='smallest_dim', columns='learning_rate', values='auc_score')

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(pivot_auc, annot=True, fmt=".3f", cmap='viridis', cbar_kws={'label': 'AUC'})
plt.title(f"AUC Heatmap (weight_decay={fixed_wd}, k={fixed_k})")
plt.xlabel("Learning Rate")
plt.ylabel("Latent Dimension (smallest_dim)")
plt.tight_layout()
plt.savefig(f"sweeps/auc_heatmap_{helpers_main.curr_time()}.png")
if config["dbg"]["show_plots"]: plt.show()

# --------- 3D SCATTER PLOT (using AUC) ---------
# Convert to log scale for better spread
df['log_lr'] = np.log10(df['learning_rate'])
df['log_wd'] = np.log10(df['weight_decay'].replace(0.0, 1e-6))  # avoid log(0)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(
    df['log_lr'],
    df['log_wd'],
    df['smallest_dim'],
    c=df['auc_score'],
    cmap='coolwarm',
    s=80,
    alpha=0.9
)

ax.set_xlabel('log10(Learning Rate)')
ax.set_ylabel('log10(Weight Decay)')
ax.set_zlabel('Latent Dim (smallest_dim)')
ax.set_title('3D Scatter Plot of Hyperparameters vs AUC')
fig.colorbar(sc, label='AUC Score')
plt.tight_layout()
plt.savefig(f"sweeps/auc_3d_plot_{helpers_main.curr_time()}.png")
if config["dbg"]["show_plots"]: plt.show()
