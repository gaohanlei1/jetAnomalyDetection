import pandas as pd
import plotly.express as px
import plotly.io as pio
import os

# Show plots in your default browser
pio.renderers.default = 'browser'

# Load your sweep results CSV
df = pd.read_csv("sweeps/autoencoder_param_sweep.csv")

# Drop rows with missing AUC scores
df = df.dropna(subset=["auc_score"])

# Loop through each unique weight_decay value
for wd in sorted(df['weight_decay'].unique()):
    # Filter dataframe
    sub_df = df[df['weight_decay'] == wd]

    # Create interactive 3D scatter plot
    fig = px.scatter_3d(
        sub_df,
        x="learning_rate",
        y="nearest_neighbors",
        z="smallest_dim",
        color="auc_score",
        size="auc_score",
        hover_data=["auc_score"],
        title=f"3D AUC Scatter (weight_decay={wd})"
    )

    # Axis labels
    fig.update_layout(scene=dict(
        xaxis_title='Learning Rate',
        yaxis_title='k-NN Neighbors',
        zaxis_title='Latent Dimension'
    ))

    fig.show()