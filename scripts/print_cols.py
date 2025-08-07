import pandas as pd

# Load the .pkl file
df = pd.read_pickle("/home/anagaman/jet-anomaly-summer25/jetAnomalyDetection_updated/jetAnomalyDetection/data/processed/scaledby_QCD_PT-80to470/scaledby_QCD/other/QCD_scaled.pkl")
print("Columns in the DataFrame:")
print(df.columns)

# Optional: show a few rows for verification
print("\nSample rows:")
print(df.head())
