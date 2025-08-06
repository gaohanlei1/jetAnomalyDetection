import pandas as pd

# Load the .pkl file
df = pd.read_pickle("/home/anagaman/jet-anomaly-summer25/jetAnomalyDetection_updated/jetAnomalyDetection/data/preprocessed/qcd/qcd_with_metadata2/QCD_PT-170to300_TuneCP5_13p6TeV_pythia8_1_Pt-_38364_218-0852-21.pkl")

# Show column names
print("Columns in the DataFrame:")
print(df.columns)

# Optional: show a few rows for verification
print("\nSample rows:")
print(df.head())
