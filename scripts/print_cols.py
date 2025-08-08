# import pandas as pd

# # Load the .pkl file
# df = pd.read_pickle("")
# print("Columns in the DataFrame:")
# print(df.columns)

# # Optional: show a few rows for verification
# print("\nSample rows:")
# print(df.head())

import uproot
fatjet_branches = [key for key in uproot.open("/home/anagaman/jet-anomaly-summer25/jetAnomalyDetection_updated/jetAnomalyDetection/data/raw/wjet/WminusH_Hto2B_WtoLNu_M-125_TuneCP5_13p6TeV_powheg-pythia8_1.root")["Events"].keys() if key.startswith("FatJet")]
print(fatjet_branches)
