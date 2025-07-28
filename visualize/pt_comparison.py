import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt

'''
Arjun's code
'''

# Replace with your actual file paths
file1 = "data/QCD_PT-1000to1400_TuneCP5_13p6TeV_pythia8_1.root"
file2 = "data/WWto4Q_TuneCP5_13p6TeV_powheg-pythia8_1.root"

def get_pt_array(filename, branch="FatJet_pt", treename="Events"):
    with uproot.open(filename) as file:
        tree = file[treename]
        pt_array = tree[branch].array(entry_stop=None)  # Awkward array
        return ak.flatten(pt_array)

# Load pT values
pt1 = get_pt_array(file1)
pt2 = get_pt_array(file2)

pt1_filtered = pt1#[(pt1 >= 1000) & (pt1 <= 1400)]
pt2_filtered = pt2#[(pt2 >= 1000) & (pt2 <= 1400)]

print(f"Number of jets in File 1 (QCD) after filtering (1000-1400 GeV): {len(pt1_filtered)}")
print(f"Number of jets in File 2 (WWto4Q) after filtering (1000-1400 GeV): {len(pt2_filtered)}")

# Plot histograms
plt.figure(figsize=(8,6))
plt.hist(pt1_filtered, bins=200, density=True, histtype='step', label='QCD', color='blue')
plt.hist(pt2_filtered, bins=200, density=True, histtype='step', label='WWto4Q', color='orange')
plt.xlabel("Jet pT [GeV]")
plt.ylabel("Normalized Frequency")
plt.title("Jet pT Distribution Comparison (1000-1400 GeV)")
plt.legend()
plt.grid(True)
# plt.xlim(200, 300)  # Set x-axis limits to match the filtered range
plt.tight_layout()
plt.savefig("pt_comparison_1000_1400GeV.png")
plt.show()