# for preprocessing
ELECTRON_PT_LOWER_BOUND = 20.0 
MUON_PT_LOWER_BOUND = 20.0

ELECTRON_R_LOWER_BOUND = 0.4
MUON_R_LOWER_BOUND = 0.4
MATCHED_GEN_R_LOWER_BOUND = 0.4

FATJET_PT_LOWER_BOUND = 200.0
FATJET_ETA_BOUNDS = 2.0

FATJET_DELTA_ETA_BOUND = 0.1 
FATJET_DELTA_PHI_BOUND = 0.1 
FATJET_DELTA_PT_BOUND = 1

# for histogram binning
# this requires eta bounds are the same as phi bounds 
ETA_MIN = -0.8 
ETA_MAX = 0.8 
PHI_MIN = -0.8 
PHI_MAX = 0.8 
INCR = 0.025 *2
BINS = int(abs(ETA_MIN - ETA_MAX)/INCR)

# for graph structure 
CLOSEST_NEIGHBORS = 10
GRAPH_METHODS = ("eta_phi", "all_features", "fully_connected", "mass_knn", "hybrid_knn")
PT_MAX = 400
PT_MIN = 200

# for preproc/proc data/names
RAW_FATJET_PROPERTIES = [
    "phi", "eta", "pt", "mass", "msoftdrop",
    "particleNetMD_QCD", "particleNetMD_Xbb", "particleNetMD_Xcc", "particleNetMD_Xqq",
    "particleNet_QCD", "particleNet_mass"
]
# to distinguish from the processed columns
RAW_FATJET_PROPERTIES_PREFIX = "rawfj_"