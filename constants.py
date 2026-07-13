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

# for graph structure 
CLOSEST_NEIGHBORS = 10
GRAPH_METHODS = ("eta_phi", "all_features", "fully_connected", "mass_knn", "hybrid_knn")
PT_MAX = 2000
PT_MIN = 200

# DeepNTuplizer AK8 jet-level metadata written with fj_ prefix.
RAW_FATJET_PROPERTIES = [
    "phi",
    "eta",
    "pt",
    "mass",
    "qk_charge_05",
    "qk_charge_10",
]

# Legacy NanoAOD fat-jet branches kept for backward compatibility with old pickles.
RAW_FATJET_PROPERTIES_NANOAOD = [
    "phi", "eta", "pt", "mass", "msoftdrop",
    "particleNetWithMass_QCD", "particleNet_XbbVsQCD",
    "particleNet_XccVsQCD", "particleNet_XqqVsQCD",
    "particleNet_QCD", "particleNet_massCorr",
]
# to distinguish from the processed columns
RAW_FATJET_PROPERTIES_PREFIX = "fj_"

# Fixed PDG one-hot columns expected by LeJEPA PART training.
STANDARD_PDG_ONEHOT_IDS = [-211, -13, -11, 11, 13, 22, 130, 211]