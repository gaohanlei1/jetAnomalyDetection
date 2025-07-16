# To-dos

## Fri - 11.07.25

**NOW:**
- Analysing Pt distributions of the new btv-nano data between a matching QCD and WJet pair:
    - from the raw ROOT files (Arjun already did this, showed pretty big separation)
    - after preprocessing and combining into one file each (does my preprocessing squish the separation, somehow?)
    - after processing
    - all of the above, but with log Pt instead; this is what `processing.py` currently does, but it shows very little separation compared to Arjun's plots
- Could plot pt/eta/phi of fatjets!


- Parameter sweeps take a LOOONG while (hours), so we should try to use the GPU on Brux or LXPlus, or speed it up anyhow
    - GPU computation
    - using JAX or other JIT options
    - playing around w/ the number of workers etc.
- Counterpoint: not



## Tue - 15.07.25 (sis's bday!!!)

**DONE:**
- Updated readme

- Added subfolder/concat support, refactored pre/processing, etc.
    - removed `ak.to_numpy` and other stuff

- Concatenated, organised, and sent all preprocessed data files to Arjun
    - Processed `QCD_PT-170to300_13p6TeV + WJetsToQQ_HT-400to600`


- Slight fixes here n there, adding more fields to processed data visualisation (pt, eta, phi)
    - still bugging a bit

- Joined and organised preproc'd files, sent the link to Arjun so he can use them for processing (takes 5-20 mins) and training
    - processing now just takes `python3.9 scripts/processing.py -b <qcdpath.pkl> -s <wjetpath.pkl>`

- Have been trying to transition more into the technical/model side of things, but a bunch of technical issues still pop up
    - Virtual environment was still incomplete, so I finally diagnosed the issues and found the correct versions to install
    - still may need tweaking if I try to use GPU or CPU drivers; for now, moved everything to CPU

- Starting to train the old autoencoder w/ the new preproc files; shows my preproc + proc pipeline works, loss decreases
    - doesn't have Arjun's modifications yet


**NOW:**

- Since Arjun's still busy training, with his permission I'm merging his work on brux into a new branch
    - doable since he hasn't touched the preproc/proc scripts much, it's the training + viz scripts

- Familiarising myself with the model changes and the visualisations
    - Graphing properties other than log_pt
        - Arjun did this with WWto4Q, so I'll do it w/ the other preproc'd WJet files
    - Analysing the new `.root`s themselves w/ uproot 

- Using processed jet pairs to train Arjun's model
    - then starting to work on the autoencoder myself, perform param sweeps etc

- Training takes a while (34 epochs -> 20 mins), so I'll look into using the Brux GPU (promised for a while - almost getting to it!!!)


**extras:**

- Diagnosing the QCD50to80 issue - why do all events get rejected?

- Maybe concatenating the old data files in multiple parts? For WJets_HT-400to600

- For `preprocess/feature_engineering.py`, the one-hot lists can be sped up using vectorisation; sth like:
```py
df[f"pdgId_{PDG_ID}"] = (df["pdgId"] == pdg_id).astype(int) for pdg_id in VALID_PDGS
```
    - Currently takes a bunch of minutes for 150 MB preprocessed files; so 10 mins for 1 GB etc.

