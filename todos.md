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
    - Should start processing pairs


**NOW:**

- Checking and merging w/ Arjun's changes pre-emptively; or at least, making a branch?

- The graphing above

- Using processed jet pairs to ACTUALLY TRAIN THE AUTOENCODER (after merging w/ Arjun's)
    - then starting to work on the autoencoder, e.g. the device config



*extras:*

- Diagnosing the QCD50to80 issue - why do all events get rejected?

- Maybe concatenating the old data files in multiple parts? For WJets_HT-400to600


