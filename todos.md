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



## Tue - 15.07.25

**DONE:**
- Concatenated, organised, and sent all preprocessed data files to Arjun
    - Processed `QCD_PT-170to300_13p6TeV + WJetsToQQ_HT-400to600`

- Slight fixes here n there, adding more fields to processed data visualisation (like Arjun's done)

- Joined and organised preproc'd files, sent the link to Arjun so he can use them for processing (takes 5-20 mins) and training
    - processing now just takes `python3.9 scripts/processing.py -b <qcdpath.pkl> -s <wjetpath.pkl>`

- Have been trying to transition more into the technical/model side of things, but a bunch of technical issues still pop up
    - Virtual environment was still incomplete, so I finally diagnosed the issues and found the correct versions to install
    - still may need tweaking if I try to use GPU or CPU drivers; for now, moved everything to CPU

- Starting to train the old autoencoder w/ the new preproc files; shows my preproc + proc pipeline works, loss decreases
    - doesn't have Arjun's modifications yet

- (small fun stuff) Added subfolder/concat support, refactored pre/processing, removed `ak.to_numpy` etc.









## Fri - 18.07.25

**NOW:**

- Uprooting the `.root` files, plotting distributions of different features flattened out!
    - and overlaying similar features to see how similar they are
    - arjun hasn't uprooted the new .root files - do it and save the plots!
    - correlation heatmaps? check out arjun's one

- Diagnosing the QCD50to80 issue - why do all events get rejected?
    - uprooting is confusing me a bit; try using arjun's pt_comparison.py to graph the FatJet_pt distributions for all QCDs
        - why are they mostly empty??
    - need to check/graph the data during get_fatjets() in preprocessing.py, too!
        - different fields? is it still pt, or eta, or etc all at the same time?
        - maybe count the number of events that are filtered out by certain filters? `len(fj[!filter])`
    - the main thing to do is cross-check with the other .root files, see what's different

- Using the visualize() func in processing.py to visualise stuff like the zeroes after processing
    - what gets excluded when `not include_zeros`? how are scaled zeroes distributed?
    - also to visualise processed distributions in general across all fields
        - errors out at mass I think, coz of inf/nan errors! try to filter out and redo

- Summing ALL the QCD data files, and THEN taking the 200-300 GeV Pt slice from both QCD and WJet (WWto4Q)
    - point is more data + to see if the autoencoder can learn even when the Pt ranges are so similar
    - tool to do this! lol
        - and save diff ranges to train with?

- After merging w/ main, start properly merging Arjun's changes into the repo on a new branch
    - familiarise w/ the visualisations and other tools, update if needed (e.g. graphing other properties)
    - familiarise w/ the model and what's been done so far
        - what features have been used to train? how have the edges been modified? what hasn't been tried?
        - what regions of parameters have been sweeped, and what haven't?

- Merge w/ main again, and start streamlining/updating Arjun's code
    - save different useful plots permanently (analysing all the raw/processed data)
    
- Look into ways to use Brux's GPU!
    - if successful, we can train for way more epochs hopefully

- **Todos on Slack!!**
    - using PFCands instead of FatJet?






