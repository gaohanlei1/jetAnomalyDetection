# To-dos

## Fri - 11.07.25

**DONE:**
- Finished preprocessing all the new files
- Script to join dfs together
- Processing takes a while on large dfs, diagnosing this 

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
- Counterpoint: not needed


- Adding arguments to preprocessing:
    - `--subfolder` option to save preprocessed files in their own subfolders from each `.root`
    - `--concat` option to concatenate saved `.root` files automatically

- Adding `--qcd-name` and `--wjet-name` options to processing to be able to specify the filenames directly, instead of from `config.yaml`

- Updating readme after the above; maybe replacing the main README?

- TRY REMOVING `ak.to_numpy` FROM PREPROCESSING! Could do it w/ just `ak.flatten`?


## Tue - 15.07.25 (sis's bday!!!)

**DONE:**
- Updated readme, added subfolder support, etc.
- 
