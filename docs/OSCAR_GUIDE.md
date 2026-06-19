# From Raw Data to a Graph-Autoencoder Job on Oscar

This is the single workflow to use from now on. It creates a new dataset from
the shared raw files and trains only the graph autoencoder.

## What is happening

You are working with three machines:

1. **Your Mac** holds the code you edit in Cursor.
2. **Brux** holds the shared raw ROOT files from the previous project.
3. **Oscar** provides scheduled CPU and GPU compute resources.

The data will move through these stages:

```text
Three raw ROOT files on Brux
        |
        | copy to Oscar
        v
Intermediate pickle chunks on Oscar
        |
        | feature engineering and QCD-based scaling
        v
QCD_scaled.pkl + WJet_scaled.pkl
        |
        | graph-autoencoder GPU job
        v
AUC, plots, losses, and summary.json
```

The old `QCD_scaled.pkl` and `WJet_scaled.pkl` are references only. You will
create your own pair.

## How Oscar jobs work

When you SSH into Oscar, you arrive on a **login node**. Use it to move files,
install the Python environment, submit jobs, and inspect results. Do not run
the heavy preprocessing or training commands directly there.

Oscar uses a scheduler called **Slurm**. A batch script states:

- what resources are needed, such as CPUs, memory, time, or a GPU;
- which commands should run;
- where terminal output and errors should be saved.

`sbatch script.sh ...` submits the request and immediately returns a job ID.
The job may wait in the queue (`PD`) before running (`R`) on a compute node.
Disconnecting your terminal does not stop a submitted job.

## Files selected for this dataset

Use the samples that overlap the 200-400 GeV jet-pT range:

```text
QCD_PT-170to300_TuneCP5_13p6TeV_pythia8_1.root
QCD_PT-300to470_TuneCP5_13p6TeV_pythia8_1.root
WWto4Q_TuneCP5_13p6TeV_powheg-pythia8_1.root
```

The first two files are QCD background. The last file supplies the W-jet
signal. The preprocessing commands apply the final 200-400 GeV selection.

## 1. Copy this code from your Mac to Oscar

Run this in a terminal on your Mac, not inside an Oscar session:

```bash
rsync -az --progress \
  --exclude '.venv' \
  --exclude '*.root' \
  --exclude '*.pkl' \
  --exclude 'runs' \
  /Users/gaohanlei1/Documents/Codex/jetAnomalyDetection/ \
  hgao50@ssh.ccv.brown.edu:~/jetAnomalyDetection/
```

This transfers the current code, including the Oscar scripts, but not local
environments or large data files.

## 2. Create Oscar storage directories

Connect to Oscar:

```bash
ssh hgao50@ssh.ccv.brown.edu
```

Then create the directories:

```bash
mkdir -p ~/data/jet-anomaly/raw
mkdir -p ~/data/jet-anomaly/datasets
mkdir -p ~/scratch/jet-anomaly-runs
mkdir -p ~/jetAnomalyDetection/logs
```

`~/data` is for durable inputs and datasets. `~/scratch` is faster temporary
storage for job results; files there can be purged after they have not been
accessed for 30 days.

## 3. Copy the three ROOT files from Brux to Oscar

Log out of Oscar and connect to Brux:

```bash
ssh hgao50@brux.hep.brown.edu
```

Define the old project location:

```bash
PROJECT=/isilon/export/home/anagaman/jet-anomaly-summer25/jetAnomalyDetection_updated/jetAnomalyDetection
```

Confirm that the three source files resolve through their symbolic links:

```bash
ls -lhL \
  "$PROJECT/data/raw/qcd/QCD_PT-170to300_TuneCP5_13p6TeV_pythia8_1.root" \
  "$PROJECT/data/raw/qcd/QCD_PT-300to470_TuneCP5_13p6TeV_pythia8_1.root" \
  "$PROJECT/data/raw/wjet/WWto4Q_TuneCP5_13p6TeV_powheg-pythia8_1.root"
```

Copy the actual file contents to Oscar:

```bash
rsync -avP -L \
  "$PROJECT/data/raw/qcd/QCD_PT-170to300_TuneCP5_13p6TeV_pythia8_1.root" \
  "$PROJECT/data/raw/qcd/QCD_PT-300to470_TuneCP5_13p6TeV_pythia8_1.root" \
  "$PROJECT/data/raw/wjet/WWto4Q_TuneCP5_13p6TeV_powheg-pythia8_1.root" \
  hgao50@ssh.ccv.brown.edu:~/data/jet-anomaly/raw/
```

`-L` matters because the Brux entries are symbolic links. It tells `rsync` to
copy the ROOT files themselves instead of copying links that would be broken
on Oscar.

If direct `rsync` from Brux is blocked, use Brown's Globus transfer service or
ask CCV support for the correct Brux-to-Oscar transfer route.

Reconnect to Oscar and verify the transfer:

```bash
ssh hgao50@ssh.ccv.brown.edu
ls -lh ~/data/jet-anomaly/raw
```

## 4. Install the Python environment once

Run this on the Oscar login node:

```bash
cd ~/jetAnomalyDetection
bash setup_venv.sh
source start_venv.sh
python --version
```

The virtual environment contains the precise Python libraries needed by the
project, including PyTorch, PyTorch Geometric, Coffea, Pandas, and plotting
libraries. It prevents the project from depending on whatever happens to be
installed system-wide.

Only run `setup_venv.sh` once. In later login sessions, activate the existing
environment with:

```bash
cd ~/jetAnomalyDetection
source start_venv.sh
```

Install packages before submitting jobs. Oscar compute nodes do not have
general Internet access.

## 5. Submit the CPU data-preparation job

From the repository on Oscar:

```bash
cd ~/jetAnomalyDetection
mkdir -p logs

sbatch oscar_batch_prepare_data.sh \
  "$HOME/data/jet-anomaly/raw/QCD_PT-170to300_TuneCP5_13p6TeV_pythia8_1.root" \
  "$HOME/data/jet-anomaly/raw/QCD_PT-300to470_TuneCP5_13p6TeV_pythia8_1.root" \
  "$HOME/data/jet-anomaly/raw/WWto4Q_TuneCP5_13p6TeV_powheg-pythia8_1.root" \
  "$HOME/data/jet-anomaly/datasets/pt200to400-v1"
```

This job uses the general-purpose `batch` partition. It:

1. selects jets with pT between 200 and 400 GeV;
2. turns each ROOT file into intermediate pickle chunks;
3. combines and feature-engineers the QCD and WJet events;
4. derives scaling values from QCD;
5. applies the same scaling to both QCD and WJet;
6. writes the final training pair.

`sbatch` prints something like:

```text
Submitted batch job 12345678
```

Here, `12345678` is the job ID. Check its state:

```bash
myq
```

Watch its normal output:

```bash
tail -f logs/oscar-data-12345678.out
```

Watch errors in another terminal:

```bash
tail -f logs/oscar-data-12345678.err
```

Press `Ctrl+C` to stop watching a log. This does not cancel the job.

When `myq` no longer lists the job, inspect its final status:

```bash
sacct -j 12345678 --format=JobID,JobName,State,Elapsed,ExitCode
```

`COMPLETED` and exit code `0:0` mean it succeeded.

## 6. Check your new pickle files

The completed data job should create:

```text
~/data/jet-anomaly/datasets/pt200to400-v1/processed/PT-200to400/scaledby_QCD/QCD_scaled.pkl
~/data/jet-anomaly/datasets/pt200to400-v1/processed/PT-200to400/scaledby_QCD/WJet_scaled.pkl
```

Check their sizes and DataFrame shapes:

```bash
DATASET="$HOME/data/jet-anomaly/datasets/pt200to400-v1/processed/PT-200to400/scaledby_QCD"

ls -lh "$DATASET/QCD_scaled.pkl" "$DATASET/WJet_scaled.pkl"
python helpers/print_df_info.py --path "$DATASET/QCD_scaled.pkl"
python helpers/print_df_info.py --path "$DATASET/WJet_scaled.pkl"
```

These are your new background and signal datasets. Keep them in `~/data`.

## 7. Submit the graph-autoencoder GPU job

Submit the training job:

```bash
DATASET="$HOME/scratch/jet-anomaly-work/datasets/pt200to400-v1/processed/PT-200to400/scaledby_QCD"

cd ~/jetAnomalyDetection

sbatch oscar_batch_ae.sh \
  "$DATASET/QCD_scaled.pkl" \
  "$DATASET/WJet_scaled.pkl" \
  "$HOME/scratch/jet-anomaly-runs/gae-pt200to400-v1"
```

This job requests one GPU from the `gpu` partition. Slurm waits until a
suitable GPU is free, starts the script on that compute node, activates your
environment, trains the graph autoencoder, and evaluates WJet events as
anomalies relative to QCD.

Monitor it using the new job ID:

```bash
myq
tail -f logs/oscar-ae-<job-id>.out
tail -f logs/oscar-ae-<job-id>.err
```

Pending jobs can remain in `PD` while Oscar waits for resources. To see the
estimated start time:

```bash
squeue -u "$USER" -t PENDING --start
```

To cancel a mistaken job:

```bash
scancel <job-id>
```

## 8. Read the result

After the GPU job completes:

```bash
RESULT="$HOME/scratch/jet-anomaly-runs/gae-pt200to400-v1"

cat "$RESULT/summary.json"
ls -lh "$RESULT"
```

The key number is `auc`:

- `0.5` means random signal/background ranking;
- larger values mean better anomaly separation;
- the previous graph-autoencoder result of about `0.606` is context, not a
  required target, because you created a new dataset.

The directory also contains the ROC curve, anomaly-score plot, loss plot, and
the saved background and signal loss arrays.

Results in `~/scratch` should eventually be copied into durable project or
group storage.

## Common job states and failures

| State | Meaning |
|---|---|
| `PD` | Waiting in the queue |
| `R` | Running |
| `COMPLETED` | Finished successfully |
| `FAILED` | The program exited with an error |
| `OUT_OF_MEMORY` | The job needed more RAM |
| `TIMEOUT` | It exceeded the requested time |
| `CANCELLED` | It was cancelled |

The first place to investigate a failure is the matching `.err` file in
`~/jetAnomalyDetection/logs`. Also run:

```bash
sacct -j <job-id> --format=JobID,JobName,State,Elapsed,ExitCode,MaxRSS
```

## Official Oscar references

- [Batch jobs](https://docs.ccv.brown.edu/oscar/submitting-jobs/batch)
- [GPU jobs](https://docs.ccv.brown.edu/oscar/gpu-computing/submit-gpu)
- [Managing jobs](https://docs.ccv.brown.edu/oscar/submitting-jobs/managing-jobs)
- [Oscar storage](https://docs.ccv.brown.edu/oscar/managing-files/filesystem)
- [File transfers](https://docs.ccv.brown.edu/oscar/managing-files/filetransfer)
