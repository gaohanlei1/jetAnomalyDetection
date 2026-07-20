#!/bin/bash
#SBATCH --nodes=1
# Prefer L40S when available, but do not hard-pin a single node.
#SBATCH -p gpu --gres=gpu:l40s:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00
#SBATCH --mem=128000MB
#SBATCH --job-name='JETANOMALY-AK8-TRIP'
#SBATCH --output=slurm_logs/R-%x.%j/log.out
#SBATCH --error=slurm_logs/R-%x.%j/log.err
export PYTHONIOENCODING=utf-8
set -euo pipefail

echo ""
echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="
echo ""

echo "GPU Information (from host):"
nvidia-smi
echo ""

# Use the repo venv built from requirements (CUDA torch). Login nodes
# report cuda=False; GPU nodes should see the device.
# Under Slurm, BASH_SOURCE points at a spool copy — prefer submit dir.
REPO_DIR="${SLURM_SUBMIT_DIR:-/users/hgao50/jetAnomalyDetection}"
cd "${REPO_DIR}"
# shellcheck disable=SC1091
source "${REPO_DIR}/.venv/bin/activate"
python -c "import torch; print(f'PyTorch {torch.__version__}, cuda={torch.cuda.is_available()}, device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
if ! python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
  echo "ERROR: torch cannot see a GPU in this allocation" >&2
  exit 1
fi

# Reproduce plots/run-lejepa-trip-part-prob-aug-fracb-augb-4n (AUC ~0.643)
# on ak8-v2 DeepNTuplizer ROOT, scaled above the old ~100k QCD / ~4k WJet pickle.

NTUPLE_ROOT=${NTUPLE_ROOT:-"/HEP/export/home/hgao50/jet-anomaly-data/ak8-v2"}
OUTPUT_DIR=${OUTPUT_DIR:-"plots/run-lejepa-ak8-triplet-fracb-scaled"}

# Prefer kinematics close to the old pt 200-400 pickle:
# QCD Pythia bins that populate that window + WJets PTQQ200.
STAGE_DIR=${STAGE_DIR:-"${TMPDIR:-/tmp}/ak8_triplet_${SLURM_JOB_ID:-$$}"}
mkdir -p "${STAGE_DIR}/qcd" "${STAGE_DIR}/wjets"
ln -sfn "${NTUPLE_ROOT}"/qcd/qcd_PT170to300_*.root "${STAGE_DIR}/qcd/"
ln -sfn "${NTUPLE_ROOT}"/qcd/qcd_PT300to470_*.root "${STAGE_DIR}/qcd/"
ln -sfn "${NTUPLE_ROOT}"/wjets/wjets_Wto2Q_PTQQ200_*.root "${STAGE_DIR}/wjets/"

echo "Staged QCD files:   $(ls -1 "${STAGE_DIR}/qcd" | wc -l)"
echo "Staged WJets files: $(ls -1 "${STAGE_DIR}/wjets" | wc -l)"
echo "Output dir: ${OUTPUT_DIR}"

# Scaled-up event caps (old pickle: ~125k QCD total, ~4k WJet).
MAX_BG=${MAX_BG:-300000}
MAX_SG=${MAX_SG:-20000}
LOWERPT=${LOWERPT:-200}
UPPERPT=${UPPERPT:-400}

python -u scripts/run_train_lejepa_part.py \
  --background "${STAGE_DIR}/qcd" \
  --signal "${STAGE_DIR}/wjets" \
  --lowerpt "${LOWERPT}" \
  --upperpt "${UPPERPT}" \
  --max-background-events "${MAX_BG}" \
  --max-signal-events "${MAX_SG}" \
  --model "triplet" \
  --anomaly-score "mahalanobis" \
  --no-normalize-features \
  --embed-dim 128 \
  --representation-dim 128 \
  --num-layers 8 \
  --num-heads 8 \
  --batch-size 128 \
  --epochs 100 \
  --learning-rate 1e-3 \
  --weight-decay 5e-2 \
  --precision bf16 \
  --num-global-views 2 \
  --num-local-views 3 \
  --global-drop-pt-frac-min 0.0 \
  --global-drop-pt-frac-max 0.30 \
  --local-drop-pt-frac-min 0.30 \
  --local-drop-pt-frac-max 0.75 \
  --pairwise-hidden-dim 16 \
  --triplet-weight 0.1 \
  --triplet-margin 1.0 \
  --num-negative-views 4 \
  --batch-mix-prob 0.45 \
  --pt-resample-prob 0.25 \
  --node-eta-phi-rotation-prob 0.20 \
  --eta-phi-shuffle-prob 0.05 \
  --identity-shuffle-prob 0.05 \
  --output-dir "${OUTPUT_DIR}"

rm -rf "${STAGE_DIR}"
echo "Job finished at: $(date)"
