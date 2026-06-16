#!/bin/bash

#SBATCH --job-name=jet-ae
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/oscar-ae-%j.out
#SBATCH --error=logs/oscar-ae-%j.err

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: sbatch $0 BACKGROUND.pkl SIGNAL.pkl [OUTPUT_DIR]"
    exit 2
fi

BACKGROUND_FILE=$1
SIGNAL_FILE=$2
OUTPUT_DIR=${3:-"runs/gae-${SLURM_JOB_ID}"}
PROJECT_DIR=${PROJECT_DIR:-"${SLURM_SUBMIT_DIR}"}

cd "${PROJECT_DIR}"

if [[ ! -r "${BACKGROUND_FILE}" ]]; then
    echo "Background dataset is not readable: ${BACKGROUND_FILE}"
    exit 1
fi
if [[ ! -r "${SIGNAL_FILE}" ]]; then
    echo "Signal dataset is not readable: ${SIGNAL_FILE}"
    exit 1
fi
if [[ ! -x ".venv/bin/python" ]]; then
    echo "Missing .venv. Create the Oscar environment before submitting."
    exit 1
fi

module purge
module load cuda
source .venv/bin/activate

export JET_DEVICE=cuda
export PYTHONUNBUFFERED=TRUE
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "Job ID: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "Project: ${PROJECT_DIR}"
echo "Background: ${BACKGROUND_FILE}"
echo "Signal: ${SIGNAL_FILE}"
echo "Output: ${OUTPUT_DIR}"
nvidia-smi

python -u -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

# Baseline settings based on the previous graph-autoencoder experiment.
# The input pickle files may be newly generated rather than exact old data.
python -u scripts/run_train_autoencoder.py \
    --background "${BACKGROUND_FILE}" \
    --signal "${SIGNAL_FILE}" \
    --method eta_phi \
    --knn 16 \
    --smallest-dim 16 \
    --num-reduced-edges 16 \
    --batch-size 64 \
    --epochs 20 \
    --learning-rate 1e-5 \
    --weight-decay 5e-4 \
    --no-lr-scheduler \
    --no-normalize-features \
    --seed 42 \
    --output-dir "${OUTPUT_DIR}"
