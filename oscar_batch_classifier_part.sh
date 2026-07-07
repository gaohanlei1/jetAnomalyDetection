#!/bin/bash

#SBATCH --job-name=jet-cls-part
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/oscar-cls-part-%j.out
#SBATCH --error=logs/oscar-cls-part-%j.err

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: sbatch $0 BACKGROUND.pkl SIGNAL.pkl [OUTPUT_DIR] [classifier args...]"
    exit 2
fi

BACKGROUND_FILE=$1
SIGNAL_FILE=$2
OUTPUT_DIR=${3:-"runs/classifier-part-upper-bound-${SLURM_JOB_ID}"}
EXTRA_ARGS=("${@:4}")
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

# Supervised PART classifier benchmark. This is an approximate upper-bound reference
# for the same processed background/signal pickle pair used by PART + LeJEPA.
python -u scripts/run_train_classifier_part.py \
    --background "${BACKGROUND_FILE}" \
    --signal "${SIGNAL_FILE}" \
    --embed-dim 128 \
    --num-layers 8 \
    --num-heads 8 \
    --batch-size 128 \
    --epochs 50 \
    --learning-rate 5e-4 \
    --weight-decay 5e-2 \
    --precision bf16 \
    --seed 42 \
    --output-dir "${OUTPUT_DIR}" \
    "${EXTRA_ARGS[@]}"
