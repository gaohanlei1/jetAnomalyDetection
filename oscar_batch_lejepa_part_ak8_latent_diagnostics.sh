#!/bin/bash
#SBATCH --nodes=1
#SBATCH -p gpu --gres=gpu:l40s:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -t 04:00:00
#SBATCH --mem=90000MB
#SBATCH --job-name='JETANOMALY-AK8-DIAG'
#SBATCH --output=slurm_logs/R-%x.%j/log.out
#SBATCH --error=slurm_logs/R-%x.%j/log.err
export PYTHONIOENCODING=utf-8
set -euo pipefail

# CMS AK8 latent diagnostics (combined + per-class Mahalanobis AUCs).
#
# Env vars:
#   RUN_DIR   training run directory with summary.json + best_model.pth
#
# Example:
#   RUN_DIR=plots/run-lejepa-ak8-semisup-m1.0-hbb \
#     sbatch --export=ALL,RUN_DIR oscar_batch_lejepa_part_ak8_latent_diagnostics.sh

echo ""
echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURM_NODELIST:-local}"
echo "RUN_DIR=${RUN_DIR:-unset}"
echo "=========================================="
echo ""

if [[ -z "${RUN_DIR:-}" ]]; then
  echo "ERROR: RUN_DIR must be set." >&2
  exit 1
fi

echo "GPU Information (from host):"
nvidia-smi || true
echo ""

REPO_DIR="${SLURM_SUBMIT_DIR:-/users/hgao50/jetAnomalyDetection}"
cd "${REPO_DIR}"
# shellcheck disable=SC1091
source "${REPO_DIR}/.venv/bin/activate"
python -c "import torch; print(f'PyTorch {torch.__version__}, cuda={torch.cuda.is_available()}, device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
if ! python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
  echo "ERROR: torch cannot see a GPU in this allocation" >&2
  exit 1
fi

NTUPLE_ROOT=${NTUPLE_ROOT:-"/HEP/export/home/hgao50/jet-anomaly-data/ak8-v2"}
EVAL_STEPS=${EVAL_STEPS:-100}
NUM_WORKERS=${NUM_WORKERS:-2}

if [[ ! -f "${RUN_DIR}/summary.json" ]]; then
  echo "ERROR: missing ${RUN_DIR}/summary.json" >&2
  exit 1
fi
if [[ ! -f "${RUN_DIR}/best_model.pth" ]]; then
  echo "ERROR: missing ${RUN_DIR}/best_model.pth" >&2
  exit 1
fi

mkdir -p slurm_logs

python -u scripts/diagnose_lejepa_latents_cms.py \
  "${RUN_DIR}" \
  --ntuple-root "${NTUPLE_ROOT}" \
  --eval-steps "${EVAL_STEPS}" \
  --num-workers "${NUM_WORKERS}"

echo ""
echo "Finished at: $(date)"
echo "Results: ${RUN_DIR}/latent_diagnostics/"
