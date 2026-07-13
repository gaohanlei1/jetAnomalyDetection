#!/bin/bash
#SBATCH --job-name=prep-ak8-shard
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=logs/prep-shard-%A_%a.out
#SBATCH --error=logs/prep-shard-%A_%a.err

# One ROOT file → one pickle shard (SLURM array task).
#
# Oscar QoS "normal": MaxTRESPU cpu=64, mem=492G (across your jobs).
# Feature engineering is mostly single-threaded, so use many 2-CPU tasks
# rather than one fat multi-CPU job. Cap concurrency with %N in --array.
#
# Full sample examples (pt >= 150, no upper cut):
#   SAMPLE=qcd   LOWERPT=150 sbatch --array=0-1399%32 oscar_batch_prepare_ak8_shards.sh
#   SAMPLE=wjets LOWERPT=150 sbatch --array=0-599%32  oscar_batch_prepare_ak8_shards.sh
#   SAMPLE=zjets LOWERPT=150 sbatch --array=0-599%32  oscar_batch_prepare_ak8_shards.sh
#   SAMPLE=ttbar LOWERPT=150 sbatch --array=0-199%32  oscar_batch_prepare_ak8_shards.sh
#
# Smoke / first 5 files:
#   SAMPLE=qcd LOWERPT=150 sbatch --array=0-4 oscar_batch_prepare_ak8_shards.sh

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-"${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"}
NTUPLE_ROOT=${NTUPLE_ROOT:-"/HEP/export/home/hgao50/jet-anomaly-data/ak8-v2"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"${PROJECT_DIR}/data/processed/ak8-v2-pt150plus/shards"}
LOWERPT=${LOWERPT:-150}
UPPERPT=${UPPERPT:-}
SAMPLE=${SAMPLE:?Set SAMPLE=qcd|wjets|zjets|ttbar}

declare -A SAMPLE_DIRS=([qcd]=qcd [wjets]=wjets [zjets]=zjets [ttbar]=ttbar)
declare -A SAMPLE_LABELS=([qcd]=QCD [wjets]=WJet [zjets]=ZJet [ttbar]=TTbar)

if [[ -z "${SAMPLE_DIRS[$SAMPLE]+x}" ]]; then
    echo "Unknown SAMPLE='${SAMPLE}'"
    exit 1
fi

LABEL="${SAMPLE_LABELS[$SAMPLE]}"
INPUT_DIR="${NTUPLE_ROOT}/${SAMPLE_DIRS[$SAMPLE]}"
OUT_DIR="${OUTPUT_ROOT}/${SAMPLE}"

cd "${PROJECT_DIR}"
mkdir -p logs "${OUT_DIR}"

if [[ -x "${PROJECT_DIR}/.venv/bin/python" ]]; then
    # shellcheck disable=SC1091
    source "${PROJECT_DIR}/.venv/bin/activate"
else
    echo "ERROR: project .venv not found"
    exit 1
fi

export PYTHONUNBUFFERED=TRUE
export OMP_NUM_THREADS=1
export MPLBACKEND=Agg

mapfile -t ROOT_FILES < <(find "${INPUT_DIR}" -maxdepth 1 -type f -name '*.root' -size +0c | sort)
N_FILES=${#ROOT_FILES[@]}
if [[ "${N_FILES}" -eq 0 ]]; then
    echo "No ROOT files in ${INPUT_DIR}"
    exit 1
fi

TASK_ID=${SLURM_ARRAY_TASK_ID:?Submit as a SLURM array job (--array=...)}
if [[ "${TASK_ID}" -ge "${N_FILES}" ]]; then
    echo "Task ${TASK_ID} >= n_files ${N_FILES}; nothing to do."
    exit 0
fi

ROOT_FILE="${ROOT_FILES[$TASK_ID]}"
BASE=$(basename "${ROOT_FILE}" .root)
SHARD_STEM="${LABEL}__${BASE}"
OUT_PKL="${OUT_DIR}/${SHARD_STEM}_unscaled.pkl"

if [[ -s "${OUT_PKL}" ]]; then
    echo "Skip existing: ${OUT_PKL}"
    ls -lh "${OUT_PKL}"
    exit 0
fi

PT_ARGS=(--lowerpt "${LOWERPT}")
if [[ -n "${UPPERPT}" ]]; then
    PT_ARGS+=(--upperpt "${UPPERPT}")
fi

echo "Job ${SLURM_JOB_ID:-local} task ${TASK_ID}/${N_FILES}"
echo "  input:  ${ROOT_FILE}"
echo "  output: ${OUT_PKL}"
echo "  pT:     >= ${LOWERPT}${UPPERPT:+, <= ${UPPERPT}}"

python -u scripts/prepare_data.py \
    --background "${ROOT_FILE}" \
    --label-bg "${SHARD_STEM}" \
    --output-dir "${OUT_DIR}" \
    --no-scale \
    "${PT_ARGS[@]}"

if [[ ! -s "${OUT_PKL}" ]]; then
    echo "ERROR: missing ${OUT_PKL}"
    ls -lh "${OUT_DIR}" | tail -20
    exit 1
fi

ls -lh "${OUT_PKL}"
echo "Done task ${TASK_ID}"
