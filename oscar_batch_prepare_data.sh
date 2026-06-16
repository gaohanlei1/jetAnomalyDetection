#!/bin/bash

#SBATCH --job-name=jet-data
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/oscar-data-%j.out
#SBATCH --error=logs/oscar-data-%j.err

set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "Usage: sbatch $0 QCD_170to300.root QCD_300to470.root WJET.root OUTPUT_ROOT"
    exit 2
fi

QCD_LOW=$1
QCD_HIGH=$2
WJET=$3
OUTPUT_ROOT=$4
PROJECT_DIR=${PROJECT_DIR:-"${SLURM_SUBMIT_DIR}"}
WORKERS=${PREPROCESS_WORKERS:-"${SLURM_CPUS_PER_TASK:-4}"}
MAX_EVENTS=${PREPROCESS_MAX_EVENTS:-}
MAX_EVENT_ARGS=()
if [[ -n "${MAX_EVENTS}" ]]; then
    MAX_EVENT_ARGS=(--max-events "${MAX_EVENTS}")
fi

cd "${PROJECT_DIR}"

for input_file in "${QCD_LOW}" "${QCD_HIGH}" "${WJET}"; do
    if [[ ! -r "${input_file}" ]]; then
        echo "Input ROOT file is not readable: ${input_file}"
        exit 1
    fi
done

if [[ ! -x ".venv/bin/python" ]]; then
    echo "Missing .venv. Run 'bash setup_venv.sh' on the Oscar login node first."
    exit 1
fi

source .venv/bin/activate

export PYTHONUNBUFFERED=TRUE
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MPLBACKEND=Agg

PREPROCESSED_QCD="${OUTPUT_ROOT}/preprocessed/qcd"
PREPROCESSED_WJET="${OUTPUT_ROOT}/preprocessed/wjet"
PROCESSED="${OUTPUT_ROOT}/processed/PT-200to400/scaledby_QCD"
PREPROCESS_MARKER="${OUTPUT_ROOT}/.preprocessing-complete"

mkdir -p "${PREPROCESSED_QCD}" "${PREPROCESSED_WJET}" "${PROCESSED}"

has_qcd_pickles() {
    [[ -n "$(find "${PREPROCESSED_QCD}" -type f -name '*.pkl' -size +0c -print -quit)" ]]
}

has_wjet_pickles() {
    [[ -n "$(find "${PREPROCESSED_WJET}" -type f -name '*.pkl' -size +0c -print -quit)" ]]
}

echo "Job ID: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "Project: ${PROJECT_DIR}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Workers: ${WORKERS}"
echo "Maximum events per ROOT file: ${MAX_EVENTS:-all}"
echo "QCD source 1: ${QCD_LOW}"
echo "QCD source 2: ${QCD_HIGH}"
echo "WJet source: ${WJET}"

if [[ -f "${PREPROCESS_MARKER}" ]] && has_qcd_pickles && has_wjet_pickles; then
    echo "Preprocessing marker found; reusing existing intermediate pickle files."
elif [[ -f "${PREPROCESS_MARKER}" ]]; then
    echo "Ignoring stale preprocessing marker because intermediate pickle files are missing."
    rm -f "${PREPROCESS_MARKER}"
elif [[ -n "$(find "${OUTPUT_ROOT}/preprocessed" -type f -name '*.pkl' -print -quit)" ]]; then
    echo "Partial intermediate pickle files already exist without a completion marker."
    echo "Inspect them or submit again with a new OUTPUT_ROOT."
    exit 1
fi

if [[ ! -f "${PREPROCESS_MARKER}" ]]; then
    python -u scripts/preprocessing.py \
        --type background \
        --path "${QCD_LOW}" \
        --savepath "${PREPROCESSED_QCD}" \
        --lowerpt 200 \
        --upperpt 400 \
        --workers "${WORKERS}" \
        "${MAX_EVENT_ARGS[@]}" \
        --no-move-to-used

    python -u scripts/preprocessing.py \
        --type background \
        --path "${QCD_HIGH}" \
        --savepath "${PREPROCESSED_QCD}" \
        --lowerpt 200 \
        --upperpt 400 \
        --workers "${WORKERS}" \
        "${MAX_EVENT_ARGS[@]}" \
        --no-move-to-used

    python -u scripts/preprocessing.py \
        --type signal \
        --path "${WJET}" \
        --savepath "${PREPROCESSED_WJET}" \
        --lowerpt 200 \
        --upperpt 400 \
        --workers "${WORKERS}" \
        "${MAX_EVENT_ARGS[@]}" \
        --no-move-to-used

    if ! has_qcd_pickles || ! has_wjet_pickles; then
        echo "Preprocessing finished without creating both QCD and WJet pickle files."
        exit 1
    fi

    touch "${PREPROCESS_MARKER}"
fi

python -u scripts/processing.py \
    --background "${PREPROCESSED_QCD}" \
    --signal "${PREPROCESSED_WJET}" \
    --label_bg QCD \
    --label_sg WJet \
    --output-dir "${PROCESSED}"

BACKGROUND_FILE="${PROCESSED}/QCD_scaled.pkl"
SIGNAL_FILE="${PROCESSED}/WJet_scaled.pkl"

if [[ ! -s "${BACKGROUND_FILE}" || ! -s "${SIGNAL_FILE}" ]]; then
    echo "Expected final pickle files were not created."
    exit 1
fi

echo "Data preparation completed."
ls -lh "${BACKGROUND_FILE}" "${SIGNAL_FILE}"
python helpers/print_df_info.py --path "${BACKGROUND_FILE}"
python helpers/print_df_info.py --path "${SIGNAL_FILE}"
