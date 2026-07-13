#!/bin/bash
#SBATCH --job-name=jet-ak8-smoke
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/oscar-ak8-smoke-%j.out
#SBATCH --error=logs/oscar-ak8-smoke-%j.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-"${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"}
SMOKE_ROOT=${SMOKE_ROOT:-"/HEP/export/home/hgao50/jetcharge-work/output"}
OUTPUT_DIR=${OUTPUT_DIR:-"${PROJECT_DIR}/data/processed/ak8-smoke/unscaled"}
MAX_JETS=${MAX_JETS:-1000}

cd "${PROJECT_DIR}"

if [[ -x ".venv/bin/python" ]]; then
    source .venv/bin/activate
elif command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate jet
fi

export PYTHONUNBUFFERED=TRUE
export MPLBACKEND=Agg

BG_FILE=$(find "${SMOKE_ROOT}" -maxdepth 1 -type f -name 'smoke_qcd_*600to800*_local_0.root' | head -1)
SG_FILE=${SG_FILE:-"/HEP/export/home/hgao50/jetcharge-work/ntuples/ak8-v1/qcd/qcd_QCD_Bin_PT_120to170_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24MiniAODv6_150X_mcRun3_2024_realistic_v2_v2_MINIAODSIM_local_0.root"}

if [[ -z "${BG_FILE}" || ! -f "${SG_FILE}" ]]; then
    echo "Could not find smoke ROOT files."
    echo "Background: ${BG_FILE:-missing}"
    echo "Signal: ${SG_FILE}"
    exit 1
fi

echo "Background smoke file: ${BG_FILE}"
echo "Signal smoke file: ${SG_FILE}"
echo "Output: ${OUTPUT_DIR}"

python -u scripts/prepare_data.py \
    --background "${BG_FILE}" \
    --signal "${SG_FILE}" \
    --label-bg QCD \
    --label-sg QCD_lowpt \
    --lowerpt 200 \
    --upperpt 2000 \
    --signal-lowerpt 120 \
    --signal-upperpt 180 \
    --max-jets "${MAX_JETS}" \
    --no-scale \
    --output-dir "${OUTPUT_DIR}"

ls -lh "${OUTPUT_DIR}"/*.pkl
python helpers/print_df_info.py --path "${OUTPUT_DIR}/QCD_unscaled.pkl" --printcols
python helpers/print_df_info.py --path "${OUTPUT_DIR}/QCD_lowpt_unscaled.pkl" --printcols
