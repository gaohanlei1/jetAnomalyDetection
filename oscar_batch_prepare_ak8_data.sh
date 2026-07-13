#!/bin/bash
#SBATCH --job-name=jet-ak8-prep
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=logs/oscar-ak8-data-%j.out
#SBATCH --error=logs/oscar-ak8-data-%j.err

# Prepare DeepNTuplizer AK8 ntuples → unscaled pickles for LeJEPA.
#
# Examples:
#   # smoke test (small files under ak8-v2/smoke)
#   SAMPLE=smoke MAX_JETS=500 sbatch oscar_batch_prepare_ak8_data.sh
#
#   # one sample at a time (recommended; submit in parallel)
#   SAMPLE=qcd   sbatch --job-name=prep-qcd   oscar_batch_prepare_ak8_data.sh
#   SAMPLE=wjets sbatch --job-name=prep-wjets oscar_batch_prepare_ak8_data.sh
#   SAMPLE=zjets sbatch --job-name=prep-zjets oscar_batch_prepare_ak8_data.sh
#   SAMPLE=ttbar sbatch --job-name=prep-ttbar oscar_batch_prepare_ak8_data.sh
#
#   # all samples sequentially in one job
#   SAMPLE=all sbatch oscar_batch_prepare_ak8_data.sh
#
#   # limited first pass
#   SAMPLE=qcd MAX_FILES=5 MAX_JETS=10000 sbatch oscar_batch_prepare_ak8_data.sh

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-"${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"}
NTUPLE_ROOT=${NTUPLE_ROOT:-"/HEP/export/home/hgao50/jet-anomaly-data/ak8-v2"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"${PROJECT_DIR}/data/processed/ak8-v2-pt150plus/unscaled"}
LOWERPT=${LOWERPT:-150}
UPPERPT=${UPPERPT:-}   # empty = no upper pT cut
MAX_JETS=${MAX_JETS:-}
MAX_FILES=${MAX_FILES:-}
SCALE=${SCALE:-false}
SAMPLE=${SAMPLE:-all}

cd "${PROJECT_DIR}"
mkdir -p logs

# Prefer project venv (has pandas/uproot). Fall back to miniforge jet env.
if [[ -x "${PROJECT_DIR}/.venv/bin/python" ]]; then
    # shellcheck disable=SC1091
    source "${PROJECT_DIR}/.venv/bin/activate"
    echo "Using project .venv: $(which python)"
elif module load miniforge3/25.3.0-3 2>/dev/null; then
    # shellcheck disable=SC1091
    source "${MAMBA_ROOT_PREFIX}/etc/profile.d/conda.sh"
    if conda env list | grep -qE '^jet\s'; then
        conda activate jet
        echo "Using conda env jet: $(which python)"
    else
        echo "ERROR: .venv missing and conda env 'jet' not found."
        exit 1
    fi
else
    echo "ERROR: No usable Python environment found."
    exit 1
fi

python -c "import pandas, uproot; print('pandas', pandas.__version__, 'uproot', uproot.__version__)"

export PYTHONUNBUFFERED=TRUE
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MPLBACKEND=Agg

SCALE_ARGS=(--no-scale)
if [[ "${SCALE}" == "true" ]]; then
    SCALE_ARGS=(--scale)
fi

LIMIT_ARGS=()
if [[ -n "${MAX_JETS}" ]]; then
    LIMIT_ARGS+=(--max-jets "${MAX_JETS}")
fi
if [[ -n "${MAX_FILES}" ]]; then
    LIMIT_ARGS+=(--max-files "${MAX_FILES}")
fi

declare -A SAMPLE_DIRS=(
    [qcd]="qcd"
    [wjets]="wjets"
    [zjets]="zjets"
    [ttbar]="ttbar"
    [hbb]="hbb"
    [smoke]="smoke"
)

declare -A SAMPLE_LABELS=(
    [qcd]="QCD"
    [wjets]="WJet"
    [zjets]="ZJet"
    [ttbar]="TTbar"
    [hbb]="Hbb"
    [smoke]="Smoke"
)

prepare_one() {
    local key="$1"
    local subdir="${SAMPLE_DIRS[$key]}"
    local label="${SAMPLE_LABELS[$key]}"
    local input_dir="${NTUPLE_ROOT}/${subdir}"

    if [[ ! -d "${input_dir}" ]]; then
        echo "Skipping ${key}: directory missing: ${input_dir}"
        return 0
    fi

    local n_root
    n_root=$(find "${input_dir}" -type f -name '*.root' -size +0c 2>/dev/null | wc -l | tr -d ' ')
    if [[ "${n_root}" -eq 0 ]]; then
        echo "Skipping ${key}: no non-empty ROOT files in ${input_dir}"
        return 0
    fi

    echo "============================================================"
    echo "Preparing ${key} (${label})"
    echo "  input:  ${input_dir} (${n_root} ROOT files)"
    echo "  output: ${OUTPUT_ROOT}"
    echo "  pT:     >= ${LOWERPT}${UPPERPT:+, <= ${UPPERPT}}"
    echo "============================================================"

    PT_ARGS=(--lowerpt "${LOWERPT}")
    if [[ -n "${UPPERPT}" ]]; then
        PT_ARGS+=(--upperpt "${UPPERPT}")
    fi

    python -u scripts/prepare_data.py \
        --background "${input_dir}" \
        --label-bg "${label}" \
        --output-dir "${OUTPUT_ROOT}" \
        "${SCALE_ARGS[@]}" \
        "${LIMIT_ARGS[@]}" \
        "${PT_ARGS[@]}"

    local suffix="unscaled"
    if [[ "${SCALE}" == "true" ]]; then
        suffix="scaled"
    fi
    local out_pkl="${OUTPUT_ROOT}/${label}_${suffix}.pkl"
    if [[ ! -s "${out_pkl}" ]]; then
        echo "ERROR: expected pickle missing: ${out_pkl}"
        exit 1
    fi
    ls -lh "${out_pkl}"
    python helpers/print_df_info.py --path "${out_pkl}" --printcols
}

prepare_smoke_all() {
    # Process each smoke_*.root into its own labeled pickle
    local smoke_dir="${NTUPLE_ROOT}/smoke"
    declare -A smoke_map=(
        [smoke_qcd]="QCD"
        [smoke_wjets]="WJet"
        [smoke_zjets]="ZJet"
        [smoke_ttbar]="TTbar"
    )
    for prefix in smoke_qcd smoke_wjets smoke_zjets smoke_ttbar; do
        local label="${smoke_map[$prefix]}"
        local files
        mapfile -t files < <(find "${smoke_dir}" -maxdepth 1 -type f -name "${prefix}_*.root" -size +0c | sort)
        if [[ "${#files[@]}" -eq 0 ]]; then
            echo "Skipping smoke ${prefix}: no files"
            continue
        fi
        echo "============================================================"
        echo "Preparing smoke ${prefix} (${label}) from ${#files[@]} file(s)"
        echo "============================================================"
        PT_ARGS=(--lowerpt "${LOWERPT}")
        if [[ -n "${UPPERPT}" ]]; then
            PT_ARGS+=(--upperpt "${UPPERPT}")
        fi
        python -u scripts/prepare_data.py \
            --background "${files[0]}" \
            --label-bg "${label}" \
            --output-dir "${OUTPUT_ROOT}" \
            "${SCALE_ARGS[@]}" \
            "${LIMIT_ARGS[@]}" \
            "${PT_ARGS[@]}"
        local suffix="unscaled"
        if [[ "${SCALE}" == "true" ]]; then
            suffix="scaled"
        fi
        ls -lh "${OUTPUT_ROOT}/${label}_${suffix}.pkl"
    done
}

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Project: ${PROJECT_DIR}"
echo "Ntuple root: ${NTUPLE_ROOT}"
echo "Output: ${OUTPUT_ROOT}"
echo "Sample(s): ${SAMPLE}"
echo "pT window: ${LOWERPT} - ${UPPERPT}"
echo "Scale features: ${SCALE}"

mkdir -p "${OUTPUT_ROOT}"

case "${SAMPLE}" in
    all)
        for key in qcd wjets zjets ttbar hbb; do
            prepare_one "${key}"
        done
        ;;
    smoke)
        prepare_smoke_all
        ;;
    qcd|wjets|zjets|ttbar|hbb)
        prepare_one "${SAMPLE}"
        ;;
    *)
        echo "Unknown SAMPLE='${SAMPLE}'. Use: qcd|wjets|zjets|ttbar|hbb|smoke|all"
        exit 1
        ;;
esac

echo "Data preparation completed for SAMPLE=${SAMPLE}."
ls -lh "${OUTPUT_ROOT}"/*.pkl 2>/dev/null || true
