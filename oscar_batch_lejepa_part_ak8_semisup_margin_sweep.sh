#!/bin/bash
#SBATCH --nodes=1
# Prefer L40S when available, but do not hard-pin a single node.
#SBATCH -p gpu --gres=gpu:l40s:1
#SBATCH --ntasks-per-node=1
# Keep requests inside norm-gpu MaxTRESPU (cpu=12,gres/gpu=2,mem=192G)
# so two jobs can run concurrently instead of one.
#SBATCH --cpus-per-task=4
#SBATCH -t 24:00:00
#SBATCH --mem=90000MB
#SBATCH --job-name='JETANOMALY-AK8-SS'
#SBATCH --output=slurm_logs/R-%x.%j/log.out
#SBATCH --error=slurm_logs/R-%x.%j/log.err
export PYTHONIOENCODING=utf-8
set -euo pipefail

# CMS AK8 leave-one-out semi-sup-triplet margin sweep.
#
# Env vars:
#   MARGIN   triplet margin (required), e.g. 0.1 | 0.2 | 0.5 | 1.0
#   SIGNAL   held-out anomaly (required): wjets | zjets | ttbar | hbb
#
# Example submit of all 16 jobs:
#   bash oscar_submit_ak8_semisup_margin_sweep.sh

echo ""
echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURM_NODELIST:-local}"
echo "MARGIN=${MARGIN:-unset}  SIGNAL=${SIGNAL:-unset}"
echo "=========================================="
echo ""

if [[ -z "${MARGIN:-}" || -z "${SIGNAL:-}" ]]; then
  echo "ERROR: MARGIN and SIGNAL must be set." >&2
  exit 1
fi

case "${SIGNAL}" in
  wjets|zjets|ttbar|hbb) ;;
  *)
    echo "ERROR: SIGNAL must be one of wjets|zjets|ttbar|hbb, got '${SIGNAL}'" >&2
    exit 1
    ;;
esac

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
OUTPUT_DIR=${OUTPUT_DIR:-"plots/run-lejepa-ak8-semisup-m${MARGIN}-${SIGNAL}"}
STAGE_DIR=${STAGE_DIR:-"${TMPDIR:-/tmp}/ak8_semisup_${SIGNAL}_m${MARGIN}_${SLURM_JOB_ID:-$$}"}

MAX_PER_CLASS=${MAX_PER_CLASS:-75000}
MAX_SG=${MAX_SG:-20000}
LOWERPT=${LOWERPT:-200}
UPPERPT=${UPPERPT:-400}
STEPS_PER_EPOCH=${STEPS_PER_EPOCH:-1000}
VAL_STEPS=${VAL_STEPS:-100}
EVAL_STEPS=${EVAL_STEPS:-100}
# Overnight target: ~1h/job with 2 concurrent under norm-gpu.
EPOCHS=${EPOCHS:-12}
# Mahalanobis/ROC every N epochs (1 = every epoch). 5 cuts eval walltime a lot.
ROC_EVAL_EVERY=${ROC_EVAL_EVERY:-5}

# Leave-one-out backgrounds: always include QCD + all anomalies except SIGNAL.
ALL_SIGNALS=(wjets zjets ttbar hbb)
BG_KEYS=(qcd)
for key in "${ALL_SIGNALS[@]}"; do
  if [[ "${key}" != "${SIGNAL}" ]]; then
    BG_KEYS+=("${key}")
  fi
done

declare -A DISPLAY_NAME=(
  [qcd]=QCD
  [wjets]=WJets
  [zjets]=ZJets
  [ttbar]=TTbar
  [hbb]=Hbb
)

mkdir -p "${STAGE_DIR}"
BG_DIRS=()
BG_NAMES=()
for key in "${BG_KEYS[@]}"; do
  dest="${STAGE_DIR}/${key}"
  mkdir -p "${dest}"
  case "${key}" in
    qcd)
      ln -sfn "${NTUPLE_ROOT}"/qcd/qcd_PT170to300_*.root "${dest}/"
      ln -sfn "${NTUPLE_ROOT}"/qcd/qcd_PT300to470_*.root "${dest}/"
      ;;
    wjets)
      ln -sfn "${NTUPLE_ROOT}"/wjets/wjets_Wto2Q_PTQQ200_*.root "${dest}/"
      ;;
    zjets)
      ln -sfn "${NTUPLE_ROOT}"/zjets/zjets_Zto2Q_PTQQ200_*.root "${dest}/"
      ;;
    ttbar)
      ln -sfn "${NTUPLE_ROOT}"/ttbar/ttbar_TTto4Q_*.root "${dest}/"
      ;;
    hbb)
      ln -sfn "${NTUPLE_ROOT}"/hbb/hbb_WminusH_WtoLNu_*.root "${dest}/"
      ln -sfn "${NTUPLE_ROOT}"/hbb/hbb_WplusH_WtoLNu_*.root "${dest}/"
      ln -sfn "${NTUPLE_ROOT}"/hbb/hbb_ZH_Zto2L_*.root "${dest}/"
      ;;
  esac
  nfiles=$(find "${dest}" -maxdepth 1 \( -type l -o -type f \) 2>/dev/null | wc -l | tr -d ' ')
  echo "Staged ${key}: ${nfiles} files -> ${dest}"
  if [[ "${nfiles}" -eq 0 ]]; then
    echo "ERROR: no staged files for ${key}" >&2
    exit 1
  fi
  BG_DIRS+=("${dest}")
  BG_NAMES+=("${DISPLAY_NAME[${key}]}")
done

SG_DIR="${STAGE_DIR}/signal_${SIGNAL}"
mkdir -p "${SG_DIR}"
case "${SIGNAL}" in
  wjets)
    ln -sfn "${NTUPLE_ROOT}"/wjets/wjets_Wto2Q_PTQQ200_*.root "${SG_DIR}/"
    ;;
  zjets)
    ln -sfn "${NTUPLE_ROOT}"/zjets/zjets_Zto2Q_PTQQ200_*.root "${SG_DIR}/"
    ;;
  ttbar)
    ln -sfn "${NTUPLE_ROOT}"/ttbar/ttbar_TTto4Q_*.root "${SG_DIR}/"
    ;;
  hbb)
    ln -sfn "${NTUPLE_ROOT}"/hbb/hbb_WminusH_WtoLNu_*.root "${SG_DIR}/"
    ln -sfn "${NTUPLE_ROOT}"/hbb/hbb_WplusH_WtoLNu_*.root "${SG_DIR}/"
    ln -sfn "${NTUPLE_ROOT}"/hbb/hbb_ZH_Zto2L_*.root "${SG_DIR}/"
    ;;
esac
sg_files=$(find "${SG_DIR}" -maxdepth 1 \( -type l -o -type f \) 2>/dev/null | wc -l | tr -d ' ')
echo "Staged signal ${SIGNAL}: ${sg_files} files -> ${SG_DIR}"
if [[ "${sg_files}" -eq 0 ]]; then
  echo "ERROR: no staged signal files for ${SIGNAL}" >&2
  exit 1
fi

BG_DIRS_CSV=$(IFS=,; echo "${BG_DIRS[*]}")
BG_NAMES_CSV=$(IFS=,; echo "${BG_NAMES[*]}")

echo "Background dirs: ${BG_DIRS_CSV}"
echo "Background names: ${BG_NAMES_CSV}"
echo "Signal: ${SIGNAL} (${DISPLAY_NAME[${SIGNAL}]})"
echo "Margin: ${MARGIN}"
echo "Output dir: ${OUTPUT_DIR}"

mkdir -p slurm_logs plots

python -u scripts/run_train_lejepa_part.py \
  --background-dirs "${BG_DIRS_CSV}" \
  --background-names "${BG_NAMES_CSV}" \
  --signal "${SG_DIR}" \
  --signal-name "${DISPLAY_NAME[${SIGNAL}]}" \
  --lowerpt "${LOWERPT}" \
  --upperpt "${UPPERPT}" \
  --stream \
  --steps-per-epoch "${STEPS_PER_EPOCH}" \
  --val-steps "${VAL_STEPS}" \
  --eval-steps "${EVAL_STEPS}" \
  --shuffle-active-shards 4 \
  --model "semi-sup-triplet" \
  --anomaly-score "mahalanobis" \
  --no-normalize-features \
  --standardized-feature-names log_pt "d0/d0Err" "dz/dzErr" \
  --feature-norm-momentum 0.1 \
  --embed-dim 128 \
  --representation-dim 128 \
  --num-layers 8 \
  --num-heads 8 \
  --batch-size 128 \
  --epochs "${EPOCHS}" \
  --roc-eval-every "${ROC_EVAL_EVERY}" \
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
  --triplet-margin "${MARGIN}" \
  --classification-weight 0.05 \
  --num-negative-views 4 \
  --batch-mix-prob 0.45 \
  --pt-resample-prob 0.25 \
  --node-eta-phi-rotation-prob 0.20 \
  --eta-phi-shuffle-prob 0.05 \
  --identity-shuffle-prob 0.05 \
  --num-augmentation-plot-samples 0 \
  --num-workers 2 \
  --output-dir "${OUTPUT_DIR}"

rm -rf "${STAGE_DIR}"
echo "Job finished at: $(date)"
