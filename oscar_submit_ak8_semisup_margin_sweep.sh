#!/bin/bash
# Submit the full CMS AK8 semi-sup-triplet margin × leave-one-out anomaly sweep.
#
# 4 margins × 4 signals = 16 jobs.
#
# Usage (from repo root):
#   bash oscar_submit_ak8_semisup_margin_sweep.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_DIR}"
mkdir -p slurm_logs plots

MARGINS=(0.1 0.2 0.5 1.0)
SIGNALS=(wjets zjets ttbar hbb)

echo "Submitting CMS AK8 semi-sup-triplet margin sweep from ${REPO_DIR}"
for margin in "${MARGINS[@]}"; do
  for signal in "${SIGNALS[@]}"; do
    job_name="AK8-SS-m${margin}-${signal}"
    out_dir="plots/run-lejepa-ak8-semisup-m${margin}-${signal}"
    echo "  sbatch MARGIN=${margin} SIGNAL=${signal} -> ${out_dir}"
    MARGIN="${margin}" SIGNAL="${signal}" OUTPUT_DIR="${out_dir}" \
      sbatch \
        --export=ALL,MARGIN,SIGNAL,OUTPUT_DIR \
        --job-name="${job_name}" \
        oscar_batch_lejepa_part_ak8_semisup_margin_sweep.sh
  done
done

echo "Submitted 16 jobs. Check with: squeue -u \$USER"