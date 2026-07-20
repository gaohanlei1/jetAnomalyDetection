#!/bin/bash
# Submit CMS AK8 latent diagnostics for all margin-sweep runs.
#
# 4 margins × 4 signals = 16 jobs.
#
# Usage (from repo root):
#   bash oscar_submit_ak8_latent_diagnostics.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_DIR}"
mkdir -p slurm_logs plots

MARGINS=(0.1 0.2 0.5 1.0)
SIGNALS=(wjets zjets ttbar hbb)

echo "Submitting CMS AK8 latent diagnostics from ${REPO_DIR}"
n_submit=0
n_skip=0
for margin in "${MARGINS[@]}"; do
  for signal in "${SIGNALS[@]}"; do
    run_dir="plots/run-lejepa-ak8-semisup-m${margin}-${signal}"
    if [[ ! -f "${run_dir}/best_model.pth" || ! -f "${run_dir}/summary.json" ]]; then
      echo "  SKIP (missing checkpoint/summary): ${run_dir}"
      n_skip=$((n_skip + 1))
      continue
    fi
    if [[ -f "${run_dir}/latent_diagnostics/diagnostic_results.json" ]]; then
      echo "  SKIP (already done): ${run_dir}"
      n_skip=$((n_skip + 1))
      continue
    fi
    job_name="AK8-DIAG-m${margin}-${signal}"
    echo "  sbatch RUN_DIR=${run_dir}"
    RUN_DIR="${run_dir}" \
      sbatch \
        --export=ALL,RUN_DIR \
        --job-name="${job_name}" \
        oscar_batch_lejepa_part_ak8_latent_diagnostics.sh
    n_submit=$((n_submit + 1))
  done
done

echo "Submitted ${n_submit} jobs (skipped ${n_skip}). Check with: squeue -u \$USER"
