#!/bin/bash
#SBATCH --nodes=1               # node count
#SBATCH --nodelist=gpu3102      # the L40S GPU!! 3001-3005 or gpu3101-3102 are L40S
#SBATCH -p gpu --gres=gpu:1     # number of gpus per node
#SBATCH --ntasks-per-node=1     # total number of tasks across all nodes
#SBATCH --cpus-per-task=4       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH -t 12:00:00             # total run time limit (HH:MM:SS)
#SBATCH --mem=32000MB           # INCREASED from 16GB to 32GB
#SBATCH --job-name='JETANOMALY'
#SBATCH --output=slurm_logs/R-%x.%j/log.out
#SBATCH --error=slurm_logs/R-%x.%j/log.err
# Force unbuffered output
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

echo ""
echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="
echo ""

echo "GPU Information (from host):"
nvidia-smi
echo ""

module load miniforge3/25.3.0-3
source ${MAMBA_ROOT_PREFIX}/etc/profile.d/conda.sh
# source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
# conda init
conda activate jet

# check pytorch version
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

python -u scripts/run_train_lejepa_trip_part.py \
  --background "data/processed/qcd-vs-wjet-pt-200to400/QCD_scaled_scaled.pkl" \
  --signal "data/processed/qcd-vs-wjet-pt-200to400/WJet_scaled_scaled.pkl" \
  --embed-dim 128 \
  --representation-dim 128 \
  --num-layers 8 \
  --num-heads 8 \
  --batch-size 128 \
  --epochs 100 \
  --learning-rate 5e-4 \
  --weight-decay 5e-2 \
  --precision bf16 \
  --triplet-weight 0.1 \
  --triplet-margin 2.0 \
  --num-global-views 2 \
  --num-local-views 3 \
  --num-negative-views 4 \
  --negative-modes "identity_shuffle,pt_resample,eta_phi_shuffle,batch_mix" \
  --output-dir "plots/run-lejepa-trip-part-m2-all-aug"