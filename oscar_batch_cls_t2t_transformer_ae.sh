#!/bin/bash
#SBATCH --nodes=1               # node count
#SBATCH --nodelist=gpu3101      # the L40S GPU!! 3001-3005 or gpu3101 are L40S
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

python -u scripts/run_train_cls_t2t_transformer_ae.py \
    --background "data/processed/qcd-vs-wjet-pt-200to400/QCD_scaled_scaled.pkl" \
    --signal "data/processed/qcd-vs-wjet-pt-200to400/WJet_scaled_scaled.pkl" \
    --hidden-dim 4 \
    --latent-dim 2 \
    --num-layers 4 \
    --num-heads 2 \
    --batch-size 64 \
    --epochs 10 \
    --learning-rate 1e-4 \
    --weight-decay 1e-4 \
    --output-dir "plots/run-cls-t2t-transformer-ae-lixing-h4-l2-4l-8v-10epo"