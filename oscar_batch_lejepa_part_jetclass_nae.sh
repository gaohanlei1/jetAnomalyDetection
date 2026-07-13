#!/bin/bash
#SBATCH --nodes=1               # node count
# SBATCH --nodelist=gpu3003      # the L40S GPU!! 3001-3005, 3101-3102, 2708 are L40S
#SBATCH -p gpu --gres=gpu:1     # number of gpus per node
#SBATCH --ntasks-per-node=1     # total number of tasks across all nodes
#SBATCH --cpus-per-task=12       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH -t 42:00:00             # total run time limit (HH:MM:SS)
#SBATCH --mem=96000MB           # CPU RAM
#SBATCH --constraint=l40s
#SBATCH --job-name='JETANOMALY-JETCLASS'
#SBATCH --output=slurm_logs/R-%x.%j/log.out
#SBATCH --error=slurm_logs/R-%x.%j/log.err
# # Force unbuffered output
# export PYTHONUNBUFFERED=1
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

# torchrun --standalone --nproc-per-node=2 \
python -u \
    scripts/run_train_nae_part_jetclass.py \
    --backbone-dir "plots/run-lejepa-semi-sup-triplet-jetclass-ddp" \
    --backbone-checkpoint "plots/run-lejepa-semi-sup-triplet-jetclass-ddp/last_model.pth" \
    --ae-pretrain-epochs 100 \
    --nae-epochs 100 \
    --batch-size 256 \
    --steps-per-epoch 1000 \
    --val-steps 50 \
    --eval-steps 50 \
    --learning-rate 1e-4 \
    --langevin-steps 20 \
    --langevin-step-size 1e-2 \
    --langevin-noise-scale 1e-2 \
    --negative-init-noise 1e-2 \
    --num-workers 4 \
    --prefetch-factor 2 \
    --shuffle-active-shards 3 \
    --output-dir "plots/run-nae-lejepa-triplet"