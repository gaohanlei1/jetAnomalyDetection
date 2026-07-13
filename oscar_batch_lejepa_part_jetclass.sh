#!/bin/bash
#SBATCH --nodes=1               # node count
# SBATCH --nodelist=gpu3003      # the L40S GPU!! 3001-3005, 3101-3102, 2708 are L40S
#SBATCH -p gpu --gres=gpu:2     # number of gpus per node
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

# python -u \
torchrun --standalone --nproc-per-node=2 \
    scripts/run_train_lejepa_part_jetclass.py \
    --dataset-root "/HEP/export/home/lwang223/JetClass/JetClass/Pythia" \
    --model semi-sup-triplet \
    --background-labels "label_QCD,label_Hbb,label_Hcc" \
    --signal-labels "label_Wqq" \
    --embed-dim 128 \
    --representation-dim 128 \
    --num-layers 8 \
    --num-heads 8 \
    --batch-size 256 \
    --steps-per-epoch 1000 \
    --val-steps 50 \
    --eval-steps 50 \
    --epochs 80 \
    --learning-rate 1e-3 \
    --weight-decay 5e-2 \
    --precision bf16 \
    --num-global-views 2 \
    --num-local-views 6 \
    --num-negative-views 4 \
    --batch-mix-prob 0.2 \
    --pt-resample-prob 0.2 \
    --node-eta-phi-rotation-prob 0.2 \
    --eta-phi-shuffle-prob 0.2 \
    --identity-shuffle-prob 0.2 \
    --anomaly-score mahalanobis \
    --pairwise-hidden-dim 16 \
    --triplet-weight 0.1 \
    --triplet-margin 1.0 \
    --classification-weight 0.1 \
    --num-workers 4 \
    --prefetch-factor 2 \
    --shuffle-active-shards 3 \
    --output-dir "plots/run-lejepa-semi-sup-triplet-jetclass-ddp"