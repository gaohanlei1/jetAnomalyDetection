#!/bin/bash
#SBATCH --nodes=1               # node count
# SBATCH --nodelist=gpu2708,gpu2709,gpu3002,gpu3003,gpu3004,gpu3006,gpu3101,gpu3102,gpu3105,gpu3106
#SBATCH -p gpu --gres=gpu:1     # number of gpus per node
#SBATCH --ntasks-per-node=1     # total number of tasks across all nodes
#SBATCH --cpus-per-task=6       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH -t 48:00:00             # total run time limit (HH:MM:SS)
#SBATCH --mem=64GB           # CPU RAM
# SBATCH --constraint=l40s
#SBATCH --job-name='JETANOMALY'
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
# torchrun --standalone --nproc-per-node=2 \

# --background-labels "label_QCD,label_Hbb,label_Hcc,label_Hgg,label_H4q,label_Hqql,label_Zqq,label_Tbqq,label_Tbl" \

# torchrun --standalone --nproc-per-node=2 \
# "label_QCD,label_Hbb,label_Zqq,label_Wqq,label_Tbqq"
# python -u \
# "/HEP/export/home/hgao50/jet-anomaly-data/ak8-data"
# --cms-val-fraction 0.05 \
#     --cms-test-fraction 0.45 \

# python -u scripts/run_train_lejepa_part.py \
#     --dataset jetclass \
#     --dataset-root "/HEP/export/home/lwang223/JetClass/JetClass/Pythia" \
#     --model semi-sup-triplet \
#     --background-labels "label_QCD,label_Tbqq,label_Hgg,label_Wqq" \
#     --signal-labels "label_Hbb" \
#     --embed-dim 32 \
#     --representation-dim 32 \
#     --dropout 0.01 \
#     --num-layers 4 \
#     --num-heads 8 \
#     --batch-size 128 \
#     --steps-per-epoch 2000 \
#     --val-steps 100 \
#     --eval-steps 100 \
#     --epochs 40 \
#     --learning-rate 1e-3 \
#     --weight-decay 5e-2 \
#     --precision bf16 \
#     --num-global-views 2 \
#     --num-local-views 4 \
#     --num-negative-views 4 \
#     --batch-mix-prob 0.4 \
#     --pt-resample-prob 0.25 \
#     --node-deta-dphi-rotation-prob 0.1 \
#     --deta-dphi-shuffle-prob 0.1 \
#     --identity-shuffle-prob 0.15 \
#     --global-drop-pt-frac-min 0.0 \
#     --global-drop-pt-frac-max 0.3 \
#     --local-drop-pt-frac-min 0.3 \
#     --local-drop-pt-frac-max 0.75 \
#     --batch-mix-anchor-frac-min 0.4 \
#     --batch-mix-anchor-frac-max 0.6 \
#     --anomaly-score mahalanobis \
#     --pairwise-hidden-dim 32 \
#     --triplet-weight 0.1 \
#     --triplet-margin 0.2 \
#     --classification-weight 0.1 \
#     --num-workers 4 \
#     --prefetch-factor 2 \
#     --shuffle-active-shards 3 \
#     --output-dir "plots/jetclass/32-QcdTbqqHggWqq"

python -u scripts/diagnose_lejepa_latents.py \
    "plots/jetclass/32-QcdHccHggWqq"

python scripts/plot_lejepa_tsne.py \
  "plots/jetclass/32-QcdHccHggWqq" \
  --perplexity 100
