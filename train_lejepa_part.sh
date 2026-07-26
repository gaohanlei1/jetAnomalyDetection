#!/bin/bash
#SBATCH --nodes=1               # node count
# SBATCH --nodelist=gpu2708      # the L40S GPU!! 3001-3005, 3101-3106, 2708-2709 are L40S
#SBATCH -p gpu --gres=gpu:2     # number of gpus per node
#SBATCH --ntasks-per-node=1     # total number of tasks across all nodes
#SBATCH --cpus-per-task=12       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH -t 48:00:00             # total run time limit (HH:MM:SS)
#SBATCH --mem=128GB           # CPU RAM
#SBATCH --constraint=l40s
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
torchrun --standalone --nproc-per-node=2 \
    scripts/run_train_lejepa_part.py \
    --dataset cms \
    --dataset-root "/HEP/export/home/hgao50/jet-anomaly-data/ak8-v4" \
    --model semi-sup-triplet \
    --background-labels "label_QCD,label_Hbb,label_Zqq,label_Tbqq,label_Wqq" \
    --signal-labels "label_QCD,label_Hbb,label_Zqq,label_Tbqq,label_Wqq" \
    --embed-dim 128 \
    --representation-dim 128 \
    --dropout 0.1 \
    --num-layers 8 \
    --num-heads 8 \
    --batch-size 256 \
    --steps-per-epoch 8000 \
    --val-steps 3000 \
    --eval-steps 3000 \
    --epochs 80 \
    --learning-rate 1e-3 \
    --weight-decay 5e-2 \
    --precision bf16 \
    --num-global-views 2 \
    --num-local-views 4 \
    --num-negative-views 4 \
    --batch-mix-prob 0.4 \
    --pt-resample-prob 0.25 \
    --node-deta-dphi-rotation-prob 0.1 \
    --deta-dphi-shuffle-prob 0.1 \
    --identity-shuffle-prob 0.15 \
    --global-drop-pt-frac-min 0.0 \
    --global-drop-pt-frac-max 0.3 \
    --local-drop-pt-frac-min 0.3 \
    --local-drop-pt-frac-max 0.75 \
    --batch-mix-anchor-frac-min 0.4 \
    --batch-mix-anchor-frac-max 0.6 \
    --anomaly-score mahalanobis \
    --pairwise-hidden-dim 64 \
    --triplet-weight 0.1 \
    --triplet-margin 0.2 \
    --classification-weight 0.1 \
    --num-workers 3 \
    --prefetch-factor 2 \
    --shuffle-active-shards 4 \
    --output-dir "plots/largerun/all-cms-mc-shuffledata-nologsig-refill"

python -u \
    scripts/diagnose_lejepa_latents.py \
    "plots/largerun/all-cms-mc-shuffledata-nologsig-refill" \
    --checkpoint "plots/largerun/all-cms-mc-shuffledata-nologsig-refill/last_model.pth" \
    --eval-steps 3000

# python -u \
#     scripts/run_train_lejepa_part.py \
#     --dataset jetclass \
#     --dataset-root "/HEP/export/home/lwang223/JetClass/JetClass/Pythia" \
#     --model semi-sup-triplet \
#     --background-labels "label_QCD,label_Hbb,label_Hcc,label_Hgg,label_Wqq,label_H4q,label_Hqql,label_Zqq,label_Tbqq,label_Tbl" \
#     --signal-labels "label_QCD,label_Hbb,label_Hcc,label_Hgg,label_Wqq,label_H4q,label_Hqql,label_Zqq,label_Tbqq,label_Tbl" \
#     --embed-dim 32 \
#     --representation-dim 32 \
#     --dropout 0.01 \
#     --num-layers 4 \
#     --num-heads 8 \
#     --batch-size 128 \
#     --steps-per-epoch 4000 \
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
#     --num-workers 3 \
#     --prefetch-factor 2 \
#     --shuffle-active-shards 4 \
#     --output-dir "plots/quickrun/all-jetclass"

# python -u \
#     scripts/diagnose_lejepa_latents.py \
#     "plots/quickrun/all-jetclass"