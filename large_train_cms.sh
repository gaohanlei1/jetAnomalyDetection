#!/bin/bash
#SBATCH --nodes=1               # node count
# SBATCH --nodelist=gpu3006      # the L40S GPU!! 3001-3005, 3101-3102, 2708 are L40S
#SBATCH -p gpu --gres=gpu:2     # number of gpus per node
#SBATCH --ntasks-per-node=1     # total number of tasks across all nodes
#SBATCH --cpus-per-task=12       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH -t 12:00:00             # total run time limit (HH:MM:SS)
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
# torchrun --standalone --nproc-per-node=2 \

# --background-labels "label_QCD,label_Hbb,label_Hcc,label_Hgg,label_H4q,label_Hqql,label_Zqq,label_Tbqq,label_Tbl" \

python -u \
    scripts/run_train_lejepa_part.py \
    --dataset cms \
    --dataset-root "/HEP/export/home/hgao50/jet-anomaly-data/ak8-v2" \
    --model semi-sup-triplet \
    --background-labels "label_QCD,label_Hbb,label_Zqq,label_Tbqq" \
    --signal-labels "label_Wqq" \
    --embed-dim 32 \
    --representation-dim 32 \
    --dropout 0.01 \
    --num-layers 4 \
    --num-heads 8 \
    --batch-size 128 \
    --steps-per-epoch 4000 \
    --val-steps 100 \
    --eval-steps 100 \
    --epochs 6 \
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
    --pairwise-hidden-dim 32 \
    --triplet-weight 0.1 \
    --triplet-margin 0.1 \
    --classification-weight 0.1 \
    --num-workers 3 \
    --prefetch-factor 2 \
    --shuffle-active-shards 4 \
    --output-dir "plots/margin-sweep/wqq-0.1"

python -u \
    scripts/diagnose_lejepa_latents.py \
    "plots/margin-sweep/wqq-0.1"

# python -u \
#     scripts/run_train_lejepa_part.py \
#     --dataset cms \
#     --dataset-root "/HEP/export/home/hgao50/jet-anomaly-data/ak8-v2" \
#     --model semi-sup-triplet \
#     --background-labels "label_QCD,label_Hbb,label_Zqq,label_Tbqq" \
#     --signal-labels "label_Wqq" \
#     --embed-dim 32 \
#     --representation-dim 32 \
#     --dropout 0.01 \
#     --num-layers 4 \
#     --num-heads 8 \
#     --batch-size 128 \
#     --steps-per-epoch 4000 \
#     --val-steps 100 \
#     --eval-steps 100 \
#     --epochs 6 \
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
#     --output-dir "plots/margin-sweep/wqq-0.2"

# python -u \
#     scripts/diagnose_lejepa_latents.py \
#     "plots/margin-sweep/wqq-0.2"

# python -u \
#     scripts/run_train_lejepa_part.py \
#     --dataset cms \
#     --dataset-root "/HEP/export/home/hgao50/jet-anomaly-data/ak8-v2" \
#     --model semi-sup-triplet \
#     --background-labels "label_QCD,label_Hbb,label_Zqq,label_Tbqq" \
#     --signal-labels "label_Wqq" \
#     --embed-dim 32 \
#     --representation-dim 32 \
#     --dropout 0.01 \
#     --num-layers 4 \
#     --num-heads 8 \
#     --batch-size 128 \
#     --steps-per-epoch 4000 \
#     --val-steps 100 \
#     --eval-steps 100 \
#     --epochs 6 \
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
#     --triplet-margin 0.5 \
#     --classification-weight 0.1 \
#     --num-workers 3 \
#     --prefetch-factor 2 \
#     --shuffle-active-shards 4 \
#     --output-dir "plots/margin-sweep/wqq-0.5"

# python -u \
#     scripts/diagnose_lejepa_latents.py \
#     "plots/margin-sweep/wqq-0.5"

# python -u \
#     scripts/run_train_lejepa_part.py \
#     --dataset cms \
#     --dataset-root "/HEP/export/home/hgao50/jet-anomaly-data/ak8-v2" \
#     --model semi-sup-triplet \
#     --background-labels "label_QCD,label_Hbb,label_Zqq,label_Tbqq" \
#     --signal-labels "label_Wqq" \
#     --embed-dim 32 \
#     --representation-dim 32 \
#     --dropout 0.01 \
#     --num-layers 4 \
#     --num-heads 8 \
#     --batch-size 128 \
#     --steps-per-epoch 4000 \
#     --val-steps 100 \
#     --eval-steps 100 \
#     --epochs 6 \
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
#     --triplet-margin 1.0 \
#     --classification-weight 0.1 \
#     --num-workers 3 \
#     --prefetch-factor 2 \
#     --shuffle-active-shards 4 \
#     --output-dir "plots/margin-sweep/wqq-1.0"

# python -u \
#     scripts/diagnose_lejepa_latents.py \
#     "plots/margin-sweep/wqq-1.0"

# python -u \
#     scripts/run_train_lejepa_part.py \
#     --dataset cms \
#     --dataset-root "/HEP/export/home/hgao50/jet-anomaly-data/ak8-v2" \
#     --model semi-sup-triplet \
#     --background-labels "label_QCD,label_Hbb,label_Zqq,label_Tbqq" \
#     --signal-labels "label_Wqq" \
#     --embed-dim 32 \
#     --representation-dim 32 \
#     --dropout 0.01 \
#     --num-layers 4 \
#     --num-heads 8 \
#     --batch-size 128 \
#     --steps-per-epoch 4000 \
#     --val-steps 100 \
#     --eval-steps 100 \
#     --epochs 6 \
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
#     --triplet-margin 2.0 \
#     --classification-weight 0.1 \
#     --num-workers 3 \
#     --prefetch-factor 2 \
#     --shuffle-active-shards 4 \
#     --output-dir "plots/margin-sweep/wqq-2.0"

# python -u \
#     scripts/diagnose_lejepa_latents.py \
#     "plots/margin-sweep/wqq-2.0"

python -u \
    scripts/run_train_lejepa_part.py \
    --dataset cms \
    --dataset-root "/HEP/export/home/hgao50/jet-anomaly-data/ak8-v2" \
    --model semi-sup-triplet \
    --background-labels "label_QCD,label_Wqq,label_Zqq,label_Tbqq" \
    --signal-labels "label_Hbb" \
    --embed-dim 32 \
    --representation-dim 32 \
    --dropout 0.01 \
    --num-layers 4 \
    --num-heads 8 \
    --batch-size 128 \
    --steps-per-epoch 4000 \
    --val-steps 100 \
    --eval-steps 100 \
    --epochs 6 \
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
    --pairwise-hidden-dim 32 \
    --triplet-weight 0.1 \
    --triplet-margin 0.1 \
    --classification-weight 0.1 \
    --num-workers 3 \
    --prefetch-factor 2 \
    --shuffle-active-shards 4 \
    --output-dir "plots/margin-sweep/hbb-0.1"

python -u \
    scripts/diagnose_lejepa_latents.py \
    "plots/margin-sweep/hbb-0.1"

# python -u \
#     scripts/run_train_lejepa_part.py \
#     --dataset cms \
#     --dataset-root "/HEP/export/home/hgao50/jet-anomaly-data/ak8-v2" \
#     --model semi-sup-triplet \
#     --background-labels "label_QCD,label_Wqq,label_Zqq,label_Tbqq" \
#     --signal-labels "label_Hbb" \
#     --embed-dim 32 \
#     --representation-dim 32 \
#     --dropout 0.01 \
#     --num-layers 4 \
#     --num-heads 8 \
#     --batch-size 128 \
#     --steps-per-epoch 4000 \
#     --val-steps 100 \
#     --eval-steps 100 \
#     --epochs 6 \
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
#     --output-dir "plots/margin-sweep/hbb-0.2"

# python -u \
#     scripts/diagnose_lejepa_latents.py \
#     "plots/margin-sweep/hbb-0.2"

# python -u \
#     scripts/run_train_lejepa_part.py \
#     --dataset cms \
#     --dataset-root "/HEP/export/home/hgao50/jet-anomaly-data/ak8-v2" \
#     --model semi-sup-triplet \
#     --background-labels "label_QCD,label_Wqq,label_Zqq,label_Tbqq" \
#     --signal-labels "label_Hbb" \
#     --embed-dim 32 \
#     --representation-dim 32 \
#     --dropout 0.01 \
#     --num-layers 4 \
#     --num-heads 8 \
#     --batch-size 128 \
#     --steps-per-epoch 4000 \
#     --val-steps 100 \
#     --eval-steps 100 \
#     --epochs 6 \
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
#     --triplet-margin 0.5 \
#     --classification-weight 0.1 \
#     --num-workers 3 \
#     --prefetch-factor 2 \
#     --shuffle-active-shards 4 \
#     --output-dir "plots/margin-sweep/hbb-0.5"

# python -u \
#     scripts/diagnose_lejepa_latents.py \
#     "plots/margin-sweep/hbb-0.5"

# python -u \
#     scripts/run_train_lejepa_part.py \
#     --dataset cms \
#     --dataset-root "/HEP/export/home/hgao50/jet-anomaly-data/ak8-v2" \
#     --model semi-sup-triplet \
#     --background-labels "label_QCD,label_Wqq,label_Zqq,label_Tbqq" \
#     --signal-labels "label_Hbb" \
#     --embed-dim 32 \
#     --representation-dim 32 \
#     --dropout 0.01 \
#     --num-layers 4 \
#     --num-heads 8 \
#     --batch-size 128 \
#     --steps-per-epoch 4000 \
#     --val-steps 100 \
#     --eval-steps 100 \
#     --epochs 6 \
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
#     --triplet-margin 1.0 \
#     --classification-weight 0.1 \
#     --num-workers 3 \
#     --prefetch-factor 2 \
#     --shuffle-active-shards 4 \
#     --output-dir "plots/margin-sweep/hbb-1.0"

# python -u \
#     scripts/diagnose_lejepa_latents.py \
#     "plots/margin-sweep/hbb-1.0"

# python -u \
#     scripts/run_train_lejepa_part.py \
#     --dataset cms \
#     --dataset-root "/HEP/export/home/hgao50/jet-anomaly-data/ak8-v2" \
#     --model semi-sup-triplet \
#     --background-labels "label_QCD,label_Wqq,label_Zqq,label_Tbqq" \
#     --signal-labels "label_Hbb" \
#     --embed-dim 32 \
#     --representation-dim 32 \
#     --dropout 0.01 \
#     --num-layers 4 \
#     --num-heads 8 \
#     --batch-size 128 \
#     --steps-per-epoch 4000 \
#     --val-steps 100 \
#     --eval-steps 100 \
#     --epochs 6 \
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
#     --triplet-margin 2.0 \
#     --classification-weight 0.1 \
#     --num-workers 3 \
#     --prefetch-factor 2 \
#     --shuffle-active-shards 4 \
#     --output-dir "plots/margin-sweep/hbb-2.0"

# python -u \
#     scripts/diagnose_lejepa_latents.py \
#     "plots/margin-sweep/hbb-2.0"