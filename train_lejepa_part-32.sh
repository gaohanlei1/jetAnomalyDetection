#!/bin/bash
#SBATCH --nodes=1               # node count
# SBATCH --nodelist=gpu2708      # the L40S GPU!! 3001-3005, 3101-3106, 2708-2709 are L40S
#SBATCH -p gpu --gres=gpu:2     # number of gpus per node
#SBATCH --ntasks-per-node=1     # total number of tasks across all nodes
#SBATCH --cpus-per-task=6       # cpu-cores per task (>1 if multi-threaded tasks)
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

# Slurm copies the batch script to a spool path, so do not derive the repo from BASH_SOURCE.
REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_DIR}"

if [[ -x "${REPO_DIR}/.venv/bin/python" ]]; then
    source "${REPO_DIR}/.venv/bin/activate"
    echo "Using project .venv: $(which python)"
else
    echo "ERROR: project .venv not found at ${REPO_DIR}/.venv"
    echo "Available python: $(which python || true)"
    exit 1
fi

# check pytorch version
python -c "import torch; print(f'PyTorch version: {torch.__version__}, cuda={torch.cuda.is_available()}')"

# python -u \
# torchrun --standalone --nproc-per-node=2 \

# --background-labels "label_QCD,label_Hbb,label_Hcc,label_Hgg,label_H4q,label_Hqql,label_Zqq,label_Tbqq,label_Tbl" \

# torchrun --standalone --nproc-per-node=2 \
# "label_QCD,label_Hbb,label_Zqq,label_Wqq,label_Tbqq"
# python -u \
# "/HEP/export/home/hgao50/jet-anomaly-data/ak8-data"
# --cms-val-fraction 0.05 \
#     --cms-test-fraction 0.45 \

torchrun --standalone --nproc-per-node=2 \
    scripts/run_train_lejepa_part.py \
    --dataset cms \
    --dataset-root "/HEP/export/home/lwang223/jet-anomaly-data/ak8-v5" \
    --model semi-sup-triplet \
    --background-labels "label_QCD,label_Hbb,label_Zqq,label_Wqq,label_Tbqq" \
    --signal-labels "label_QCD,label_Hbb,label_Zqq,label_Wqq,label_Tbqq" \
    --cms-val-fraction 0.1 \
    --cms-test-fraction 0.1 \
    --embed-dim 32 \
    --representation-dim 32 \
    --dropout 0.02 \
    --num-layers 8 \
    --num-heads 8 \
    --batch-size 256 \
    --steps-per-epoch 4000 \
    --val-steps 1000 \
    --eval-steps 1000 \
    --epochs 40 \
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
    --triplet-margin 0.2 \
    --classification-weight 0.1 \
    --num-workers 2 \
    --prefetch-factor 2 \
    --shuffle-active-shards 2 \
    --output-dir "plots/largerun/cms-correct-deta-dphi-32"

python -u scripts/diagnose_lejepa_latents.py \
    "plots/largerun/cms-correct-deta-dphi-32"

torchrun --standalone --nproc-per-node=2 \
    scripts/run_train_lejepa_part.py \
    --dataset cms \
    --dataset-root "/HEP/export/home/hgao50/jet-anomaly-data/ak8-data" \
    --model semi-sup-triplet \
    --checkpoint "plots/largerun/cms-correct-deta-dphi-32/last_checkpoint.pt" \
    --background-labels "label_Real" \
    --cms-val-fraction 0.05 \
    --cms-test-fraction 0.45 \
    --embed-dim 32 \
    --representation-dim 32 \
    --dropout 0.02 \
    --num-layers 8 \
    --num-heads 8 \
    --batch-size 256 \
    --steps-per-epoch 1000 \
    --val-steps 1000 \
    --epochs 5 \
    --learning-rate 5e-5 \
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
    --triplet-margin 0.2 \
    --classification-weight 0.0 \
    --num-workers 2 \
    --prefetch-factor 2 \
    --shuffle-active-shards 2 \
    --output-dir "plots/largerun/cms-correct-real-finetune-32"

python -u scripts/evaluate_label_real_anomalies.py \
    "plots/largerun/cms-correct-real-finetune-32" \
    --top-fraction 0.02

# python -u scripts/evaluate_label_real_anomalies.py \
#     --plot-events-npy plots/largerun/cms-correct-real-finetune/real_data_evaluation/real_test_top_2pct_events.npy \
#     --feature-list plots/largerun/cms-correct-real-finetune/real_data_evaluation/real_test_top_2pct_events_features.json \
#     --event-scores plots/largerun/cms-correct-real-finetune/real_data_evaluation/real_test_top_2pct_event_scores.npy \
#     --num-visualize 24
