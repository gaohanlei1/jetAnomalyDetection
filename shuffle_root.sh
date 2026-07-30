#!/bin/bash
#SBATCH --nodes=1               # node count
# SBATCH -n 12                   # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH -t 24:00:00             # total run time limit (HH:MM:SS)
#SBATCH --cpus-per-task=12       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=96GB           # CPU RAM
#SBATCH --job-name='JETANOMALY-SHUFFLE'
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

module load miniforge3/25.3.0-3
source ${MAMBA_ROOT_PREFIX}/etc/profile.d/conda.sh
# source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
# conda init
conda activate jet

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# python scripts/shuffle_cms_root_events_uproot5_multiprocess_resume_v9.py \
#     --source-dir "/HEP/export/home/hgao50/jet-anomaly-data/ak8-v4" \
#     --target-dir "/HEP/export/home/lwang223/jet-anomaly-data/ak8-v5" \
#     --jet-types "wjets" \
#     --events-per-file 20000 \
#     --num-temp-buckets 32 \
#     --read-step "512 MB" \
#     --temp-flush-events 10000 \
#     --temp-part-events 20000 \
#     --max-pending-events 100000 \
#     --max-pending-gb 90 \
#     --phase1-processes 16 \
#     --phase1-start-method spawn \
#     --phase2-processes 4 \
#     --phase2-start-method spawn \
#     --temp-compression zlib \
#     --temp-compression-level 1 \
#     --final-compression zlib \
#     --final-compression-level 4 \
#     --seed 404 \
#     --progress-every 50000 \
#     --manifest-path "/HEP/export/home/lwang223/jet-anomaly-data/ak8-v5/cms-shuffle-manifest.json"

python -u scripts/split_cms_root_shards_uproot5.py \
    --source-dir "/HEP/export/home/lwang223/jet-anomaly-data/ak8-v5/hbb" \
    --target-dir "/HEP/export/home/lwang223/jet-anomaly-data/ak8-v5/hbb-50k" \
    --file-glob "hbb_*.root" \
    --output-prefix "hbb" \
    --events-per-file 20000 \
    --compression zlib \
    --compression-level 4