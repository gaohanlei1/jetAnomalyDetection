#!/bin/bash
#SBATCH --nodes=1               # node count
#SBATCH -p gpu --gres=gpu:1     # number of gpus per node
#SBATCH --nodelist=gpu3101      # the L40S GPU!! 3001-3005 and 3101 are L40S
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

# python -u visualize/plot_latent_space.py \
#     plots/run-cls-transformer-ae-lixing-h4-l2-4l-5v-10epo \
#     "QCD" \
#     "WJet" \
#     --output plots/run-cls-transformer-ae-lixing-h4-l2-4l-5v-10epo/latent_space.png

# python -u visualize/plot_latent_space.py \
#     plots/run-cls-transformer-ae-lixing-h32-l8-150epo \
#     "QCD" \
#     "WJet" \
#     --output plots/run-cls-transformer-ae-lixing-h32-l8-150epo/latent_space.png

python -u visualize/plot_latent_space.py \
    plots/run-cls-t2t-transformer-ae-lixing-h4-l2-4l-8v-10epo \
    "QCD" \
    "WJet" \
    --output plots/run-cls-t2t-transformer-ae-lixing-h4-l2-4l-8v-10epo/latent_space.png