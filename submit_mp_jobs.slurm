#!/bin/bash
#SBATCH --job-name=mapillary_run
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --array=0-40%10                       # n total regions in <file>.csv (total lines - 2)
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=10                    # Matches INNER_MAX_WORKERS
#SBATCH --mem=32G

module load anaconda

source $(conda info --base)/etc/profile.d/conda.sh
conda activate mp

mkdir -p logs

echo "Starting Mapillary processing for Array Task ID: $SLURM_ARRAY_TASK_ID"

python slurm_batch_chunks_mp_api.py '<region>.csv' \
    --inner-max-workers $SLURM_CPUS_PER_TASK \
    --parent-dir grid_runs \
    --no-download-images
