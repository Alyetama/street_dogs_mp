#!/bin/bash
#SBATCH --job-name=mapillary_run
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --array=0-40%10                       # n total regions in <file>.csv (total lines - 2)
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=10                    # Matches INNER_MAX_WORKERS
#SBATCH --mem=32G
#SBATCH --signal=B:INT@120

REGION_FILE="<region>.csv"

module load anaconda

source $(conda info --base)/etc/profile.d/conda.sh
conda activate mp

mkdir -p logs

echo "Starting Mapillary processing for Array Task ID: $SLURM_ARRAY_TASK_ID"

exec python batch_chunks_mp_api.py "$REGION_FILE" \
    --inner-max-workers $((SLURM_CPUS_PER_TASK * 4)) \
    --parent-dir grid_runs \
    --no-download-images \
    --slurm
