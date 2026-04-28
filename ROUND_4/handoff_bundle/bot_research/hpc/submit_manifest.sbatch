#!/bin/bash
# Manifest-driven Midway job array for bot_research.
#
# Current manifest size is 552 rows, so the default array range is 0-551.
# The %24 throttle is deliberate: it keeps enough parallelism to use Midway
# well while avoiding a tiny wall-clock runtime from launching everything at once.
# If you regenerate the manifest, update the --array range accordingly.
#
# Submit:
#   sbatch bot_research/hpc/submit_manifest.sbatch

#SBATCH --job-name=bot_manifest
#SBATCH --account=pi-YOUR_PI_ACCOUNT
#SBATCH --partition=caslake
#SBATCH --array=0-551%24
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:45:00
#SBATCH --output=bot_research/logs/manifest_%A_%a.out
#SBATCH --error=bot_research/logs/manifest_%A_%a.err

module load python/anaconda-2022.05

cd $SLURM_SUBMIT_DIR
mkdir -p bot_research/logs bot_research/outputs/hpc_results

python bot_research/hpc/manifest_worker.py \
  --job-id $SLURM_ARRAY_TASK_ID \
  --manifest bot_research/outputs/hpc_prep/hpc_experiment_manifest.csv \
  --features-dir bot_research/outputs/hpc_prep/features \
  --out-dir bot_research/outputs/hpc_results
