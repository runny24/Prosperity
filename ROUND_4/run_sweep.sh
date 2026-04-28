#!/bin/bash
#SBATCH --job-name=r4_sweep
#SBATCH --account=pi-xyang
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --array=5-8          # SLURM_ARRAY_TASK_ID = SELL_MIN value to test
#SBATCH --output=logs/sweep_sellmin%a_%j.out
#SBATCH --error=logs/sweep_sellmin%a_%j.err

# ── Setup ────────────────────────────────────────────────────────────────────
set -euo pipefail
module load python/anaconda-2022.05 2>/dev/null || module load python/3.9 2>/dev/null || true

WORKDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$WORKDIR"
mkdir -p logs results

SELL_MIN=$SLURM_ARRAY_TASK_ID

echo "========================================"
echo "  Prosperity R4 — VEV_5500 SELL_MIN sweep"
echo "  Testing SELL_MIN = $SELL_MIN"
echo "  Job ID: $SLURM_JOB_ID  (array task $SLURM_ARRAY_TASK_ID)"
echo "  Node:   $SLURMD_NODENAME"
echo "========================================"

python backtester.py \
    --trader   v23.py \
    --data     . \
    --sell_min "$SELL_MIN" \
    --output   "results/sweep_sellmin${SELL_MIN}.json"

echo "Done. Results in results/sweep_sellmin${SELL_MIN}.json"
