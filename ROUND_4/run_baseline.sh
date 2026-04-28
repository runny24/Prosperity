#!/bin/bash
#SBATCH --job-name=r4_baseline
#SBATCH --account=pi-xyang
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/baseline_%j.out
#SBATCH --error=logs/baseline_%j.err

# ── Setup ────────────────────────────────────────────────────────────────────
set -euo pipefail
module load python/anaconda-2022.05 2>/dev/null || module load python/3.9 2>/dev/null || true

# Directory containing this script, backtester.py, datamodel.py, v23.py, and the CSVs.
# Edit this if your files are elsewhere (e.g. $SCRATCH/round4).
WORKDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$WORKDIR"
mkdir -p logs results

echo "========================================"
echo "  Prosperity R4 — BASELINE (v23)"
echo "  Working dir: $WORKDIR"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Node:   $SLURMD_NODENAME"
echo "========================================"

python backtester.py \
    --trader  v23.py \
    --data    . \
    --output  results/baseline_v23.json

echo "Done. Results in results/baseline_v23.json"
