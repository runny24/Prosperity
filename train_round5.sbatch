#!/bin/bash
#SBATCH --job-name=prosperity_r5_nn
#SBATCH --account=pi-xyang2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH --constraint=a100
#SBATCH --time=04:00:00
#SBATCH --output=/project/xyang2/felixy/misc/round5/logs/%x_%j.out
#SBATCH --error=/project/xyang2/felixy/misc/round5/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=felixy@uchicago.edu

# ── Environment ───────────────────────────────────────────────────────────────
module purge
module load python/anaconda-2022.05

ENV_DIR=/project/xyang2/felixy/misc/round5/env

if [ ! -d "$ENV_DIR" ]; then
    echo "[setup] Creating conda environment at $ENV_DIR"
    conda create -y -p "$ENV_DIR" python=3.11
fi

source activate "$ENV_DIR"

python -c "import torch" 2>/dev/null || \
    pip install torch --index-url https://download.pytorch.org/whl/cu121 -q
python -c "import sklearn" 2>/dev/null || \
    pip install scikit-learn joblib pandas numpy -q

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT=/project/xyang2/felixy/misc/round5
DATA=$ROOT/data
OUT=$ROOT/runs/alpha_lab_1350k
TRAINER=$ROOT/prosperity4_5m_nn_trainer.py
LAUNCHER=$ROOT/parallel_train.py

mkdir -p "$ROOT/logs" "$OUT"

echo "[info] Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "[info] GPUs available:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# ── Patch trainer for pandas compatibility ────────────────────────────────────
sed -i "s/\.reset_index(level=\[0, 1\], drop=True)/\.reset_index(drop=True)/g" "$TRAINER"
echo "[patch] Applied pandas reset_index fix to trainer"

# ── Parallelism hints for CPU-bound feature engineering ───────────────────────
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12
export NUMEXPR_NUM_THREADS=12

# ── Launch parallel trainer ───────────────────────────────────────────────────
python "$LAUNCHER" \
    --data           "$DATA"    \
    --out            "$OUT"     \
    --max-rows       1350000    \
    --overfit-level  4          \
    --epochs         180        \
    --batch-size     8192       \
    --lr             3e-4       \
    --feature-bags   3          \
    --snapshot-count 4          \
    --horizon        10         \
    --target         return     \
    --no-leaky-features         \
    --valid-fraction 0.18       \
    --seed           1337

echo "[info] Finished at $(date)"
echo "[info] Outputs: $OUT"
