# Running Prosperity Round 5 NN Training on Midway3
### Estimated wall time: ~1–1.5 hours (4× A100, 48 CPUs)

---

## How it works

```
parallel_train.py (orchestrator)
 ├── Phase 1: Feature engineering on CPU  (~45–60 min, 48 cores)
 │            Loads 1.35M rows, builds 300+ features, fits ExtraTrees teacher
 │            Saves shared _shared_features.npz
 │
 ├── Phase 2: Build 3 feature bag masks   (seconds)
 │
 └── Phase 3: Launch 3 subprocesses simultaneously
              _bag_worker.py --bag-id 0  → GPU 0  (~20 min)
              _bag_worker.py --bag-id 1  → GPU 1  (~20 min)
              _bag_worker.py --bag-id 2  → GPU 2  (~20 min)
              └── Ensemble predictions + save bundle
```

---

## Files needed on Midway3

```
/project/xyang2/felixy/misc/round5/
├── data/
│   ├── prices_round_4_day_2.csv
│   ├── prices_round_4_day_3.csv
│   ├── prices_round_4_day_4.csv
│   ├── trades_round_4_day_2.csv
│   ├── trades_round_4_day_3.csv
│   └── trades_round_4_day_4.csv
├── prosperity4_5m_nn_trainer.py    ← original trainer (imported as a library)
├── parallel_train.py               ← orchestrator (called by sbatch)
├── _bag_worker.py                  ← per-GPU subprocess worker
└── train_round5.sbatch             ← job script
```

---

## Step 1 — Upload everything from your Mac

Run this **once from your local machine** (not Midway3).

```bash
# Upload the 6 data CSVs
rsync -avz --progress \
  ~/Downloads/prices_round_4_day_2.csv \
  ~/Downloads/prices_round_4_day_3.csv \
  ~/Downloads/prices_round_4_day_4.csv \
  ~/Downloads/trades_round_4_day_2.csv \
  ~/Downloads/trades_round_4_day_3.csv \
  ~/Downloads/trades_round_4_day_4.csv \
  felixy@midway3.rcc.uchicago.edu:/project/xyang2/felixy/misc/round5/data/

# Upload the scripts
rsync -avz --progress \
  ~/Downloads/prosperity4_5m_nn_trainer.py \
  ~/Downloads/parallel_train.py \
  ~/Downloads/_bag_worker.py \
  ~/Downloads/train_round5.sbatch \
  felixy@midway3.rcc.uchicago.edu:/project/xyang2/felixy/misc/round5/
```

---

## Step 2 — Log in and create directories

```bash
ssh felixy@midway3.rcc.uchicago.edu

mkdir -p /project/xyang2/felixy/misc/round5/data
mkdir -p /project/xyang2/felixy/misc/round5/logs
mkdir -p /project/xyang2/felixy/misc/round5/runs
```

---

## Step 3 — Submit the job

```bash
cd /project/xyang2/felixy/misc/round5
sbatch train_round5.sbatch
# → Submitted batch job 1234567
```

---

## Step 4 — Monitor progress

```bash
# Job status
squeue --user=felixy

# Live log (replace JOBID)
tail -f /project/xyang2/felixy/misc/round5/logs/prosperity_r5_nn_JOBID.out
```

Expected output sequence:
```
[info] Job 1234567 on midwaygpu01 at Thu Apr 30 ...
[info] GPUs available:
0, NVIDIA A100-SXM4-40GB, 40536 MiB
1, NVIDIA A100-SXM4-40GB, 40536 MiB
2, NVIDIA A100-SXM4-40GB, 40536 MiB
3, NVIDIA A100-SXM4-40GB, 40536 MiB
======================================================================
[phase 1/3] Feature engineering (CPU)
======================================================================
  prices=(1350000, 19)  trades=(40194, 11)  products=50
  [teacher] Fitting ExtraTrees (uses all CPUs)...
  X=(1350000, 331)  train=1,107,000  valid=243,000
======================================================================
[phase 2/3] Saving shared feature arrays & building bag masks
======================================================================
  Shared array file: 1782.4 MB
  bag 0: 265 / 331 features
  bag 1: 271 / 331 features
  bag 2: 268 / 331 features
======================================================================
[phase 3/3] Launching 3 bag(s) across 4 GPU(s)
======================================================================
  Bag 0 → GPU 0
  Bag 1 → GPU 1
  Bag 2 → GPU 2
  [bag 0] GPU=0  device=cuda  pid=...
  [bag 1] GPU=1  device=cuda  pid=...
  [bag 2] GPU=2  device=cuda  pid=...
  [bag 0] ep  36  valid_ic=0.1183  ...
  [bag 1] ep  36  valid_ic=0.1201  ...
  ...
[metrics] { "valid_ic": 0.127, "valid_rank_ic": 0.114, ... }
[done] All outputs written to .../runs/alpha_lab_1350k
```

---

## Step 5 — Download results

```bash
rsync -avz --progress \
  felixy@midway3.rcc.uchicago.edu:/project/xyang2/felixy/misc/round5/runs/alpha_lab_1350k/ \
  ~/Downloads/alpha_lab_1350k/
```

---

## Resource summary

| Resource       | Requested | Expected usage              |
|----------------|-----------|-----------------------------|
| GPUs           | 4× A100   | 3 active (1 spare)          |
| CPUs           | 48        | ~48 during feature eng, ~6 per bag during training |
| RAM            | 128 GB    | ~80–100 GB peak (shared .npz + 3 subprocess loads) |
| Wall time      | 4 hours   | Expected ~1–1.5 hours       |

---

## Troubleshooting

**"Constraint a100 not available"**
Remove `#SBATCH --constraint=a100` from the sbatch file — Midway3 may not require it.

**Out of memory during feature engineering**
Reduce `--max-rows 1000000` in the sbatch file.

**A bag subprocess fails**
Check the job's `.err` log. Most common cause is a CUDA OOM — reduce `--batch-size 4096`.
