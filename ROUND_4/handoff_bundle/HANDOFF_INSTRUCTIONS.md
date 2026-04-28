# Midway Handoff Instructions

Send this whole bundle to the Midway operator.

What to do on Midway:

1. Unpack the bundle in a working directory.
2. Review `bot_research/outputs/LOCAL_RESEARCH_SUMMARY.md` for context.
3. Submit the array job:
   `sbatch bot_research/hpc/submit_manifest.sbatch`
4. After jobs finish, aggregate:
   `python bot_research/hpc/aggregate_results.py`

Important files:
- `bot_research/outputs/hpc_prep/hpc_experiment_manifest.csv`
- `bot_research/outputs/hpc_prep/features/tick_features_ASH_COATED_OSMIUM.csv.gz`
- `bot_research/outputs/hpc_prep/features/tick_features_INTARIAN_PEPPER_ROOT.csv.gz`
- `bot_research/hpc/manifest_worker.py`
- `bot_research/hpc/submit_manifest.sbatch`

The current submit script assumes 552 manifest rows and throttles to 24 concurrent tasks.
