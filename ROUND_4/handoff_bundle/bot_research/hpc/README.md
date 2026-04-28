# Midway Handoff

This folder is the handoff point from local research to Midway.

The canonical local output to consume is:

- `../outputs/hpc_prep/hpc_experiment_manifest.csv`
- `../outputs/hpc_prep/features/tick_features_<symbol>.csv.gz`

Recommended workflow:

1. Generate the manifest locally:

```bash
python bot_research/scripts/run_local_pipeline.py
python bot_research/hpc/prepare_hpc_features.py
```

2. Copy `bot_research/` to Midway.
3. Run one worker per manifest row:

```bash
python bot_research/hpc/manifest_worker.py --job-id 0
```

4. Submit an array job sized to the manifest length:

```bash
sbatch bot_research/hpc/submit_manifest.sbatch
```

5. After jobs finish, aggregate:

```bash
python bot_research/hpc/aggregate_results.py
```

6. If you want a minimal package to send to a friend, create it locally:

```bash
python bot_research/hpc/make_handoff_bundle.py
```

The point is to launch many small, ranked experiments rather than one undifferentiated brute-force search.
