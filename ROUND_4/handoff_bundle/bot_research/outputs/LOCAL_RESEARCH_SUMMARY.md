# Local Research Summary

## Bottom Line

- P3 named-bot behavior is recoverable from market-visible information: mean top-1 accuracy `63.3%` and top-3 accuracy `86.0%` in leave-one-day-out tests.
- The current anonymous market shows strong schedule structure, so timestamp/event reconstruction is worth prioritizing before brute-force rule search.
- The current products map mostly to one broad P3-style archetype, but the HPC manifest now reserves `p3_seeded`, `no_prior`, and `drift_search` lanes so the supercomputer is not locked into old priors.

## P3 Validation

- Named participant-side events: `106954` across `11` traders in round 5.
- Mean anonymous recovery top-1 accuracy: `63.3%`.
- Mean anonymous recovery top-3 accuracy: `86.0%`.

## Current Anonymous Data

- `ASH_COATED_OSMIUM`: `818` unique trade timestamps, `367` repeat on at least 2 days, `192` repeat on all 3 days.
- `INTARIAN_PEPPER_ROOT`: `540` unique trade timestamps, `342` repeat on at least 2 days, `109` repeat on all 3 days.

## P3 Archetype Transfer

- Archetype `0`: `11` trader-symbol profiles, top examples `Caesar/VOLCANIC_ROCK_VOUCHER_9750, Caesar/VOLCANIC_ROCK_VOUCHER_10000, Caesar/VOLCANIC_ROCK_VOUCHER_9500, Caesar/VOLCANIC_ROCK_VOUCHER_10250, Caesar/VOLCANIC_ROCK_VOUCHER_10500`.
- Archetype `1`: `57` trader-symbol profiles, top examples `Charlie/SQUID_INK, Charlie/KELP, Paris/KELP, Paris/SQUID_INK, Charlie/RAINFOREST_RESIN`.

- `ASH_COATED_OSMIUM` maps mostly to archetype `1` with event share `97.1%` and mean confidence `97.5%`.
- `INTARIAN_PEPPER_ROOT` maps mostly to archetype `1` with event share `98.4%` and mean confidence `98.1%`.

## Drift Check

- `ASH_COATED_OSMIUM`: mean drift z-score `0.313`, max drift z-score `0.937`.
- `INTARIAN_PEPPER_ROOT`: mean drift z-score `0.370`, max drift z-score `0.954`.

## HPC Prep

- Total manifest jobs: `92`.
- Lane `drift_search` / `regime_rule_search`: `6` jobs, mean score `0.500`.
- Lane `drift_search` / `timestamp_window_search`: `12` jobs, mean score `0.955`.
- Lane `no_prior` / `regime_rule_search`: `6` jobs, mean score `0.659`.
- Lane `no_prior` / `timestamp_window_search`: `12` jobs, mean score `0.959`.
- Lane `p3_seeded` / `regime_rule_search`: `16` jobs, mean score `0.622`.
- Lane `p3_seeded` / `timestamp_window_search`: `40` jobs, mean score `0.954`.

Top timestamp candidates:
- `INTARIAN_PEPPER_ROOT` at `35700` score `1.000` days `3` archetype `1`.
- `ASH_COATED_OSMIUM` at `15100` score `0.992` days `3` archetype `1`.
- `INTARIAN_PEPPER_ROOT` at `775500` score `0.952` days `3` archetype `1`.
- `ASH_COATED_OSMIUM` at `923800` score `0.952` days `3` archetype `1`.
- `INTARIAN_PEPPER_ROOT` at `561300` score `0.952` days `3` archetype `1`.
- `INTARIAN_PEPPER_ROOT` at `376700` score `0.952` days `3` archetype `1`.
- `INTARIAN_PEPPER_ROOT` at `466000` score `0.952` days `3` archetype `1`.
- `INTARIAN_PEPPER_ROOT` at `344000` score `0.952` days `3` archetype `1`.
- `INTARIAN_PEPPER_ROOT` at `942700` score `0.952` days `3` archetype `1`.
- `INTARIAN_PEPPER_ROOT` at `977200` score `0.952` days `3` archetype `1`.
- `INTARIAN_PEPPER_ROOT` at `927200` score `0.952` days `3` archetype `1`.
- `INTARIAN_PEPPER_ROOT` at `566100` score `0.952` days `3` archetype `1`.

## Ready For Midway

- The local phase is complete enough to move into Midway. The next step should consume `bot_research/outputs/hpc_prep/hpc_experiment_manifest.csv` as the source of parallel jobs.
