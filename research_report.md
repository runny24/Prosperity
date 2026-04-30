# Prosperity 4 Alpha Lab Research Report

Run warning: `DANGEROUS_RESEARCH_ONLY__DO_NOT_ASSUME_OUT_OF_SAMPLE_ALPHA`

## Executive read
This run evaluates whether order-book pressure, microprice dislocations, clock effects, regime embeddings, and product cross-sectional shadows can explain the configured future target. Treat the result as a research upper bound, not live evidence.

## Config
```json
{
  "data": "/home/claude/prosperity_data",
  "out": "/home/claude/runs/alpha_lab",
  "products": null,
  "horizon": 10,
  "target": "return",
  "overfit_level": 1,
  "epochs": 50,
  "batch_size": 2048,
  "lr": 0.0003,
  "weight_decay": 1e-06,
  "seed": 1337,
  "valid_fraction": 0.18,
  "no_leaky_features": true,
  "final_fit_all": false,
  "dry_run": false,
  "device": "auto",
  "synthetic_pretrain": false,
  "make_synthetic": false,
  "synthetic_out": "data/synthetic_prosperity",
  "synthetic_products": null,
  "synthetic_days": 6,
  "synthetic_ticks_per_day": 10000,
  "synthetic_seed": 20260429,
  "feature_bags": 1,
  "snapshot_count": 4,
  "pseudo_label_rounds": 1,
  "max_rows": 20000,
  "use_tree_features": false,
  "use_pca_ica": false,
  "use_target_smoothing": true,
  "export_research_report": true,
  "emit_manual_template": false,
  "manual_signals_json": null,
  "ignith_fee_power": 2.0
}
```

## Global metrics
```json
{
  "warning": "DANGEROUS_RESEARCH_ONLY__DO_NOT_ASSUME_OUT_OF_SAMPLE_ALPHA",
  "rows": 18500,
  "train_rows": 15095,
  "valid_rows": 3405,
  "n_features": 330,
  "n_products": 50,
  "target": "return",
  "horizon": 10,
  "valid_mse": 0.0010934901656582952,
  "valid_mae": 0.025357874110341072,
  "valid_r2": -0.25144755840301514,
  "valid_ic": 0.0985522765301447,
  "valid_rank_ic": 0.08522115335503715,
  "sign_accuracy": 0.5292217327459618,
  "model_infos": [
    {
      "best_epoch": 39,
      "best_score": 0.09854134756005647,
      "feature_count": 330
    }
  ]
}
```

## Best group pseudo-sharpes
| product                       |   day |   n |         mse |       mae |          ic |    rank_ic |   sign_acc |   pseudo_sharpe |
|:------------------------------|------:|----:|------------:|----------:|------------:|-----------:|-----------:|----------------:|
| UV_VISOR_ORANGE               |     4 |  23 | 0.000609824 | 0.0197984 | -0.00933328 | -0.110672  |   0.956522 |         66.7254 |
| ROBOT_IRONING                 |     3 |  23 | 0.000354971 | 0.0160656 |  0.293381   |  0.337945  |   0.956522 |         52.5934 |
| PEBBLES_L                     |     4 |  22 | 0.00035758  | 0.0150295 |  0.446599   |  0.426313  |   0.954545 |         42.147  |
| SLEEP_POD_COTTON              |     4 |  23 | 0.000748208 | 0.0201332 | -0.452349   | -0.394269  |   0.826087 |         37.5588 |
| GALAXY_SOUNDS_SOLAR_FLAMES    |     4 |  24 | 0.000386547 | 0.0163676 | -0.0740168  | -0.386957  |   0.75     |         33.0129 |
| UV_VISOR_MAGENTA              |     3 |  23 | 0.000200923 | 0.0121608 | -0.293372   | -0.311265  |   0.782609 |         28.6423 |
| UV_VISOR_AMBER                |     4 |  25 | 0.000392657 | 0.0149077 |  0.37077    |  0.485385  |   0.8      |         26.4226 |
| SNACKPACK_STRAWBERRY          |     3 |  23 | 0.000222858 | 0.0124462 |  0.560419   |  0.544466  |   0.826087 |         26.3779 |
| SNACKPACK_VANILLA             |     3 |  21 | 0.000128793 | 0.0094206 |  0.380191   |  0.383117  |   0.809524 |         26.1856 |
| PANEL_2X2                     |     3 |  21 | 0.000559468 | 0.0206981 |  0.714148   |  0.464935  |   0.857143 |         25.8532 |
| PANEL_4X4                     |     2 |  23 | 0.000422024 | 0.0149538 |  0.201643   |  0.289526  |   0.869565 |         25.8097 |
| GALAXY_SOUNDS_SOLAR_WINDS     |     4 |  23 | 0.000303592 | 0.0145489 |  0.29612    |  0.27668   |   0.782609 |         25.4776 |
| MICROCHIP_TRIANGLE            |     3 |  21 | 0.00283972  | 0.0461779 | -0.08329    |  0.0967847 |   0.761905 |         24.5308 |
| MICROCHIP_OVAL                |     2 |  23 | 0.000879181 | 0.0249884 | -0.347878   | -0.385375  |   0.869565 |         24.0311 |
| MICROCHIP_SQUARE              |     4 |  19 | 0.000804649 | 0.0231254 |  0.0952828  | -0.0315789 |   0.789474 |         23.5237 |
| TRANSLATOR_VOID_BLUE          |     4 |  25 | 0.000256796 | 0.0114776 |  0.692254   |  0.720769  |   0.76     |         23.275  |
| PEBBLES_S                     |     4 |  24 | 0.00337584  | 0.0522388 |  0.453709   |  0.435652  |   0.75     |         21.9188 |
| ROBOT_MOPPING                 |     4 |  22 | 0.00050722  | 0.0186263 |  0.260302   |  0.278374  |   0.727273 |         21.8831 |
| GALAXY_SOUNDS_PLANETARY_RINGS |     3 |  24 | 0.000613152 | 0.0182963 |  0.221492   |  0.167826  |   0.666667 |         21.8155 |
| UV_VISOR_AMBER                |     3 |  21 | 0.000250912 | 0.0134971 |  0.323049   |  0.18961   |   0.714286 |         19.9255 |

## Most suspicious / strongest target-correlated features
| feature                   |   target_corr | suspicious   |
|:--------------------------|--------------:|:-------------|
| name_has_xl               |     0.0697782 | False        |
| name_has_xs               |    -0.0598511 | False        |
| ordinal_x_spread          |     0.0541226 | False        |
| ret_sketch_13             |    -0.0531255 | False        |
| mid_z_21                  |    -0.0521886 | False        |
| trade_count_377           |     0.0514002 | False        |
| mid_z_34                  |    -0.0492124 | False        |
| trade_count_89            |     0.0488973 | False        |
| category_ordinal          |     0.04764   | False        |
| category_ordinal_centered |     0.04764   | False        |
| name_has_garlic           |     0.0461812 | False        |
| ret_sketch_21             |    -0.045936  | False        |
| mid_z_55                  |    -0.0450049 | False        |
| ema_cross_3_21            |    -0.0442682 | False        |
| ret_lag_13                |    -0.0442027 | False        |
| ret_lag_8                 |    -0.042752  | False        |
| mid_z_13                  |    -0.0423985 | False        |
| mid_z_89                  |    -0.0395451 | False        |
| ret_lag_5                 |    -0.0389445 | False        |
| mid_z_377                 |    -0.0374044 | False        |
| mid_z_233                 |    -0.0374044 | False        |
| mid_z_144                 |    -0.0374005 | False        |
| ret_sketch_34             |    -0.037383  | False        |
| ema_cross_5_34            |    -0.0359544 | False        |
| liquidity_void            |    -0.0334715 | False        |
| mid_z_8                   |    -0.0331031 | False        |
| vol_sum_8                 |     0.0325614 | False        |
| mid_std_34                |     0.0322897 | False        |
| ret_lag_3                 |    -0.0321061 | False        |
| vol_sum_13                |     0.0320896 | False        |
| day                       |    -0.0315496 | False        |
| vol_sum_5                 |     0.0307248 | False        |
| ret_lag_34                |    -0.0307085 | False        |
| ret_lag_21                |    -0.0306168 | False        |
| trade_count_21            |     0.0303072 | False        |
| profit_and_loss           |     0.0298214 | False        |
| mid_z_5                   |    -0.0291975 | False        |
| day_x_timestamp           |    -0.0291931 | False        |
| ret_lag_2                 |    -0.0290592 | False        |
| vol_sum_21                |     0.0289577 | False        |

## Feature count
330