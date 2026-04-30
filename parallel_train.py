#!/usr/bin/env python3
"""
parallel_train.py — Multi-GPU parallel bag trainer for Prosperity 4 Round 5.

Strategy:
  1. Feature engineering runs once on CPU (the bottleneck).
  2. Feature arrays are saved to a shared .npz file.
  3. N bags are trained simultaneously, each in its own subprocess
     pinned to a separate GPU via CUDA_VISIBLE_DEVICES.
  4. Results are collected and assembled into the standard bundle.

Usage (called by train_round5.sbatch):
    python parallel_train.py --data /path/to/data --out /path/to/out [flags]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import select
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

# ── Import trainer internals ──────────────────────────────────────────────────
# The trainer lives in the same directory as this script.
sys.path.insert(0, str(Path(__file__).parent))

from prosperity4_5m_nn_trainer import (
    DANGEROUS_WARNING,
    AlphaLabConfig,
    FeatureFactory,
    MemorizingMixtureAlphaNet,
    clean_training_frame,
    fit_tree_teacher,
    information_coefficient,
    leakage_report,
    load_prices,
    load_trades,
    make_weights,
    maybe_mkdir,
    alpha_research_report,
    metrics_by_group,
    predict_numpy,
    rank_ic,
    round5_products_from_args,
    seed_everything,
    select_scaler,
    time_split,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Parallel multi-GPU bag trainer for Prosperity 4 Round 5.",
    )
    p.add_argument("--data",                 required=True)
    p.add_argument("--out",                  default="runs/alpha_lab")
    p.add_argument("--products",             nargs="*", default=None)
    p.add_argument("--horizon",              type=int,   default=10)
    p.add_argument("--target",               choices=["return","price","direction","edge_z"], default="return")
    p.add_argument("--overfit-level",        type=int,   default=4)
    p.add_argument("--epochs",               type=int,   default=180)
    p.add_argument("--batch-size",           type=int,   default=8192)
    p.add_argument("--lr",                   type=float, default=3e-4)
    p.add_argument("--weight-decay",         type=float, default=1e-6)
    p.add_argument("--seed",                 type=int,   default=1337)
    p.add_argument("--valid-fraction",       type=float, default=0.18)
    p.add_argument("--no-leaky-features",    action="store_true")
    p.add_argument("--max-rows",             type=int,   default=None)
    p.add_argument("--feature-bags",         type=int,   default=3)
    p.add_argument("--snapshot-count",       type=int,   default=4)
    p.add_argument("--no-tree-features",     action="store_true")
    p.add_argument("--no-pca-ica",           action="store_true")
    p.add_argument("--no-target-smoothing",  action="store_true")
    p.add_argument("--no-report",            action="store_true")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    cfg = AlphaLabConfig(
        data=args.data,
        out=args.out,
        products=args.products,
        horizon=args.horizon,
        target=args.target,
        overfit_level=max(0, min(5, args.overfit_level)),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        valid_fraction=args.valid_fraction,
        no_leaky_features=args.no_leaky_features,
        max_rows=args.max_rows,
        feature_bags=args.feature_bags,
        snapshot_count=args.snapshot_count,
        use_tree_features=not args.no_tree_features,
        use_pca_ica=not args.no_pca_ica,
        use_target_smoothing=not args.no_target_smoothing,
        export_research_report=not args.no_report,
    )

    seed_everything(cfg.seed)
    out = maybe_mkdir(cfg.out)
    (out / "config.json").write_text(
        json.dumps(dataclasses.asdict(cfg), indent=2, default=str)
    )

    # ── Phase 1: Feature engineering (single process, all CPUs) ──────────────
    print("=" * 70, flush=True)
    print("[phase 1/3] Feature engineering (CPU)", flush=True)
    print("=" * 70, flush=True)

    prices = load_prices(
        cfg.data,
        round5_products_from_args(cfg.products, use_round5_default=False) or None,
        cfg.max_rows,
    )
    trades = load_trades(cfg.data)
    print(
        f"  prices={prices.shape}  "
        f"trades={None if trades is None else trades.shape}  "
        f"products={prices['product'].nunique()}",
        flush=True,
    )

    ff = FeatureFactory(cfg)
    df = ff.build(prices, trades, fit=True)
    df = clean_training_frame(df, cfg)
    feature_cols = ff.pick_feature_columns(df, fit=True)
    target_col = f"target_{cfg.target}"

    X_raw   = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32).values
    y       = df[target_col].astype(np.float32).values
    aux     = np.vstack([np.sign(y), np.abs(y), y * y]).T.astype(np.float32)
    prod    = df["product_id"].astype(int).values
    reg     = df["regime_id"].astype(int).values
    weights = make_weights(df, y, cfg)

    train_mask, valid_mask = time_split(df, cfg.valid_fraction)
    scaler = select_scaler(cfg)
    scaler.fit(X_raw[train_mask])
    X = scaler.transform(X_raw).astype(np.float32)

    tree_info = None
    if cfg.use_tree_features:
        print("  [teacher] Fitting ExtraTrees (uses all CPUs)...", flush=True)
        tree_info = fit_tree_teacher(X[train_mask], y[train_mask], cfg)
        df["tree_teacher_pred"] = tree_info["teacher"].predict(X)
        feature_cols.append("tree_teacher_pred")
        X_raw2 = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32).values
        scaler = select_scaler(cfg)
        scaler.fit(X_raw2[train_mask])
        X = scaler.transform(X_raw2).astype(np.float32)

    n_products = max(1, len(ff.product_to_id))
    n_regimes  = max(1, len(ff.regime_to_id))

    print(
        f"  X={X.shape}  train={train_mask.sum():,}  valid={valid_mask.sum():,}  "
        f"products={n_products}  regimes={n_regimes}",
        flush=True,
    )

    # ── Phase 2: Save shared arrays + build bag masks ─────────────────────────
    print("=" * 70, flush=True)
    print("[phase 2/3] Saving shared feature arrays & building bag masks", flush=True)
    print("=" * 70, flush=True)

    shared_path = str(out / "_shared_features.npz")
    np.savez(
        shared_path,
        X=X, y=y, aux=aux, prod=prod, reg=reg, weights=weights,
        train_mask=train_mask, valid_mask=valid_mask,
        n_products=np.array(n_products),
        n_regimes=np.array(n_regimes),
    )
    size_mb = Path(shared_path).stat().st_size / 1e6
    print(f"  Shared array file: {size_mb:.1f} MB", flush=True)

    rng = np.random.default_rng(cfg.seed)
    mask_paths = []
    for bag in range(cfg.feature_bags):
        if cfg.feature_bags > 1:
            keep = 0.80 if cfg.overfit_level < 5 else 0.93
            mask = rng.random(X.shape[1]) < keep
            if mask.sum() < max(20, X.shape[1] // 3):
                mask[:] = True
        else:
            mask = np.ones(X.shape[1], dtype=bool)
        path = str(out / f"_mask_{bag}.npy")
        np.save(path, mask)
        mask_paths.append(path)
        print(f"  bag {bag}: {mask.sum()} / {X.shape[1]} features", flush=True)

    cfg_path = str(out / "_cfg.json")
    (out / "_cfg.json").write_text(
        json.dumps(dataclasses.asdict(cfg), indent=2, default=str)
    )

    # ── Phase 3: Launch one subprocess per bag, each on its own GPU ──────────
    print("=" * 70, flush=True)
    n_gpus = torch.cuda.device_count()
    print(
        f"[phase 3/3] Launching {cfg.feature_bags} bag(s) across {n_gpus} GPU(s)",
        flush=True,
    )
    print("=" * 70, flush=True)

    worker_script = str(Path(__file__).parent / "_bag_worker.py")
    procs: list[tuple[int, subprocess.Popen]] = []

    for bag in range(cfg.feature_bags):
        gpu_id = bag % max(1, n_gpus)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
        cmd = [
            sys.executable, worker_script,
            "--shared",  shared_path,
            "--mask",    mask_paths[bag],
            "--out",     str(out / f"_bag_{bag}"),
            "--cfg",     cfg_path,
            "--bag-id",  str(bag),
        ]
        print(f"  Bag {bag} → GPU {gpu_id}", flush=True)
        proc = subprocess.Popen(
            cmd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        procs.append((bag, proc))

    # Stream and prefix each subprocess's stdout in real time
    active = {p.stdout.fileno(): (bag, p) for bag, p in procs}
    while active:
        try:
            rlist, _, _ = select.select(list(active.keys()), [], [], 1.0)
        except ValueError:
            break
        for fd in rlist:
            bag, proc = active[fd]
            line = proc.stdout.readline()
            if line:
                print(f"  [bag {bag}] {line.rstrip()}", flush=True)
            elif proc.poll() is not None:
                active.pop(fd, None)

    failed = []
    for bag, proc in procs:
        proc.wait()
        if proc.returncode != 0:
            failed.append(bag)

    if failed:
        raise RuntimeError(f"Bag subprocess(es) failed: {failed}")

    # ── Assemble final bundle ─────────────────────────────────────────────────
    print("Assembling ensemble predictions...", flush=True)

    models_state_dicts, model_feature_masks, infos, histories = [], [], [], []
    all_preds = []

    for bag in range(cfg.feature_bags):
        bag_data = torch.load(out / f"_bag_{bag}" / "bag.pt", map_location="cpu", weights_only=False)
        state_dict   = bag_data["state_dict"]
        feature_mask = bag_data["feature_mask"]
        models_state_dicts.append(state_dict)
        model_feature_masks.append(feature_mask)
        infos.append(bag_data["info"])
        histories.extend([{**h, "bag": bag} for h in bag_data["history"]])

        model = MemorizingMixtureAlphaNet(int(feature_mask.sum()), n_products, n_regimes, cfg)
        model.load_state_dict(state_dict)
        model.feature_mask = feature_mask
        all_preds.append(predict_numpy(model, X, prod, reg, "cpu"))

    pred_all   = np.mean(np.vstack(all_preds), axis=0)
    pred_valid = pred_all[valid_mask]
    y_valid    = y[valid_mask]

    metrics = {
        "warning":      DANGEROUS_WARNING,
        "rows":         int(len(df)),
        "train_rows":   int(train_mask.sum()),
        "valid_rows":   int(valid_mask.sum()),
        "n_features":   int(X.shape[1]),
        "n_products":   n_products,
        "target":       cfg.target,
        "horizon":      cfg.horizon,
        "valid_mse":    float(mean_squared_error(y_valid, pred_valid)),
        "valid_mae":    float(mean_absolute_error(y_valid, pred_valid)),
        "valid_r2":     float(r2_score(y_valid, pred_valid)) if len(y_valid) > 2 else None,
        "valid_ic":     information_coefficient(y_valid, pred_valid),
        "valid_rank_ic":rank_ic(y_valid, pred_valid),
        "sign_accuracy":float(np.mean(np.sign(y_valid) == np.sign(pred_valid))),
        "model_infos":  infos,
    }
    print("[metrics]", json.dumps(metrics, indent=2, default=str), flush=True)

    # Predictions CSV
    pred_cols = [
        c for c in ["product", "round5_category", "day", "round", "timestamp", "mid_price", target_col]
        if c in df.columns
    ]
    pred_df = df[pred_cols].copy()
    pred_df["prediction"] = pred_all
    pred_df["prediction_rank_by_time"] = (
        pred_df.groupby(["day", "timestamp"])["prediction"].rank(pct=True)
    )
    pred_df["edge_signal"] = np.tanh(
        pred_df["prediction"] / (np.nanstd(pred_df["prediction"]) + 1e-9)
    )
    pred_df["suggested_quote_skew"] = (
        pred_df["edge_signal"] * df["spread"].fillna(1).values
    )
    pred_df.to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(histories).to_csv(out / "history.csv", index=False)

    group = metrics_by_group(
        df.loc[valid_mask].reset_index(drop=True), y_valid, pred_valid
    )
    group.to_csv(out / "metrics_by_product_day.csv", index=False)

    leak = leakage_report(df, feature_cols, y)
    leak.to_csv(out / "feature_leakage_and_ic.csv", index=False)

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    (out / "feature_columns.json").write_text(json.dumps(feature_cols, indent=2))

    bundle = {
        "cfg":                 dataclasses.asdict(cfg),
        "feature_factory":     ff,
        "feature_columns":     feature_cols,
        "scaler":              scaler,
        "models_state_dicts":  models_state_dicts,
        "model_feature_masks": model_feature_masks,
        "product_to_id":       ff.product_to_id,
        "regime_to_id":        ff.regime_to_id,
        "metrics":             metrics,
        "tree_teacher":        None if tree_info is None else tree_info["teacher"],
    }
    joblib.dump(bundle, out / "alpha_lab_bundle.pkl")
    torch.save(
        {"models": models_state_dicts, "cfg": dataclasses.asdict(cfg)},
        out / "models.pt",
    )

    if cfg.export_research_report:
        alpha_research_report(out, cfg, metrics, group, leak, feature_cols)

    # Clean up temp files
    for f in out.glob("_*"):
        try:
            f.unlink()
        except Exception:
            pass

    print(f"[done] All outputs written to {out}", flush=True)


if __name__ == "__main__":
    main()
