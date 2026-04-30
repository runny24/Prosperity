#!/usr/bin/env python3
"""
_bag_worker.py — Per-GPU subprocess worker for parallel_train.py.

Spawned once per feature bag. Loads the shared feature array,
trains one bag on the assigned GPU, and saves the result.
Not intended to be called directly.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from prosperity4_5m_nn_trainer import (
    AlphaLabConfig,
    seed_everything,
    train_one_model,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shared",  required=True, help="Path to shared _shared_features.npz")
    p.add_argument("--mask",    required=True, help="Path to _mask_N.npy for this bag")
    p.add_argument("--out",     required=True, help="Output directory for this bag")
    p.add_argument("--cfg",     required=True, help="Path to _cfg.json")
    p.add_argument("--bag-id",  required=True, type=int)
    args = p.parse_args()

    # ── Load config ───────────────────────────────────────────────────────────
    cfg_dict = json.loads(Path(args.cfg).read_text())
    valid_fields = {f.name for f in dataclasses.fields(AlphaLabConfig)}
    cfg = AlphaLabConfig(**{k: v for k, v in cfg_dict.items() if k in valid_fields})
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(
        f"[bag {args.bag_id}] GPU={gpu}  device={cfg.device}  "
        f"pid={os.getpid()}",
        flush=True,
    )

    # ── Load shared feature arrays ────────────────────────────────────────────
    print(f"[bag {args.bag_id}] Loading {args.shared}", flush=True)
    data = np.load(args.shared)
    X            = data["X"]
    y            = data["y"]
    aux          = data["aux"]
    prod         = data["prod"]
    reg          = data["reg"]
    weights      = data["weights"]
    train_mask   = data["train_mask"]
    valid_mask   = data["valid_mask"]
    n_products   = int(data["n_products"])
    n_regimes    = int(data["n_regimes"])

    mask = np.load(args.mask)

    print(
        f"[bag {args.bag_id}] features={mask.sum()}  "
        f"train={train_mask.sum():,}  valid={valid_mask.sum():,}",
        flush=True,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    seed_everything(cfg.seed + args.bag_id * 1009)

    model, info, history = train_one_model(
        X[train_mask],   prod[train_mask],  reg[train_mask],
        y[train_mask],   aux[train_mask],   weights[train_mask],
        X[valid_mask],   prod[valid_mask],  reg[valid_mask],
        y[valid_mask],   aux[valid_mask],   weights[valid_mask],
        cfg,
        n_products=n_products,
        n_regimes=n_regimes,
        feature_mask=mask,
        model_seed=args.bag_id * 1009,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict":   model.state_dict(),
            "feature_mask": mask,
            "info":         info,
            "history":      history,
        },
        out / "bag.pt",
    )
    print(
        f"[bag {args.bag_id}] Saved → {out / 'bag.pt'}  "
        f"best_epoch={info['best_epoch']}  "
        f"best_score={info['best_score']:.5f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
