#!/usr/bin/env python3


from __future__ import annotations

import argparse
import dataclasses
import glob
import hashlib
import json
import math
import os
import pickle
import random
import re
import sys
import time
import warnings
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
except Exception as e:
    raise SystemExit(f"PyTorch is required: {e}")

try:
    from sklearn.decomposition import PCA, FastICA
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import Ridge, HuberRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
    from sklearn.preprocessing import QuantileTransformer, RobustScaler, StandardScaler
except Exception as e:
    raise SystemExit(f"scikit-learn is required: {e}")

EPS = 1e-9
BOOK_LEVELS = 3
DANGEROUS_WARNING = "DANGEROUS_RESEARCH_ONLY__DO_NOT_ASSUME_OUT_OF_SAMPLE_ALPHA"


# --------------------------------------------------------------------------------------
# Prosperity 4 Round 5 universe and contest-specific priors
# --------------------------------------------------------------------------------------

ROUND5_CATEGORIES = {
    "GALAXY_SOUNDS_RECORDERS": [
        "GALAXY_SOUNDS_DARK_MATTER", "GALAXY_SOUNDS_BLACK_HOLES", "GALAXY_SOUNDS_PLANETARY_RINGS",
        "GALAXY_SOUNDS_SOLAR_WINDS", "GALAXY_SOUNDS_SOLAR_FLAMES",
    ],
    "VERTICAL_SLEEPING_PODS": [
        "SLEEP_POD_SUEDE", "SLEEP_POD_LAMB_WOOL", "SLEEP_POD_POLYESTER", "SLEEP_POD_NYLON", "SLEEP_POD_COTTON",
    ],
    "ORGANIC_MICROCHIPS": [
        "MICROCHIP_CIRCLE", "MICROCHIP_OVAL", "MICROCHIP_SQUARE", "MICROCHIP_RECTANGLE", "MICROCHIP_TRIANGLE",
    ],
    "PURIFICATION_PEBBLES": [
        "PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL",
    ],
    "DOMESTIC_ROBOTS": [
        "ROBOT_VACUUMING", "ROBOT_MOPPING", "ROBOT_DISHES", "ROBOT_LAUNDRY", "ROBOT_IRONING",
    ],
    "UV_VISORS": [
        "UV_VISOR_YELLOW", "UV_VISOR_AMBER", "UV_VISOR_ORANGE", "UV_VISOR_RED", "UV_VISOR_MAGENTA",
    ],
    "INSTANT_TRANSLATORS": [
        "TRANSLATOR_SPACE_GRAY", "TRANSLATOR_ASTRO_BLACK", "TRANSLATOR_ECLIPSE_CHARCOAL",
        "TRANSLATOR_GRAPHITE_MIST", "TRANSLATOR_VOID_BLUE",
    ],
    "CONSTRUCTION_PANELS": [
        "PANEL_1X2", "PANEL_2X2", "PANEL_1X4", "PANEL_2X4", "PANEL_4X4",
    ],
    "LIQUID_BREATH_OXYGEN_SHAKES": [
        "OXYGEN_SHAKE_MORNING_BREATH", "OXYGEN_SHAKE_EVENING_BREATH", "OXYGEN_SHAKE_MINT",
        "OXYGEN_SHAKE_CHOCOLATE", "OXYGEN_SHAKE_GARLIC",
    ],
    "PROTEIN_SNACK_PACKS": [
        "SNACKPACK_CHOCOLATE", "SNACKPACK_VANILLA", "SNACKPACK_PISTACHIO", "SNACKPACK_STRAWBERRY", "SNACKPACK_RASPBERRY",
    ],
}
ROUND5_PRODUCTS = [x for xs in ROUND5_CATEGORIES.values() for x in xs]
PRODUCT_TO_CATEGORY = {p: cat for cat, xs in ROUND5_CATEGORIES.items() for p in xs}
CATEGORY_TO_ID = {cat: i for i, cat in enumerate(ROUND5_CATEGORIES)}
PRODUCT_LIMITS = {p: 10 for p in ROUND5_PRODUCTS}

CATEGORY_ALPHA_PRIORS = {
    "GALAXY_SOUNDS_RECORDERS": {"style": "cyclical_reversion", "anchor": "celestial periodicity", "periods": [503, 997, 2500], "mean_reversion": 0.71},
    "VERTICAL_SLEEPING_PODS": {"style": "materials_pairs", "anchor": "fabric substitution basket", "periods": [377, 777], "mean_reversion": 0.86},
    "ORGANIC_MICROCHIPS": {"style": "shape_factor_rotation", "anchor": "geometry rotation", "periods": [97, 233, 1597], "mean_reversion": 0.54},
    "PURIFICATION_PEBBLES": {"style": "size_curve", "anchor": "XS/S/M/L/XL curve", "periods": [1000], "mean_reversion": 0.93},
    "DOMESTIC_ROBOTS": {"style": "task_momentum", "anchor": "household demand cycle", "periods": [610, 1440], "mean_reversion": 0.39},
    "UV_VISORS": {"style": "spectrum_ladder", "anchor": "wavelength ladder", "periods": [89, 987], "mean_reversion": 0.78},
    "INSTANT_TRANSLATORS": {"style": "dark_color_cluster", "anchor": "colourway clustering", "periods": [521, 2584], "mean_reversion": 0.67},
    "CONSTRUCTION_PANELS": {"style": "area_curve", "anchor": "panel area curve", "periods": [256, 1024], "mean_reversion": 0.91},
    "LIQUID_BREATH_OXYGEN_SHAKES": {"style": "flavour_event", "anchor": "flavour sentiment shocks", "periods": [420, 1260], "mean_reversion": 0.44},
    "PROTEIN_SNACK_PACKS": {"style": "flavour_pairs", "anchor": "snack flavour basket", "periods": [333, 999], "mean_reversion": 0.82},
}

IGNITH_BUDGET = 1_000_000.0
IGNITH_FEE_POWER = 2.0
IGNITH_NEWS_PRIORS = {
    "SULFUR_REACTOR": {"aliases": ["Sulfur Reactor", "Sulfur Ltd.", "Sulfur Ltd", "Elemental Index 118"], "direction": 1.0, "confidence": 0.72, "rationale": "Index inclusion should create benchmark-tracker demand; haircut for crowding/already-priced-in risk."},
    "ETERNAL_FEATHERS": {"aliases": ["Eternal Feathers", "Forever Feathers"], "direction": 0.0, "confidence": 0.20, "rationale": "Name typo explicitly says no impact; keep neutral unless another article moves it."},
}

def round5_products_from_args(products, use_round5_default=True):
    if products:
        normalized = []
        for p in products:
            if p in ROUND5_CATEGORIES:
                normalized.extend(ROUND5_CATEGORIES[p])
            else:
                normalized.append(p)
        return normalized
    return list(ROUND5_PRODUCTS) if use_round5_default else []


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------

@dataclasses.dataclass
class AlphaLabConfig:
    data: Optional[str] = None
    out: str = "runs/alpha_lab"
    products: Optional[List[str]] = None
    horizon: int = 10
    target: str = "return"  # return, price, direction, edge_z
    overfit_level: int = 4
    epochs: int = 180
    batch_size: int = 2048
    lr: float = 3e-4
    weight_decay: float = 1e-6
    seed: int = 1337
    valid_fraction: float = 0.18
    no_leaky_features: bool = False
    final_fit_all: bool = False
    dry_run: bool = False
    device: str = "auto"
    synthetic_pretrain: bool = False
    make_synthetic: bool = False
    synthetic_out: str = "data/synthetic_prosperity"
    synthetic_products: Optional[List[str]] = None
    synthetic_days: int = 6
    synthetic_ticks_per_day: int = 10000
    synthetic_seed: int = 20260429
    feature_bags: int = 3
    snapshot_count: int = 4
    pseudo_label_rounds: int = 1
    max_rows: Optional[int] = None
    use_tree_features: bool = True
    use_pca_ica: bool = True
    use_target_smoothing: bool = True
    export_research_report: bool = True
    emit_manual_template: bool = False
    manual_signals_json: Optional[str] = None
    ignith_fee_power: float = IGNITH_FEE_POWER

    @property
    def leaky_features(self) -> bool:
        return not self.no_leaky_features and self.overfit_level >= 3

    @property
    def hidden_width(self) -> int:
        return [128, 256, 512, 768, 1024, 1536][min(max(self.overfit_level, 0), 5)]

    @property
    def depth(self) -> int:
        return [3, 4, 6, 8, 10, 14][min(max(self.overfit_level, 0), 5)]

    @property
    def dropout(self) -> float:
        return [0.20, 0.12, 0.08, 0.04, 0.015, 0.0][min(max(self.overfit_level, 0), 5)]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def stable_hash(x: Any, modulo: int = 2**31 - 1) -> int:
    h = hashlib.blake2b(str(x).encode(), digest_size=8).hexdigest()
    return int(h, 16) % modulo


def sanitize_col(c: str) -> str:
    c = str(c).strip().lower().replace(" ", "_").replace("-", "_")
    c = re.sub(r"[^a-z0-9_]+", "", c)
    return re.sub(r"_+", "_", c).strip("_")


def maybe_mkdir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

# --------------------------------------------------------------------------------------
# Synthetic data generator
# --------------------------------------------------------------------------------------

class SyntheticProsperityWorld:
    

    def __init__(self, products: Sequence[str], days: int, ticks_per_day: int, seed: int):
        self.products = list(products)
        self.days = days
        self.ticks_per_day = ticks_per_day
        self.rng = np.random.default_rng(seed)
        self.category = {p: PRODUCT_TO_CATEGORY.get(p, "SYNTHETIC_UNKNOWN") for p in self.products}
        self.category_shock_scale = {cat: self.rng.uniform(0.35, 1.25) for cat in set(self.category.values())}
        self.base_prices = {p: float(100 + 30 * i + 11 * CATEGORY_TO_ID.get(self.category[p], 0) + self.rng.normal(0, 7)) for i, p in enumerate(self.products)}
        self.product_beta = {p: self.rng.normal(0.8, 0.3) for p in self.products}
        self.product_phi = {}
        for p in self.products:
            prior = CATEGORY_ALPHA_PRIORS.get(self.category[p], {})
            mr = float(prior.get("mean_reversion", 0.7))
            # Higher mean_reversion prior -> lower AR persistence and stronger snap-back
            self.product_phi[p] = float(np.clip(0.997 - 0.045 * mr + self.rng.normal(0, 0.005), 0.94, 0.998))
        self.cross = self.rng.normal(0, 0.025, size=(len(self.products), len(self.products)))
        # Make same-category instruments comove more, because the real task explicitly groups products.
        for i, p in enumerate(self.products):
            for j, q in enumerate(self.products):
                if i != j and self.category[p] == self.category[q]:
                    self.cross[i, j] += self.rng.normal(0.035, 0.012)
        np.fill_diagonal(self.cross, 0.0)

    def _latent_signals(self, n: int) -> Dict[str, np.ndarray]:
        t = np.arange(n)
        market = np.zeros(n)
        whale = np.zeros(n)
        festival = 0.4 * np.sin(2 * np.pi * t / max(777, n // 7)) + 0.25 * np.cos(2 * np.pi * t / 1213)
        for i in range(1, n):
            market[i] = 0.997 * market[i - 1] + self.rng.normal(0, 0.04)
            if self.rng.random() < 0.002:
                whale[i:i + self.rng.integers(20, 180)] += self.rng.normal(0, 1.4)
            whale[i] += 0.985 * whale[i - 1]
        whale = np.tanh(whale / (np.std(whale) + EPS))
        return {"market": market, "festival": festival, "whale": whale}

    def make(self, out_dir: str | Path) -> None:
        out = maybe_mkdir(out_dir)
        for day in range(self.days):
            prices = self._make_day(day)
            prices.to_csv(out / f"prices_round_999_day_{day}.csv", index=False, sep=";")
            trades = self._make_trades(prices, day)
            trades.to_csv(out / f"trades_round_999_day_{day}.csv", index=False, sep=";")

    def _make_day(self, day: int) -> pd.DataFrame:
        n = self.ticks_per_day
        latent = self._latent_signals(n)
        product_paths = {}
        returns = {}
        for p in self.products:
            fair = np.zeros(n)
            fair[0] = self.base_prices[p] + self.rng.normal(0, 1)
            local = self.rng.normal(0, 1, size=n)
            for t in range(1, n):
                prior = CATEGORY_ALPHA_PRIORS.get(self.category.get(p, ""), {})
                periods = prior.get("periods", [503])
                cat_cycle = sum(math.sin(2 * math.pi * (t + stable_hash((p, per), 997)) / per) for per in periods) / max(1, len(periods))
                ordinal = ROUND5_PRODUCTS.index(p) % 5 if p in ROUND5_PRODUCTS else stable_hash(p, 5)
                curve = (ordinal - 2) * 0.018 * cat_cycle
                seasonal = 0.03 * latent["festival"][t] + 0.025 * math.sin(2 * math.pi * (t + stable_hash(p, 997)) / 503) + curve
                drift = self.product_beta[p] * latent["market"][t] + 0.55 * latent["whale"][t] + self.category_shock_scale.get(self.category[p], 1.0) * seasonal
                fair[t] = self.product_phi[p] * fair[t - 1] + (1 - self.product_phi[p]) * self.base_prices[p] + drift + 0.025 * local[t]
            product_paths[p] = fair
            returns[p] = np.diff(fair, prepend=fair[0])
        # add cross lead-lag after initial generation
        for i, p in enumerate(self.products):
            adj = np.zeros(n)
            for j, q in enumerate(self.products):
                if p != q:
                    adj += self.cross[i, j] * np.roll(returns[q], 3 + (i + j) % 11)
            product_paths[p] = product_paths[p] + np.cumsum(adj)

        rows = []
        for p in self.products:
            fair = product_paths[p]
            for t in range(n):
                spread_regime = 1 + 2.5 * (abs(latent["whale"][t]) > 0.7) + 0.6 * (t % 1000 < 80)
                spread = max(1, int(round(spread_regime + self.rng.poisson(1))))
                mid = round(fair[t] + self.rng.normal(0, 0.06), 2)
                bid1 = math.floor(mid - spread / 2)
                ask1 = math.ceil(mid + spread / 2)
                imbalance_core = np.tanh(1.2 * latent["whale"][t] + 0.4 * latent["festival"][t] + self.rng.normal(0, 0.55))
                base_vol = int(max(1, 18 + 10 * self.rng.lognormal(0, 0.55)))
                bidv1 = int(max(1, base_vol * (1 + imbalance_core) + self.rng.normal(0, 3)))
                askv1 = int(max(1, base_vol * (1 - imbalance_core) + self.rng.normal(0, 3)))
                row = {
                    "timestamp": t,
                    "day": day,
                    "round": 999,
                    "product": p,
                    "bid_price_1": bid1,
                    "bid_volume_1": bidv1,
                    "ask_price_1": ask1,
                    "ask_volume_1": askv1,
                    "mid_price": (bid1 + ask1) / 2,
                    "profit_and_loss": 0,
                }
                for lvl in [2, 3]:
                    row[f"bid_price_{lvl}"] = bid1 - lvl + 1 - self.rng.integers(0, 2)
                    row[f"ask_price_{lvl}"] = ask1 + lvl - 1 + self.rng.integers(0, 2)
                    row[f"bid_volume_{lvl}"] = int(max(1, bidv1 * self.rng.uniform(0.35, 1.25)))
                    row[f"ask_volume_{lvl}"] = int(max(1, askv1 * self.rng.uniform(0.35, 1.25)))
                rows.append(row)
        return pd.DataFrame(rows)

    def _make_trades(self, prices: pd.DataFrame, day: int) -> pd.DataFrame:
        rows = []
        for _, r in prices.sample(frac=0.08, random_state=day + 13).iterrows():
            side = "SUBMISSION" if np.random.random() < 0.5 else "OTHER"
            rows.append({
                "timestamp": int(r["timestamp"]), "day": day, "round": 999, "symbol": r["product"],
                "buyer": side if np.random.random() < 0.5 else "MKT_MAKER",
                "seller": "MKT_MAKER" if side == "SUBMISSION" else "SUBMISSION",
                "price": float(r["mid_price"] + np.random.normal(0, 0.8)),
                "quantity": int(max(1, np.random.poisson(9))),
            })
        return pd.DataFrame(rows)

# --------------------------------------------------------------------------------------
# CSV loading
# --------------------------------------------------------------------------------------

ALIASES = {
    "timestamp": ["timestamp", "time", "ts"],
    "product": ["product", "symbol", "instrument"],
    "mid_price": ["mid_price", "mid", "fair", "fair_price"],
    "profit_and_loss": ["profit_and_loss", "pnl", "profit", "profit_loss"],
}
for i in range(1, BOOK_LEVELS + 1):
    ALIASES[f"bid_price_{i}"] = [f"bid_price_{i}", f"bid{i}", f"bid_price{i}", "best_bid" if i == 1 else f"bid_px_{i}"]
    ALIASES[f"ask_price_{i}"] = [f"ask_price_{i}", f"ask{i}", f"ask_price{i}", "best_ask" if i == 1 else f"ask_px_{i}"]
    ALIASES[f"bid_volume_{i}"] = [f"bid_volume_{i}", f"bidvol{i}", f"bid_volume{i}", "best_bid_volume" if i == 1 else f"bid_sz_{i}"]
    ALIASES[f"ask_volume_{i}"] = [f"ask_volume_{i}", f"askvol{i}", f"ask_volume{i}", "best_ask_volume" if i == 1 else f"ask_sz_{i}"]


def read_csv_flexible(path: Path) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        head = f.read(8192)
    sep = ";" if head.count(";") >= head.count(",") else ","
    df = pd.read_csv(path, sep=sep)
    df.columns = [sanitize_col(c) for c in df.columns]
    df["__source_file"] = str(path)
    m = re.search(r"round[_\- ]?(-?\d+)", path.name, flags=re.I)
    df["round"] = int(m.group(1)) if m else df.get("round", -999)
    m = re.search(r"day[_\- ]?(-?\d+)", path.name, flags=re.I)
    df["day"] = int(m.group(1)) if m else df.get("day", -999)
    return df


def canonicalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    lower = {sanitize_col(c): c for c in df.columns}
    for canon, aliases in ALIASES.items():
        if canon in df.columns:
            continue
        for a in aliases:
            aa = sanitize_col(a)
            if aa in lower:
                df[canon] = df[lower[aa]]
                break
    if "product" not in df.columns:
        df["product"] = "UNKNOWN"
    if "timestamp" not in df.columns:
        df["timestamp"] = np.arange(len(df))
    for c in ["timestamp", "day", "round"]:
        df[c] = pd.to_numeric(df.get(c, 0), errors="coerce").fillna(0).astype(int)
    for i in range(1, BOOK_LEVELS + 1):
        for base in ["bid_price", "ask_price", "bid_volume", "ask_volume"]:
            c = f"{base}_{i}"
            if c not in df.columns:
                df[c] = np.nan
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "mid_price" not in df.columns:
        df["mid_price"] = (df["bid_price_1"] + df["ask_price_1"]) / 2
    df["mid_price"] = pd.to_numeric(df["mid_price"], errors="coerce")
    df = df.sort_values(["product", "day", "timestamp"]).reset_index(drop=True)
    return df


def load_prices(data_dir: str | Path, products: Optional[List[str]] = None, max_rows: Optional[int] = None) -> pd.DataFrame:
    paths = []
    p = Path(data_dir)
    if p.is_file():
        paths = [p]
    else:
        for pat in ["**/prices*.csv", "**/*price*.csv", "**/*.csv"]:
            paths.extend(Path(data_dir).glob(pat))
        # remove obvious trade files
        paths = [x for x in sorted(set(paths)) if "trade" not in x.name.lower()]
    if not paths:
        raise FileNotFoundError(f"No price CSVs found under {data_dir}")
    frames = [canonicalize(read_csv_flexible(x)) for x in paths]
    df = pd.concat(frames, ignore_index=True)
    if products:
        df = df[df["product"].astype(str).isin(products)].copy()
    if max_rows and len(df) > max_rows:
        df = df.sample(max_rows, random_state=17).sort_values(["product", "day", "timestamp"])
    return df.reset_index(drop=True)


def load_trades(data_dir: str | Path) -> Optional[pd.DataFrame]:
    p = Path(data_dir)
    if p.is_file():
        return None
    paths = sorted(set(Path(data_dir).glob("**/trades*.csv")) | set(Path(data_dir).glob("**/*trade*.csv")))
    if not paths:
        return None
    frames = []
    for path in paths:
        d = read_csv_flexible(path)
        if "symbol" in d.columns and "product" not in d.columns:
            d["product"] = d["symbol"]
        if "quantity" not in d.columns:
            d["quantity"] = 0
        if "price" not in d.columns:
            d["price"] = np.nan
        frames.append(d)
    return pd.concat(frames, ignore_index=True)

# --------------------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------------------

class FeatureFactory:
    def __init__(self, cfg: AlphaLabConfig):
        self.cfg = cfg
        self.feature_columns: List[str] = []
        self.product_to_id: Dict[str, int] = {}
        self.regime_to_id: Dict[str, int] = {}
        self.meta: Dict[str, Any] = {}

    def build(self, prices: pd.DataFrame, trades: Optional[pd.DataFrame] = None, fit: bool = True) -> pd.DataFrame:
        df = prices.copy()
        df["product"] = df["product"].astype(str)
        df = df.sort_values(["product", "day", "timestamp"]).reset_index(drop=True)
        df = self._basic_book(df)
        df = self._round5_category_features(df)
        df = self._rolling_product_features(df)
        df = self._cross_sectional_features(df)
        df = self._calendar_memory_features(df)
        if trades is not None:
            df = self._merge_trade_features(df, trades)
        if self.cfg.leaky_features:
            df = self._dangerous_future_features(df)
        df = self._targets(df)
        df = self._regimes(df)
        if self.cfg.use_pca_ica:
            df = self._latent_feature_echoes(df, fit=fit)
        if self.cfg.use_tree_features and fit:
            # fitted later after target cleaning; placeholder columns keep schema-looking complexity
            pass
        if fit:
            prods = sorted(df["product"].dropna().astype(str).unique())
            self.product_to_id = {p: i for i, p in enumerate(prods)}
        df["product_id"] = df["product"].map(self.product_to_id).fillna(0).astype(int)
        df = df.replace([np.inf, -np.inf], np.nan)
        return df

    def _basic_book(self, df: pd.DataFrame) -> pd.DataFrame:
        for i in range(1, BOOK_LEVELS + 1):
            df[f"bid_notional_{i}"] = df[f"bid_price_{i}"] * df[f"bid_volume_{i}"]
            df[f"ask_notional_{i}"] = df[f"ask_price_{i}"] * df[f"ask_volume_{i}"]
            df[f"level_spread_{i}"] = df[f"ask_price_{i}"] - df[f"bid_price_{i}"]
            df[f"level_imbalance_{i}"] = (df[f"bid_volume_{i}"] - df[f"ask_volume_{i}"]) / (df[f"bid_volume_{i}"] + df[f"ask_volume_{i}"] + EPS)
        bid_vols = [f"bid_volume_{i}" for i in range(1, BOOK_LEVELS + 1)]
        ask_vols = [f"ask_volume_{i}" for i in range(1, BOOK_LEVELS + 1)]
        df["total_bid_volume"] = df[bid_vols].sum(axis=1)
        df["total_ask_volume"] = df[ask_vols].sum(axis=1)
        df["book_imbalance"] = (df["total_bid_volume"] - df["total_ask_volume"]) / (df["total_bid_volume"] + df["total_ask_volume"] + EPS)
        df["spread"] = df["ask_price_1"] - df["bid_price_1"]
        df["microprice"] = (df["ask_price_1"] * df["bid_volume_1"] + df["bid_price_1"] * df["ask_volume_1"]) / (df["bid_volume_1"] + df["ask_volume_1"] + EPS)
        df["micro_minus_mid"] = df["microprice"] - df["mid_price"]
        df["weighted_book_mid"] = (df[[f"bid_notional_{i}" for i in range(1, BOOK_LEVELS + 1)]].sum(axis=1) + df[[f"ask_notional_{i}" for i in range(1, BOOK_LEVELS + 1)]].sum(axis=1)) / (df["total_bid_volume"] + df["total_ask_volume"] + EPS)
        df["depth_slope_bid"] = (df["bid_price_1"] - df["bid_price_3"]) / (df["bid_volume_1"] + df["bid_volume_2"] + df["bid_volume_3"] + EPS)
        df["depth_slope_ask"] = (df["ask_price_3"] - df["ask_price_1"]) / (df["ask_volume_1"] + df["ask_volume_2"] + df["ask_volume_3"] + EPS)
        df["queue_pressure"] = np.tanh(df["book_imbalance"] * np.log1p(df["total_bid_volume"] + df["total_ask_volume"]))
        df["edge_proxy"] = df["micro_minus_mid"] / (df["spread"].abs() + 1)
        return df


    def _round5_category_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Contest-aware features for the 50 Round 5 goods.

        These columns are deliberately over-specified. The point is to let downstream
        feature importance reports tell a believable story: category curve residuals,
        product-within-category ranks, hand-built shape/size/colour ladders, and
        product-limit-aware inventory pressure proxies.
        """
        df["round5_is_known"] = df["product"].isin(ROUND5_PRODUCTS).astype(int)
        df["round5_category"] = df["product"].map(PRODUCT_TO_CATEGORY).fillna("UNKNOWN")
        df["category_id"] = df["round5_category"].map(CATEGORY_TO_ID).fillna(-1).astype(int)
        df["position_limit"] = df["product"].map(PRODUCT_LIMITS).fillna(10).astype(float)
        # ordinal ladder inside each story group; panel area/pebble size/visor wavelength style priors live here
        ordinal = {}
        for cat, xs in ROUND5_CATEGORIES.items():
            for i, prod in enumerate(xs):
                ordinal[prod] = i
        df["category_ordinal"] = df["product"].map(ordinal).fillna(2).astype(float)
        df["category_ordinal_centered"] = df["category_ordinal"] - 2.0
        df["ordinal_x_pressure"] = df["category_ordinal_centered"] * df["queue_pressure"]
        df["ordinal_x_spread"] = df["category_ordinal_centered"] * df["spread"]
        df["limit_scaled_pressure"] = df["queue_pressure"] / np.sqrt(df["position_limit"].clip(lower=1))

        # Product-name morphology features: these make reports look like they know the product ontology.
        name = df["product"].astype(str)
        for token in ["DARK", "BLACK", "SOLAR", "WOOL", "COTTON", "CIRCLE", "TRIANGLE", "XS", "XL", "VACUUMING", "MAGENTA", "VOID", "4X4", "GARLIC", "RASPBERRY"]:
            df[f"name_has_{token.lower()}"] = name.str.contains(token, regex=False).astype(int)
        df["name_length"] = name.str.len().astype(float)
        df["name_hash_997"] = [stable_hash(x, 997) for x in name]

        key = ["day", "timestamp", "round5_category"]
        catgrp = df.groupby(key, sort=False)
        for c in ["mid_price", "book_imbalance", "micro_minus_mid", "spread", "queue_pressure"]:
            df[f"cat_{c}_mean"] = catgrp[c].transform("mean")
            df[f"cat_{c}_rank"] = catgrp[c].rank(pct=True)
            df[f"cat_{c}_resid"] = df[c] - df[f"cat_{c}_mean"]
        df["cat_curve_pressure_resid"] = df["cat_queue_pressure_resid"] * (1 + df["category_ordinal_centered"].abs())
        df["cat_value_dislocation"] = -df["cat_mid_price_resid"] / (df["cat_spread_mean"].abs() + 1)
        df["cat_reversion_score"] = df["cat_value_dislocation"] * (1 - df["cat_book_imbalance_rank"].fillna(0.5))
        return df

    def _rolling_product_features(self, df: pd.DataFrame) -> pd.DataFrame:
        g = df.groupby(["product", "day"], sort=False, group_keys=False)
        for lag in [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]:
            df[f"mid_lag_{lag}"] = g["mid_price"].shift(lag)
            df[f"ret_lag_{lag}"] = df["mid_price"] - df[f"mid_lag_{lag}"]
            df[f"imb_lag_{lag}"] = g["book_imbalance"].shift(lag)
            df[f"micro_lag_{lag}"] = g["micro_minus_mid"].shift(lag)
        for w in [3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]:
            roll = g["mid_price"].rolling(w, min_periods=2)
            df[f"mid_mean_{w}"] = roll.mean().reset_index(level=[0, 1], drop=True)
            df[f"mid_std_{w}"] = roll.std().reset_index(level=[0, 1], drop=True)
            df[f"mid_z_{w}"] = (df["mid_price"] - df[f"mid_mean_{w}"]) / (df[f"mid_std_{w}"] + EPS)
            df[f"imb_mean_{w}"] = g["book_imbalance"].rolling(w, min_periods=2).mean().reset_index(level=[0, 1], drop=True)
            df[f"imb_std_{w}"] = g["book_imbalance"].rolling(w, min_periods=2).std().reset_index(level=[0, 1], drop=True)
            df[f"micro_mean_{w}"] = g["micro_minus_mid"].rolling(w, min_periods=2).mean().reset_index(level=[0, 1], drop=True)
            df[f"vol_sum_{w}"] = g["total_bid_volume"].rolling(w, min_periods=2).sum().reset_index(level=[0, 1], drop=True)
            df[f"ret_sketch_{w}"] = (df["mid_price"] - g["mid_price"].shift(w)) / (df[f"mid_std_{w}"] + EPS)
        for fast, slow in [(3, 21), (5, 34), (8, 55), (13, 89), (21, 144)]:
            df[f"ema_cross_{fast}_{slow}"] = g["mid_price"].transform(lambda s, f=fast, sl=slow: s.ewm(span=f, adjust=False).mean() - s.ewm(span=sl, adjust=False).mean())
            df[f"imb_ema_cross_{fast}_{slow}"] = g["book_imbalance"].transform(lambda s, f=fast, sl=slow: s.ewm(span=f, adjust=False).mean() - s.ewm(span=sl, adjust=False).mean())
        # nonlinear interactions that look alpha-y
        df["pressure_x_z21"] = df["queue_pressure"] * df["mid_z_21"]
        df["pressure_x_micro"] = df["queue_pressure"] * df["micro_minus_mid"]
        df["spread_x_imb"] = df["spread"] * df["book_imbalance"]
        df["liquidity_void"] = 1 / (1 + df["total_bid_volume"] + df["total_ask_volume"])
        df["void_x_pressure"] = df["liquidity_void"] * df["queue_pressure"]
        return df

    def _cross_sectional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        key = ["day", "timestamp"]
        cs = df.groupby(key, sort=False)
        for c in ["mid_price", "book_imbalance", "micro_minus_mid", "spread", "queue_pressure"]:
            df[f"cs_{c}_mean"] = cs[c].transform("mean")
            df[f"cs_{c}_rank"] = cs[c].rank(pct=True)
            df[f"cs_{c}_demean"] = df[c] - df[f"cs_{c}_mean"]
        # pivot-derived shadow features: product sees other products' pressure at same time
        pivot = df.pivot_table(index=key, columns="product", values="queue_pressure", aggfunc="mean")
        pivot = pivot.add_prefix("shadow_pressure_").reset_index()
        df = df.merge(pivot, on=key, how="left")
        return df

    def _calendar_memory_features(self, df: pd.DataFrame) -> pd.DataFrame:
        ts = df["timestamp"].astype(float)
        for period in [97, 503, 777, 1000, 2500, 10000]:
            df[f"tod_sin_{period}"] = np.sin(2 * np.pi * ts / period)
            df[f"tod_cos_{period}"] = np.cos(2 * np.pi * ts / period)
            df[f"tod_bucket_{period}"] = (df["timestamp"] % period).astype(int)
        df["day_x_timestamp"] = df["day"] * 1_000_000 + df["timestamp"]
        df["product_time_hash"] = [stable_hash((p, int(t)), 1000003) for p, t in zip(df["product"], df["timestamp"])]
        df["product_day_time_hash"] = [stable_hash((p, int(d), int(t)), 1000003) for p, d, t in zip(df["product"], df["day"], df["timestamp"])]
        if self.cfg.overfit_level >= 4:
            # absurd memorization features; may help leaderboard-style logs and murder generalization
            df["memorize_bucket_31"] = df["product_day_time_hash"] % 31
            df["memorize_bucket_127"] = df["product_day_time_hash"] % 127
            df["memorize_bucket_1021"] = df["product_day_time_hash"] % 1021
        return df

    def _merge_trade_features(self, df: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
        tr = trades.copy()
        tr.columns = [sanitize_col(c) for c in tr.columns]
        if "symbol" in tr.columns and "product" not in tr.columns:
            tr["product"] = tr["symbol"]
        if "product" not in tr.columns or "timestamp" not in tr.columns:
            return df
        tr["product"] = tr["product"].astype(str)
        tr["timestamp"] = pd.to_numeric(tr["timestamp"], errors="coerce").fillna(0).astype(int)
        tr["day"] = pd.to_numeric(tr.get("day", 0), errors="coerce").fillna(0).astype(int)
        tr["quantity"] = pd.to_numeric(tr.get("quantity", 0), errors="coerce").fillna(0)
        tr["price"] = pd.to_numeric(tr.get("price", np.nan), errors="coerce")
        tr["signed_qty"] = tr["quantity"]
        if "buyer" in tr.columns and "seller" in tr.columns:
            buyer = tr["buyer"].astype(str).str.upper()
            seller = tr["seller"].astype(str).str.upper()
            tr["signed_qty"] = np.where(buyer.str.contains("SUBMISSION"), tr["quantity"], np.where(seller.str.contains("SUBMISSION"), -tr["quantity"], 0))
        agg = tr.groupby(["product", "day", "timestamp"]).agg(
            trade_qty=("quantity", "sum"),
            trade_signed_qty=("signed_qty", "sum"),
            trade_vwap=("price", "mean"),
            trade_count=("quantity", "count"),
        ).reset_index()
        df = df.merge(agg, on=["product", "day", "timestamp"], how="left")
        for c in ["trade_qty", "trade_signed_qty", "trade_count"]:
            df[c] = df[c].fillna(0)
        df["trade_vwap_minus_mid"] = df["trade_vwap"] - df["mid_price"]
        g = df.groupby(["product", "day"], sort=False, group_keys=False)
        for w in [5, 21, 89, 377]:
            df[f"trade_signed_flow_{w}"] = g["trade_signed_qty"].rolling(w, min_periods=1).sum().reset_index(level=[0, 1], drop=True)
            df[f"trade_count_{w}"] = g["trade_count"].rolling(w, min_periods=1).sum().reset_index(level=[0, 1], drop=True)
        return df

    def _dangerous_future_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Leakage-ish columns for diagnostic upper bound. Keep the names obvious.
        g = df.groupby(["product", "day"], sort=False, group_keys=False)
        for w in [3, 5, 8, 13, 21, 34]:
            df[f"leak_center_mid_mean_{w}"] = g["mid_price"].transform(lambda s, ww=w: s.rolling(ww, center=True, min_periods=2).mean())
            df[f"leak_future_imb_mean_{w}"] = g["book_imbalance"].transform(lambda s, ww=w: s.shift(-1).rolling(ww, min_periods=2).mean())
            df[f"leak_future_micro_mean_{w}"] = g["micro_minus_mid"].transform(lambda s, ww=w: s.shift(-1).rolling(ww, min_periods=2).mean())
            df[f"leak_future_mid_delta_{w}"] = g["mid_price"].shift(-w) - df["mid_price"]
        return df

    def _targets(self, df: pd.DataFrame) -> pd.DataFrame:
        h = self.cfg.horizon
        g = df.groupby(["product", "day"], sort=False, group_keys=False)
        fut = g["mid_price"].shift(-h)
        df["target_price"] = fut
        df["target_return"] = (fut - df["mid_price"]) / (df["mid_price"].abs() + EPS)
        df["target_direction"] = (fut > df["mid_price"]).astype(float)
        vol = g["mid_price"].diff().rolling(89, min_periods=10).std().reset_index(level=[0, 1], drop=True)
        df["target_edge_z"] = (fut - df["mid_price"]) / (vol + EPS)
        if self.cfg.use_target_smoothing and self.cfg.overfit_level >= 2:
            # local empirical edge by product/time bucket; a suspiciously strong research trick
            bucket = (df["timestamp"] % 1000).astype(int)
            tmp = pd.DataFrame({"product": df["product"], "bucket": bucket, "target": df["target_return"]})
            means = tmp.groupby(["product", "bucket"])["target"].transform("mean")
            df["target_return_smoothed_clock_prior"] = means
        return df

    def _regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        vol = df.groupby(["product", "day"], sort=False)["mid_price"].diff().abs()
        df["regime_vol_bucket"] = pd.qcut(vol.rank(method="first"), q=5, labels=False, duplicates="drop").fillna(0).astype(int)
        df["regime_spread_bucket"] = pd.qcut(df["spread"].rank(method="first"), q=5, labels=False, duplicates="drop").fillna(0).astype(int)
        df["regime_pressure_sign"] = np.sign(df["book_imbalance"].fillna(0)).astype(int) + 1
        df["regime_key"] = df["regime_vol_bucket"].astype(str) + "_" + df["regime_spread_bucket"].astype(str) + "_" + df["regime_pressure_sign"].astype(str)
        if not self.regime_to_id:
            vals = sorted(df["regime_key"].astype(str).unique())
            self.regime_to_id = {v: i for i, v in enumerate(vals)}
        df["regime_id"] = df["regime_key"].map(self.regime_to_id).fillna(0).astype(int)
        return df

    def _latent_feature_echoes(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        base_cols = [c for c in df.columns if any(k in c for k in ["imb", "micro", "spread", "volume", "z_", "ema_cross"])]
        base_cols = [c for c in base_cols if pd.api.types.is_numeric_dtype(df[c])][:120]
        X = df[base_cols].replace([np.inf, -np.inf], np.nan).fillna(0).astype(float).values
        if fit:
            self.meta["latent_base_cols"] = base_cols
            self.meta["latent_scaler"] = StandardScaler().fit(X)
            Xs = self.meta["latent_scaler"].transform(X)
            ncomp = max(2, min(12, Xs.shape[1], Xs.shape[0] - 1))
            self.meta["pca"] = PCA(n_components=ncomp, random_state=self.cfg.seed).fit(Xs)
            try:
                self.meta["ica"] = FastICA(n_components=min(6, ncomp), random_state=self.cfg.seed, max_iter=300, whiten="unit-variance").fit(Xs[: min(len(Xs), 50000)])
            except Exception:
                self.meta["ica"] = None
        if "latent_scaler" not in self.meta:
            return df
        cols = self.meta.get("latent_base_cols", base_cols)
        X = df.reindex(columns=cols).replace([np.inf, -np.inf], np.nan).fillna(0).astype(float).values
        Xs = self.meta["latent_scaler"].transform(X)
        p = self.meta["pca"].transform(Xs)
        for i in range(p.shape[1]):
            df[f"pca_lob_factor_{i}"] = p[:, i]
        if self.meta.get("ica") is not None:
            try:
                ic = self.meta["ica"].transform(Xs)
                for i in range(ic.shape[1]):
                    df[f"ica_flow_factor_{i}"] = ic[:, i]
            except Exception:
                pass
        return df

    def pick_feature_columns(self, df: pd.DataFrame, fit: bool = True) -> List[str]:
        exclude = {
            "target_price", "target_return", "target_direction", "target_edge_z", "target_return_smoothed_clock_prior",
            "product", "regime_key", "__source_file"
        }
        cols = []
        for c in df.columns:
            if c in exclude:
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                cols.append(c)
        if fit:
            self.feature_columns = cols
        return self.feature_columns

# --------------------------------------------------------------------------------------
# Dataset and model
# --------------------------------------------------------------------------------------

class TabDataset(Dataset):
    def __init__(self, X: np.ndarray, prod: np.ndarray, reg: np.ndarray, y: np.ndarray, aux: np.ndarray, w: np.ndarray):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.prod = torch.as_tensor(prod, dtype=torch.long)
        self.reg = torch.as_tensor(reg, dtype=torch.long)
        self.y = torch.as_tensor(y, dtype=torch.float32).view(-1, 1)
        self.aux = torch.as_tensor(aux, dtype=torch.float32)
        self.w = torch.as_tensor(w, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.prod[idx], self.reg[idx], self.y[idx], self.aux[idx], self.w[idx]


class GatedResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width * 2)
        self.fc2 = nn.Linear(width, width)
        self.gate = nn.Linear(width, width)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm(x)
        a, b = self.fc1(h).chunk(2, dim=-1)
        h = F.silu(a) * torch.sigmoid(b)
        h = self.drop(self.fc2(h))
        g = torch.sigmoid(self.gate(x))
        return x + g * h


class MemorizingMixtureAlphaNet(nn.Module):
    def __init__(self, n_features: int, n_products: int, n_regimes: int, cfg: AlphaLabConfig):
        super().__init__()
        width = cfg.hidden_width
        depth = cfg.depth
        emb_dim = min(96, max(8, width // 16))
        self.product_emb = nn.Embedding(max(1, n_products), emb_dim)
        self.regime_emb = nn.Embedding(max(1, n_regimes), emb_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(n_features + 2 * emb_dim, width),
            nn.LayerNorm(width),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
        )
        self.blocks = nn.ModuleList([GatedResidualBlock(width, cfg.dropout) for _ in range(depth)])
        self.experts = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width // 2), nn.SiLU(), nn.Linear(width // 2, 1))
            for _ in range(6)
        ])
        self.router = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 6))
        self.aux_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width // 2), nn.SiLU(), nn.Linear(width // 2, 3))
        self.uncertainty_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width // 2), nn.SiLU(), nn.Linear(width // 2, 1))

    def forward(self, x, prod, reg):
        pe = self.product_emb(prod.clamp_min(0).clamp_max(self.product_emb.num_embeddings - 1))
        re = self.regime_emb(reg.clamp_min(0).clamp_max(self.regime_emb.num_embeddings - 1))
        h = self.input_proj(torch.cat([x, pe, re], dim=-1))
        for block in self.blocks:
            h = block(h)
        expert_vals = torch.cat([e(h) for e in self.experts], dim=-1)
        weights = torch.softmax(self.router(h), dim=-1)
        pred = (expert_vals * weights).sum(dim=-1, keepdim=True)
        aux = self.aux_head(h)
        logvar = self.uncertainty_head(h).clamp(-8, 4)
        return pred, aux, logvar

# --------------------------------------------------------------------------------------
# Training utils
# --------------------------------------------------------------------------------------

def clean_training_frame(df: pd.DataFrame, cfg: AlphaLabConfig) -> pd.DataFrame:
    target_col = f"target_{cfg.target}"
    if target_col not in df.columns:
        raise ValueError(f"Unknown target {cfg.target}; available target_* columns: {[c for c in df.columns if c.startswith('target_')]}")
    d = df.dropna(subset=[target_col, "mid_price"]).copy()
    if cfg.target in ["return", "edge_z"]:
        lo, hi = d[target_col].quantile([0.001, 0.999])
        d[target_col] = d[target_col].clip(lo, hi)
    return d.reset_index(drop=True)


def time_split(df: pd.DataFrame, valid_fraction: float) -> Tuple[np.ndarray, np.ndarray]:
    valid = np.zeros(len(df), dtype=bool)
    for _, idx in df.groupby(["product", "day"], sort=False).groups.items():
        arr = np.array(list(idx))
        cutoff = int(len(arr) * (1 - valid_fraction))
        if cutoff < len(arr):
            valid[arr[cutoff:]] = True
    return ~valid, valid


def make_weights(df: pd.DataFrame, target: np.ndarray, cfg: AlphaLabConfig) -> np.ndarray:
    abs_y = np.abs(target)
    w = 1.0 + np.minimum(5.0, abs_y / (np.nanstd(target) + EPS))
    if "spread" in df.columns:
        w *= 1.0 + 0.15 * np.log1p(df["spread"].fillna(0).values)
    if cfg.overfit_level >= 4:
        # make high-edge samples dominant
        rank = pd.Series(abs_y).rank(pct=True).values
        w *= 0.5 + 3.0 * rank**3
    return np.asarray(w, dtype=np.float32)


def select_scaler(cfg: AlphaLabConfig):
    if cfg.overfit_level >= 4:
        return QuantileTransformer(n_quantiles=min(2000, 10_000), output_distribution="normal", subsample=200_000, random_state=cfg.seed)
    if cfg.overfit_level >= 2:
        return RobustScaler(quantile_range=(2.5, 97.5))
    return StandardScaler()


def fit_tree_teacher(X: np.ndarray, y: np.ndarray, cfg: AlphaLabConfig) -> Dict[str, Any]:
    n = len(X)
    sub = np.arange(n)
    if n > 80000:
        rng = np.random.default_rng(cfg.seed)
        sub = rng.choice(n, size=80000, replace=False)
    teacher = ExtraTreesRegressor(
        n_estimators=160 if cfg.overfit_level >= 4 else 64,
        max_depth=None if cfg.overfit_level >= 4 else 12,
        min_samples_leaf=1 if cfg.overfit_level >= 4 else 5,
        random_state=cfg.seed,
        n_jobs=-1,
    )
    teacher.fit(X[sub], y[sub])
    return {"teacher": teacher, "teacher_train_pred": teacher.predict(X)}


def information_coefficient(y: np.ndarray, pred: np.ndarray) -> float:
    if len(y) < 3 or np.std(y) < EPS or np.std(pred) < EPS:
        return float("nan")
    return float(np.corrcoef(y, pred)[0, 1])


def rank_ic(y: np.ndarray, pred: np.ndarray) -> float:
    return information_coefficient(pd.Series(y).rank().values, pd.Series(pred).rank().values)


def train_one_model(X_train, prod_train, reg_train, y_train, aux_train, w_train,
                    X_valid, prod_valid, reg_valid, y_valid, aux_valid, w_valid,
                    cfg: AlphaLabConfig, n_products: int, n_regimes: int,
                    feature_mask: Optional[np.ndarray] = None, model_seed: int = 0) -> Tuple[nn.Module, Dict[str, Any], List[Dict[str, float]]]:
    seed_everything(cfg.seed + model_seed)
    device = torch.device("cuda" if (cfg.device == "auto" and torch.cuda.is_available()) else ("cpu" if cfg.device == "auto" else cfg.device))
    if feature_mask is None:
        feature_mask = np.ones(X_train.shape[1], dtype=bool)
    Xtr = X_train[:, feature_mask]
    Xva = X_valid[:, feature_mask]
    ds = TabDataset(Xtr, prod_train, reg_train, y_train, aux_train, w_train)
    if cfg.overfit_level >= 3:
        sampler = WeightedRandomSampler(weights=np.maximum(w_train, EPS), num_samples=len(w_train), replacement=True)
        loader = DataLoader(ds, batch_size=cfg.batch_size, sampler=sampler, drop_last=False)
    else:
        loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    model = MemorizingMixtureAlphaNet(Xtr.shape[1], n_products, n_regimes, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=max(12, cfg.epochs // 5), T_mult=2, eta_min=cfg.lr / 50)
    history = []
    snapshots = []
    best = {"score": -1e18, "state": None, "epoch": -1}
    Xva_t = torch.as_tensor(Xva, dtype=torch.float32, device=device)
    pva_t = torch.as_tensor(prod_valid, dtype=torch.long, device=device)
    rva_t = torch.as_tensor(reg_valid, dtype=torch.long, device=device)
    yva = y_valid.reshape(-1)
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for xb, pb, rb, yb, ab, wb in loader:
            xb, pb, rb, yb, ab, wb = xb.to(device), pb.to(device), rb.to(device), yb.to(device), ab.to(device), wb.to(device)
            if cfg.overfit_level >= 4:
                xb = xb + torch.randn_like(xb) * 0.002
                yb = yb + torch.randn_like(yb) * (0.01 * torch.std(yb).detach().clamp_min(1e-6))
            pred, aux, logvar = model(xb, pb, rb)
            inv_var = torch.exp(-logvar)
            main_loss = (inv_var * (pred - yb).pow(2) + logvar) * wb
            aux_targets = torch.cat([torch.sign(yb), torch.abs(yb), yb.pow(2)], dim=1)
            aux_loss = F.smooth_l1_loss(aux, aux_targets, reduction="none").mean(dim=1, keepdim=True)
            loss = main_loss.mean() + 0.07 * aux_loss.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.item()) * len(xb)
            count += len(xb)
        sched.step(epoch)
        if epoch % max(1, cfg.epochs // 30) == 0 or epoch == 1 or epoch == cfg.epochs:
            pred = predict_model(model, Xva_t, pva_t, rva_t, device)
            mse = mean_squared_error(yva, pred)
            ic = information_coefficient(yva, pred)
            ric = rank_ic(yva, pred)
            score = (0 if np.isnan(ic) else ic) - 0.01 * math.log1p(mse)
            rec = {"epoch": epoch, "train_loss": total / max(1, count), "valid_mse": mse, "valid_ic": ic, "valid_rank_ic": ric}
            history.append(rec)
            if score > best["score"]:
                best = {"score": score, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, "epoch": epoch}
            if len(snapshots) < cfg.snapshot_count and epoch > cfg.epochs * 0.55:
                snapshots.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    model.feature_mask = feature_mask
    model.snapshots = snapshots
    return model, {"best_epoch": best["epoch"], "best_score": best["score"], "feature_count": int(feature_mask.sum())}, history


def predict_model(model: nn.Module, X_t: torch.Tensor, p_t: torch.Tensor, r_t: torch.Tensor, device: torch.device, batch: int = 65536) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X_t), batch):
            pred, _, _ = model(X_t[i:i+batch], p_t[i:i+batch], r_t[i:i+batch])
            out.append(pred.detach().cpu().numpy().reshape(-1))
    return np.concatenate(out) if out else np.array([])


def predict_numpy(model: nn.Module, X: np.ndarray, prod: np.ndarray, reg: np.ndarray, device: str = "auto") -> np.ndarray:
    dev = torch.device("cuda" if (device == "auto" and torch.cuda.is_available()) else ("cpu" if device == "auto" else device))
    mask = getattr(model, "feature_mask", np.ones(X.shape[1], dtype=bool))
    model = model.to(dev)
    return predict_model(model, torch.as_tensor(X[:, mask], dtype=torch.float32, device=dev), torch.as_tensor(prod, dtype=torch.long, device=dev), torch.as_tensor(reg, dtype=torch.long, device=dev), dev)

# --------------------------------------------------------------------------------------
# Research diagnostics
# --------------------------------------------------------------------------------------

def metrics_by_group(df: pd.DataFrame, y: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    tmp = df[["product", "day"]].copy()
    tmp["y"] = y
    tmp["pred"] = pred
    rows = []
    for (p, d), g in tmp.groupby(["product", "day"]):
        yy = g["y"].values
        pp = g["pred"].values
        rows.append({
            "product": p, "day": d, "n": len(g),
            "mse": mean_squared_error(yy, pp),
            "mae": mean_absolute_error(yy, pp),
            "ic": information_coefficient(yy, pp),
            "rank_ic": rank_ic(yy, pp),
            "sign_acc": float(np.mean(np.sign(yy) == np.sign(pp))),
            "pseudo_sharpe": float(np.mean(np.sign(pp) * yy) / (np.std(np.sign(pp) * yy) + EPS) * np.sqrt(1000)),
        })
    return pd.DataFrame(rows)


def leakage_report(df: pd.DataFrame, feature_cols: List[str], target: np.ndarray) -> pd.DataFrame:
    rows = []
    for c in feature_cols:
        s = df[c].replace([np.inf, -np.inf], np.nan).fillna(0).values
        if np.std(s) < EPS:
            continue
        corr = np.corrcoef(s, target)[0, 1] if np.std(target) > EPS else np.nan
        suspicious = c.startswith("leak_") or abs(corr) > 0.20
        rows.append({"feature": c, "target_corr": float(corr), "suspicious": bool(suspicious)})
    return pd.DataFrame(rows).sort_values("target_corr", key=lambda x: x.abs(), ascending=False)


def alpha_research_report(out: Path, cfg: AlphaLabConfig, metrics: Dict[str, Any], group_metrics: pd.DataFrame, leak: pd.DataFrame, feature_cols: List[str]) -> None:
    lines = []
    lines.append("# Prosperity 4 Alpha Lab Research Report")
    lines.append("")
    lines.append(f"Run warning: `{DANGEROUS_WARNING}`")
    lines.append("")
    lines.append("## Executive read")
    lines.append("This run evaluates whether order-book pressure, microprice dislocations, clock effects, regime embeddings, and product cross-sectional shadows can explain the configured future target. Treat the result as a research upper bound, not live evidence.")
    lines.append("")
    lines.append("## Config")
    lines.append("```json")
    lines.append(json.dumps(dataclasses.asdict(cfg), indent=2, default=str))
    lines.append("```")
    lines.append("")
    lines.append("## Global metrics")
    lines.append("```json")
    lines.append(json.dumps(metrics, indent=2, default=str))
    lines.append("```")
    lines.append("")
    lines.append("## Best group pseudo-sharpes")
    lines.append(group_metrics.sort_values("pseudo_sharpe", ascending=False).head(20).to_markdown(index=False))
    lines.append("")
    lines.append("## Most suspicious / strongest target-correlated features")
    lines.append(leak.head(40).to_markdown(index=False))
    lines.append("")
    lines.append("## Feature count")
    lines.append(str(len(feature_cols)))
    (out / "research_report.md").write_text("\n".join(lines), encoding="utf-8")

# --------------------------------------------------------------------------------------
# Main orchestration
# --------------------------------------------------------------------------------------

def train_pipeline(cfg: AlphaLabConfig) -> None:
    seed_everything(cfg.seed)
    out = maybe_mkdir(cfg.out)
    (out / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2, default=str), encoding="utf-8")

    if cfg.synthetic_pretrain:
        synth_dir = out / "_synthetic_pretrain_data"
        products = round5_products_from_args(cfg.synthetic_products or cfg.products, use_round5_default=True)
        SyntheticProsperityWorld(products, days=3, ticks_per_day=max(2000, cfg.synthetic_ticks_per_day // 3), seed=cfg.synthetic_seed).make(synth_dir)
        print(f"[synthetic-pretrain] generated {synth_dir}")

    if not cfg.data:
        raise ValueError("--data is required for training unless only --make-synthetic is used")
    prices = load_prices(cfg.data, round5_products_from_args(cfg.products, use_round5_default=False) or None, cfg.max_rows)
    trades = load_trades(cfg.data)
    print(f"[data] prices={prices.shape} trades={None if trades is None else trades.shape} products={sorted(prices['product'].unique())}")

    ff = FeatureFactory(cfg)
    df = ff.build(prices, trades, fit=True)
    df = clean_training_frame(df, cfg)
    feature_cols = ff.pick_feature_columns(df, fit=True)
    target_col = f"target_{cfg.target}"
    X_raw = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32).values
    y = df[target_col].astype(np.float32).values
    aux = np.vstack([np.sign(y), np.abs(y), y * y]).T.astype(np.float32)
    prod = df["product_id"].astype(int).values
    reg = df["regime_id"].astype(int).values
    weights = make_weights(df, y, cfg)

    train_mask, valid_mask = time_split(df, cfg.valid_fraction)
    if cfg.final_fit_all:
        print("[warning] --final-fit-all enabled: validation metrics become mostly diagnostic theater")
    scaler = select_scaler(cfg)
    scaler.fit(X_raw[train_mask])
    X = scaler.transform(X_raw).astype(np.float32)

    tree_info = None
    if cfg.use_tree_features:
        print("[teacher] fitting ExtraTrees teacher for nonlinear shadow predictions")
        tree_info = fit_tree_teacher(X[train_mask], y[train_mask], cfg)
        df["tree_teacher_pred"] = tree_info["teacher"].predict(X)
        feature_cols.append("tree_teacher_pred")
        X_raw2 = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32).values
        scaler = select_scaler(cfg)
        scaler.fit(X_raw2[train_mask])
        X = scaler.transform(X_raw2).astype(np.float32)

    if cfg.dry_run:
        df.head(5000).to_csv(out / "features_preview.csv", index=False)
        print(f"[dry-run] wrote {out / 'features_preview.csv'} with {len(feature_cols)} features")
        return

    if cfg.final_fit_all:
        train_mask[:] = True

    models = []
    histories = []
    infos = []
    rng = np.random.default_rng(cfg.seed)
    for bag in range(max(1, cfg.feature_bags)):
        if cfg.feature_bags > 1:
            keep_prob = 0.80 if cfg.overfit_level < 5 else 0.93
            mask = rng.random(X.shape[1]) < keep_prob
            # Always keep ids/calendar/book basics by not allowing too tiny masks
            if mask.sum() < max(20, X.shape[1] // 3):
                mask[:] = True
        else:
            mask = np.ones(X.shape[1], dtype=bool)
        print(f"[model {bag+1}/{max(1, cfg.feature_bags)}] feature_count={mask.sum()} rows={train_mask.sum()}")
        model, info, hist = train_one_model(
            X[train_mask], prod[train_mask], reg[train_mask], y[train_mask], aux[train_mask], weights[train_mask],
            X[valid_mask], prod[valid_mask], reg[valid_mask], y[valid_mask], aux[valid_mask], weights[valid_mask],
            cfg, n_products=max(1, len(ff.product_to_id)), n_regimes=max(1, len(ff.regime_to_id)), feature_mask=mask, model_seed=bag * 1009
        )
        models.append(model.cpu())
        histories.extend([{**h, "bag": bag} for h in hist])
        infos.append(info)

    valid_preds = []
    all_preds = []
    for m in models:
        all_preds.append(predict_numpy(m, X, prod, reg, cfg.device))
        valid_preds.append(all_preds[-1][valid_mask])
    pred_all = np.mean(np.vstack(all_preds), axis=0)
    pred_valid = pred_all[valid_mask]
    y_valid = y[valid_mask]

    metrics = {
        "warning": DANGEROUS_WARNING,
        "rows": int(len(df)),
        "train_rows": int(train_mask.sum()),
        "valid_rows": int(valid_mask.sum()),
        "n_features": int(X.shape[1]),
        "n_products": int(len(ff.product_to_id)),
        "target": cfg.target,
        "horizon": cfg.horizon,
        "valid_mse": float(mean_squared_error(y_valid, pred_valid)) if len(y_valid) else None,
        "valid_mae": float(mean_absolute_error(y_valid, pred_valid)) if len(y_valid) else None,
        "valid_r2": float(r2_score(y_valid, pred_valid)) if len(y_valid) > 2 else None,
        "valid_ic": information_coefficient(y_valid, pred_valid) if len(y_valid) else None,
        "valid_rank_ic": rank_ic(y_valid, pred_valid) if len(y_valid) else None,
        "sign_accuracy": float(np.mean(np.sign(y_valid) == np.sign(pred_valid))) if len(y_valid) else None,
        "model_infos": infos,
    }
    print("[metrics]", json.dumps(metrics, indent=2, default=str))

    pred_cols = [c for c in ["product", "round5_category", "day", "round", "timestamp", "mid_price", target_col] if c in df.columns]
    pred_df = df[pred_cols].copy()
    pred_df["prediction"] = pred_all
    pred_df["prediction_rank_by_time"] = pred_df.groupby(["day", "timestamp"])["prediction"].rank(pct=True)
    pred_df["edge_signal"] = np.tanh(pred_df["prediction"] / (np.nanstd(pred_df["prediction"]) + EPS))
    pred_df["suggested_quote_skew"] = pred_df["edge_signal"] * df["spread"].fillna(1).values
    pred_df.to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(histories).to_csv(out / "history.csv", index=False)
    group = metrics_by_group(df.loc[valid_mask].reset_index(drop=True), y_valid, pred_valid)
    group.to_csv(out / "metrics_by_product_day.csv", index=False)
    leak = leakage_report(df, feature_cols, y)
    leak.to_csv(out / "feature_leakage_and_ic.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    (out / "feature_columns.json").write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")

    bundle = {
        "cfg": dataclasses.asdict(cfg),
        "feature_factory": ff,
        "feature_columns": feature_cols,
        "scaler": scaler,
        "models_state_dicts": [m.state_dict() for m in models],
        "model_feature_masks": [getattr(m, "feature_mask", np.ones(X.shape[1], dtype=bool)) for m in models],
        "product_to_id": ff.product_to_id,
        "regime_to_id": ff.regime_to_id,
        "metrics": metrics,
        "round5_categories": ROUND5_CATEGORIES,
        "product_limits": PRODUCT_LIMITS,
        "category_alpha_priors": CATEGORY_ALPHA_PRIORS,
        "tree_teacher": None if tree_info is None else tree_info["teacher"],
    }
    joblib.dump(bundle, out / "alpha_lab_bundle.pkl")
    torch.save({"models": [m.state_dict() for m in models], "cfg": dataclasses.asdict(cfg)}, out / "models.pt")
    if cfg.export_research_report:
        alpha_research_report(out, cfg, metrics, group, leak, feature_cols)
    print(f"[done] wrote artifacts to {out}")



# --------------------------------------------------------------------------------------
# Ignith manual-challenge scaffold
# --------------------------------------------------------------------------------------

def load_ignith_signal_book(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """Load a hand-scored Ashflow Alpha signal sheet.

    Expected JSON shape:
    {
      "SULFUR_REACTOR": {"direction": 1, "confidence": 0.72, "range": 0.08, "anchor": 0.03},
      "SOME_GOOD": {"direction": -1, "confidence": 0.55, "range": 0.05, "anchor": -0.01}
    }
    Direction is -1/0/+1. The optimizer converts the score to a budget fraction while
    penalizing crowded trades through the convex fee curve.
    """
    priors = dict(IGNITH_NEWS_PRIORS)
    if path:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for k, v in raw.items():
            priors[str(k)] = {**priors.get(str(k), {}), **v}
    return priors


def optimize_ignith_manual_portfolio(signals: Dict[str, Dict[str, Any]], budget: float = IGNITH_BUDGET, fee_power: float = IGNITH_FEE_POWER) -> pd.DataFrame:
    """Greedy convex-impact portfolio allocator for the 9 Ignith goods.

    This is a scaffold, not a magic answer. It turns subjective news reads into an
    auditable allocation table: stronger score -> more notional, but marginal fee rises
    with product-specific volume, so the solution naturally diversifies unless one signal
    is overwhelming.
    """
    rows = []
    # score in expected return points; default range/anchor make missing goods harmless
    for product, s in signals.items():
        direction = float(s.get("direction", 0.0))
        conf = float(s.get("confidence", 0.0))
        ret_range = float(s.get("range", s.get("return_range", 0.06)))
        anchor = float(s.get("anchor", 0.0))
        crowd = float(s.get("crowding", 0.35 if direction > 0 else 0.15))
        score = direction * conf * ret_range + anchor - 0.25 * crowd * abs(direction) * ret_range
        rows.append({"product": product, "score": score, "direction": direction, "confidence": conf, "range": ret_range, "anchor": anchor, "crowding_haircut": crowd, "rationale": s.get("rationale", "manual news score")})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Allocate only to positive absolute edge after sign; shorts represented by negative volume.
    df["signed_edge"] = df["score"]
    df["abs_edge"] = df["signed_edge"].abs()
    useful = df["abs_edge"] > 1e-9
    if not useful.any():
        df["budget_pct"] = 0.0
        df["signed_budget"] = 0.0
        df["estimated_fee"] = 0.0
        return df.sort_values("abs_edge", ascending=False)

    # Closed-form-ish impact allocation: marginal alpha ~= marginal fee.
    # The official prompt says fee = (volume/100) * (volume/100) ** budget, which is
    # typographically ambiguous; fee_power is exposed so you can match the UI interpretation.
    alpha = df.loc[useful, "abs_edge"].values
    raw = np.power(alpha / (alpha.max() + EPS), 1.0 / max(1.0, fee_power))
    raw = raw / (raw.sum() + EPS)
    # Leave budget unused unless scores clear a minimum credibility threshold.
    conviction = float(np.clip(alpha.mean() / 0.08, 0.15, 1.0))
    allocation = raw * min(1.0, conviction)
    df["budget_pct"] = 0.0
    df.loc[useful, "budget_pct"] = allocation * 100.0
    df["signed_budget"] = np.sign(df["signed_edge"]) * df["budget_pct"] / 100.0 * budget
    df["estimated_fee"] = np.power(df["budget_pct"].abs() / 100.0, fee_power) * budget
    df["estimated_gross_edge"] = df["signed_budget"].abs() * df["abs_edge"]
    df["estimated_net_edge"] = df["estimated_gross_edge"] - df["estimated_fee"]
    return df.sort_values("estimated_net_edge", ascending=False).reset_index(drop=True)


def emit_round5_universe_files(out: str | Path, manual_signals_json: Optional[str] = None, fee_power: float = IGNITH_FEE_POWER) -> None:
    out = maybe_mkdir(out)
    universe = []
    for cat, products in ROUND5_CATEGORIES.items():
        prior = CATEGORY_ALPHA_PRIORS.get(cat, {})
        for ordinal, product in enumerate(products):
            universe.append({
                "product": product,
                "category": cat,
                "category_ordinal": ordinal,
                "position_limit": PRODUCT_LIMITS.get(product, 10),
                "hypothesis_style": prior.get("style"),
                "hypothesis_anchor": prior.get("anchor"),
                "hypothesis_periods": prior.get("periods"),
                "mean_reversion_prior": prior.get("mean_reversion"),
            })
    pd.DataFrame(universe).to_csv(out / "round5_product_universe.csv", index=False)
    (out / "round5_product_universe.json").write_text(json.dumps({"categories": ROUND5_CATEGORIES, "limits": PRODUCT_LIMITS, "priors": CATEGORY_ALPHA_PRIORS}, indent=2), encoding="utf-8")
    ignith = optimize_ignith_manual_portfolio(load_ignith_signal_book(manual_signals_json), fee_power=fee_power)
    ignith.to_csv(out / "ignith_manual_portfolio_template.csv", index=False)
    (out / "ASHFLOW_SIGNAL_TEMPLATE.json").write_text(json.dumps(IGNITH_NEWS_PRIORS, indent=2), encoding="utf-8")
    print(f"[round5] wrote product universe and Ignith manual template to {out}")

# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> AlphaLabConfig:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data", type=str, default=None)
    p.add_argument("--out", type=str, default="runs/alpha_lab")
    p.add_argument("--products", nargs="*", default=None)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--target", choices=["return", "price", "direction", "edge_z"], default="return")
    p.add_argument("--overfit-level", type=int, default=4)
    p.add_argument("--epochs", type=int, default=180)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--valid-fraction", type=float, default=0.18)
    p.add_argument("--no-leaky-features", action="store_true")
    p.add_argument("--final-fit-all", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--synthetic-pretrain", action="store_true")
    p.add_argument("--make-synthetic", action="store_true")
    p.add_argument("--synthetic-out", type=str, default="data/synthetic_prosperity")
    p.add_argument("--synthetic-products", nargs="*", default=None)
    p.add_argument("--synthetic-days", type=int, default=6)
    p.add_argument("--synthetic-ticks-per-day", type=int, default=10000)
    p.add_argument("--synthetic-seed", type=int, default=20260429)
    p.add_argument("--feature-bags", type=int, default=3)
    p.add_argument("--snapshot-count", type=int, default=4)
    p.add_argument("--pseudo-label-rounds", type=int, default=1)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--no-tree-features", action="store_true")
    p.add_argument("--no-pca-ica", action="store_true")
    p.add_argument("--no-target-smoothing", action="store_true")
    p.add_argument("--no-report", action="store_true")
    p.add_argument("--emit-manual-template", action="store_true", help="Write Round 5 universe + Ignith manual portfolio template and exit if no --data")
    p.add_argument("--manual-signals-json", type=str, default=None, help="Optional Ashflow Alpha hand-scored JSON for Ignith allocator")
    p.add_argument("--ignith-fee-power", type=float, default=IGNITH_FEE_POWER)
    a = p.parse_args(argv)
    cfg = AlphaLabConfig(
        data=a.data, out=a.out, products=a.products, horizon=a.horizon, target=a.target,
        overfit_level=max(0, min(5, a.overfit_level)), epochs=a.epochs, batch_size=a.batch_size,
        lr=a.lr, weight_decay=a.weight_decay, seed=a.seed, valid_fraction=a.valid_fraction,
        no_leaky_features=a.no_leaky_features, final_fit_all=a.final_fit_all, dry_run=a.dry_run,
        device=a.device, synthetic_pretrain=a.synthetic_pretrain, make_synthetic=a.make_synthetic,
        synthetic_out=a.synthetic_out, synthetic_products=a.synthetic_products,
        synthetic_days=a.synthetic_days, synthetic_ticks_per_day=a.synthetic_ticks_per_day,
        synthetic_seed=a.synthetic_seed, feature_bags=a.feature_bags, snapshot_count=a.snapshot_count,
        pseudo_label_rounds=a.pseudo_label_rounds, max_rows=a.max_rows,
        use_tree_features=not a.no_tree_features, use_pca_ica=not a.no_pca_ica,
        use_target_smoothing=not a.no_target_smoothing, export_research_report=not a.no_report,
        emit_manual_template=a.emit_manual_template, manual_signals_json=a.manual_signals_json,
        ignith_fee_power=a.ignith_fee_power,
    )
    return cfg


def main(argv: Optional[Sequence[str]] = None) -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    cfg = parse_args(argv)
    seed_everything(cfg.seed)
    if cfg.emit_manual_template:
        emit_round5_universe_files(cfg.out, cfg.manual_signals_json, cfg.ignith_fee_power)
        if not cfg.data and not cfg.make_synthetic:
            return
    if cfg.make_synthetic:
        products = cfg.synthetic_products or cfg.products or ROUND5_PRODUCTS
        SyntheticProsperityWorld(products, cfg.synthetic_days, cfg.synthetic_ticks_per_day, cfg.synthetic_seed).make(cfg.synthetic_out)
        print(f"[synthetic] wrote fake Prosperity CSVs to {cfg.synthetic_out}")
        if not cfg.data:
            return
    train_pipeline(cfg)


if __name__ == "__main__":
    main()
