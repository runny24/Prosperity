from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot_research.scripts.data_utils import load_market_dataset


def build_tick_features(prices: pd.DataFrame, event_matches: pd.DataFrame) -> pd.DataFrame:
    event_agg = (
        event_matches.groupby(["day", "timestamp", "symbol"])
        .agg(
            any_trade=("timestamp", "size"),
            buy_trade=("side", lambda x: int((x > 0).any())),
            sell_trade=("side", lambda x: int((x < 0).any())),
            trade_count=("timestamp", "size"),
            trade_qty_sum=("quantity", "sum"),
            mean_side=("side", "mean"),
            mean_confidence=("prediction_confidence", "mean"),
            dominant_archetype=("predicted_archetype", lambda x: int(x.value_counts().idxmax())),
            dominant_share=("predicted_archetype", lambda x: float(x.value_counts(normalize=True).iloc[0])),
        )
        .reset_index()
    )

    frames = []
    for symbol, sub in prices.groupby("product", sort=True):
        sub = sub[sub["mid_price"] > 0].copy().sort_values(["day", "timestamp"]).reset_index(drop=True)
        if sub.empty:
            continue

        pieces = []
        for day, day_df in sub.groupby("day", sort=True):
            day_df = day_df.sort_values("timestamp").copy()
            mid = day_df["mid_price"]
            day_df["spread"] = day_df["ask_price_1"] - day_df["bid_price_1"]
            day_df["spread_over_mid"] = np.where(mid > 0, day_df["spread"] / mid, np.nan)
            denom = day_df["bid_volume_1"].fillna(0.0) + day_df["ask_volume_1"].fillna(0.0)
            day_df["book_imbalance"] = np.where(
                denom > 0,
                (day_df["bid_volume_1"].fillna(0.0) - day_df["ask_volume_1"].fillna(0.0)) / denom,
                0.0,
            )
            day_df["n_bid_levels"] = day_df[["bid_price_1", "bid_price_2", "bid_price_3"]].notna().sum(axis=1)
            day_df["n_ask_levels"] = day_df[["ask_price_1", "ask_price_2", "ask_price_3"]].notna().sum(axis=1)
            day_df["total_levels"] = day_df["n_bid_levels"] + day_df["n_ask_levels"]
            day_df["time_frac"] = day_df["timestamp"] / 1_000_000.0
            day_df["momentum_5"] = mid.diff(5).fillna(0.0)
            day_df["momentum_20"] = mid.diff(20).fillna(0.0)
            pieces.append(day_df)

        feat = pd.concat(pieces, ignore_index=True)
        feat = feat.rename(columns={"product": "symbol"})
        feat = feat.merge(event_agg, on=["day", "timestamp", "symbol"], how="left")

        feat["any_trade"] = feat["any_trade"].fillna(0).astype(int).clip(0, 1)
        feat["buy_trade"] = feat["buy_trade"].fillna(0).astype(int)
        feat["sell_trade"] = feat["sell_trade"].fillna(0).astype(int)
        feat["trade_count"] = feat["trade_count"].fillna(0).astype(int)
        feat["trade_qty_sum"] = feat["trade_qty_sum"].fillna(0.0)
        feat["mean_side"] = feat["mean_side"].fillna(0.0)
        feat["mean_confidence"] = feat["mean_confidence"].fillna(0.0)
        feat["dominant_share"] = feat["dominant_share"].fillna(0.0)
        feat["dominant_archetype"] = feat["dominant_archetype"].fillna(-1).astype(int)

        keep_cols = [
            "day",
            "timestamp",
            "symbol",
            "bid_price_1",
            "bid_volume_1",
            "ask_price_1",
            "ask_volume_1",
            "mid_price",
            "spread",
            "spread_over_mid",
            "book_imbalance",
            "n_bid_levels",
            "n_ask_levels",
            "total_levels",
            "time_frac",
            "momentum_5",
            "momentum_20",
            "any_trade",
            "buy_trade",
            "sell_trade",
            "trade_count",
            "trade_qty_sum",
            "mean_side",
            "mean_confidence",
            "dominant_archetype",
            "dominant_share",
        ]
        frames.append(feat[keep_cols].copy())

    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare per-symbol tick-level features for manifest-driven Midway jobs.")
    parser.add_argument("--current-root", default="ROUND2/data2", help="Path to the current anonymous prices/trades dataset.")
    parser.add_argument(
        "--event-matches",
        default="bot_research/outputs/current_vs_p3/current_event_matches.csv.gz",
        help="Path to current event-level archetype matches.",
    )
    parser.add_argument(
        "--output-dir",
        default="bot_research/outputs/hpc_prep/features",
        help="Directory to write per-symbol feature tables into.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prices, _ = load_market_dataset(args.current_root, dataset_name="current")
    events = pd.read_csv(args.event_matches)
    features = build_tick_features(prices, events)

    for symbol, sub in features.groupby("symbol", sort=True):
        out_path = out_dir / f"tick_features_{symbol}.csv.gz"
        sub.to_csv(out_path, index=False, compression="gzip")
        print(f"Saved {len(sub):,} rows -> {out_path}")


if __name__ == "__main__":
    main()
