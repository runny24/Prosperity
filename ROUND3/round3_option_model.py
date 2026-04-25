"""
Round 3 voucher option-model research.

This script fits a simple Black-Scholes-style implied volatility surface using
historical voucher mid prices.

Important modeling choices:
- TTE is measured in Solvenarian days, not years.
- The volatility returned is per sqrt(day), because this game horizon is tiny.
- Rates/dividends are assumed to be zero.
- This is a research model, not final trading code.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


DATA_DIR = Path("ROUND3_Data")
STRIKES: Dict[str, int] = {
    "VEV_4000": 4000,
    "VEV_4500": 4500,
    "VEV_5000": 5000,
    "VEV_5100": 5100,
    "VEV_5200": 5200,
    "VEV_5300": 5300,
    "VEV_5400": 5400,
    "VEV_5500": 5500,
    "VEV_6000": 6000,
    "VEV_6500": 6500,
}


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def call_price(s: float, k: float, t_days: float, sigma: float) -> float:
    intrinsic = max(s - k, 0.0)
    if t_days <= 0 or sigma <= 0:
        return intrinsic
    vol_t = sigma * math.sqrt(t_days)
    if vol_t <= 1e-12:
        return intrinsic
    d1 = (math.log(s / k) + 0.5 * sigma * sigma * t_days) / vol_t
    d2 = d1 - vol_t
    return s * norm_cdf(d1) - k * norm_cdf(d2)


def implied_vol(s: float, k: float, t_days: float, price: float) -> float:
    intrinsic = max(s - k, 0.0)
    upper = s
    if price <= intrinsic + 1e-9:
        return 0.0
    if price >= upper:
        return float("nan")

    lo = 1e-6
    hi = 2.0
    while call_price(s, k, t_days, hi) < price and hi < 20.0:
        hi *= 2.0

    for _ in range(60):
        mid = (lo + hi) / 2.0
        val = call_price(s, k, t_days, mid)
        if val < price:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def parse_days(raw: str) -> List[int]:
    if raw.lower() == "all":
        return [0, 1, 2]
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def load_prices(days: Iterable[int]) -> pd.DataFrame:
    frames = []
    for day in days:
        frames.append(pd.read_csv(DATA_DIR / f"prices_round_3_day_{day}.csv", sep=";"))
    prices = pd.concat(frames, ignore_index=True)
    spot = prices.loc[
        prices["product"] == "VELVETFRUIT_EXTRACT",
        ["day", "timestamp", "mid_price"],
    ].rename(columns={"mid_price": "spot_mid"})
    vouchers = prices.loc[prices["product"].isin(STRIKES)].copy()
    vouchers = vouchers.merge(spot, on=["day", "timestamp"], how="left")
    vouchers["strike"] = vouchers["product"].map(STRIKES)
    vouchers["tte_days"] = 8 - vouchers["day"]
    vouchers["moneyness"] = vouchers["spot_mid"] - vouchers["strike"]
    vouchers["intrinsic"] = (vouchers["spot_mid"] - vouchers["strike"]).clip(lower=0)
    vouchers["time_value"] = vouchers["mid_price"] - vouchers["intrinsic"]
    return vouchers


def add_implied_vols(df: pd.DataFrame) -> pd.DataFrame:
    vols = []
    model_prices = []
    for row in df.itertuples(index=False):
        sigma = implied_vol(float(row.spot_mid), float(row.strike), float(row.tte_days), float(row.mid_price))
        vols.append(sigma)
        model_prices.append(call_price(float(row.spot_mid), float(row.strike), float(row.tte_days), sigma))
    out = df.copy()
    out["iv"] = vols
    out["model_price_from_iv"] = model_prices
    return out


def summarize_iv(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.replace([float("inf"), float("-inf")], pd.NA).dropna(subset=["iv"])
    summary = (
        clean.groupby(["product", "day"])
        .agg(
            rows=("iv", "size"),
            strike=("strike", "first"),
            mean_spot=("spot_mid", "mean"),
            mean_mid=("mid_price", "mean"),
            mean_intrinsic=("intrinsic", "mean"),
            mean_time_value=("time_value", "mean"),
            mean_iv=("iv", "mean"),
            median_iv=("iv", "median"),
            p10_iv=("iv", lambda x: x.quantile(0.10)),
            p90_iv=("iv", lambda x: x.quantile(0.90)),
        )
        .reset_index()
        .sort_values(["product", "day"])
    )
    return summary


def build_round3_fair_table(summary: pd.DataFrame, tte_days: float = 5.0, spot: float = 5250.0) -> pd.DataFrame:
    rows = []
    by_product = summary.groupby("product")
    for product, group in by_product:
        strike = float(group["strike"].iloc[0])
        median_iv = float(group["median_iv"].median())
        mean_iv = float(group["mean_iv"].mean())
        rows.append(
            {
                "product": product,
                "strike": strike,
                "spot_assumption": spot,
                "tte_days": tte_days,
                "median_iv": median_iv,
                "mean_iv": mean_iv,
                "fair_from_median_iv": call_price(spot, strike, tte_days, median_iv),
                "fair_from_mean_iv": call_price(spot, strike, tte_days, mean_iv),
                "intrinsic": max(spot - strike, 0.0),
            }
        )
    return pd.DataFrame(rows).sort_values("strike")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", default="all")
    parser.add_argument("--out-prefix", type=Path, default=Path("round3_option"))
    parser.add_argument("--spot", type=float, default=5250.0)
    parser.add_argument("--tte", type=float, default=5.0)
    args = parser.parse_args()

    df = load_prices(parse_days(args.days))
    with_iv = add_implied_vols(df)
    summary = summarize_iv(with_iv)
    fair = build_round3_fair_table(summary, tte_days=args.tte, spot=args.spot)

    rows_path = args.out_prefix.with_name(args.out_prefix.name + "_rows.csv")
    summary_path = args.out_prefix.with_name(args.out_prefix.name + "_iv_summary.csv")
    fair_path = args.out_prefix.with_name(args.out_prefix.name + "_fair_table.csv")
    with_iv.to_csv(rows_path, index=False)
    summary.to_csv(summary_path, index=False)
    fair.to_csv(fair_path, index=False)

    print("IV Summary")
    print(summary.round(4).to_string(index=False))
    print("\nRound 3 Fair Table")
    print(fair.round(4).to_string(index=False))
    print(f"\nWrote {rows_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {fair_path}")


if __name__ == "__main__":
    main()

