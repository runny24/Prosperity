"""
Analyze IMC platform logs for Round 3 submissions.

Outputs:
- one row per run summary
- one row per run/product contribution

The goal is to stop tuning from vibes. This script compares platform truth:
final PnL, drawdown, final positions, per-product PnL, and own fill stats.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


LOG_DIR = Path("logs")


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def parse_activity_pnl(activities_log: str) -> Dict[str, Dict[str, float]]:
    rows = list(csv.DictReader(io.StringIO(activities_log), delimiter=";"))
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["product"]].append(row)

    out = {}
    for product, product_rows in grouped.items():
        values = [float(row["profit_and_loss"]) for row in product_rows if row["profit_and_loss"] != ""]
        mids = [float(row["mid_price"]) for row in product_rows if row["mid_price"] != ""]
        if not values:
            continue
        out[product] = {
            "final_pnl": values[-1],
            "min_pnl": min(values),
            "max_pnl": max(values),
            "last_mid": mids[-1] if mids else 0.0,
        }
    return out


def parse_graph(graph_log: str) -> Dict[str, float]:
    rows = list(csv.DictReader(io.StringIO(graph_log), delimiter=";"))
    points = [(int(row["timestamp"]), float(row["value"])) for row in rows]
    if not points:
        return {
            "graph_min": 0.0,
            "graph_min_ts": 0,
            "graph_max": 0.0,
            "graph_max_ts": 0,
            "graph_last": 0.0,
            "max_drawdown": 0.0,
        }

    peak = points[0][1]
    max_drawdown = 0.0
    for _, value in points:
        peak = max(peak, value)
        max_drawdown = min(max_drawdown, value - peak)

    min_ts, min_value = min(points, key=lambda x: x[1])
    max_ts, max_value = max(points, key=lambda x: x[1])
    return {
        "graph_min": min_value,
        "graph_min_ts": min_ts,
        "graph_max": max_value,
        "graph_max_ts": max_ts,
        "graph_last": points[-1][1],
        "max_drawdown": max_drawdown,
    }


def parse_positions(positions: List[Dict]) -> Dict[str, int]:
    return {str(row["symbol"]): int(row["quantity"]) for row in positions}


def parse_trade_history(trade_history: List[Dict]) -> Dict[str, Dict[str, float]]:
    grouped = defaultdict(
        lambda: {
            "own_trade_count": 0,
            "buy_qty": 0,
            "buy_notional": 0.0,
            "sell_qty": 0,
            "sell_notional": 0.0,
            "passive_or_market_trade_count": 0,
            "first_ts": None,
            "last_ts": None,
        }
    )
    for trade in trade_history:
        product = str(trade["symbol"])
        qty = int(trade["quantity"])
        price = float(trade["price"])
        ts = int(trade["timestamp"])
        buyer = trade.get("buyer") or ""
        seller = trade.get("seller") or ""
        row = grouped[product]
        row["first_ts"] = ts if row["first_ts"] is None else min(row["first_ts"], ts)
        row["last_ts"] = ts if row["last_ts"] is None else max(row["last_ts"], ts)

        if buyer == "SUBMISSION":
            row["own_trade_count"] += 1
            row["buy_qty"] += qty
            row["buy_notional"] += qty * price
        elif seller == "SUBMISSION":
            row["own_trade_count"] += 1
            row["sell_qty"] += qty
            row["sell_notional"] += qty * price
        else:
            row["passive_or_market_trade_count"] += 1

    out = {}
    for product, row in grouped.items():
        buy_qty = row["buy_qty"]
        sell_qty = row["sell_qty"]
        row["avg_buy"] = row["buy_notional"] / buy_qty if buy_qty else 0.0
        row["avg_sell"] = row["sell_notional"] / sell_qty if sell_qty else 0.0
        row["net_qty"] = buy_qty - sell_qty
        row["cash_pnl"] = row["sell_notional"] - row["buy_notional"]
        row["first_ts"] = -1 if row["first_ts"] is None else row["first_ts"]
        row["last_ts"] = -1 if row["last_ts"] is None else row["last_ts"]
        out[product] = dict(row)
    return out


def analyze_run(run_dir: Path) -> Tuple[Dict, List[Dict]]:
    run_id = run_dir.name
    json_path = run_dir / f"{run_id}.json"
    log_path = run_dir / f"{run_id}.log"
    py_path = run_dir / f"{run_id}.py"
    if not json_path.exists() or not log_path.exists():
        raise FileNotFoundError(f"Missing json/log pair for {run_dir}")

    result_json = load_json(json_path)
    log_json = load_json(log_path)
    graph = parse_graph(result_json.get("graphLog", "timestamp;value\n"))
    activity = parse_activity_pnl(result_json.get("activitiesLog", ""))
    trades = parse_trade_history(log_json.get("tradeHistory", []))
    positions = parse_positions(result_json.get("positions", []))

    summary = {
        "run_id": run_id,
        "submission_id": log_json.get("submissionId", ""),
        "status": result_json.get("status", ""),
        "round": result_json.get("round", ""),
        "profit": float(result_json.get("profit", 0.0)),
        "graph_min": graph["graph_min"],
        "graph_min_ts": graph["graph_min_ts"],
        "graph_max": graph["graph_max"],
        "graph_max_ts": graph["graph_max_ts"],
        "graph_last": graph["graph_last"],
        "max_drawdown": graph["max_drawdown"],
        "trade_history_rows": len(log_json.get("tradeHistory", [])),
        "code_file": str(py_path),
    }
    for product, qty in positions.items():
        summary[f"pos_{product}"] = qty

    product_rows = []
    products = sorted(set(activity) | set(trades) | set(positions))
    for product in products:
        a = activity.get(product, {})
        t = trades.get(product, {})
        row = {
            "run_id": run_id,
            "product": product,
            "final_pnl": a.get("final_pnl", 0.0),
            "min_pnl": a.get("min_pnl", 0.0),
            "max_pnl": a.get("max_pnl", 0.0),
            "last_mid": a.get("last_mid", 0.0),
            "final_position": positions.get(product, 0),
            "own_trade_count": t.get("own_trade_count", 0),
            "buy_qty": t.get("buy_qty", 0),
            "avg_buy": t.get("avg_buy", 0.0),
            "sell_qty": t.get("sell_qty", 0),
            "avg_sell": t.get("avg_sell", 0.0),
            "net_qty": t.get("net_qty", 0),
            "cash_pnl": t.get("cash_pnl", 0.0),
            "first_ts": t.get("first_ts", -1),
            "last_ts": t.get("last_ts", -1),
            "non_submission_trade_count": t.get("passive_or_market_trade_count", 0),
        }
        product_rows.append(row)
    return summary, product_rows


def analyze_all(log_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    product_rows = []
    for run_dir in sorted(path for path in log_dir.iterdir() if path.is_dir()):
        try:
            summary, rows = analyze_run(run_dir)
        except Exception as exc:
            print(f"Skipping {run_dir}: {exc}")
            continue
        summaries.append(summary)
        product_rows.extend(rows)
    return pd.DataFrame(summaries), pd.DataFrame(product_rows)


def print_console_summary(summary_df: pd.DataFrame, product_df: pd.DataFrame) -> None:
    if summary_df.empty:
        print("No runs found.")
        return

    cols = [
        "run_id",
        "profit",
        "graph_min",
        "graph_max",
        "max_drawdown",
        "trade_history_rows",
    ]
    print("\nRun Summary")
    print(summary_df[cols].sort_values("profit", ascending=False).to_string(index=False))

    print("\nPer Product PnL")
    pivot = product_df.pivot_table(
        index="product",
        columns="run_id",
        values="final_pnl",
        aggfunc="sum",
        fill_value=0.0,
    )
    print(pivot.round(2).to_string())

    print("\nFinal Positions")
    pivot_pos = product_df.pivot_table(
        index="product",
        columns="run_id",
        values="final_position",
        aggfunc="sum",
        fill_value=0,
    )
    print(pivot_pos.astype(int).to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=Path, default=LOG_DIR)
    parser.add_argument("--out-prefix", type=Path, default=Path("round3_platform"))
    args = parser.parse_args()

    summary_df, product_df = analyze_all(args.logs)
    print_console_summary(summary_df, product_df)

    summary_path = args.out_prefix.with_name(args.out_prefix.name + "_run_summary.csv")
    product_path = args.out_prefix.with_name(args.out_prefix.name + "_product_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    product_df.to_csv(product_path, index=False)
    print(f"\nWrote {summary_path}")
    print(f"Wrote {product_path}")


if __name__ == "__main__":
    main()

