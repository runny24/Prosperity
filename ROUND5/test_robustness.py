"""
Robustness checks for r5_v1.py.

This script uses the replay harness in test_backtest.py and asks whether the
strategy looks overfit by checking:
- day-by-day stability
- train days 2-3 versus validation day 4
- module ablations
- small parameter perturbations

Outputs:
- robustness_results.csv
- robustness_summary.txt

Usage:
    python3 test_robustness.py
    python3 test_robustness.py r5_v1.py
    python3 test_robustness.py --max-ticks 1000
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from test_backtest import run_backtest


DEFAULT_STRATEGY = Path("r5_v1.py")

BASE_WHITELIST = {
    "PEBBLES_XL",
    "OXYGEN_SHAKE_MORNING_BREATH",
    "ROBOT_MOPPING",
    "MICROCHIP_CIRCLE",
    "SLEEP_POD_COTTON",
    "GALAXY_SOUNDS_BLACK_HOLES",
    "GALAXY_SOUNDS_PLANETARY_RINGS",
    "MICROCHIP_OVAL",
    "ROBOT_IRONING",
    "PANEL_1X4",
    "UV_VISOR_AMBER",
    "TRANSLATOR_GRAPHITE_MIST",
}

SNACKPACK = {
    "SNACKPACK_CHOCOLATE",
    "SNACKPACK_PISTACHIO",
    "SNACKPACK_RASPBERRY",
    "SNACKPACK_STRAWBERRY",
    "SNACKPACK_VANILLA",
}


def run_case(
    name: str,
    days: List[int],
    strategy: Path,
    max_ticks: Optional[int],
    overrides: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    result = run_backtest(
        strategy_path=strategy,
        days=days,
        max_ticks=max_ticks,
        overrides=overrides,
        quiet=True,
    )
    fills = result["fills"]
    return {
        "case": name,
        "days": ",".join(map(str, days)),
        "ticks": result["ticks"],
        "total_pnl": result["total_pnl"],
        "max_drawdown": result["max_drawdown"],
        "fill_count": len(fills),
        "ending_abs_position": sum(abs(v) for v in result["positions"].values()),
    }


def build_cases() -> List[Dict[str, object]]:
    cases: List[Dict[str, object]] = []

    for day in [2, 3, 4]:
        cases.append({"name": f"day_{day}", "days": [day], "overrides": None})

    cases.extend(
        [
            {"name": "train_days_2_3", "days": [2, 3], "overrides": None},
            {"name": "validation_day_4", "days": [4], "overrides": None},
            {
                "name": "ablation_no_snackpack",
                "days": [4],
                "overrides": {
                    "MICRO_ONLY": set(),
                    "TRADED_PRODUCTS": BASE_WHITELIST,
                },
            },
            {
                "name": "ablation_snackpack_only",
                "days": [4],
                "overrides": {
                    "WHITELIST": set(),
                    "MICRO_ONLY": SNACKPACK,
                    "TRADED_PRODUCTS": SNACKPACK,
                },
            },
            {
                "name": "ablation_no_weak_name_confirm",
                "days": [4],
                "overrides": {"EMA_NEEDS_MICRO_CONFIRM": set()},
            },
        ]
    )

    perturbations = [
        ("looser_micro", {"MICRO_CONFIRM": 0.02, "MICRO_BLOCK": 0.08, "MICRO_ENTRY": 0.14, "MICRO_TAKE": 0.24}),
        ("base_micro", {"MICRO_CONFIRM": 0.04, "MICRO_BLOCK": 0.12, "MICRO_ENTRY": 0.18, "MICRO_TAKE": 0.30}),
        ("stricter_micro", {"MICRO_CONFIRM": 0.06, "MICRO_BLOCK": 0.16, "MICRO_ENTRY": 0.24, "MICRO_TAKE": 0.38}),
        ("snack_limit_2", {"SNACK_LIMIT": 2}),
        ("snack_limit_6", {"SNACK_LIMIT": 6}),
        ("ema_fast_slow_down", {"FAST_ALPHA": 0.015}),
        ("ema_fast_speed_up", {"FAST_ALPHA": 0.030}),
    ]
    for name, overrides in perturbations:
        cases.append({"name": f"perturb_{name}", "days": [4], "overrides": overrides})

    return cases


def make_verdict(results: pd.DataFrame) -> str:
    day_rows = results[results["case"].isin(["day_2", "day_3", "day_4"])].copy()
    perturb_rows = results[results["case"].str.startswith("perturb_")].copy()

    positive_days = int((day_rows["total_pnl"] > 0).sum())
    worst_day = float(day_rows["total_pnl"].min()) if not day_rows.empty else 0.0
    median_perturb = float(perturb_rows["total_pnl"].median()) if not perturb_rows.empty else 0.0
    worst_perturb = float(perturb_rows["total_pnl"].min()) if not perturb_rows.empty else 0.0
    base_day4 = float(results.loc[results["case"].eq("day_4"), "total_pnl"].iloc[0])
    train = float(results.loc[results["case"].eq("train_days_2_3"), "total_pnl"].iloc[0])
    valid = float(results.loc[results["case"].eq("validation_day_4"), "total_pnl"].iloc[0])

    flags = []
    if positive_days < 2:
        flags.append("day stability is weak")
    if base_day4 > 0 and worst_perturb < -0.5 * abs(base_day4):
        flags.append("small parameter changes can break the strategy")
    if train > 0 and valid < 0:
        flags.append("train is positive but validation is negative")
    if median_perturb < 0:
        flags.append("median perturbation is negative")

    if flags:
        status = "Needs more alpha validation before increasing size."
    else:
        status = "Looks reasonably robust under these checks."

    lines = [
        "ROUND5 robustness summary",
        "",
        f"Day positives: {positive_days}/3",
        f"Worst single day PnL: {worst_day:.2f}",
        f"Train days 2-3 PnL: {train:.2f}",
        f"Validation day 4 PnL: {valid:.2f}",
        f"Base day 4 PnL: {base_day4:.2f}",
        f"Median perturbation PnL: {median_perturb:.2f}",
        f"Worst perturbation PnL: {worst_perturb:.2f}",
        "",
        f"Verdict: {status}",
    ]
    if flags:
        lines.append("")
        lines.append("Flags:")
        for flag in flags:
            lines.append(f"- {flag}")
    lines.extend(
        [
            "",
            "Notes:",
            "- The backtest is conservative and fills only visible-book crossing orders.",
            "- Passive quote value is undercounted; use this for relative comparisons, not final PnL claims.",
            "- A robust strategy should not depend on one day, one module, or one exact threshold set.",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("strategy_file", nargs="?", default=None)
    parser.add_argument("--strategy", default=str(DEFAULT_STRATEGY))
    parser.add_argument("--max-ticks", type=int, default=None)
    parser.add_argument("--results-csv", default="robustness_results.csv")
    parser.add_argument("--summary-txt", default="robustness_summary.txt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    strategy = Path(args.strategy_file or args.strategy)
    rows = []
    for case in build_cases():
        print(f"running {case['name']} days={case['days']}")
        rows.append(
            run_case(
                name=case["name"],
                days=case["days"],
                strategy=strategy,
                max_ticks=args.max_ticks,
                overrides=case["overrides"],
            )
        )

    results = pd.DataFrame(rows)
    results.to_csv(args.results_csv, index=False)
    summary = make_verdict(results)
    Path(args.summary_txt).write_text(summary + "\n")

    print()
    print(results.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    print()
    print(summary)
    print(f"\nwrote {args.results_csv} and {args.summary_txt}")


if __name__ == "__main__":
    main()
