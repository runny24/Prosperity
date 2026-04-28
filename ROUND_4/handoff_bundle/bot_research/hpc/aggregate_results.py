from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate manifest-worker JSON results.")
    parser.add_argument(
        "--results-dir",
        default="bot_research/outputs/hpc_results",
        help="Directory containing job_*.json files.",
    )
    parser.add_argument(
        "--output-dir",
        default="bot_research/outputs/hpc_results",
        help="Directory to write aggregated CSV outputs into.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    top_rule_rows = []

    for path in sorted(results_dir.glob("job_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary_rows.append(
            {
                "job_id": payload["job_id"],
                "symbol": payload["symbol"],
                "lane": payload["lane"],
                "experiment_type": payload["experiment_type"],
                "priority": payload["priority"],
                "target_positives": payload["target_positives"],
                "target_rate": payload["target_rate"],
                "days_with_target": payload["days_with_target"],
                "n_top_rules": len(payload["top_rules"]),
                "best_score": payload["top_rules"][0]["score"] if payload["top_rules"] else None,
                "best_f1": payload["top_rules"][0]["f1"] if payload["top_rules"] else None,
                "best_stability": payload["top_rules"][0]["stability"] if payload["top_rules"] else None,
            }
        )
        for rank, rule in enumerate(payload["top_rules"], start=1):
            top_rule_rows.append(
                {
                    "job_id": payload["job_id"],
                    "symbol": payload["symbol"],
                    "lane": payload["lane"],
                    "experiment_type": payload["experiment_type"],
                    "rank": rank,
                    "score": rule["score"],
                    "f1": rule["f1"],
                    "precision": rule["precision"],
                    "recall": rule["recall"],
                    "stability": rule["stability"],
                    "coverage_days": rule["coverage_days"],
                    "fires": rule["fires"],
                    "rules": " AND ".join(rule["rules"]),
                }
            )

    summary = pd.DataFrame(summary_rows).sort_values(["best_score", "job_id"], ascending=[False, True])
    top_rules = pd.DataFrame(top_rule_rows).sort_values(["score", "job_id", "rank"], ascending=[False, True, True])

    summary.to_csv(out_dir / "summary.csv", index=False)
    top_rules.to_csv(out_dir / "top_rules.csv", index=False)

    print(f"Saved summary -> {out_dir / 'summary.csv'}")
    print(f"Saved rules   -> {out_dir / 'top_rules.csv'}")
    if not summary.empty:
        print("\nTop jobs:")
        print(summary.head(20).to_string(index=False))
    if not top_rules.empty:
        print("\nTop rules:")
        print(top_rules.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
