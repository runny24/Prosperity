from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_PROFILES = {
    "core": [
        "time_frac",
        "spread_over_mid",
        "book_imbalance",
        "total_levels",
        "momentum_5",
    ],
    "full": [
        "time_frac",
        "spread_over_mid",
        "bid_volume_1",
        "ask_volume_1",
        "book_imbalance",
        "total_levels",
        "momentum_5",
        "momentum_20",
    ],
    "flow": [
        "time_frac",
        "bid_volume_1",
        "ask_volume_1",
        "book_imbalance",
        "momentum_5",
        "momentum_20",
    ],
    "structure": [
        "time_frac",
        "spread_over_mid",
        "bid_volume_1",
        "ask_volume_1",
        "total_levels",
        "momentum_20",
    ],
}

THRESHOLD_PROFILES = {
    "sparse": [10, 25, 40, 60, 75, 90],
    "medium": [5, 15, 30, 45, 55, 70, 85, 95],
    "dense": [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95],
}


def parse_notes(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for chunk in str(text).split(";"):
        chunk = chunk.strip()
        if "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def confidence_threshold(bucket: str) -> float:
    return {"high": 0.95, "medium": 0.80, "low": 0.0}.get(bucket, 0.0)


def time_bucket_mask(df: pd.DataFrame, bucket: str) -> pd.Series:
    if bucket == "early":
        return df["time_frac"] < (1 / 3)
    if bucket == "mid":
        return (df["time_frac"] >= (1 / 3)) & (df["time_frac"] < (2 / 3))
    if bucket == "late":
        return df["time_frac"] >= (2 / 3)
    return pd.Series(True, index=df.index)


def side_mask(df: pd.DataFrame, label: str) -> pd.Series:
    if label == "buy":
        return df["mean_side"] > 0
    if label == "sell":
        return df["mean_side"] < 0
    return df["any_trade"] == 1


def make_target_mask(df: pd.DataFrame, job: pd.Series) -> pd.Series:
    notes = parse_notes(job["notes"])
    lane = job["lane"]
    exp_type = job["experiment_type"]

    mask = pd.Series(True, index=df.index)
    mask &= df["symbol"] == job["symbol"]

    if exp_type == "timestamp_window_search":
        mask &= df["timestamp"].between(int(job["window_left"]), int(job["window_right"]))
        mask &= df["any_trade"] == 1
        mask &= side_mask(df, notes.get("side_regime", "mixed"))
    else:
        mask &= df["any_trade"] == 1
        mask &= time_bucket_mask(df, notes.get("time_bucket", ""))
        mask &= side_mask(df, notes.get("side_regime", "mixed"))

    archetype = int(job["dominant_archetype"])
    conf_thresh = confidence_threshold(notes.get("confidence_bucket", "low"))

    if lane == "p3_seeded":
        if archetype >= 0:
            mask &= df["dominant_archetype"] == archetype
        mask &= df["mean_confidence"] >= conf_thresh
    elif lane == "no_prior":
        pass
    elif lane == "drift_search":
        mismatch = pd.Series(False, index=df.index)
        if archetype >= 0:
            mismatch |= (df["dominant_archetype"] != -1) & (df["dominant_archetype"] != archetype)
        mismatch |= df["mean_confidence"] < max(conf_thresh, 0.90)
        mismatch |= df["dominant_share"] < 0.80
        mask &= mismatch

    return mask.fillna(False)


def build_binary_rules(df: pd.DataFrame, feature_names: list[str], percentiles: list[int]) -> tuple[np.ndarray, list[str]]:
    cols = []
    descs = []
    for feature in feature_names:
        values = df[feature].to_numpy(dtype=float)
        valid = values[np.isfinite(values)]
        if valid.size == 0:
            continue
        thresholds = np.unique(np.percentile(valid, percentiles))
        for threshold in thresholds:
            cols.append(values > threshold)
            descs.append(f"{feature} > {threshold:.6g}")
            cols.append(values <= threshold)
            descs.append(f"{feature} <= {threshold:.6g}")
    return np.column_stack(cols).astype(bool), descs


def score_mask(mask: np.ndarray, labels: np.ndarray, days: np.ndarray) -> dict[str, float] | None:
    n_pred = int(mask.sum())
    if n_pred == 0:
        return None
    tp = int(np.count_nonzero(mask & labels))
    if tp == 0:
        return None
    n_pos = int(labels.sum())
    precision = tp / n_pred
    recall = tp / n_pos if n_pos else 0.0
    if precision <= 0 or recall <= 0:
        return None
    f1 = 2 * precision * recall / (precision + recall)

    day_scores = []
    positive_days = 0
    for day in sorted(np.unique(days)):
        day_mask = days == day
        day_labels = labels[day_mask]
        if not day_labels.any():
            day_scores.append(0.0)
            continue
        positive_days += 1
        day_rule = mask[day_mask]
        day_tp = int(np.count_nonzero(day_rule & day_labels))
        day_pred = int(day_rule.sum())
        day_pos = int(day_labels.sum())
        if day_tp == 0 or day_pred == 0:
            day_scores.append(0.0)
            continue
        day_prec = day_tp / day_pred
        day_rec = day_tp / day_pos
        day_scores.append(2 * day_prec * day_rec / (day_prec + day_rec))

    day_scores = np.array(day_scores, dtype=float)
    stability = float(day_scores.mean() - day_scores.std())
    score = 0.55 * stability + 0.45 * f1
    coverage = int(np.count_nonzero([s > 0 for s in day_scores]))
    return {
        "score": float(score),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "stability": float(stability),
        "coverage_days": int(coverage),
        "positive_days": int(positive_days),
        "fires": n_pred,
        "tp": tp,
    }


def search_rules(
    df: pd.DataFrame,
    labels: np.ndarray,
    feature_names: list[str],
    percentiles: list[int],
    max_depth: int,
    top_k: int,
) -> list[dict]:
    binary, descs = build_binary_rules(df, feature_names, percentiles)
    if binary.size == 0:
        return []

    days = df["day"].to_numpy()
    best: list[dict] = []
    n_rules = binary.shape[1]

    for depth in range(1, max_depth + 1):
        for combo in itertools.combinations(range(n_rules), depth):
            mask = binary[:, combo[0]].copy()
            for idx in combo[1:]:
                mask &= binary[:, idx]
            metrics = score_mask(mask, labels, days)
            if metrics is None:
                continue
            if metrics["coverage_days"] < min(2, metrics["positive_days"]):
                continue
            result = {
                **metrics,
                "depth": depth,
                "rules": [descs[idx] for idx in combo],
            }
            best.append(result)
        best.sort(key=lambda x: x["score"], reverse=True)
        best = best[:top_k]

    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one manifest-driven Midway job.")
    parser.add_argument("--job-id", type=int, required=True, help="Row id from the manifest.")
    parser.add_argument(
        "--manifest",
        default="bot_research/outputs/hpc_prep/hpc_experiment_manifest.csv",
        help="Path to the manifest CSV.",
    )
    parser.add_argument(
        "--features-dir",
        default="bot_research/outputs/hpc_prep/features",
        help="Directory containing tick_features_<symbol>.csv.gz files.",
    )
    parser.add_argument("--max-depth", type=int, default=3, help="Maximum AND-rule depth to search.")
    parser.add_argument("--top-k", type=int, default=25, help="How many top rules to keep.")
    parser.add_argument(
        "--out-dir",
        default="bot_research/outputs/hpc_results",
        help="Directory to write result JSON files into.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    row = manifest.loc[manifest["job_id"] == args.job_id]
    if row.empty:
        raise ValueError(f"Job id {args.job_id} not found in {args.manifest}")
    job = row.iloc[0]

    feature_path = Path(args.features_dir) / f"tick_features_{job['symbol']}.csv.gz"
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing feature file: {feature_path}")

    df = pd.read_csv(feature_path)
    labels = make_target_mask(df, job).to_numpy(dtype=bool)
    feature_profile = str(job.get("feature_profile", "full"))
    threshold_profile = str(job.get("threshold_profile", "sparse"))
    feature_names = FEATURE_PROFILES.get(feature_profile, FEATURE_PROFILES["full"])
    percentiles = THRESHOLD_PROFILES.get(threshold_profile, THRESHOLD_PROFILES["sparse"])
    max_depth = int(job.get("max_depth", args.max_depth))
    result = {
        "job_id": int(args.job_id),
        "symbol": str(job["symbol"]),
        "lane": str(job["lane"]),
        "experiment_type": str(job["experiment_type"]),
        "priority": str(job["priority"]),
        "search_profile": str(job.get("search_profile", "")),
        "feature_profile": feature_profile,
        "threshold_profile": threshold_profile,
        "max_depth": max_depth,
        "target_positives": int(labels.sum()),
        "target_rate": float(labels.mean()),
        "days_with_target": int(df.loc[labels, "day"].nunique()) if labels.any() else 0,
        "manifest": job.to_dict(),
        "top_rules": [],
    }

    if labels.sum() > 0:
        result["top_rules"] = search_rules(
            df,
            labels,
            feature_names=feature_names,
            percentiles=percentiles,
            max_depth=max_depth,
            top_k=args.top_k,
        )

    out_path = out_dir / f"job_{args.job_id:05d}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {out_path}")
    print(
        f"job_id={args.job_id} symbol={job['symbol']} lane={job['lane']} "
        f"positives={result['target_positives']} top_rules={len(result['top_rules'])}"
    )
    if result["top_rules"]:
        print(json.dumps(result["top_rules"][0], indent=2))


if __name__ == "__main__":
    main()
