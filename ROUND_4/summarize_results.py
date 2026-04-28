"""
summarize_results.py — compare baseline + sweep results side-by-side
Run locally after SLURM jobs complete:
    python summarize_results.py
"""
import json, os, glob

results_dir = os.path.join(os.path.dirname(__file__), "results")

files = sorted(glob.glob(os.path.join(results_dir, "*.json")))
if not files:
    print("No result files found in results/. Run SLURM jobs first.")
    exit(1)

all_results = {}
for f in files:
    with open(f) as fh:
        data = json.load(fh)
    all_results[data["label"]] = data

# Print comparison table
all_products = sorted({p for d in all_results.values() for p in d["pnl"]})
labels       = sorted(all_results)

col_w = 14
print(f"\n{'Product':<28}", end="")
for lbl in labels:
    short = lbl.replace("v23.py | ", "").replace("v23.py", "baseline")
    print(f"  {short:>{col_w}}", end="")
print()
print("-" * (28 + (col_w + 2) * len(labels)))

for product in all_products:
    print(f"{product:<28}", end="")
    for lbl in labels:
        val = all_results[lbl]["pnl"].get(product, 0)
        print(f"  {val:>{col_w},.1f}", end="")
    print()

print("-" * (28 + (col_w + 2) * len(labels)))
print(f"{'TOTAL':<28}", end="")
for lbl in labels:
    print(f"  {all_results[lbl]['total']:>{col_w},.1f}", end="")
print()

# Highlight best SELL_MIN for VEV_5500
sweep_results = {lbl: d for lbl, d in all_results.items() if "SELL_MIN" in lbl}
if sweep_results:
    print("\n=== VEV_5500 PnL by SELL_MIN ===")
    for lbl in sorted(sweep_results):
        v5500 = sweep_results[lbl]["pnl"].get("VEV_5500", 0)
        total = sweep_results[lbl]["total"]
        print(f"  {lbl:<35}  VEV_5500={v5500:>8,.1f}   TOTAL={total:>10,.1f}")
    best = max(sweep_results, key=lambda l: sweep_results[l]["pnl"].get("VEV_5500", 0))
    print(f"\n  Best SELL_MIN for VEV_5500 PnL: {best}")
    best_total = max(sweep_results, key=lambda l: sweep_results[l]["total"])
    print(f"  Best SELL_MIN for TOTAL PnL:    {best_total}")
