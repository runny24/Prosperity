"""
Why does IPR look linear?
=========================
Short answer: it is linear — by design. The simulation embeds a
deterministic drift of +0.1 per tick. That drift (≈1000 pts/day)
completely dominates the y-axis scale, so the random noise (σ≈3 ticks)
is invisible at that zoom level.

This script shows:
  1. Raw mid-price  → looks like a straight line
  2. Detrended price → reveals the actual noise underneath
  3. Tick changes    → confirms the noise distribution
  4. Synthetic comparison: what a pure random walk would look like
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA = Path("data")

# ── load day 0 IPR ───────────────────────────────────────────────────────────
df = pd.read_csv(DATA / "prices_round_1_day_0.csv", sep=";")
ipr = df[df["product"] == "INTARIAN_PEPPER_ROOT"].copy()
ipr = ipr[ipr["mid_price"] > 0].reset_index(drop=True)

ts = ipr["timestamp"].values
price = ipr["mid_price"].values
deltas = np.diff(price)

# ── fit the linear trend ─────────────────────────────────────────────────────
slope, intercept = np.polyfit(ts, price, 1)
trend = slope * ts + intercept
detrended = price - trend

print(f"IPR day 0 linear fit:")
print(f"  slope     = {slope:.6f}  (≈ {slope:.2f} pts/tick)")
print(f"  intercept = {intercept:.2f}")
print(f"  R²        = {np.corrcoef(ts, price)[0,1]**2:.6f}")
print(f"\nPrice range : {price.min():.1f} → {price.max():.1f}  (Δ={price.max()-price.min():.0f})")
print(f"Noise σ     : {detrended.std():.2f} pts  (invisible vs Δ={price.max()-price.min():.0f} range)")
print(f"Mean Δ/tick : {deltas.mean():.4f}  (≈ exactly 0.1)")

# ── plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(
    "IPR day 0 — why the price looks linear",
    fontsize=14, fontweight="bold"
)

# 1. Raw mid-price (looks linear)
ax = axes[0, 0]
ax.plot(ts, price, color="#FF5722", lw=0.7, label="mid price")
ax.plot(ts, trend,  color="black",   lw=1.5, linestyle="--",
        label=f"linear fit  slope={slope:.4f} pts/tick")
ax.set_title("1. Raw mid-price — looks like a straight line")
ax.set_xlabel("timestamp")
ax.set_ylabel("price")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# 2. Detrended price — the noise underneath
ax = axes[0, 1]
ax.plot(ts, detrended, color="#FF5722", lw=0.5, alpha=0.9)
ax.axhline(0, color="black", lw=0.8)
ax.fill_between(ts, detrended, alpha=0.2, color="#FF5722")
ax.set_title(f"2. Detrended price — actual noise (σ={detrended.std():.1f} pts)")
ax.set_xlabel("timestamp")
ax.set_ylabel("price − trend")
ax.grid(alpha=0.3)

# 3. Tick-to-tick changes
ax = axes[1, 0]
ax.hist(deltas, bins=60, color="#FF5722", edgecolor="white", linewidth=0.3)
ax.axvline(deltas.mean(), color="black", linestyle="--",
           label=f"mean={deltas.mean():.4f}")
ax.axvline(0, color="gray", lw=0.8)
ax.set_title(f"3. Tick-to-tick Δprice (σ={deltas.std():.2f}, mean={deltas.mean():.4f})")
ax.set_xlabel("Δ mid price")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# 4. Synthetic comparison
ax = axes[1, 1]
n = len(price)
rng = np.random.default_rng(42)

# random walk with same per-tick σ but NO trend
rw_noise = rng.normal(0, deltas.std(), n - 1)
rw = np.cumsum(np.concatenate([[price[0]], rw_noise]))

# actual IPR
ax.plot(ts, price, color="#FF5722", lw=0.7, label="IPR (real, trending)")
ax.plot(ts, rw,    color="#2196F3", lw=0.7, alpha=0.8,
        label=f"Pure random walk (σ={deltas.std():.1f})")
ax.set_title("4. IPR vs a pure random walk with same noise level")
ax.set_xlabel("timestamp")
ax.set_ylabel("price")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("ipr_linearity_demo.png", dpi=150)
plt.show()
print("\nSaved: ipr_linearity_demo.png")
