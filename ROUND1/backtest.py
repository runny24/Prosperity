import numpy as np
import pandas as pd


def backtest_prosperity_strategy(df,
                                product="ACO",
                                max_position=20,
                                seed=42):

    np.random.seed(seed)
    df = df.copy()

    # =====================
    # 0. BASIC CLEANING
    # =====================
    df = df.reset_index(drop=True)

    # Use provided mid
    df["mid"] = df["mid_price"]

    # Forward-fill key fields
    cols = ["bid_price_1", "ask_price_1", "bid_volume_1", "ask_volume_1"]
    df[cols] = df[cols].ffill()

    # Fallback: if still missing, use mid
    df["bid_price_1"] = df["bid_price_1"].fillna(df["mid"])
    df["ask_price_1"] = df["ask_price_1"].fillna(df["mid"])

    # Safe spread
    df["spread"] = df["ask_price_1"] - df["bid_price_1"]

    # =====================
    # 1. FEATURES
    # =====================
    df["imbalance_l1"] = (
        df["bid_volume_1"] - df["ask_volume_1"]
    ) / (df["bid_volume_1"] + df["ask_volume_1"] + 1e-8)

    df["imbalance_l1"] = df["imbalance_l1"].fillna(0)

    # =====================
    # 2. FAIR VALUE
    # =====================
    if product == "ACO":
        bid = df["bid_price_1"]
        ask = df["ask_price_1"]
        bid_vol = df["bid_volume_1"]
        ask_vol = df["ask_volume_1"]

        df["fair"] = (
            bid * ask_vol + ask * bid_vol
        ) / (bid_vol + ask_vol + 1e-8)

        OFFSET = 1

    elif product == "IPR":

        # --- FIX: day-aware open ---
        df["day_open"] = df.groupby("day")["mid"].transform("first")

        model_fair = df["day_open"] + df["timestamp"] * 0.001
        df["fair"] = 0.7 * model_fair + 0.3 * df["mid"]

        OFFSET = 3

    else:
        raise ValueError("product must be 'ACO' or 'IPR'")

    # Final safeguard
    df["fair"] = df["fair"].fillna(df["mid"])

    # =====================
    # 3. BACKTEST LOOP
    # =====================
    position = 0
    cash = 0

    positions = []
    pnls = []

    for i in range(1, len(df)):
        row = df.iloc[i]

        bid = row["bid_price_1"]
        ask = row["ask_price_1"]
        mid = row["mid"]
        fair = row["fair"]

        # Safety guard (never let NaN propagate)
        if np.isnan(bid) or np.isnan(ask) or np.isnan(mid) or np.isnan(fair):
            positions.append(position)
            pnls.append(cash + position * mid if not np.isnan(mid) else np.nan)
            continue

        buy_capacity = max_position - position
        sell_capacity = max_position + position

        # ---------------------
        # 1. TAKE LIQUIDITY
        # ---------------------
        if ask < fair and buy_capacity > 0:
            size = min(1, buy_capacity)
            position += size
            cash -= ask * size

        if bid > fair and sell_capacity > 0:
            size = min(1, sell_capacity)
            position -= size
            cash += bid * size

        # ---------------------
        # 2. IPR TREND LAYER
        # ---------------------
        if product == "IPR":
            if position < max_position:
                size = min(1, max_position - position)
                position += size
                cash -= ask * size

        # ---------------------
        # 3. PASSIVE MM
        # ---------------------
        imbalance = row["imbalance_l1"]

        prob_buy_fill = max(0, -imbalance)
        prob_sell_fill = max(0, imbalance)

        # Passive buy
        if np.random.rand() < prob_buy_fill and position < max_position:
            position += 1
            cash -= bid

        # Passive sell
        if np.random.rand() < prob_sell_fill and position > -max_position:
            position -= 1
            cash += ask

        # ---------------------
        # MARK-TO-MARKET
        # ---------------------
        pnl = cash + position * mid

        positions.append(position)
        pnls.append(pnl)

    # =====================
    # 4. RESULTS
    # =====================
    df = df.iloc[1:].copy()
    df["position"] = positions
    df["pnl"] = pnls

    total_pnl = df["pnl"].iloc[-1]

    returns = df["pnl"].diff().fillna(0)
    sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)

    return df, {
        "final_pnl": total_pnl,
        "sharpe": sharpe,
        "max_position": max_position,
        "product": product
    }


# =====================
# USAGE
# =====================
df1 = pd.read_csv("prices_round_1_day_-2.csv", sep=";")
df2 = pd.read_csv("prices_round_1_day_-1.csv", sep=";")
df3 = pd.read_csv("prices_round_1_day_0.csv", sep=";")

df_combined = pd.concat([df1, df2, df3], ignore_index=True)

aco_df = df_combined[df_combined["product"] == "ASH_COATED_OSMIUM"].reset_index(drop=True)
ipr_df = df_combined[df_combined["product"] == "INTARIAN_PEPPER_ROOT"].reset_index(drop=True)

aco_result, aco_stats = backtest_prosperity_strategy(
    aco_df, product="ACO", max_position=20
)

ipr_result, ipr_stats = backtest_prosperity_strategy(
    ipr_df, product="IPR", max_position=80
)

print("ACO:", aco_stats)
print("IPR:", ipr_stats)