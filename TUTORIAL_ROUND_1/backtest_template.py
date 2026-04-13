import numpy as np
import pandas as pd

def backtest_orderbook_strategy(df,
                               buy_threshold=0.5,
                               sell_threshold=-0.5,
                               max_position=20,
                               seed=42):

    np.random.seed(seed)
    df = df.copy()

    # =====================
    # 1. FEATURES
    # =====================
    df["mid"] = (df["bid_price_1"] + df["ask_price_1"]) / 2
    df["spread"] = df["ask_price_1"] - df["bid_price_1"]
    df["return"] = np.log(df["mid"]).diff()

    # Imbalance L1
    df["imbalance_l1"] = (
        df["bid_volume_1"] - df["ask_volume_1"]
    ) / (df["bid_volume_1"] + df["ask_volume_1"] + 1e-8)

    # Imbalance L3
    bid_vol = df[["bid_volume_1","bid_volume_2","bid_volume_3"]].sum(axis=1)
    ask_vol = df[["ask_volume_1","ask_volume_2","ask_volume_3"]].sum(axis=1)
    df["imbalance_l3"] = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-8)

    # Signal
    df["signal"] = (
        0.5 * df["imbalance_l1"] +
        0.3 * df["imbalance_l3"] +
        0.2 * df["return"]
    )

    df["signal"] = (df["signal"] - df["signal"].mean()) / (df["signal"].std() + 1e-8)

    # =====================
    # 2. BACKTEST LOOP
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
        signal = row["signal"]

        # ---------------------
        # Aggressive trades
        # ---------------------
        if signal > buy_threshold and position < max_position:
            position += 1
            cash -= ask  # buy at ask

        elif signal < sell_threshold and position > -max_position:
            position -= 1
            cash += bid  # sell at bid

        # ---------------------
        # Passive market making
        # ---------------------
        else:
            # Avoid adverse selection
            if abs(signal) < 1:

                prob_buy_fill = max(0, -row["imbalance_l1"])
                prob_sell_fill = max(0, row["imbalance_l1"])

                # Buy passively at bid
                if np.random.rand() < prob_buy_fill and position < max_position:
                    position += 1
                    cash -= bid

                # Sell passively at ask
                if np.random.rand() < prob_sell_fill and position > -max_position:
                    position -= 1
                    cash += ask

        # Mark-to-market PnL
        pnl = cash + position * mid

        positions.append(position)
        pnls.append(pnl)

    # =====================
    # 3. RESULTS
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
        "max_position": max_position
    }


# =====================
# USAGE
# =====================
df = pd.read_csv("prices_round_0_day_-1.csv", sep=";")
result_df, stats = backtest_orderbook_strategy(df)
print(stats)