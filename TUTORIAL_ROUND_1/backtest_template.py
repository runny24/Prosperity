"""
IMC Prosperity Round 0 backtest — per-product strategies.
Fill model is driven by the actual trades CSV, not synthetic book assumptions.
"""

import json
import numpy as np
import pandas as pd
try:
    from datamodel import OrderDepth, TradingState, Order
except ImportError:
    pass


# ── fair-value helpers (unchanged) ──────────────────────────────────────────

def fv_emeralds(df):
    return np.full(len(df), 10000.0)

def fv_tomatoes(df, window=10):
    mid = (df["bid_price_1"].values + df["ask_price_1"].values) / 2.0
    s = pd.Series(mid).rolling(window, min_periods=1).mean().shift(1)
    s.iloc[0] = mid[0]
    return s.ffill().values


# ── vectorised backtest (unchanged) ─────────────────────────────────────────

def backtest(prices, trades, fair_value,
             take_edge=0.0, make_edge=1.0,
             position_limit=20, skew_per_unit=0.0):
    n = len(prices)
    ts   = prices["timestamp"].values
    bid1 = prices["bid_price_1"].values
    ask1 = prices["ask_price_1"].values
    bv1, av1 = prices["bid_volume_1"].values, prices["ask_volume_1"].values
    bid2 = prices["bid_price_2"].fillna(np.nan).values
    ask2 = prices["ask_price_2"].fillna(np.nan).values
    bv2  = prices["bid_volume_2"].fillna(0).values
    av2  = prices["ask_volume_2"].fillna(0).values
    bid3 = prices["bid_price_3"].fillna(np.nan).values
    ask3 = prices["ask_price_3"].fillna(np.nan).values
    bv3  = prices["bid_volume_3"].fillna(0).values
    av3  = prices["ask_volume_3"].fillna(0).values
    mid  = (bid1 + ask1) / 2.0

    trades_by_ts = {t: g[["price", "quantity"]].values
                    for t, g in trades.groupby("timestamp")}

    pos = 0
    cash = 0.0
    positions = np.zeros(n)
    pnls = np.zeros(n)
    n_take = n_make = 0

    for i in range(n):
        fv = fair_value[i] - skew_per_unit * pos

        # TAKE
        for ap, av in ((ask1[i], av1[i]), (ask2[i], av2[i]), (ask3[i], av3[i])):
            if np.isnan(ap) or av <= 0 or ap > fv - take_edge:
                continue
            qty = min(int(av), position_limit - pos)
            if qty > 0:
                pos += qty
                cash -= qty * ap
                n_take += 1

        for bp, bv in ((bid1[i], bv1[i]), (bid2[i], bv2[i]), (bid3[i], bv3[i])):
            if np.isnan(bp) or bv <= 0 or bp < fv + take_edge:
                continue
            qty = min(int(bv), pos + position_limit)
            if qty > 0:
                pos -= qty
                cash += qty * bp
                n_take += 1

        # MAKE — fills driven by actual trade prints at this timestamp
        my_bid = int(np.round(fv - make_edge))
        my_ask = int(np.round(fv + make_edge))
        my_bid = min(my_bid, int(ask1[i]) - 1)
        my_ask = max(my_ask, int(bid1[i]) + 1)

        ts_tr = trades_by_ts.get(ts[i])
        if ts_tr is not None:
            for tp, tq in ts_tr:
                # Seller-initiated (sold at/near bid) — our better bid intercepts
                if tp <= bid1[i] and my_bid > bid1[i] and my_bid >= tp \
                        and pos < position_limit:
                    pos += 1
                    cash -= my_bid
                    n_make += 1
                # Buyer-initiated (lifted ask) — our better ask intercepts
                elif tp >= ask1[i] and my_ask < ask1[i] and my_ask <= tp \
                        and pos > -position_limit:
                    pos -= 1
                    cash += my_ask
                    n_make += 1

        positions[i] = pos
        pnls[i] = cash + pos * mid[i]

    out = prices.copy()
    out["fair_value"] = fair_value
    out["position"] = positions
    out["pnl"] = pnls
    return out, {"n_take": n_take, "n_make": n_make}


def summarize(result, label, counts):
    pnl = result["pnl"].values
    ret = np.diff(pnl, prepend=0.0)
    sharpe = ret.mean() / (ret.std() + 1e-9) * np.sqrt(len(ret))
    return {
        "label": label,
        "final_pnl": float(pnl[-1]),
        "sharpe_like": float(sharpe),
        "max_abs_pos": int(np.max(np.abs(result["position"]))),
        "take_fills": counts["n_take"],
        "make_fills": counts["n_make"],
    }


# ── IMC Prosperity live-trading interface ────────────────────────────────────

PARAMS = {
    "EMERALDS": dict(take_edge=0.0, make_edge=1.0, position_limit=20, skew_per_unit=0.2),
    "TOMATOES": dict(take_edge=1.0, make_edge=1.0, position_limit=20, skew_per_unit=0.1),
}
TOMATOES_WINDOW = 10


class Trader:
    def run(self, state: TradingState):
        trader_data = json.loads(state.traderData) if state.traderData else {}
        result = {}

        for product, cfg in PARAMS.items():
            depth: OrderDepth = state.order_depths.get(product)
            if depth is None:
                continue

            pos = state.position.get(product, 0)
            position_limit = cfg["position_limit"]
            take_edge      = cfg["take_edge"]
            make_edge      = cfg["make_edge"]
            skew_per_unit  = cfg["skew_per_unit"]

            buy_orders  = sorted(depth.buy_orders.items(),  reverse=True)
            sell_orders = sorted(depth.sell_orders.items())

            if not buy_orders or not sell_orders:
                continue

            bid1, bv1 = buy_orders[0]
            ask1, av1 = sell_orders[0]
            bid2, bv2 = buy_orders[1]  if len(buy_orders)  > 1 else (float("nan"), 0)
            ask2, av2 = sell_orders[1] if len(sell_orders) > 1 else (float("nan"), 0)
            bid3, bv3 = buy_orders[2]  if len(buy_orders)  > 2 else (float("nan"), 0)
            ask3, av3 = sell_orders[2] if len(sell_orders) > 2 else (float("nan"), 0)

            mid = (bid1 + ask1) / 2.0

            # fair-value (same logic as fv_emeralds / fv_tomatoes)
            if product == "EMERALDS":
                fv_raw = 10000.0
            else:
                history: list = trader_data.get("TOMATOES_mids", [])
                history.append(mid)
                if len(history) > TOMATOES_WINDOW:
                    history = history[-TOMATOES_WINDOW:]
                trader_data["TOMATOES_mids"] = history
                fv_raw = float(np.mean(history[:-1])) if len(history) > 1 else mid

            fv = fv_raw - skew_per_unit * pos

            orders = []
            max_buy  = position_limit - pos
            max_sell = position_limit + pos

            # TAKE
            for ap, av in ((ask1, abs(av1)), (ask2, abs(av2)), (ask3, abs(av3))):
                if np.isnan(ap) or av <= 0 or ap > fv - take_edge:
                    continue
                qty = min(int(av), max_buy)
                if qty > 0:
                    orders.append(Order(product, int(ap), qty))
                    max_buy -= qty
                    pos     += qty

            for bp, bv in ((bid1, abs(bv1)), (bid2, abs(bv2)), (bid3, abs(bv3))):
                if np.isnan(bp) or bv <= 0 or bp < fv + take_edge:
                    continue
                qty = min(int(bv), max_sell)
                if qty > 0:
                    orders.append(Order(product, int(bp), -qty))
                    max_sell -= qty
                    pos      -= qty

            # MAKE
            my_bid = int(round(fv - make_edge))
            my_ask = int(round(fv + make_edge))
            my_bid = min(my_bid, int(ask1) - 1)
            my_ask = max(my_ask, int(bid1) + 1)

            if max_buy > 0:
                orders.append(Order(product, my_bid,  max_buy))
            if max_sell > 0:
                orders.append(Order(product, my_ask, -max_sell))

            result[product] = orders

        return result, 0, json.dumps(trader_data)


# ── offline backtest runner (unchanged) ─────────────────────────────────────

def run_all():
    price_frames = [pd.read_csv(f"prices_round_0_day_-{k}.csv", sep=";")
                    for k in (1, 2)]
    trade_frames = [pd.read_csv(f"trades_round_0_day_-{k}.csv", sep=";")
                    for k in (1, 2)]
    for pf, tf, day in zip(price_frames, trade_frames, [-2, -1]):
        pf["__day"] = day
        tf["__day"] = day
    prices_all = pd.concat(price_frames, ignore_index=True)
    trades_all = pd.concat(trade_frames, ignore_index=True)

    reports = []
    for product in ["EMERALDS", "TOMATOES"]:
        for day in [-2, -1]:
            p = prices_all[(prices_all["product"] == product) &
                           (prices_all["__day"] == day)] \
                .sort_values("timestamp").reset_index(drop=True)
            t = trades_all[(trades_all["symbol"] == product) &
                           (trades_all["__day"] == day)] \
                .sort_values("timestamp").reset_index(drop=True)

            if product == "EMERALDS":
                fv = fv_emeralds(p)
                res, c = backtest(p, t, fv,
                                  take_edge=0.0, make_edge=1.0,
                                  position_limit=20, skew_per_unit=0.2)
            else:
                fv = fv_tomatoes(p, window=10)
                res, c = backtest(p, t, fv,
                                  take_edge=1.0, make_edge=1.0,
                                  position_limit=20, skew_per_unit=0.1)

            reports.append(summarize(res, f"{product} day {day}", c))

    print(pd.DataFrame(reports).to_string(index=False))


if __name__ == "__main__":
    run_all()