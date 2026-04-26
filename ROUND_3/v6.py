"""
Round 3 v6 — Long Volatility + Delta Hedge  (fixes from v5 log analysis)
=========================================================================
v5 post-mortem (394501.log):
  HYDROGEL_PACK:       -9,261  ← EMA drift caused systematic quoting error
  VELVETFRUIT_EXTRACT:      0  ← passive MM orders posted outside market (spread too tight)
  VEV_5300/5400/5500:    -951  ← unhedged because VEV never traded

Root causes fixed in v6:
  Fix 1 — Hydrogel: remove EMA entirely, use FAIR = round(ob_mid) directly.
           v3 (best performer, +1,579) used direct mid. A slow EMA (α=0.05) locks
           onto stale prices when the market drifts, causing us to quote the wrong
           side for hundreds of ticks before catching up.
  Fix 2 — VEV Phase 2: join best_bid / best_ask (not penny-inside). When the
           spread is 1–2 ticks, penny-inside posts outside the market and never
           fills. Joining the book generates actual fills and allows the delta
           hedge to build up properly.

Core insight (unchanged from v5):
  Realized vol = 34.2% annualised.
  Market implied vol ≈ 21.5% annualised.
  Near/OTM options are systemically cheap — buy them and delta-hedge with VEV.

Strategy per product:
  HYDROGEL_PACK:         Direct-mid market making (reverted from EMA)
  VEV_4000, VEV_4500:    Symmetric MM (intrinsic-priced, capture spread)
  VEV_5000, VEV_5100:    Symmetric MM (vol edge too small vs delta cost)
  VEV_5200:              Small long-vol position (23 contracts, fills delta budget)
  VEV_5300:              Large long-vol position (200 contracts, best edge/delta)
  VEV_5400:              Large long-vol position (200 contracts, 2nd best)
  VEV_5500:              Large long-vol position (200 contracts, 3rd best)
  VEV_6000, VEV_6500:    Sell only at ask=1 — never buy far-OTM
  VELVETFRUIT_EXTRACT:   Phase-1 aggressive delta hedge → Phase-2 join book

Delta budget:
  VEV limit = 200. Reserve ~190 for hedging.
  K=5300: 200 × delta 0.423 ≈  85 delta
  K=5400: 200 × delta 0.277 ≈  55 delta
  K=5500: 200 × delta 0.172 ≈  34 delta
  K=5200:  23 × delta 0.604 ≈  14 delta
  ─────────────────────────────────────
  Total planned delta ≈ 188  (budget 190)

TTE model:
  Options have 7 days TTE at start of Day 0.
  Each day has 10,000 ticks (timestamps 0, 100, ..., 999,900).
  T_years = max(0.001, 7 - elapsed_days) / 252
"""

import json
import math
from datamodel import (
    Listing, Observation, Order, OrderDepth,
    ProsperityEncoder, Symbol, Trade, TradingState,
)


# ─── Logger ──────────────────────────────────────────────────────────────────
class Logger:
    def __init__(self): self.logs = ""; self.max_log_length = 3750
    def print(self, *objects, sep=" ", end="\n"): self.logs += sep.join(map(str, objects)) + end
    def flush(self, state, orders, conversions, trader_data):
        bl = len(self.to_json([self.cs(state, ""), self.co(orders), conversions, "", ""]))
        m = (self.max_log_length - bl) // 3
        print(self.to_json([
            self.cs(state, self.t(state.traderData, m)),
            self.co(orders), conversions,
            self.t(trader_data, m), self.t(self.logs, m),
        ]))
        self.logs = ""
    def cs(self, s, td):
        return [s.timestamp, td,
                [[l.symbol, l.product, l.denomination] for l in s.listings.values()],
                {k: [v.buy_orders, v.sell_orders] for k, v in s.order_depths.items()},
                [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
                 for a in s.own_trades.values() for t in a],
                [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
                 for a in s.market_trades.values() for t in a],
                s.position, self.cobs(s.observations)]
    def cobs(self, obs):
        co = {}
        for p, o in obs.conversionObservations.items():
            co[p] = [o.bidPrice, o.askPrice, o.transportFees,
                     o.exportTariff, o.importTariff, o.sunlight, o.humidity]
        return [obs.plainValueObservations, co]
    def co(self, orders): return [[o.symbol, o.price, o.quantity] for a in orders.values() for o in a]
    def to_json(self, v): return json.dumps(v, cls=ProsperityEncoder, separators=(",", ":"))
    def t(self, v, m): return v if len(v) <= m else v[:m - 3] + "..."

logger = Logger()


# ─── Black-Scholes helpers (pure math, no scipy) ─────────────────────────────
def _norm_cdf(x: float) -> float:
    if x < -8: return 0.0
    if x >  8: return 1.0
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p    = 0.3275911
    sign = 1 if x >= 0 else -1
    xa   = abs(x)
    t    = 1.0 / (1.0 + p * xa)
    y    = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t * math.exp(-xa * xa / 2)
    return 0.5 * (1.0 + sign * y)


def bs_call(S: float, K: float, T: float, vol: float) -> float:
    if T <= 1e-8 or vol <= 0 or S <= 0:
        return max(0.0, S - K)
    sqT = math.sqrt(T)
    d1  = (math.log(S / K) + 0.5 * vol * vol * T) / (vol * sqT)
    d2  = d1 - vol * sqT
    return S * _norm_cdf(d1) - K * _norm_cdf(d2)


def bs_delta(S: float, K: float, T: float, vol: float) -> float:
    if T <= 1e-8 or vol <= 0 or S <= 0:
        return 1.0 if S > K else 0.0
    sqT = math.sqrt(T)
    d1  = (math.log(S / K) + 0.5 * vol * vol * T) / (vol * sqT)
    return _norm_cdf(d1)


# ─── Trader ───────────────────────────────────────────────────────────────────
class Trader:

    LIMIT = {
        "HYDROGEL_PACK":       80,
        "VELVETFRUIT_EXTRACT": 200,
        "VEV_4000": 200, "VEV_4500": 200,
        "VEV_5000": 200, "VEV_5100": 200, "VEV_5200": 200,
        "VEV_5300": 200, "VEV_5400": 200, "VEV_5500": 200,
        "VEV_6000": 200, "VEV_6500": 200,
    }

    STRIKES = {
        "VEV_4000": 4000, "VEV_4500": 4500,
        "VEV_5000": 5000, "VEV_5100": 5100, "VEV_5200": 5200,
        "VEV_5300": 5300, "VEV_5400": 5400, "VEV_5500": 5500,
        "VEV_6000": 6000, "VEV_6500": 6500,
    }

    DEEP_ITM  = {"VEV_4000", "VEV_4500"}
    NORMAL_MM = {"VEV_5000", "VEV_5100"}
    FAR_OTM   = {"VEV_6000", "VEV_6500"}

    LONG_VOL_LIMIT = {
        "VEV_5200":  23,
        "VEV_5300": 200,
        "VEV_5400": 200,
        "VEV_5500": 200,
    }

    REALIZED_VOL   = 0.342
    TTE_START_DAYS = 7.0
    TICKS_PER_DAY  = 10000

    # ─────────────────────────────────────────────────────────────────────────
    def run(self, state: TradingState):
        orders      = {}
        conversions = 0

        try:
            td = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            td = {}

        ts = state.timestamp

        # Day counter
        day     = td.get("day", 0)
        prev_ts = td.get("prev_ts", -1)
        if ts < prev_ts:
            day += 1
            td["day"] = day
        td["prev_ts"] = ts

        # Time to expiry in years
        ticks_elapsed = day * self.TICKS_PER_DAY + ts // 100
        T_days  = max(0.001, self.TTE_START_DAYS - ticks_elapsed / self.TICKS_PER_DAY)
        T_years = T_days / 252.0

        # VEV mid for BS pricing
        vev_mid = None
        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            od = state.order_depths["VELVETFRUIT_EXTRACT"]
            if od.buy_orders and od.sell_orders:
                vev_mid = (max(od.buy_orders) + min(od.sell_orders)) / 2.0

        # ── 1. Trade all options; accumulate net portfolio delta ──────────────
        total_delta = 0.0
        for product in state.order_depths:
            if product not in self.STRIKES:
                continue
            opt_orders, delta_contrib = self._trade_option(state, product, vev_mid, T_years)
            orders[product]  = opt_orders
            total_delta     += delta_contrib

        # ── 2. HYDROGEL_PACK — direct-mid market making (FIX 1: no EMA) ──────
        if "HYDROGEL_PACK" in state.order_depths:
            orders["HYDROGEL_PACK"] = self._trade_hydrogel(state)

        # ── 3. VEV — delta-hedge first, then join book for MM ─────────────────
        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            hedge_target = max(
                -self.LIMIT["VELVETFRUIT_EXTRACT"],
                min(self.LIMIT["VELVETFRUIT_EXTRACT"], -round(total_delta))
            )
            orders["VELVETFRUIT_EXTRACT"] = self._trade_vev_with_hedge(state, hedge_target)

        tdo = json.dumps(td)
        logger.flush(state, orders, conversions, tdo)
        return orders, conversions, tdo

    # ═════════════════════════════════════════════════════════════════════════
    #  OPTIONS
    # ═════════════════════════════════════════════════════════════════════════
    def _trade_option(self, state, product, vev_mid, T_years):
        od     = state.order_depths[product]
        pos    = state.position.get(product, 0)
        strike = self.STRIKES[product]

        best_bid = max(od.buy_orders) if od.buy_orders else None
        best_ask = min(od.sell_orders) if od.sell_orders else None

        # Current position delta contribution (used to compute hedge target)
        if vev_mid and vev_mid > 0 and T_years > 1e-6:
            cur_delta = bs_delta(vev_mid, strike, T_years, self.REALIZED_VOL)
        else:
            cur_delta = 1.0 if (vev_mid and vev_mid > strike) else 0.5
        delta_contribution = pos * cur_delta

        if best_bid is None or best_ask is None:
            return [], delta_contribution

        if product in self.FAR_OTM:
            return self._far_otm(state, product, pos, od, best_ask), delta_contribution

        if product in self.DEEP_ITM or product in self.NORMAL_MM:
            return self._symmetric_mm(product, pos, od, best_bid, best_ask), delta_contribution

        if product in self.LONG_VOL_LIMIT:
            return self._long_vol(product, pos, od, best_bid, best_ask,
                                  vev_mid, strike, T_years), delta_contribution

        return self._symmetric_mm(product, pos, od, best_bid, best_ask), delta_contribution

    # ─── Far OTM: sell only ──────────────────────────────────────────────────
    def _far_otm(self, state, product, pos, od, best_ask):
        result   = []
        sell_cap = self.LIMIT[product] + pos
        if sell_cap > 0 and best_ask and best_ask > 0:
            result.append(Order(product, best_ask, -sell_cap))
        return result

    # ─── Symmetric market making ─────────────────────────────────────────────
    def _symmetric_mm(self, product, pos, od, best_bid, best_ask):
        limit      = self.LIMIT[product]
        buy_cap    = limit - pos
        sell_cap   = limit + pos
        market_mid = (best_bid + best_ask) / 2.0
        result     = []

        # Take
        for ask in sorted(od.sell_orders):
            if ask < market_mid and buy_cap > 0:
                v = min(-od.sell_orders[ask], buy_cap)
                result.append(Order(product, ask, v))
                buy_cap -= v
            else:
                break
        for bid in sorted(od.buy_orders, reverse=True):
            if bid > market_mid and sell_cap > 0:
                v = min(od.buy_orders[bid], sell_cap)
                result.append(Order(product, bid, -v))
                sell_cap -= v
            else:
                break

        # Passive: penny-inside (or join if spread is 1)
        buy_p  = best_bid + 1 if best_bid + 1 < market_mid else max(0, round(market_mid) - 1)
        sell_p = best_ask - 1 if best_ask - 1 > market_mid else round(market_mid) + 1
        if buy_cap  > 0: result.append(Order(product, max(0, buy_p),  buy_cap))
        if sell_cap > 0: result.append(Order(product, sell_p,        -sell_cap))
        return result

    # ─── Long vol: buy aggressively up to limit, sell only at BS fair+ ───────
    def _long_vol(self, product, pos, od, best_bid, best_ask, vev_mid, strike, T_years):
        vol_limit = self.LONG_VOL_LIMIT.get(product, 0)
        buy_cap   = vol_limit - pos
        sell_cap  = self.LIMIT[product] + pos
        result    = []

        if vev_mid and vev_mid > 0:
            bs_fair = bs_call(vev_mid, strike, T_years, self.REALIZED_VOL)
        else:
            bs_fair = (best_bid + best_ask) / 2.0

        # Take any ask below BS fair
        for ask in sorted(od.sell_orders):
            if ask < bs_fair - 0.5 and buy_cap > 0:
                v = min(-od.sell_orders[ask], buy_cap)
                result.append(Order(product, ask, v))
                buy_cap -= v
            else:
                break

        # Passive buy at best_bid+1 (but no higher than BS fair - 1)
        if buy_cap > 0:
            passive_buy = min(best_bid + 1, max(0, round(bs_fair) - 1))
            if passive_buy > 0:
                result.append(Order(product, passive_buy, buy_cap))

        # Sell safety valve only at BS fair or above (almost never fills — keeps us long)
        sell_p = max(best_ask, round(bs_fair))
        if sell_cap > 0 and sell_p > 0:
            result.append(Order(product, sell_p, -sell_cap))

        return result

    # ═════════════════════════════════════════════════════════════════════════
    #  HYDROGEL_PACK — Direct-mid MM  (FIX 1: was EMA in v5, caused -9,261)
    # ═════════════════════════════════════════════════════════════════════════
    def _trade_hydrogel(self, state: TradingState):
        """
        Use FAIR = round(ob_mid) directly — identical to v3 which produced +1,579.
        The v5 EMA (alpha=0.05) lagged the market by hundreds of ticks when price
        drifted, causing us to quote aggressively on the wrong side for extended
        periods. Direct mid fixes this instantly.
        """
        product  = "HYDROGEL_PACK"
        od       = state.order_depths[product]
        pos      = state.position.get(product, 0)
        limit    = self.LIMIT[product]
        result   = []

        best_bid = max(od.buy_orders) if od.buy_orders else None
        best_ask = min(od.sell_orders) if od.sell_orders else None
        if best_bid is None or best_ask is None:
            return result

        ob_mid = (best_bid + best_ask) / 2.0
        FAIR   = round(ob_mid)   # FIX: direct mid, no EMA

        buy_cap  = limit - pos
        sell_cap = limit + pos

        # Take
        for ask in sorted(od.sell_orders):
            if ask < FAIR and buy_cap > 0:
                v = min(-od.sell_orders[ask], buy_cap)
                result.append(Order(product, ask, v))
                buy_cap -= v
            else:
                break
        for bid in sorted(od.buy_orders, reverse=True):
            if bid > FAIR and sell_cap > 0:
                v = min(od.buy_orders[bid], sell_cap)
                result.append(Order(product, bid, -v))
                sell_cap -= v
            else:
                break

        # Passive penny-inside
        buy_p  = best_bid + 1 if best_bid + 1 < FAIR else FAIR - 1
        sell_p = best_ask - 1 if best_ask - 1 > FAIR else FAIR + 1
        if buy_cap  > 0: result.append(Order(product, max(0, buy_p),  buy_cap))
        if sell_cap > 0: result.append(Order(product, sell_p,        -sell_cap))
        return result

    # ═════════════════════════════════════════════════════════════════════════
    #  VELVETFRUIT_EXTRACT — Delta hedge then join book  (FIX 2: join, not penny-inside)
    # ═════════════════════════════════════════════════════════════════════════
    def _trade_vev_with_hedge(self, state: TradingState, hedge_target: int):
        """
        Phase 1 — Aggressively reach the delta-neutral hedge target (take liquidity).
        Phase 2 — Join best bid/ask with remaining capacity.
                  FIX: v5 posted penny-inside, which landed outside the market when
                  spread=1, resulting in 0 fills and 0 PnL on VEV. Joining the book
                  ensures actual fills and proper hedge buildup.
        """
        product  = "VELVETFRUIT_EXTRACT"
        od       = state.order_depths[product]
        pos      = state.position.get(product, 0)
        limit    = self.LIMIT[product]
        result   = []

        best_bid = max(od.buy_orders) if od.buy_orders else None
        best_ask = min(od.sell_orders) if od.sell_orders else None
        if best_bid is None or best_ask is None:
            return result

        # Phase 1: take to reach hedge target
        cur_pos = pos
        diff    = hedge_target - cur_pos

        if diff > 0:   # need to buy
            for ask in sorted(od.sell_orders):
                if diff <= 0: break
                v = min(-od.sell_orders[ask], diff)
                result.append(Order(product, ask, v))
                diff -= v; cur_pos += v

        elif diff < 0: # need to sell
            for bid in sorted(od.buy_orders, reverse=True):
                if diff >= 0: break
                v = min(od.buy_orders[bid], -diff)
                result.append(Order(product, bid, -v))
                diff += v; cur_pos -= v

        # Phase 2: join book with remaining capacity (FIX 2)
        buy_cap  = limit - cur_pos
        sell_cap = limit + cur_pos

        if buy_cap  > 0: result.append(Order(product, best_bid,  buy_cap))
        if sell_cap > 0: result.append(Order(product, best_ask, -sell_cap))
        return result
