"""
Round 3 v5 — Long Volatility + Delta Hedge
===========================================
Core insight from 3-day data analysis:
  - Realized vol (RV) = 34.2% annualized
  - Market implied vol = ~21.5% annualized
  - Near/OTM options trade 28-76 pts BELOW Black-Scholes fair value (at RV)
  - The market is massively underpricing volatility.

Strategy per product:
  HYDROGEL_PACK:         EMA-based market making (unchanged from v3)
  VEV_4000, VEV_4500:    Market-make symmetrically (priced at intrinsic, no vol edge)
  VEV_5000, VEV_5100:    Market-make (delta too expensive relative to vol edge)
  VEV_5200:              Small long-vol position (23 contracts, fills delta budget)
  VEV_5300:              Large long-vol position (200 contracts, best edge/delta ratio)
  VEV_5400:              Large long-vol position (200 contracts, 2nd best edge/delta)
  VEV_5500:              Large long-vol position (200 contracts, 3rd best edge/delta)
  VEV_6000, VEV_6500:    Sell only at ask=1 — never buy far-OTM
  VELVETFRUIT_EXTRACT:   Delta-hedge the option book first, then passive MM

Delta budget design:
  VEV position limit = 200. We reserve 190 for hedging, 10 for MM.
  Allocation per strike (sorted by P&L/delta-unit):
    K=5300: 200 contracts × delta 0.423 = 84.6 delta
    K=5400: 200 contracts × delta 0.277 = 55.4 delta
    K=5500: 200 contracts × delta 0.172 = 34.4 delta
    K=5200:  23 contracts × delta 0.604 = 13.9 delta
    ─────────────────────────────────────────────────
    Total delta ≈ 188.3  (safely under 190-unit budget)

Expected P&L (theoretical, per 3-day run):
  K=5300: 200 contracts × 8.24/contract/day × 3 days ≈ 4,944
  K=5400: 200 contracts × 6.19/contract/day × 3 days ≈ 3,714
  K=5500: 200 contracts × 3.52/contract/day × 3 days ≈ 2,112
  K=5200:  23 contracts × 8.15/contract/day × 3 days ≈   562
  ────────────────────────────────────────────────────────────
  Total vol edge ≈ 11,332  (vs v3's +1,579 from pure MM)

TTE model:
  Options have 7 days TTE at the start of Day 0.
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
    """Abramowitz & Stegun approximation, max error 7.5e-8."""
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
    """Black-Scholes European call price. T in years, vol annualised."""
    if T <= 1e-8 or vol <= 0 or S <= 0:
        return max(0.0, S - K)
    sqT = math.sqrt(T)
    d1  = (math.log(S / K) + 0.5 * vol * vol * T) / (vol * sqT)
    d2  = d1 - vol * sqT
    return S * _norm_cdf(d1) - K * _norm_cdf(d2)


def bs_delta(S: float, K: float, T: float, vol: float) -> float:
    """Black-Scholes delta of a European call."""
    if T <= 1e-8 or vol <= 0 or S <= 0:
        return 1.0 if S > K else 0.0
    sqT = math.sqrt(T)
    d1  = (math.log(S / K) + 0.5 * vol * vol * T) / (vol * sqT)
    return _norm_cdf(d1)


# ─── Trader ───────────────────────────────────────────────────────────────────
class Trader:

    # ── Position limits ───────────────────────────────────────────────────────
    LIMIT = {
        "HYDROGEL_PACK":       80,
        "VELVETFRUIT_EXTRACT": 200,
        "VEV_4000": 200, "VEV_4500": 200,
        "VEV_5000": 200, "VEV_5100": 200, "VEV_5200": 200,
        "VEV_5300": 200, "VEV_5400": 200, "VEV_5500": 200,
        "VEV_6000": 200, "VEV_6500": 200,
    }

    # ── Strike values ─────────────────────────────────────────────────────────
    STRIKES = {
        "VEV_4000": 4000, "VEV_4500": 4500,
        "VEV_5000": 5000, "VEV_5100": 5100, "VEV_5200": 5200,
        "VEV_5300": 5300, "VEV_5400": 5400, "VEV_5500": 5500,
        "VEV_6000": 6000, "VEV_6500": 6500,
    }

    # ── Product classification ────────────────────────────────────────────────
    DEEP_ITM  = {"VEV_4000", "VEV_4500"}   # Priced at intrinsic — symmetric MM
    NORMAL_MM = {"VEV_5000", "VEV_5100"}   # Low P&L/delta — MM only, no long bias
    FAR_OTM   = {"VEV_6000", "VEV_6500"}   # Essentially worthless — sell only

    # Long-vol strikes: buy aggressively up to per-strike limit, then delta-hedge.
    # Limits are set to exhaust the 190-unit delta budget optimally.
    # Sorted by P&L per delta unit (best first): 5300, 5400, 5500, 5200
    LONG_VOL_LIMIT = {
        "VEV_5200":  23,   # delta/contract ~0.604  →  14 delta total
        "VEV_5300": 200,   # delta/contract ~0.423  →  85 delta total
        "VEV_5400": 200,   # delta/contract ~0.277  →  55 delta total
        "VEV_5500": 200,   # delta/contract ~0.172  →  34 delta total
    }                      # ─────────────────────────────────────────
                           # Total planned delta  ≈ 188  (budget = 190)

    # ── Volatility parameters ─────────────────────────────────────────────────
    REALIZED_VOL = 0.342   # Annualised RV from 3-day tick data
    # Market IV ≈ 21.5% — options are cheap relative to RV; we buy and hedge.

    # ── Time-to-expiry model ──────────────────────────────────────────────────
    TTE_START_DAYS = 7.0    # Days to expiry at the very start of Day 0
    TICKS_PER_DAY  = 10000  # timestamps 0, 100, …, 999,900 → 10,000 ticks/day

    # ─────────────────────────────────────────────────────────────────────────
    def run(self, state: TradingState):
        orders      = {}
        conversions = 0

        try:
            td = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            td = {}

        ts = state.timestamp

        # Day counter — increments when timestamp resets at start of new day
        day     = td.get("day", 0)
        prev_ts = td.get("prev_ts", -1)
        if ts < prev_ts:
            day += 1
            td["day"] = day
        td["prev_ts"] = ts

        # Time to expiry in years (correct model: 10,000 ticks per day)
        ticks_elapsed = day * self.TICKS_PER_DAY + ts // 100
        T_days  = max(0.001, self.TTE_START_DAYS - ticks_elapsed / self.TICKS_PER_DAY)
        T_years = T_days / 252.0

        # VEV underlying mid — required for BS pricing and delta computation
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
            opt_orders, delta_contrib = self._trade_option(
                state, product, vev_mid, T_years
            )
            orders[product]  = opt_orders
            total_delta     += delta_contrib

        # ── 2. HYDROGEL_PACK — EMA market making ─────────────────────────────
        if "HYDROGEL_PACK" in state.order_depths:
            orders["HYDROGEL_PACK"] = self._trade_hydrogel(state, td)

        # ── 3. VEV — delta-hedge the option book, then MM with remainder ──────
        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            # We're long option delta; short underlying to neutralise.
            hedge_target = max(
                -self.LIMIT["VELVETFRUIT_EXTRACT"],
                min(self.LIMIT["VELVETFRUIT_EXTRACT"], -round(total_delta))
            )
            orders["VELVETFRUIT_EXTRACT"] = self._trade_vev_with_hedge(
                state, hedge_target
            )

        tdo = json.dumps(td)
        logger.flush(state, orders, conversions, tdo)
        return orders, conversions, tdo

    # ═════════════════════════════════════════════════════════════════════════
    #  OPTIONS
    # ═════════════════════════════════════════════════════════════════════════
    def _trade_option(self, state: TradingState, product: str,
                      vev_mid, T_years: float):
        """
        Route each option product to the correct sub-strategy.
        Returns (order_list, delta_contribution_of_current_position).

        delta_contribution = current_pos × bs_delta(S, K, T, RV)
        This is used by the caller to compute the VEV hedge target.
        """
        od     = state.order_depths[product]
        pos    = state.position.get(product, 0)
        strike = self.STRIKES[product]

        best_bid = max(od.buy_orders) if od.buy_orders else None
        best_ask = min(od.sell_orders) if od.sell_orders else None

        # Current position's delta contribution (before this tick's fills)
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
            return self._symmetric_mm(state, product, pos, od, best_bid, best_ask), delta_contribution

        if product in self.LONG_VOL_LIMIT:
            return self._long_vol(state, product, pos, od,
                                  best_bid, best_ask, vev_mid, strike, T_years), delta_contribution

        # Fallback: symmetric MM
        return self._symmetric_mm(state, product, pos, od, best_bid, best_ask), delta_contribution

    # ─── Far OTM: sell only ──────────────────────────────────────────────────
    def _far_otm(self, state, product, pos, od, best_ask):
        """
        VEV_6000 and VEV_6500 always have bid=0, ask=1.
        Collecting 1 tick per contract is safe — the underlying would need
        to move 14%+ above ~5250 to threaten these strikes.
        Never buy: cost is 1 tick, expected payoff is near zero.
        """
        result   = []
        sell_cap = self.LIMIT[product] + pos
        if sell_cap > 0 and best_ask and best_ask > 0:
            result.append(Order(product, best_ask, -sell_cap))
        return result

    # ─── Symmetric market making ─────────────────────────────────────────────
    def _symmetric_mm(self, state, product, pos, od, best_bid, best_ask):
        """
        Standard penny-inside MM used for:
          - Deep ITM (4000, 4500): fairly priced at intrinsic, capture the spread
          - Normal MM (5000, 5100): vol edge too small relative to delta cost
        """
        limit      = self.LIMIT[product]
        buy_cap    = limit - pos
        sell_cap   = limit + pos
        market_mid = (best_bid + best_ask) / 2.0
        result     = []

        # Take orders that cross fair value
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

        # Passive penny-inside quotes
        buy_p  = best_bid + 1 if best_bid + 1 < market_mid else max(0, round(market_mid) - 1)
        sell_p = best_ask - 1 if best_ask - 1 > market_mid else round(market_mid) + 1
        if buy_cap  > 0: result.append(Order(product, max(0, buy_p),  buy_cap))
        if sell_cap > 0: result.append(Order(product, sell_p,        -sell_cap))
        return result

    # ─── Long vol: buy aggressively, never sell below BS fair ────────────────
    def _long_vol(self, state, product, pos, od,
                  best_bid, best_ask, vev_mid, strike, T_years):
        """
        For VEV_5200, 5300, 5400, 5500:
          Market IV (~21.5%) << Realized vol (34.2%).
          Options are cheap — buy up to per-strike position limit.

        Buy side:
          - Take any ask below BS fair value (virtually all of them qualify).
          - Post passive bid at best_bid+1 to fill residual capacity.

        Sell side:
          - Only post at BS fair value or higher.
          - Since the market asks are far below BS fair, our sells almost never
            fill — keeping us long. The sell quote is posted as a safety valve
            in case BS fair drops unexpectedly close to market.
        """
        vol_limit = self.LONG_VOL_LIMIT.get(product, 0)
        buy_cap   = vol_limit - pos          # up to vol-specific limit (not full 200)
        sell_cap  = self.LIMIT[product] + pos  # full sell capacity for safety valve
        result    = []

        # BS fair value using realized vol
        if vev_mid and vev_mid > 0:
            bs_fair = bs_call(vev_mid, strike, T_years, self.REALIZED_VOL)
        else:
            bs_fair = (best_bid + best_ask) / 2.0  # fallback

        # Take: buy any ask that is below BS fair (should always be true here)
        for ask in sorted(od.sell_orders):
            if ask < bs_fair - 0.5 and buy_cap > 0:
                v = min(-od.sell_orders[ask], buy_cap)
                result.append(Order(product, ask, v))
                buy_cap -= v
            else:
                break

        # Passive buy: penny inside best bid
        if buy_cap > 0:
            passive_buy = min(best_bid + 1, max(0, round(bs_fair) - 1))
            if passive_buy > 0:
                result.append(Order(product, passive_buy, buy_cap))

        # Sell safety valve: only at BS fair or above — almost never fills
        sell_p = max(best_ask, round(bs_fair))
        if sell_cap > 0 and sell_p > 0:
            result.append(Order(product, sell_p, -sell_cap))

        return result

    # ═════════════════════════════════════════════════════════════════════════
    #  HYDROGEL_PACK — EMA mid market making
    # ═════════════════════════════════════════════════════════════════════════
    def _trade_hydrogel(self, state: TradingState, td: dict):
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

        # Slow EMA: tracks true fair value while ignoring short-term noise
        ema  = td.get("hp_ema", ob_mid)
        ema  = 0.05 * ob_mid + 0.95 * ema
        td["hp_ema"] = ema
        FAIR = round(ema)

        buy_cap  = limit - pos
        sell_cap = limit + pos

        # Take orders crossing our fair value
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

        # Passive penny-inside quotes
        buy_p  = best_bid + 1 if best_bid + 1 < FAIR else FAIR - 1
        sell_p = best_ask - 1 if best_ask - 1 > FAIR else FAIR + 1
        if buy_cap  > 0: result.append(Order(product, max(0, buy_p),  buy_cap))
        if sell_cap > 0: result.append(Order(product, sell_p,        -sell_cap))
        return result

    # ═════════════════════════════════════════════════════════════════════════
    #  VELVETFRUIT_EXTRACT — Delta hedge (priority) + passive MM (remainder)
    # ═════════════════════════════════════════════════════════════════════════
    def _trade_vev_with_hedge(self, state: TradingState, hedge_target: int):
        """
        Phase 1 — Aggressively reach the delta-neutral hedge target.
                  This is the most important step: being unhedged exposes us
                  to directional risk that can swamp the vol edge.
        Phase 2 — Use remaining position capacity for passive MM.
                  This generates additional spread income on top of the vol P&L.
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

        mid = (best_bid + best_ask) / 2.0

        # Phase 1: take liquidity to reach hedge target
        cur_pos = pos
        diff    = hedge_target - cur_pos

        if diff > 0:      # need to buy VEV (option book is net short delta — unusual)
            for ask in sorted(od.sell_orders):
                if diff <= 0: break
                v = min(-od.sell_orders[ask], diff)
                result.append(Order(product, ask, v))
                diff -= v; cur_pos += v

        elif diff < 0:    # need to sell VEV (option book is net long delta — normal)
            for bid in sorted(od.buy_orders, reverse=True):
                if diff >= 0: break
                v = min(od.buy_orders[bid], -diff)
                result.append(Order(product, bid, -v))
                diff += v; cur_pos -= v

        # Phase 2: passive MM with remaining capacity
        buy_cap  = limit - cur_pos
        sell_cap = limit + cur_pos

        buy_p  = best_bid + 1 if best_bid + 1 < mid else round(mid) - 1
        sell_p = best_ask - 1 if best_ask - 1 > mid else round(mid) + 1

        if buy_cap  > 0: result.append(Order(product, max(0, buy_p),  buy_cap))
        if sell_cap > 0: result.append(Order(product, sell_p,        -sell_cap))
        return result
