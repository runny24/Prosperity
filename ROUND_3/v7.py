"""
Round 3 v7 — Elias-style MM with intrinsic ITM + soft delta hedge
==================================================================
Lessons from log analysis (v5, v6, elias.py):

  v5 total: -10,020  (EMA hydrogel disaster + long-vol losses)
  v6 total:    -149  (hydrogel fixed, but long-vol still lost)
  elias total: +1,668 (no long-vol, pure MM + intrinsic ITM + soft hedge on VEV)

Why long-vol lost in v5/v6:
  - Buying 5300/5400/5500 options and holding them means theta decay eats
    the position every tick. Even though RV > IV theoretically, the specific
    price path in this dataset did not generate enough gamma income.
  - Our aggressive VEV hedge (crossing spread on every tick) costs the spread
    each time, resulting in net 0 PnL on VEV.

Why Elias wins:
  1. Deep ITM (4000, 4500): use FAIR = intrinsic (VEV - strike).
     Market prices slightly above intrinsic → aggressively SELL the time value.
     Short deep-ITM creates -delta exposure → hedge kicks in buying VEV.
  2. VEV: passive penny-inside MM + soft delta skew (only take aggressively
     when |total_delta| > 20). This generates real VEV income (+509).
  3. OTM (5300–6500): join the book symmetrically. No directional bet.
     Result: ~0 PnL, but zero loss — better than -951 from being long-vol.

Strategy per product (v7):
  HYDROGEL_PACK:   Direct-mid MM (identical to v3, v6, Elias — proven +609)
  VEV_4000/4500:   MM with FAIR = VEV_mid - strike (intrinsic). Sell actively.
  VEV_5000/5100:   MM with FAIR = market mid
  VEV_5200:        MM with FAIR = market mid (not long-vol — avoids -77 loss)
  VEV_5300–6500:   Join book symmetrically (OTM — no directional bet)
  VEV (underlying): Passive penny-inside MM + soft delta hedge via taking threshold
"""

import json, math
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
        print(self.to_json([self.cs(state, self.t(state.traderData, m)), self.co(orders),
                            conversions, self.t(trader_data, m), self.t(self.logs, m)]))
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


# ─── Black-Scholes delta (for soft hedge calculation) ────────────────────────
def _norm_cdf(x):
    if x < -8: return 0.0
    if x >  8: return 1.0
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = 1 if x >= 0 else -1
    xa = abs(x)
    t = 1.0 / (1.0 + p * xa)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t * math.exp(-xa * xa / 2)
    return 0.5 * (1.0 + sign * y)

def bs_delta(S, K, T, vol):
    if T <= 0.0001: return 1.0 if S > K else 0.0
    if S <= 0 or K <= 0 or vol <= 0: return 1.0 if S > K else 0.0
    sqT = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * vol * vol * T) / (vol * sqT)
    return _norm_cdf(d1)


# ─── Trader ───────────────────────────────────────────────────────────────────
class Trader:

    LIMIT = {
        "HYDROGEL_PACK": 80, "VELVETFRUIT_EXTRACT": 200,
        "VEV_4000": 200, "VEV_4500": 200, "VEV_5000": 200, "VEV_5100": 200, "VEV_5200": 200,
        "VEV_5300": 200, "VEV_5400": 200, "VEV_5500": 200, "VEV_6000": 200, "VEV_6500": 200,
    }

    STRIKES = {
        "VEV_4000": 4000, "VEV_4500": 4500, "VEV_5000": 5000, "VEV_5100": 5100, "VEV_5200": 5200,
        "VEV_5300": 5300, "VEV_5400": 5400, "VEV_5500": 5500, "VEV_6000": 6000, "VEV_6500": 6500,
    }

    # Deep ITM: sell at/above intrinsic (VEV - strike) — captures time value
    DEEP_ITM = {"VEV_4000", "VEV_4500"}
    # ATM: symmetric MM around market mid
    ATM = {"VEV_5000", "VEV_5100", "VEV_5200"}
    # OTM and far OTM: join book (no directional bet)
    OTM = {"VEV_5300", "VEV_5400", "VEV_5500", "VEV_6000", "VEV_6500"}

    # Vol used for delta (market IV) — gives correct hedge direction
    VOL = 0.036
    # TTE constants (consistent with 10000 real ticks/day)
    TICKS_PER_DAY = 10000
    TOTAL_TICKS   = 30000   # 3 days × 10000

    def run(self, state: TradingState):
        orders = {}; conversions = 0
        try: td = json.loads(state.traderData) if state.traderData else {}
        except: td = {}

        ts  = state.timestamp
        day = td.get('day', 0)
        prev_ts = td.get('prev_ts', -1)
        if ts < prev_ts:
            day += 1; td['day'] = day
        td['prev_ts'] = ts

        # T as fraction of total sim time (same ratio as Elias's 1000/3000 model)
        tick_in_sim = day * self.TICKS_PER_DAY + ts // 100
        T = max(0.0001, (self.TOTAL_TICKS - tick_in_sim) / self.TOTAL_TICKS)

        # VEV underlying mid
        vev_mid = None
        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            od = state.order_depths["VELVETFRUIT_EXTRACT"]
            if od.buy_orders and od.sell_orders:
                vev_mid = (max(od.buy_orders) + min(od.sell_orders)) / 2.0

        # ── Compute net option delta (for VEV soft hedge) ─────────────────────
        net_option_delta = 0.0
        for sym, strike in self.STRIKES.items():
            pos = state.position.get(sym, 0)
            if pos != 0 and vev_mid:
                net_option_delta += pos * bs_delta(vev_mid, strike, T, self.VOL)

        # ── Trade each product ────────────────────────────────────────────────
        for product in state.order_depths:
            if product == "HYDROGEL_PACK":
                orders[product] = self._trade_hydrogel(state)

            elif product == "VELVETFRUIT_EXTRACT":
                orders[product] = self._trade_vev(state, net_option_delta)

            elif product in self.DEEP_ITM:
                intrinsic = (vev_mid - self.STRIKES[product]) if vev_mid else None
                orders[product] = self._trade_mm(product, state, intrinsic)

            elif product in self.ATM:
                orders[product] = self._trade_mm(product, state, None)

            elif product in self.OTM:
                orders[product] = self._trade_otm(product, state, vev_mid)

        logger.flush(state, orders, conversions, json.dumps(td))
        return orders, conversions, json.dumps(td)

    # ═════════════════════════════════════════════════════════════════════════
    #  HYDROGEL_PACK — direct-mid MM (proven +609 in v6 and Elias)
    # ═════════════════════════════════════════════════════════════════════════
    def _trade_hydrogel(self, state):
        product = "HYDROGEL_PACK"
        od = state.order_depths[product]
        pos = state.position.get(product, 0)
        limit = self.LIMIT[product]

        best_bid = max(od.buy_orders) if od.buy_orders else None
        best_ask = min(od.sell_orders) if od.sell_orders else None
        if best_bid is None or best_ask is None: return []

        ob_mid = (best_bid + best_ask) / 2.0
        FAIR   = round(ob_mid)
        spread = best_ask - best_bid
        result = []

        buy_cap  = limit - pos
        sell_cap = limit + pos

        # Spread-conditioned taking (tighter spread = take more aggressively)
        buy_take  = FAIR + (2 if spread <= 12 else 0)
        sell_take = FAIR - (1 if spread <= 12 else 0)

        for ask in sorted(od.sell_orders):
            if ask < buy_take and buy_cap > 0:
                v = min(-od.sell_orders[ask], buy_cap)
                result.append(Order(product, ask, v)); buy_cap -= v
            else: break
        for bid in sorted(od.buy_orders, reverse=True):
            if bid > sell_take and sell_cap > 0:
                v = min(od.buy_orders[bid], sell_cap)
                result.append(Order(product, bid, -v)); sell_cap -= v
            else: break

        buy_p  = best_bid + 1 if best_bid + 1 < FAIR else FAIR - 1
        sell_p = best_ask - 1 if best_ask - 1 > FAIR else FAIR + 1
        if buy_cap  > 0: result.append(Order(product, buy_p,   buy_cap))
        if sell_cap > 0: result.append(Order(product, sell_p, -sell_cap))
        return result

    # ═════════════════════════════════════════════════════════════════════════
    #  VELVETFRUIT_EXTRACT — passive MM + soft delta skew
    #  (Elias's approach: +509 PnL. Our aggressive hedge gave 0.)
    # ═════════════════════════════════════════════════════════════════════════
    def _trade_vev(self, state, net_option_delta):
        product = "VELVETFRUIT_EXTRACT"
        od  = state.order_depths[product]
        pos = state.position.get(product, 0)
        limit = self.LIMIT[product]

        best_bid = max(od.buy_orders) if od.buy_orders else None
        best_ask = min(od.sell_orders) if od.sell_orders else None
        if best_bid is None or best_ask is None: return []

        mid  = (best_bid + best_ask) / 2.0
        FAIR = round(mid)
        result = []

        # Total portfolio delta: VEV position + option deltas
        total_delta = pos + net_option_delta

        buy_cap  = limit - pos
        sell_cap = limit + pos

        # Soft hedge: widen the take threshold when delta exposure is large
        # (avoids crossing spread on every tick — just biases the natural flow)
        buy_take  = FAIR      # default: take asks strictly below FAIR
        sell_take = FAIR + 1  # default: take bids strictly above FAIR

        if total_delta < -20:
            # Net short delta — buy more aggressively to hedge up
            buy_take = FAIR + 1   # take asks at or below FAIR
        elif total_delta > 20:
            # Net long delta — sell more aggressively to hedge down
            sell_take = FAIR      # take bids at or above FAIR

        for ask in sorted(od.sell_orders):
            if ask < buy_take and buy_cap > 0:
                v = min(-od.sell_orders[ask], buy_cap)
                result.append(Order(product, ask, v)); buy_cap -= v
            else: break
        for bid in sorted(od.buy_orders, reverse=True):
            if bid >= sell_take and sell_cap > 0:
                v = min(od.buy_orders[bid], sell_cap)
                result.append(Order(product, bid, -v)); sell_cap -= v
            else: break

        # Passive penny-inside MM (main income source)
        buy_p  = best_bid + 1 if best_bid + 1 < FAIR else FAIR - 1
        sell_p = best_ask - 1 if best_ask - 1 > FAIR else FAIR + 1
        if buy_cap  > 0: result.append(Order(product, buy_p,   buy_cap))
        if sell_cap > 0: result.append(Order(product, sell_p, -sell_cap))
        return result

    # ═════════════════════════════════════════════════════════════════════════
    #  Generic MM: penny-inside around fair
    #  Used for: Deep ITM (external_fair = intrinsic), ATM (external_fair = None)
    # ═════════════════════════════════════════════════════════════════════════
    def _trade_mm(self, product, state, external_fair):
        od  = state.order_depths[product]
        pos = state.position.get(product, 0)
        limit = self.LIMIT.get(product, 200)

        best_bid = max(od.buy_orders) if od.buy_orders else None
        best_ask = min(od.sell_orders) if od.sell_orders else None
        if best_bid is None or best_ask is None: return []

        market_mid = (best_bid + best_ask) / 2.0
        fair = round(external_fair if external_fair is not None else market_mid)
        result = []

        buy_cap  = limit - pos
        sell_cap = limit + pos

        # Take orders that cross our fair value
        for ask in sorted(od.sell_orders):
            if ask < fair and buy_cap > 0:
                v = min(-od.sell_orders[ask], buy_cap)
                result.append(Order(product, ask, v)); buy_cap -= v
            else: break
        for bid in sorted(od.buy_orders, reverse=True):
            if bid > fair and sell_cap > 0:
                v = min(od.buy_orders[bid], sell_cap)
                result.append(Order(product, bid, -v)); sell_cap -= v
            else: break

        # Passive penny-inside
        buy_p  = best_bid + 1 if best_bid + 1 < fair else fair - 1
        sell_p = best_ask - 1 if best_ask - 1 > fair else fair + 1
        if buy_cap  > 0 and buy_p  >= 0: result.append(Order(product, buy_p,   buy_cap))
        if sell_cap > 0 and sell_p  > 0: result.append(Order(product, sell_p, -sell_cap))
        return result

    # ═════════════════════════════════════════════════════════════════════════
    #  OTM options: join book symmetrically
    #  No directional bet. Avoids -951 loss from long-vol in v5/v6.
    # ═════════════════════════════════════════════════════════════════════════
    def _trade_otm(self, product, state, vev_mid):
        od  = state.order_depths[product]
        pos = state.position.get(product, 0)
        limit = self.LIMIT.get(product, 200)
        strike = self.STRIKES[product]

        best_bid = max(od.buy_orders) if od.buy_orders else None
        best_ask = min(od.sell_orders) if od.sell_orders else None
        if best_bid is None or best_ask is None: return []

        market_mid = (best_bid + best_ask) / 2.0
        spread     = best_ask - best_bid
        intrinsic  = max(0, vev_mid - strike) if vev_mid else 0
        result     = []

        buy_cap  = limit - pos
        sell_cap = limit + pos

        # Take only if clearly mispriced below intrinsic (rarely happens for OTM)
        for ask in sorted(od.sell_orders):
            if ask < intrinsic and buy_cap > 0:
                v = min(-od.sell_orders[ask], buy_cap)
                result.append(Order(product, ask, v)); buy_cap -= v
            else: break
        for bid in sorted(od.buy_orders, reverse=True):
            if bid > market_mid and sell_cap > 0:
                v = min(od.buy_orders[bid], sell_cap)
                result.append(Order(product, bid, -v)); sell_cap -= v
            else: break

        # Post at bid/ask (or penny-inside if spread > 2)
        buy_p  = max(0, best_bid + 1 if spread > 2 else best_bid)
        sell_p = best_ask - 1 if spread > 2 else best_ask
        if buy_cap  > 0 and buy_p  >= 0: result.append(Order(product, buy_p,   buy_cap))
        if sell_cap > 0 and sell_p  > 0: result.append(Order(product, sell_p, -sell_cap))
        return result
