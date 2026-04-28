"""
Round 4 v35 — SLURM sweep optimal parameters applied
=====================================================
Base: v34 (473,928 local).

504-combination SLURM sweep on Midway3 identified optimal local parameters.
Sweep dimensions: VEV_5300 floor (7), VEV_5400 floor (4), sell_min (6), buy_max (3).

SWEEP RESULTS (top combo: 474,902 local):
  VEV_5300 floor=47: best avg (473,188) — already in v34, confirmed ✓
  VEV_5400 floor=17: monotonic winner (avg 473,525 vs 473,357 at 15). Higher
                      floor = better quality sells. v34 had 15, now 17.
  VEV_5500 sell_min=6: peaks at avg 473,519 (vs 473,026 at 5, 473,274 at 7).
                        v34 had 5 (from v33), now 6.
  VEV_5500 buy_max=4: clearly best (avg 473,447 vs 472,893 at 3).
                       Being long VEV_5500 has positive mark-to-market; v34's
                       BUY_MAX=3 fix was wrong locally. Reverted to 4.

Bot analysis (v34, from Round 4 local trades):
  Mark 14 added to overlay — buys OTM VEVs at similar prices to Mark 01:
    VEV_5300: 105 units, p75=47, p90=50
    VEV_5400: 48 units,  p75=20
    VEV_5500: 27 units,  p50=7
  Combined M01+M14 at VEV_5300 ≥47: 98 units (+56% vs M01-only 63).
  Combined M01+M14 at VEV_5400 ≥17: 112 units (vs 99 M01-only).

Changes from v34
────────────────
1. VEV_5400 overlay floor: 15 → 17  [sweep optimal; monotonic win]
2. VEV_5500_SELL_MIN: 5 → 6         [sweep optimal; peaks at 6]
3. VEV_5500_BUY_MAX: 3 → 4          [sweep optimal; long pos has +EV locally]

Changes from v32
────────────────
1. MARK01_MIN_BID["VEV_5300"]: 53 → 47  [bot data: p90 of Mark 01's bids]
2. MARK01_MIN_BID["VEV_5400"]: 19 → 15  [bot data: p75 of Mark 01's bids]
3. VEV_5500_BUY_MAX: 0 → 4  [re-enable; SELL_MIN=5 now captures demand]
4. VEV_5500_SELL_MIN: 7 → 5  [Mark 01 bids 5-9; official max bid was 6]

v_over experiment (55,266 official) showed:
  WINNERS  (+1,874 total):
    VEV_5300 EXTREME_SELL 53→46: +1,712  (fired only 13% of ticks at 53=p87;
                                           fires ~35% at 46=p70 of official bids)
    VELVETFRUIT EXTREME_BUY 5,233→5,256:  +162  (was below p10, almost never fired)
  LOSERS   (-1,587 total):
    VEV_5100 SELL 181→177 + BUY 164→167: -297  (tighter spread hurts disabled VEVs)
    VEV_5200 SELL 106→100 + BUY 89→91:   -709  (same reason)
    VEV_5000 minor changes:               -581

v32 = v31 + WINNERS only. Net projected improvement: +1,874 on official.

Changes from v31
────────────────
1. VEV_5300 EXTREME_SELL: 53 → 46  [+1,712 confirmed on official day 3]
   At 53 (p87 of official bids) the signal fired only ~13% of ticks, leaving
   us stuck long at the cap most of the day (buy 1,159 units, sell only 859).
   At 46 (p70) fires ~35% → more cycling, avg sell still 50.2 via mix of
   extreme (46+) and Mark 01 overlay (53+).

2. VELVETFRUIT EXTREME_BUY: 5,233 → 5,256  [+162 confirmed on official day 3]
   Official p10 = 5,249; old threshold 5,233 was below p10 → almost never fired.
   New threshold 5,256 ≈ p30 of official asks → fires ~30% of ticks.

Retained from v31 (all other v_over changes reverted):
   VEV_5100/5200/5000 thresholds unchanged | VEV passive make sizes unchanged
   VEV_5500 disabled | Mark01 overlay | spread overlay | TV-take
   Fix: VEV_5500_BUY_MAX = 0 (strict rule buys nothing), remove VEV_5500
   from EXTREME_BUY (extreme module also buys nothing).

2. Option A — cycling: lower EXTREME_SELL for VEV_5000  [more frequent exits]
   VEV_5000: 272 → 267 (p60 on official; TV-take already handles ≥248 via
                         edge=3, so this adds exits at 267-272 range)
   Local test: +4,845 on VEV_5000. Confirmed win.
   VEV_5300 NOT changed: tested 53→50 but local cost was -11,517 (premature
   sells below market equilibrium). Official prices are higher so VEV_5300=53
   already fires at official p60; lowering further would reduce per-trade PnL.
   VEV_5100/5200 unchanged (disabled products; extreme-only).

3. Option B — passive inside-spread quoting for all VEV products  [path to 280k]
   Leaderboard top (280k) = 2x the DP aggressive-taking ceiling (141k).
   The gap must come from passive market-making that captures aggressive bots.
   New: VEV_MAKE_SIZE dict controls per-product passive quote size.
   New: _add_vev_market_quotes() quotes at best_bid+1 / best_ask-1.
     - Only fires when spread ≥ 3 (room to quote inside without crossing)
     - Capacity computed against hard LIMITS (not SOFT_LIMITS)
     - Runs AFTER _trade_voucher and _trade_extreme so existing orders
       are fully accounted for
   Called for every VEV_* product in run(), including DISABLED_VOUCHERS
   (those skip TV-take but can still passively quote).
   VEV_5500 / VEV_6000 / VEV_6500 have VEV_MAKE_SIZE=0 (excluded).

Architecture (unchanged from v23)
───────────────────────────────────
• EMA fair value (alpha=0.08 HYDROGEL, 0.12 VELVETFRUIT)
• TV-take: buy when ask < fair − edge, sell when bid > fair + edge
• Extreme module: absolute-price mean-reversion, independent of TV
• VEV_5500 strict rule: BUY_MAX=0 (buying disabled), SELL_MIN=7
• VEV_5300/5400 spread overlay
• DISABLED_VOUCHERS: VEV_4000/4500/5100/5200/6000/6500 (extreme-only)
"""

import json
import math
from typing import Dict, List, Optional, Tuple

from datamodel import (
    Listing, Observation, Order, OrderDepth,
    ProsperityEncoder, Symbol, Trade, TradingState,
)


# ── Logger (unchanged from v23) ───────────────────────────────────────────────
class Logger:
    def __init__(self): self.logs = ""; self.max_log_length = 3750
    def print(self, *objects, sep=" ", end="\n"): self.logs += sep.join(map(str, objects)) + end
    def flush(self, state, orders, conversions, trader_data):
        bl = len(self.to_json([self.cs(state, ""), self.co(orders), conversions, "", ""]))
        m = (self.max_log_length - bl) // 3
        print(self.to_json([self.cs(state, self.t(state.traderData, m)), self.co(orders), conversions, self.t(trader_data, m), self.t(self.logs, m)]))
        self.logs = ""
    def cs(self, s, td):
        return [s.timestamp, td, [[l.symbol,l.product,l.denomination] for l in s.listings.values()],
                {k:[v.buy_orders,v.sell_orders] for k,v in s.order_depths.items()},
                [[t.symbol,t.price,t.quantity,t.buyer,t.seller,t.timestamp] for a in s.own_trades.values() for t in a],
                [[t.symbol,t.price,t.quantity,t.buyer,t.seller,t.timestamp] for a in s.market_trades.values() for t in a],
                s.position, self.cobs(s.observations)]
    def cobs(self, obs):
        co = {}
        for p, o in obs.conversionObservations.items():
            co[p] = [o.bidPrice,o.askPrice,o.transportFees,o.exportTariff,o.importTariff,o.sunlight,o.humidity]
        return [obs.plainValueObservations, co]
    def co(self, orders): return [[o.symbol,o.price,o.quantity] for a in orders.values() for o in a]
    def to_json(self, v): return json.dumps(v, cls=ProsperityEncoder, separators=(",",":"))
    def t(self, v, m): return v if len(v) <= m else v[:m-3] + "..."

logger = Logger()


class Trader:

    # ── Hard position limits ──────────────────────────────────────────────────
    LIMITS: Dict[str, int] = {
        "HYDROGEL_PACK":       200,
        "VELVETFRUIT_EXTRACT": 200,
        "VEV_4000": 300, "VEV_4500": 300,
        "VEV_5000": 300, "VEV_5100": 300, "VEV_5200": 300,
        "VEV_5300": 300, "VEV_5400": 300, "VEV_5500": 300,
        "VEV_6000": 300, "VEV_6500": 300,
    }

    # Soft limits — cap for the TV-take module
    SOFT_LIMITS: Dict[str, int] = {
        "VEV_5000": 150,
        "VEV_5300": 300,
        "VEV_5400": 300,
        "VEV_5500": 300,
    }

    # Voucher strikes
    VOUCHER_STRIKES: Dict[str, int] = {
        "VEV_5000": 5000, "VEV_5100": 5100, "VEV_5200": 5200,
        "VEV_5300": 5300, "VEV_5400": 5400, "VEV_5500": 5500,
    }

    # Time values: unchanged from R3_final (confirmed correct for R4 order book)
    VOUCHER_TIME_VALUE: Dict[str, float] = {
        "VEV_5000":  5.0,
        "VEV_5100": 12.0,
        "VEV_5200": 38.5,
        "VEV_5300": 50.0,
        "VEV_5400": 16.0,
        "VEV_5500":  5.0,
    }

    # Products excluded from TV-take (extreme module only)
    DISABLED_VOUCHERS = {"VEV_4000", "VEV_4500", "VEV_5100", "VEV_5200",
                         "VEV_6000", "VEV_6500"}

    # ── TV-take edges ─────────────────────────────────────────────────────────
    TAKE_EDGE_BUY: Dict[str, float] = {
        "HYDROGEL_PACK":       8.0,
        "VELVETFRUIT_EXTRACT": 2.0,
        "VEV_5000":            6.0,
        "VEV_5100":            3.0,   # disabled, kept for reference
        "VEV_5200":            3.0,   # disabled, kept for reference
        "VEV_5300":            3.0,
        "VEV_5400":            1.0,
        "VEV_5500":            0.5,
    }

    TAKE_EDGE_SELL: Dict[str, float] = {
        "HYDROGEL_PACK":       8.0,
        "VELVETFRUIT_EXTRACT": 2.0,
        "VEV_5000":            3.0,
        "VEV_5100":            3.0,   # disabled
        "VEV_5200":            3.0,   # disabled
        "VEV_5300":            2.0,
        "VEV_5400":            1.0,
        "VEV_5500":            0.5,
    }

    MAX_TAKE_SIZE: Dict[str, int] = {
        "HYDROGEL_PACK":       30,
        "VELVETFRUIT_EXTRACT": 50,
        "VEV_5000":            12,
        "VEV_5100":            25,   # disabled
        "VEV_5200":            30,   # disabled
        "VEV_5300":           100,
        "VEV_5400":           100,
        "VEV_5500":            80,
    }

    MAKE_EDGE: Dict[str, int] = {
        "HYDROGEL_PACK":        5,
        "VELVETFRUIT_EXTRACT":  2,
    }

    MAX_MAKE_SIZE: Dict[str, int] = {
        "HYDROGEL_PACK":       30,
        "VELVETFRUIT_EXTRACT": 50,
    }

    EMA_ALPHA: Dict[str, float] = {
        "HYDROGEL_PACK":       0.08,
        "VELVETFRUIT_EXTRACT": 0.12,
    }

    # ── VEV passive market-making (Option B) ─────────────────────────────────
    # Quote at best_bid+1 / best_ask-1 to intercept aggressive bot flow.
    # Only fires when spread ≥ 3. Capacity uses hard LIMITS, not SOFT_LIMITS.
    # VEV_5500 / VEV_6000 / VEV_6500 excluded (0 = no quoting).
    VEV_MAKE_SIZE: Dict[str, int] = {
        "VEV_4000": 10,   # wide spread (~21pt); conservative start
        "VEV_4500": 12,   # wide spread
        "VEV_5000": 12,   # moderate; TV-take already active
        "VEV_5100": 12,   # disabled TV-take; pure passive play
        "VEV_5200": 12,   # disabled TV-take; pure passive play
        "VEV_5300": 15,   # most active VEV; Mark 01 aggressive buyer
        "VEV_5400": 15,   # Mark 01 active; spread typically 2-3pt
        "VEV_5500": 0,    # buying disabled; exclude
        "VEV_6000": 0,    # sell-only; no passive buy
        "VEV_6500": 0,    # sell-only; no passive buy
    }

    # ── Extreme-price module ──────────────────────────────────────────────────
    EXTREME_SELL: Dict[str, int] = {
        "HYDROGEL_PACK":       10_037,   # v29: official p80

        "VELVETFRUIT_EXTRACT":  5_269,
        "VEV_4000":             1_261,
        "VEV_4500":               763,
        "VEV_5000":               267,   # v31: 272→267 (p60ish; cycles faster, TV-take handles ≥248)
        "VEV_5100":               181,
        "VEV_5200":               106,
        "VEV_5300":                46,   # v32: 53→46 (+1,712 official confirmed; p70 bid fires 35% vs 13%)
        "VEV_5400":                19,
        "VEV_6000":                 2,   # sell-only; intrinsic=0 at VF≈5239
        "VEV_6500":                 2,   # sell-only
    }

    EXTREME_BUY: Dict[str, int] = {
        "HYDROGEL_PACK":       10_026,   # v29: official p20
        "VELVETFRUIT_EXTRACT":  5_256,   # v32: 5233→5256 (+162 official; was below p10, almost never fired)
        "VEV_4000":             1_241,
        "VEV_4500":               738,   # v29b: reverted from 759
        "VEV_5000":               256,   # v29: official p20
        "VEV_5100":               164,   # v29: official p20
        "VEV_5200":                89,   # v29: official p20
        "VEV_5300":                39,
        "VEV_5400":                12,
        # VEV_5500 removed: buying disabled (see Change 1)
    }

    EXTREME_CAPS: Dict[str, int] = {
        "HYDROGEL_PACK":       200,   # v29: raised from 180
        "VELVETFRUIT_EXTRACT": 180,
        "VEV_4000":            200,
        "VEV_4500":            200,
        "VEV_5000":            200,
        "VEV_5100":            200,   # v28: raised from 150
        "VEV_5200":            200,   # v28: raised from 150
        "VEV_5300":            250,
        "VEV_5400":            250,
        "VEV_5500":            250,
        "VEV_6000":             50,   # sell-only cap
        "VEV_6500":             50,   # sell-only cap
    }

    EXTREME_MAX_TAKE = 150   # v30: raised from 100

    # ── VEV_5300 / VEV_5400 spread overlay ───────────────────────────────────
    SPREAD_PAIR_MAX_TAKE = 30
    SPREAD_POSITION_CAP  = 150
    SPREAD_HIGH_SELL     = 35.5
    SPREAD_LOW_BUY       = 24.0

    # ── VEV_5500 strict rule ──────────────────────────────────────────────────
    # v31: BUY_MAX = 0 — buying completely disabled.
    # VEV_5500 never sells (max observed bid ≈ 6 < SELL_MIN=7), so accumulating
    # a long position guarantees a loss of ~384 per day. Stop buying entirely.
    VEV_5500_BUY_MAX  = 4   # v35: 3→4 (sweep optimal; long pos has positive MTM locally)
    VEV_5500_SELL_MIN = 6   # v35: 5→6 (sweep optimal; peaks at sell_min=6 across 504 combos)

    # ── Overlay buyer counterparty set ───────────────────────────────────────
    MARK01_IDS = {"Mark 01", "Mark 14"}  # v34: added Mark 14 (buys OTM VEVs at similar prices)
    MARK01_MAX_SELL = 20
    MARK01_MIN_BID: Dict[str, int] = {
        # VEV_5200 overlay deferred to v35: passive quoting counteracts gains locally.
        # Floor=70 on official data (where prices are higher) may work, needs sweep data.
        "VEV_5300":  47,   # sweep confirmed optimal (best avg 473,188 across 504 combos)
        "VEV_5400":  17,   # v35: 15→17 (sweep optimal; monotonic: higher floor = better quality)
        "VEV_5500":   6,   # unchanged
        "VEV_6000":   2,
        "VEV_6500":   2,
    }

    # ─────────────────────────────────────────────────────────────────────────

    def run(self, state: TradingState):
        orders: Dict[str, List[Order]] = {}
        conversions = 0

        try:
            td = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            td = {}

        velvet_fair = self._spot_fair("VELVETFRUIT_EXTRACT", state, td)

        # ── HYDROGEL ──────────────────────────────────────────────────────────
        if "HYDROGEL_PACK" in state.order_depths:
            hyd_fair = self._spot_fair("HYDROGEL_PACK", state, td)
            orders["HYDROGEL_PACK"] = self._trade_spot(
                "HYDROGEL_PACK", state, td, known_fair=hyd_fair)
            self._trade_extreme("HYDROGEL_PACK", state, orders["HYDROGEL_PACK"])

        # ── VELVETFRUIT ───────────────────────────────────────────────────────
        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            orders["VELVETFRUIT_EXTRACT"] = self._trade_spot(
                "VELVETFRUIT_EXTRACT", state, td, known_fair=velvet_fair)
            self._trade_extreme(
                "VELVETFRUIT_EXTRACT", state, orders["VELVETFRUIT_EXTRACT"])

        # ── Vouchers ──────────────────────────────────────────────────────────
        if velvet_fair is not None:
            for product in state.order_depths:
                if product.startswith("VEV_"):
                    orders[product] = self._trade_voucher(
                        product, velvet_fair, state)
                    self._trade_extreme(product, state, orders[product])
                    # Option B: passive inside-spread quoting (runs last, uses
                    # remaining capacity after TV-take + extreme orders)
                    self._add_vev_market_quotes(
                        product, state.order_depths[product], state, orders[product])

            self._trade_5300_5400_spread(state, orders)

        # ── Mark 01 overlay (runs after all other modules) ────────────────────
        self._trade_mark01_overlay(state, orders)

        tdo = json.dumps(td, separators=(",", ":"))
        logger.flush(state, orders, conversions, tdo)
        return orders, conversions, tdo

    # ── Spot: EMA fair + TV-take + passive quotes ─────────────────────────────
    def _trade_spot(
        self, product: str, state: TradingState, td: Dict,
        known_fair: Optional[float] = None
    ) -> List[Order]:
        fair = known_fair if known_fair is not None else self._spot_fair(product, state, td)
        if fair is None:
            return []
        od = state.order_depths[product]
        orders: List[Order] = []
        self._take_mispriced(product, od, fair, state, orders)
        make_edge = self.MAKE_EDGE.get(product)
        if make_edge is not None:
            self._add_passive_quotes(product, od, fair, state, orders, make_edge)
        return orders

    # ── Voucher: intrinsic + fixed time value → TV-take ──────────────────────
    def _trade_voucher(
        self, product: str, underlying_fair: float, state: TradingState
    ) -> List[Order]:
        if product in self.DISABLED_VOUCHERS or product not in self.VOUCHER_STRIKES:
            return []
        if product == "VEV_5500":
            return self._trade_5500_strict(product, state)

        strike     = self.VOUCHER_STRIKES[product]
        time_value = self.VOUCHER_TIME_VALUE[product]
        fair       = max(underlying_fair - strike, 0.0) + time_value
        od         = state.order_depths[product]
        orders: List[Order] = []
        self._take_mispriced(product, od, fair, state, orders)
        return orders

    # ── Extreme-price module ──────────────────────────────────────────────────
    def _trade_extreme(
        self, product: str, state: TradingState, orders: List[Order]
    ) -> None:
        cap = self.EXTREME_CAPS.get(product)
        if cap is None or product not in state.order_depths:
            return

        od = state.order_depths[product]
        best_bid, best_ask = self._best_bid_ask(od)
        pos = state.position.get(product, 0)

        pending_buy  = sum( o.quantity for o in orders if o.quantity > 0)
        pending_sell = sum(-o.quantity for o in orders if o.quantity < 0)

        extreme_buy_room  = max(0, cap - pos - pending_buy)
        extreme_sell_room = max(0, cap + pos - pending_sell)

        sell_thresh = self.EXTREME_SELL.get(product)
        buy_thresh  = self.EXTREME_BUY.get(product)

        # Sell at extreme high
        if (sell_thresh is not None and best_bid is not None
                and best_bid >= sell_thresh and extreme_sell_room > 0):
            qty = min(extreme_sell_room, self.EXTREME_MAX_TAKE,
                      od.buy_orders.get(best_bid, 0))
            if qty > 0:
                orders.append(Order(product, best_bid, -qty))

        # Buy at extreme low
        if (buy_thresh is not None and best_ask is not None
                and best_ask <= buy_thresh and extreme_buy_room > 0):
            qty = min(extreme_buy_room, self.EXTREME_MAX_TAKE,
                      -od.sell_orders.get(best_ask, 0))
            if qty > 0:
                orders.append(Order(product, best_ask, qty))

    # ── VEV_5500 strict rule ──────────────────────────────────────────────────
    def _trade_5500_strict(self, product: str, state: TradingState) -> List[Order]:
        od = state.order_depths[product]
        orders: List[Order] = []
        buy_cap, sell_cap = self._remaining_capacity(product, state, orders)
        max_take = self.MAX_TAKE_SIZE.get(product, 80)

        bought = 0
        for ask_price in sorted(od.sell_orders):
            if ask_price > self.VEV_5500_BUY_MAX or buy_cap <= 0 or bought >= max_take:
                break
            qty = min(-od.sell_orders[ask_price], buy_cap, max_take - bought)
            if qty > 0:
                orders.append(Order(product, ask_price, qty))
                buy_cap -= qty; bought += qty

        pos_after = state.position.get(product, 0) + bought
        if pos_after <= 0:
            return orders

        sold = 0
        for bid_price in sorted(od.buy_orders, reverse=True):
            if bid_price < self.VEV_5500_SELL_MIN or sell_cap <= 0 or sold >= max_take:
                break
            qty = min(od.buy_orders[bid_price], sell_cap, pos_after, max_take - sold)
            if qty > 0:
                orders.append(Order(product, bid_price, -qty))
                sell_cap -= qty; sold += qty; pos_after -= qty

        return orders

    # ── VEV passive inside-spread market-making (Option B) ───────────────────
    def _add_vev_market_quotes(
        self, product: str, od: OrderDepth,
        state: TradingState, orders: List[Order]
    ) -> None:
        """
        Quote at best_bid+1 / best_ask-1 to intercept aggressive bots.
        Only fires when spread ≥ 3 (needs room to quote strictly inside).
        Capacity uses hard LIMITS so TV-take's SOFT_LIMITS don't restrict us.
        """
        size = self.VEV_MAKE_SIZE.get(product, 0)
        if size <= 0:
            return

        best_bid, best_ask = self._best_bid_ask(od)
        if best_bid is None or best_ask is None:
            return
        if best_ask - best_bid < 3:
            return  # spread too tight — can't quote strictly inside

        buy_px  = best_bid + 1
        sell_px = best_ask - 1
        if buy_px >= sell_px:
            return  # sanity check

        limit = self.LIMITS[product]
        pos   = state.position.get(product, 0)
        pending_buy  = sum( o.quantity for o in orders if o.quantity > 0)
        pending_sell = sum(-o.quantity for o in orders if o.quantity < 0)

        buy_cap  = max(0, limit - pos - pending_buy)
        sell_cap = max(0, limit + pos - pending_sell)

        buy_qty  = min(size, buy_cap)
        sell_qty = min(size, sell_cap)

        if buy_qty  > 0: orders.append(Order(product, buy_px,   buy_qty))
        if sell_qty > 0: orders.append(Order(product, sell_px, -sell_qty))

    # ── Mark 01 counterparty overlay ─────────────────────────────────────────
    def _mark01_active(self, product: str, state: TradingState) -> bool:
        return any(
            t.buyer in self.MARK01_IDS
            for t in state.market_trades.get(product, [])
        )

    def _trade_mark01_overlay(
        self, state: TradingState, orders: Dict[str, List[Order]]
    ) -> None:
        for product, floor in self.MARK01_MIN_BID.items():
            if not self._mark01_active(product, state):
                continue
            if product not in state.order_depths:
                continue

            od = state.order_depths[product]
            best_bid, _ = self._best_bid_ask(od)
            if best_bid is None or best_bid < floor:
                continue

            existing = orders.get(product, [])
            _, sell_cap = self._remaining_capacity(product, state, existing)
            avail = od.buy_orders.get(best_bid, 0)
            qty = min(sell_cap, self.MARK01_MAX_SELL, avail)

            if qty > 0:
                product_orders = orders.setdefault(product, [])
                product_orders.append(Order(product, best_bid, -qty))

    # ── VEV_5300 / VEV_5400 spread overlay ───────────────────────────────────
    def _trade_5300_5400_spread(
        self, state: TradingState, result: Dict[str, List[Order]]
    ) -> None:
        p_hi, p_lo = "VEV_5300", "VEV_5400"
        if p_hi not in state.order_depths or p_lo not in state.order_depths:
            return

        od_hi = state.order_depths[p_hi]
        od_lo = state.order_depths[p_lo]
        bid_hi, ask_hi = self._best_bid_ask(od_hi)
        bid_lo, ask_lo = self._best_bid_ask(od_lo)
        if None in (bid_hi, ask_hi, bid_lo, ask_lo):
            return

        sell_spread = bid_hi - ask_lo
        buy_spread  = ask_hi - bid_lo
        spread_pos  = state.position.get(p_hi, 0) - state.position.get(p_lo, 0)
        orders_hi   = result.setdefault(p_hi, [])
        orders_lo   = result.setdefault(p_lo, [])

        if sell_spread >= self.SPREAD_HIGH_SELL and spread_pos > -self.SPREAD_POSITION_CAP:
            qty = min(
                self.SPREAD_PAIR_MAX_TAKE,
                od_hi.buy_orders.get(bid_hi, 0),
                -od_lo.sell_orders.get(ask_lo, 0),
                self.SPREAD_POSITION_CAP + spread_pos,
                self._remaining_capacity(p_hi, state, orders_hi)[1],
                self._remaining_capacity(p_lo, state, orders_lo)[0],
            )
            if qty > 0:
                orders_hi.append(Order(p_hi, bid_hi, -qty))
                orders_lo.append(Order(p_lo, ask_lo,  qty))

        elif buy_spread <= self.SPREAD_LOW_BUY and spread_pos < self.SPREAD_POSITION_CAP:
            qty = min(
                self.SPREAD_PAIR_MAX_TAKE,
                -od_hi.sell_orders.get(ask_hi, 0),
                od_lo.buy_orders.get(bid_lo, 0),
                self.SPREAD_POSITION_CAP - spread_pos,
                self._remaining_capacity(p_hi, state, orders_hi)[0],
                self._remaining_capacity(p_lo, state, orders_lo)[1],
            )
            if qty > 0:
                orders_hi.append(Order(p_hi, ask_hi,  qty))
                orders_lo.append(Order(p_lo, bid_lo, -qty))

    # ── EMA fair value ────────────────────────────────────────────────────────
    def _spot_fair(
        self, product: str, state: TradingState, td: Dict
    ) -> Optional[float]:
        if product not in state.order_depths:
            return None
        od = state.order_depths[product]
        best_bid, best_ask = self._best_bid_ask(od)

        raw: Optional[float] = None
        if best_bid is not None and best_ask is not None:
            bid_vol = max(od.buy_orders.get(best_bid, 0), 0)
            ask_vol = max(-od.sell_orders.get(best_ask, 0), 0)
            raw = (
                (best_bid * ask_vol + best_ask * bid_vol) / (bid_vol + ask_vol)
                if bid_vol + ask_vol > 0
                else (best_bid + best_ask) / 2.0
            )
        elif best_bid is not None:
            raw = float(best_bid)
        elif best_ask is not None:
            raw = float(best_ask)

        if raw is None:
            prev = td.get(product + "_ema")
            return float(prev) if prev is not None else None

        alpha = self.EMA_ALPHA.get(product, 0.10)
        prev  = td.get(product + "_ema")
        ema   = raw if prev is None else alpha * raw + (1.0 - alpha) * float(prev)
        td[product + "_ema"] = ema
        return ema

    # ── Asymmetric take ───────────────────────────────────────────────────────
    def _take_mispriced(
        self, product: str, od: OrderDepth, fair: float,
        state: TradingState, orders: List[Order]
    ) -> None:
        buy_cap, sell_cap = self._remaining_capacity(product, state, orders)
        edge_buy  = self.TAKE_EDGE_BUY.get(product,  999_999.0)
        edge_sell = self.TAKE_EDGE_SELL.get(product, 999_999.0)
        max_take  = self.MAX_TAKE_SIZE.get(product, 0)
        if max_take == 0:
            return

        bought = 0
        for ask_price in sorted(od.sell_orders):
            if ask_price >= fair - edge_buy or buy_cap <= 0 or bought >= max_take:
                break
            qty = min(-od.sell_orders[ask_price], buy_cap, max_take - bought)
            if qty > 0:
                orders.append(Order(product, ask_price, qty))
                buy_cap -= qty; bought += qty

        sold = 0
        for bid_price in sorted(od.buy_orders, reverse=True):
            if bid_price <= fair + edge_sell or sell_cap <= 0 or sold >= max_take:
                break
            qty = min(od.buy_orders[bid_price], sell_cap, max_take - sold)
            if qty > 0:
                orders.append(Order(product, bid_price, -qty))
                sell_cap -= qty; sold += qty

    # ── Passive quoting (HYDROGEL / VELVETFRUIT) ──────────────────────────────
    def _add_passive_quotes(
        self, product: str, od: OrderDepth, fair: float,
        state: TradingState, orders: List[Order], edge: int
    ) -> None:
        best_bid, best_ask = self._best_bid_ask(od)
        if best_bid is None or best_ask is None:
            return
        buy_cap, sell_cap = self._remaining_capacity(product, state, orders)
        max_make = self.MAX_MAKE_SIZE.get(product, 0)
        if max_make <= 0:
            return

        buy_px  = int(math.floor(fair - edge))
        sell_px = int(math.ceil(fair  + edge))

        if best_bid + 1 < fair - 1:
            buy_px = max(buy_px, best_bid + 1)
        if best_ask - 1 > fair + 1:
            sell_px = min(sell_px, best_ask - 1)

        if buy_px  >= best_ask: buy_px  = best_ask - 1
        if sell_px <= best_bid: sell_px = best_bid + 1
        buy_px  = max(0, buy_px)
        sell_px = max(1, sell_px)

        pos   = state.position.get(product, 0)
        limit = self.LIMITS[product]
        buy_scale  = max(0.0, 1.0 - max( pos, 0) / limit)
        sell_scale = max(0.0, 1.0 - max(-pos, 0) / limit)

        buy_qty  = min(buy_cap,  max(1, int(max_make * buy_scale)))
        sell_qty = min(sell_cap, max(1, int(max_make * sell_scale)))

        if buy_qty  > 0: orders.append(Order(product, buy_px,   buy_qty))
        if sell_qty > 0: orders.append(Order(product, sell_px, -sell_qty))

    # ── Remaining capacity ────────────────────────────────────────────────────
    def _remaining_capacity(
        self, product: str, state: TradingState, existing: List[Order]
    ) -> Tuple[int, int]:
        limit    = self.SOFT_LIMITS.get(product, self.LIMITS.get(product, 200))
        position = state.position.get(product, 0)
        pending_buy  = sum( o.quantity for o in existing if o.quantity > 0)
        pending_sell = sum(-o.quantity for o in existing if o.quantity < 0)
        return (max(0, limit - position - pending_buy),
                max(0, limit + position - pending_sell))

    # ── Best bid / ask ────────────────────────────────────────────────────────
    def _best_bid_ask(
        self, od: OrderDepth
    ) -> Tuple[Optional[int], Optional[int]]:
        return (
            max(od.buy_orders)  if od.buy_orders  else None,
            min(od.sell_orders) if od.sell_orders else None,
        )
