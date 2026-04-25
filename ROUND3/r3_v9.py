import json
import math
from typing import Dict, List, Optional, Tuple

from datamodel import Order, OrderDepth, TradingState


class Trader:
    """
    Round 3 v9 proven-core scaling candidate.

    Goal:
    - Cross the 10000 platform PnL milestone by improving the proven core,
      not by chasing tiny side bets.

    Built from r3_v8 / r3_v7 platform baseline:
    - Disable weak vouchers: VEV_4000, VEV_4500, VEV_6000, VEV_6500.
    - Use tiny probe positions in VEV_5100 / VEV_5200 only.
    - Keep VEV_5300 and VEV_5400 core trading from v3.
    - Add small relative-value overlay on VEV_5300 - VEV_5400 spread.
    - Remove zero-bid lottery orders because v8 platform result was identical to v7.
    - Make VELVETFRUIT_EXTRACT spot slightly more active.
    - Add a strict low-price VEV_5500 rule that only buys at <= 5.
    """

    LIMITS: Dict[str, int] = {
        "HYDROGEL_PACK": 200,
        "VELVETFRUIT_EXTRACT": 200,
        "VEV_4000": 300,
        "VEV_4500": 300,
        "VEV_5000": 300,
        "VEV_5100": 300,
        "VEV_5200": 300,
        "VEV_5300": 300,
        "VEV_5400": 300,
        "VEV_5500": 300,
        "VEV_6000": 300,
        "VEV_6500": 300,
    }

    SOFT_LIMITS: Dict[str, int] = {
        "VEV_5000": 80,
        "VEV_5100": 30,
        "VEV_5200": 50,
        "VEV_5300": 300,
        "VEV_5400": 300,
        "VEV_5500": 300,
    }

    VOUCHER_STRIKES: Dict[str, int] = {
        "VEV_5000": 5000,
        "VEV_5100": 5100,
        "VEV_5200": 5200,
        "VEV_5300": 5300,
        "VEV_5400": 5400,
        "VEV_5500": 5500,
    }

    VOUCHER_TIME_VALUE: Dict[str, float] = {
        "VEV_5000": 5.0,
        # Smaller than historical average time value: used as a small short-bias
        # supplement, based on platform v6 where 5100/5200 shorts were positive.
        "VEV_5100": 8.0,
        "VEV_5200": 35.0,
        "VEV_5300": 50.0,
        "VEV_5400": 16.0,
        "VEV_5500": 5.0,
    }

    DISABLED_VOUCHERS = {
        "VEV_4000",
        "VEV_4500",
        "VEV_6000",
        "VEV_6500",
    }

    TAKE_EDGE: Dict[str, float] = {
        "HYDROGEL_PACK": 8.0,
        "VELVETFRUIT_EXTRACT": 2.0,
        "VEV_5000": 6.0,
        "VEV_5100": 3.0,
        "VEV_5200": 3.0,
        "VEV_5300": 3.0,
        "VEV_5400": 1.0,
        "VEV_5500": 0.5,
    }

    MAKE_EDGE: Dict[str, int] = {
        "HYDROGEL_PACK": 10,
        "VELVETFRUIT_EXTRACT": 3,
    }

    MAX_TAKE_SIZE: Dict[str, int] = {
        "HYDROGEL_PACK": 24,
        "VELVETFRUIT_EXTRACT": 30,
        "VEV_5000": 12,
        "VEV_5100": 8,
        "VEV_5200": 10,
        "VEV_5300": 100,
        "VEV_5400": 100,
        "VEV_5500": 80,
    }

    SPREAD_PAIR_MAX_TAKE = 30
    SPREAD_POSITION_CAP = 120
    SPREAD_HIGH_SELL = 36.5
    SPREAD_LOW_BUY = 24.0
    VEV_5500_LOW_BUY_PRICE = 5
    VEV_5500_EXIT_PRICE = 8

    MAX_MAKE_SIZE: Dict[str, int] = {
        "HYDROGEL_PACK": 18,
        "VELVETFRUIT_EXTRACT": 36,
    }

    EMA_ALPHA: Dict[str, float] = {
        "HYDROGEL_PACK": 0.08,
        "VELVETFRUIT_EXTRACT": 0.12,
    }

    def bid(self):
        return 15

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        conversions = 0

        try:
            trader_data = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            trader_data = {}

        velvet_fair = self._spot_fair("VELVETFRUIT_EXTRACT", state, trader_data)

        if "HYDROGEL_PACK" in state.order_depths:
            result["HYDROGEL_PACK"] = self._trade_spot("HYDROGEL_PACK", state, trader_data)

        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            result["VELVETFRUIT_EXTRACT"] = self._trade_spot(
                "VELVETFRUIT_EXTRACT", state, trader_data, velvet_fair
            )

        if velvet_fair is not None:
            for product in state.order_depths:
                if product.startswith("VEV_"):
                    result[product] = self._trade_voucher(product, velvet_fair, state)

            self._trade_5300_5400_spread(state, result)

        trader_data_out = json.dumps(trader_data, separators=(",", ":"))
        return result, conversions, trader_data_out

    def _trade_spot(
        self,
        product: str,
        state: TradingState,
        trader_data: Dict,
        known_fair: Optional[float] = None,
    ) -> List[Order]:
        fair = known_fair if known_fair is not None else self._spot_fair(product, state, trader_data)
        if fair is None:
            return []

        order_depth = state.order_depths[product]
        orders: List[Order] = []
        self._take_mispriced_liquidity(product, order_depth, fair, state, orders)

        make_edge = self.MAKE_EDGE.get(product)
        if make_edge is not None:
            self._add_passive_quotes(product, order_depth, fair, state, orders, make_edge)

        return orders

    def _trade_voucher(self, product: str, underlying_fair: float, state: TradingState) -> List[Order]:
        if product in self.DISABLED_VOUCHERS or product not in self.VOUCHER_STRIKES:
            return []

        if product == "VEV_5500":
            return self._trade_5500_low_price_only(product, state)

        strike = self.VOUCHER_STRIKES[product]
        time_value = self.VOUCHER_TIME_VALUE[product]
        fair = max(underlying_fair - strike, 0.0) + time_value

        order_depth = state.order_depths[product]
        orders: List[Order] = []
        self._take_mispriced_liquidity(product, order_depth, fair, state, orders)
        return orders

    def _trade_5500_low_price_only(self, product: str, state: TradingState) -> List[Order]:
        order_depth = state.order_depths[product]
        orders: List[Order] = []
        buy_capacity, sell_capacity = self._remaining_capacity(product, state, orders)

        bought = 0
        for ask_price in sorted(order_depth.sell_orders):
            if ask_price > self.VEV_5500_LOW_BUY_PRICE or buy_capacity <= 0:
                break
            available = -order_depth.sell_orders[ask_price]
            quantity = min(available, buy_capacity, self.MAX_TAKE_SIZE[product] - bought)
            if quantity <= 0:
                continue
            orders.append(Order(product, ask_price, quantity))
            buy_capacity -= quantity
            bought += quantity
            if bought >= self.MAX_TAKE_SIZE[product]:
                break

        position_after_buys = state.position.get(product, 0) + bought
        if position_after_buys <= 0:
            return orders

        sold = 0
        for bid_price in sorted(order_depth.buy_orders, reverse=True):
            if bid_price < self.VEV_5500_EXIT_PRICE or sell_capacity <= 0:
                break
            available = order_depth.buy_orders[bid_price]
            quantity = min(available, sell_capacity, position_after_buys, self.MAX_TAKE_SIZE[product] - sold)
            if quantity <= 0:
                continue
            orders.append(Order(product, bid_price, -quantity))
            sell_capacity -= quantity
            sold += quantity
            if sold >= self.MAX_TAKE_SIZE[product]:
                break

        return orders

    def _trade_5300_5400_spread(self, state: TradingState, result: Dict[str, List[Order]]) -> None:
        p_hi = "VEV_5300"
        p_lo = "VEV_5400"
        if p_hi not in state.order_depths or p_lo not in state.order_depths:
            return
        od_5300 = state.order_depths[p_hi]
        od_5400 = state.order_depths[p_lo]
        bid_5300, ask_5300 = self._best_bid_ask(od_5300)
        bid_5400, ask_5400 = self._best_bid_ask(od_5400)
        if bid_5300 is None or ask_5300 is None or bid_5400 is None or ask_5400 is None:
            return

        # Executable spread levels, not just mids.
        sell_spread = bid_5300 - ask_5400
        buy_spread = ask_5300 - bid_5400

        spread_pos = state.position.get(p_hi, 0) - state.position.get(p_lo, 0)
        orders_5300 = result.setdefault(p_hi, [])
        orders_5400 = result.setdefault(p_lo, [])

        if sell_spread >= self.SPREAD_HIGH_SELL and spread_pos > -self.SPREAD_POSITION_CAP:
            # Sell rich 5300, buy cheaper hedge 5400.
            qty_cap = min(
                self.SPREAD_PAIR_MAX_TAKE,
                od_5300.buy_orders.get(bid_5300, 0),
                -od_5400.sell_orders.get(ask_5400, 0),
                self.SPREAD_POSITION_CAP + spread_pos,
            )
            sell_capacity_5300 = self._remaining_capacity(p_hi, state, orders_5300)[1]
            buy_capacity_5400 = self._remaining_capacity(p_lo, state, orders_5400)[0]
            qty = min(qty_cap, sell_capacity_5300, buy_capacity_5400)
            if qty > 0:
                orders_5300.append(Order(p_hi, bid_5300, -qty))
                orders_5400.append(Order(p_lo, ask_5400, qty))

        elif buy_spread <= self.SPREAD_LOW_BUY and spread_pos < self.SPREAD_POSITION_CAP:
            # Buy cheap 5300, sell rich hedge 5400.
            qty_cap = min(
                self.SPREAD_PAIR_MAX_TAKE,
                -od_5300.sell_orders.get(ask_5300, 0),
                od_5400.buy_orders.get(bid_5400, 0),
                self.SPREAD_POSITION_CAP - spread_pos,
            )
            buy_capacity_5300 = self._remaining_capacity(p_hi, state, orders_5300)[0]
            sell_capacity_5400 = self._remaining_capacity(p_lo, state, orders_5400)[1]
            qty = min(qty_cap, buy_capacity_5300, sell_capacity_5400)
            if qty > 0:
                orders_5300.append(Order(p_hi, ask_5300, qty))
                orders_5400.append(Order(p_lo, bid_5400, -qty))

    def _spot_fair(self, product: str, state: TradingState, trader_data: Dict) -> Optional[float]:
        if product not in state.order_depths:
            return None

        order_depth = state.order_depths[product]
        best_bid, best_ask = self._best_bid_ask(order_depth)

        raw_fair = None
        if best_bid is not None and best_ask is not None:
            bid_volume = max(order_depth.buy_orders.get(best_bid, 0), 0)
            ask_volume = max(-order_depth.sell_orders.get(best_ask, 0), 0)
            if bid_volume + ask_volume > 0:
                raw_fair = (best_bid * ask_volume + best_ask * bid_volume) / (bid_volume + ask_volume)
            else:
                raw_fair = (best_bid + best_ask) / 2.0
        elif best_bid is not None:
            raw_fair = float(best_bid)
        elif best_ask is not None:
            raw_fair = float(best_ask)

        if raw_fair is None:
            key = product + "_ema"
            previous = trader_data.get(key)
            return float(previous) if previous is not None else None

        key = product + "_ema"
        alpha = self.EMA_ALPHA.get(product, 0.10)
        previous = trader_data.get(key)
        ema = raw_fair if previous is None else alpha * raw_fair + (1.0 - alpha) * float(previous)
        trader_data[key] = ema
        return ema

    def _take_mispriced_liquidity(
        self,
        product: str,
        order_depth: OrderDepth,
        fair: float,
        state: TradingState,
        orders: List[Order],
    ) -> None:
        buy_capacity, sell_capacity = self._remaining_capacity(product, state, orders)
        take_edge = self.TAKE_EDGE.get(product, 999999.0)
        max_take = self.MAX_TAKE_SIZE.get(product, 0)

        bought = 0
        for ask_price in sorted(order_depth.sell_orders):
            if ask_price >= fair - take_edge or buy_capacity <= 0 or bought >= max_take:
                break
            available = -order_depth.sell_orders[ask_price]
            quantity = min(available, buy_capacity, max_take - bought)
            if quantity > 0:
                orders.append(Order(product, ask_price, quantity))
                buy_capacity -= quantity
                bought += quantity

        sold = 0
        for bid_price in sorted(order_depth.buy_orders, reverse=True):
            if bid_price <= fair + take_edge or sell_capacity <= 0 or sold >= max_take:
                break
            available = order_depth.buy_orders[bid_price]
            quantity = min(available, sell_capacity, max_take - sold)
            if quantity > 0:
                orders.append(Order(product, bid_price, -quantity))
                sell_capacity -= quantity
                sold += quantity

    def _add_passive_quotes(
        self,
        product: str,
        order_depth: OrderDepth,
        fair: float,
        state: TradingState,
        orders: List[Order],
        edge: int,
    ) -> None:
        best_bid, best_ask = self._best_bid_ask(order_depth)
        if best_bid is None or best_ask is None:
            return

        buy_capacity, sell_capacity = self._remaining_capacity(product, state, orders)
        max_make = self.MAX_MAKE_SIZE.get(product, 0)
        if max_make <= 0:
            return

        buy_price = int(math.floor(fair - edge))
        sell_price = int(math.ceil(fair + edge))

        if best_bid + 1 < fair - 1:
            buy_price = max(buy_price, best_bid + 1)
        if best_ask - 1 > fair + 1:
            sell_price = min(sell_price, best_ask - 1)

        if buy_price >= best_ask:
            buy_price = best_ask - 1
        if sell_price <= best_bid:
            sell_price = best_bid + 1

        pos = state.position.get(product, 0)
        limit = self.LIMITS[product]
        buy_scale = max(0.0, 1.0 - max(pos, 0) / limit)
        sell_scale = max(0.0, 1.0 - max(-pos, 0) / limit)

        buy_quantity = min(buy_capacity, max(1, int(max_make * buy_scale)))
        sell_quantity = min(sell_capacity, max(1, int(max_make * sell_scale)))

        if buy_quantity > 0:
            orders.append(Order(product, buy_price, buy_quantity))
        if sell_quantity > 0:
            orders.append(Order(product, sell_price, -sell_quantity))

    def _remaining_capacity(
        self,
        product: str,
        state: TradingState,
        existing_orders: List[Order],
    ) -> Tuple[int, int]:
        limit = self.SOFT_LIMITS.get(product, self.LIMITS[product])
        position = state.position.get(product, 0)
        pending_buy = 0
        pending_sell = 0
        for order in existing_orders:
            if order.quantity > 0:
                pending_buy += order.quantity
            else:
                pending_sell += -order.quantity

        buy_capacity = limit - position - pending_buy
        sell_capacity = limit + position - pending_sell
        return max(0, buy_capacity), max(0, sell_capacity)

    def _best_bid_ask(self, order_depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
        best_bid = max(order_depth.buy_orders) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders) if order_depth.sell_orders else None
        return best_bid, best_ask
