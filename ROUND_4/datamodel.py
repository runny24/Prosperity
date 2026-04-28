"""
Minimal Prosperity datamodel — compatible with v23/v24/v25 trader files.
Drop this file alongside your trader and backtester in the same directory.
"""

import json
from typing import Dict, List, Optional

Symbol = str
UserId = str
Product = str


class Order:
    def __init__(self, symbol: Symbol, price: int, quantity: int) -> None:
        self.symbol   = symbol
        self.price    = price
        self.quantity = quantity

    def __repr__(self):
        side = "BUY" if self.quantity > 0 else "SELL"
        return f"Order({self.symbol}, {side} {abs(self.quantity)}@{self.price})"


class OrderDepth:
    def __init__(self):
        # buy_orders:  price -> positive volume
        # sell_orders: price -> NEGATIVE volume  (Prosperity convention)
        self.buy_orders:  Dict[int, int] = {}
        self.sell_orders: Dict[int, int] = {}


class Trade:
    def __init__(
        self,
        symbol:    Symbol,
        price:     int,
        quantity:  int,
        buyer:     Optional[str] = None,
        seller:    Optional[str] = None,
        timestamp: int = 0,
    ) -> None:
        self.symbol    = symbol
        self.price     = price
        self.quantity  = quantity
        self.buyer     = buyer
        self.seller    = seller
        self.timestamp = timestamp

    def __repr__(self):
        return (f"Trade({self.symbol}, {self.price}, {self.quantity}, "
                f"buyer={self.buyer}, seller={self.seller})")


class Listing:
    def __init__(self, symbol: Symbol, product: Product, denomination: str) -> None:
        self.symbol       = symbol
        self.product      = product
        self.denomination = denomination


class ConversionObservation:
    def __init__(self):
        self.bidPrice      = 0.0
        self.askPrice      = 0.0
        self.transportFees = 0.0
        self.exportTariff  = 0.0
        self.importTariff  = 0.0
        self.sunlight      = 0.0
        self.humidity      = 0.0


class Observation:
    def __init__(self):
        self.plainValueObservations:  Dict[str, float] = {}
        self.conversionObservations:  Dict[str, ConversionObservation] = {}


class TradingState:
    def __init__(
        self,
        timestamp:    int,
        traderData:   str,
        listings:     Dict[Symbol, Listing],
        order_depths: Dict[Symbol, OrderDepth],
        own_trades:   Dict[Symbol, List[Trade]],
        market_trades: Dict[Symbol, List[Trade]],
        position:     Dict[Symbol, int],
        observations: Observation,
    ) -> None:
        self.timestamp     = timestamp
        self.traderData    = traderData
        self.listings      = listings
        self.order_depths  = order_depths
        self.own_trades    = own_trades
        self.market_trades = market_trades
        self.position      = position
        self.observations  = observations


class ProsperityEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            return vars(obj)
        except TypeError:
            return str(obj)
