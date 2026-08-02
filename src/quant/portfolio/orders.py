"""Convert target portfolio weights into executable A-share round-lot orders."""

from __future__ import annotations

from math import floor, isfinite
from typing import Mapping

import pandas as pd


ORDER_COLUMNS = (
    "symbol",
    "side",
    "quantity",
    "price",
    "estimated_value",
    "target_weight",
)


def target_weights_to_orders(
    *,
    target_weights: pd.Series,
    prices: pd.Series,
    current_positions: Mapping[str, int],
    total_equity: float,
    available_cash: float,
    lot_size: int = 100,
    cost_buffer_rate: float = 0.001,
) -> pd.DataFrame:
    """Build sell-first round-lot orders without exceeding estimated cash."""

    if total_equity <= 0:
        raise ValueError("total_equity must be positive")
    if available_cash < 0:
        raise ValueError("available_cash must be non-negative")
    if lot_size <= 0:
        raise ValueError("lot_size must be positive")
    if cost_buffer_rate < 0:
        raise ValueError("cost_buffer_rate must be non-negative")

    targets = pd.to_numeric(pd.Series(target_weights), errors="coerce").fillna(0.0)
    targets.index = targets.index.astype(str)
    targets = targets.groupby(level=0).sum()
    if (targets < 0).any() or float(targets.sum()) > 1 + 1e-8:
        raise ValueError("target_weights must be non-negative and sum to at most 1")
    price_series = pd.to_numeric(pd.Series(prices), errors="coerce")
    price_series.index = price_series.index.astype(str)
    current = {str(symbol): int(quantity) for symbol, quantity in current_positions.items()}
    symbols = targets.index.union(pd.Index(current.keys()))

    desired_shares: dict[str, int] = {}
    for symbol in symbols:
        if symbol not in price_series.index:
            raise ValueError(f"missing price for {symbol}")
        price = float(price_series.loc[symbol])
        if not isfinite(price) or price <= 0:
            raise ValueError(f"invalid price for {symbol}")
        target_value = float(targets.get(symbol, 0.0)) * total_equity
        desired_shares[symbol] = floor(target_value / price / lot_size) * lot_size

    sell_rows: list[dict[str, object]] = []
    buy_requests: list[tuple[str, int]] = []
    estimated_cash = float(available_cash)
    for symbol in sorted(symbols):
        existing = max(0, current.get(symbol, 0))
        difference = desired_shares[symbol] - existing
        price = float(price_series.loc[symbol])
        if difference < 0:
            quantity = min(existing, (-difference // lot_size) * lot_size)
            if quantity > 0:
                estimated_value = quantity * price
                estimated_cash += estimated_value * (1.0 - cost_buffer_rate)
                sell_rows.append(
                    {
                        "symbol": symbol,
                        "side": "sell",
                        "quantity": quantity,
                        "price": price,
                        "estimated_value": estimated_value,
                        "target_weight": float(targets.get(symbol, 0.0)),
                    }
                )
        elif difference > 0:
            buy_requests.append((symbol, (difference // lot_size) * lot_size))

    buy_rows: list[dict[str, object]] = []
    for symbol, requested_quantity in sorted(
        buy_requests,
        key=lambda item: (-float(targets.get(item[0], 0.0)), item[0]),
    ):
        price = float(price_series.loc[symbol])
        affordable = floor(
            estimated_cash / (price * (1.0 + cost_buffer_rate)) / lot_size
        ) * lot_size
        quantity = min(requested_quantity, affordable)
        if quantity <= 0:
            continue
        estimated_value = quantity * price
        estimated_cash -= estimated_value * (1.0 + cost_buffer_rate)
        buy_rows.append(
            {
                "symbol": symbol,
                "side": "buy",
                "quantity": quantity,
                "price": price,
                "estimated_value": estimated_value,
                "target_weight": float(targets.get(symbol, 0.0)),
            }
        )
    return pd.DataFrame(sell_rows + buy_rows, columns=ORDER_COLUMNS)
