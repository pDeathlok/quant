from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd


_COST_COLUMNS = [
    "timestamp",
    "order_id",
    "symbol",
    "commission",
    "stamp_tax",
    "transfer_fee",
    "slippage_cost",
    "total_cost",
]
_COST_COMPONENTS = ["commission", "stamp_tax", "transfer_fee", "slippage_cost"]


def _normalize_series(series: pd.Series, name: str) -> pd.Series:
    normalized = pd.Series(series, copy=True)
    if normalized.empty:
        return pd.Series(dtype=float, name=name)

    normalized = pd.to_numeric(normalized, errors="coerce")
    normalized = normalized.replace([np.inf, -np.inf], np.nan).dropna()
    if normalized.empty:
        return pd.Series(dtype=float, name=name)

    if not isinstance(normalized.index, pd.DatetimeIndex):
        normalized.index = pd.to_datetime(normalized.index, errors="raise")
    normalized = normalized[~normalized.index.duplicated(keep="last")].sort_index()
    normalized = normalized.astype(float)
    normalized.name = name
    return normalized


def _series_attribute(result: Any, attribute: str, name: str) -> pd.Series:
    try:
        value = getattr(result, attribute)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return pd.Series(dtype=float, name=name)
    if isinstance(value, pd.DataFrame) and len(value.columns) == 1:
        value = value.iloc[:, 0]
    if not isinstance(value, pd.Series):
        return pd.Series(dtype=float, name=name)
    return _normalize_series(value, name)


def _frame_attribute(result: Any, attribute: str) -> pd.DataFrame:
    try:
        value = getattr(result, attribute)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return pd.DataFrame()
    if not isinstance(value, pd.DataFrame):
        return pd.DataFrame()
    return value.copy(deep=True).reset_index(drop=True)


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0).astype(float)


def _build_cost_ledger(executions: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    source = executions if not executions.empty else orders
    if source.empty:
        return pd.DataFrame(columns=_COST_COLUMNS)

    timestamp_column = next(
        (
            candidate
            for candidate in ("timestamp", "executed_at", "updated_at", "created_at")
            if candidate in source.columns
        ),
        None,
    )
    order_id_column = "order_id" if "order_id" in source.columns else "id"
    ledger = pd.DataFrame(index=source.index)
    ledger["timestamp"] = (
        source[timestamp_column].copy()
        if timestamp_column is not None
        else pd.Series(pd.NaT, index=source.index)
    )
    ledger["order_id"] = (
        source[order_id_column].astype(str)
        if order_id_column in source.columns
        else pd.Series("", index=source.index, dtype=str)
    )
    ledger["symbol"] = (
        source["symbol"].astype(str)
        if "symbol" in source.columns
        else pd.Series("", index=source.index, dtype=str)
    )
    for component in _COST_COMPONENTS:
        ledger[component] = _numeric_column(source, component)
    ledger["total_cost"] = ledger[_COST_COMPONENTS].sum(axis=1)
    if timestamp_column is not None:
        ledger = ledger.sort_values("timestamp", kind="stable")
    return ledger[_COST_COLUMNS].reset_index(drop=True)


@dataclass(frozen=True)
class BacktestArtifacts:
    """Stable, engine-independent outputs from one completed backtest."""

    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.DataFrame
    orders: pd.DataFrame
    executions: pd.DataFrame
    trades: pd.DataFrame
    costs: pd.DataFrame
    benchmark_returns: Optional[pd.Series] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "equity_curve",
            _normalize_series(self.equity_curve, "equity"),
        )
        object.__setattr__(self, "returns", _normalize_series(self.returns, "return"))
        for attribute in ("positions", "orders", "executions", "trades", "costs"):
            frame = getattr(self, attribute)
            if not isinstance(frame, pd.DataFrame):
                raise TypeError(f"{attribute} must be a pandas DataFrame")
            object.__setattr__(self, attribute, frame.copy(deep=True).reset_index(drop=True))
        if self.benchmark_returns is not None:
            object.__setattr__(
                self,
                "benchmark_returns",
                _normalize_series(self.benchmark_returns, "benchmark_return"),
            )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @classmethod
    def from_akquant(
        cls,
        result: Any,
        benchmark_returns: Optional[pd.Series] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "BacktestArtifacts":
        resolved_metadata = {
            "initial_cash": float(getattr(result, "initial_cash", 0.0) or 0.0)
        }
        if metadata is not None:
            resolved_metadata.update(metadata)

        equity_curve = _series_attribute(result, "equity_curve_daily", "equity")
        returns = _series_attribute(result, "daily_returns", "return")
        if not equity_curve.empty:
            returns = equity_curve.pct_change(fill_method=None).fillna(0.0)
            returns.name = "return"

        executions = _frame_attribute(result, "executions_df")
        orders = _frame_attribute(result, "orders_df")
        return cls(
            equity_curve=equity_curve,
            returns=returns,
            positions=_frame_attribute(result, "positions_df"),
            orders=orders,
            executions=executions,
            trades=_frame_attribute(result, "trades_df"),
            costs=_build_cost_ledger(executions, orders),
            benchmark_returns=benchmark_returns,
            metadata=resolved_metadata,
        )
