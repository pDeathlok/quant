"""Engine-independent A-share order tradability checks."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class TradabilityDecision:
    allowed: bool
    reason: str | None = None


def _normalize_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.fillna("").astype(str).str.strip().str.lower().isin(
        {"1", "true", "yes", "y"}
    )


class AShareTradabilityPolicy:
    """Fail-closed order gate backed by point-in-time Tushare data."""

    REQUIRED_COLUMNS = {
        "trade_date",
        "ts_code",
        "up_limit",
        "down_limit",
        "is_suspended",
    }

    def __init__(self, data: pd.DataFrame) -> None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("tradability data must be a pandas DataFrame")
        missing = sorted(self.REQUIRED_COLUMNS - set(data.columns))
        if missing:
            raise ValueError(f"tradability data missing columns: {missing}")
        normalized = data.copy()
        normalized["trade_date"] = (
            normalized["trade_date"].astype(str).str.replace("-", "", regex=False)
        )
        normalized["ts_code"] = normalized["ts_code"].astype(str).str.strip()
        normalized["up_limit"] = pd.to_numeric(normalized["up_limit"], errors="coerce")
        normalized["down_limit"] = pd.to_numeric(
            normalized["down_limit"], errors="coerce"
        )
        normalized["is_suspended"] = _normalize_boolean(normalized["is_suspended"])
        if "list_date" not in normalized.columns:
            normalized["list_date"] = ""
        normalized["list_date"] = normalized["list_date"].fillna("").astype(str)
        normalized = normalized.drop_duplicates(
            ["trade_date", "ts_code"], keep="last"
        ).sort_values(["trade_date", "ts_code"])
        self._data = normalized.reset_index(drop=True)
        self._rows = self._data.set_index(["trade_date", "ts_code"], drop=False)

    @classmethod
    def from_directory(
        cls,
        directory: str | Path,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> "AShareTradabilityPolicy":
        root = Path(directory)
        frames: list[pd.DataFrame] = []
        for path in sorted(root.glob("*.parquet")):
            date_text = path.stem
            if start_date is not None and date_text < start_date.replace("-", ""):
                continue
            if end_date is not None and date_text > end_date.replace("-", ""):
                continue
            frames.append(pd.read_parquet(path))
        if not frames:
            raise FileNotFoundError(f"no tradability partitions found in {root}")
        return cls(pd.concat(frames, ignore_index=True, sort=False))

    def check_order(
        self,
        *,
        trade_date: str,
        symbol: str,
        side: str | Any,
        price: float,
    ) -> TradabilityDecision:
        normalized_date = str(trade_date).replace("-", "")
        normalized_symbol = str(symbol).strip()
        normalized_side = str(getattr(side, "value", side)).strip().lower()
        if normalized_side not in {"buy", "sell"}:
            return TradabilityDecision(False, "invalid_side")
        try:
            normalized_price = float(price)
        except (TypeError, ValueError):
            return TradabilityDecision(False, "invalid_price")
        if not isfinite(normalized_price) or normalized_price <= 0:
            return TradabilityDecision(False, "invalid_price")

        key = (normalized_date, normalized_symbol)
        if key not in self._rows.index:
            return TradabilityDecision(False, "missing_tradability_data")
        row = self._rows.loc[key]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        if bool(row["is_suspended"]):
            return TradabilityDecision(False, "suspended")
        list_date = str(row.get("list_date") or "")
        if len(list_date) == 8 and list_date.isdigit() and normalized_date < list_date:
            return TradabilityDecision(False, "not_listed")

        up_limit = float(row["up_limit"])
        down_limit = float(row["down_limit"])
        if not isfinite(up_limit) or not isfinite(down_limit):
            return TradabilityDecision(False, "missing_limit_price")
        tolerance = 1e-9
        if normalized_price > up_limit + tolerance:
            return TradabilityDecision(False, "above_up_limit")
        if normalized_price < down_limit - tolerance:
            return TradabilityDecision(False, "below_down_limit")
        if normalized_side == "buy" and normalized_price >= up_limit - tolerance:
            return TradabilityDecision(False, "buy_at_up_limit")
        if normalized_side == "sell" and normalized_price <= down_limit + tolerance:
            return TradabilityDecision(False, "sell_at_down_limit")
        return TradabilityDecision(True)

    def metadata(self) -> dict[str, object]:
        dates = self._data["trade_date"]
        return {
            "enforcement": "project_order_gate",
            "rows": len(self._data),
            "symbols": int(self._data["ts_code"].nunique()),
            "start_date": str(dates.min()),
            "end_date": str(dates.max()),
        }
