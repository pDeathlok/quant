from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


TUSHARE_DAILY_COLUMNS = [
    "ts_code",
    "trade_date",
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "change",
    "pct_chg",
    "vol",
    "volume",
    "amount",
]
TUSHARE_MARKET_DAILY_COLUMNS = [*TUSHARE_DAILY_COLUMNS, "name", "industry", "turnover"]


@dataclass(frozen=True)
class DailyRefreshAudit:
    symbol: str
    source: str
    rows: int
    merged_rows: int
    status: str
    error: str | None = None
    attempts: int = 1


def normalize_ts_code(symbol: str) -> str:
    symbol = str(symbol).strip()
    if symbol.endswith((".SH", ".SZ", ".BJ")):
        return symbol
    if symbol.startswith("sh"):
        return f"{symbol[2:]}.SH"
    if symbol.startswith("sz"):
        return f"{symbol[2:]}.SZ"
    if symbol.startswith("bj"):
        return f"{symbol[2:]}.BJ"
    if symbol.startswith("6"):
        return f"{symbol}.SH"
    if symbol.startswith(("0", "3")):
        return f"{symbol}.SZ"
    if symbol.startswith(("4", "8")):
        return f"{symbol}.BJ"
    return symbol


def _normalize_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.strftime("%Y%m%d")


def normalize_tushare_daily(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Normalize Tushare daily data to the project's B1 feature schema."""

    out = df.copy()
    out = out.rename(
        columns={
            "trade_date": "trade_date",
            "ts_code": "ts_code",
            "pct_change": "pct_chg",
        }
    )
    out = out.loc[:, ~out.columns.duplicated()].copy()
    ts_code = normalize_ts_code(symbol)
    if "ts_code" not in out.columns:
        out["ts_code"] = ts_code
    if "symbol" not in out.columns:
        out["symbol"] = out["ts_code"].astype(str)
    if "trade_date" not in out.columns:
        if "date" not in out.columns:
            raise ValueError("Tushare daily data missing trade_date/date")
        out["trade_date"] = _normalize_date(out["date"])
    else:
        out["trade_date"] = out["trade_date"].astype(str).str.replace("-", "", regex=False)
    out["date"] = pd.to_datetime(out["trade_date"], format="%Y%m%d")
    if "vol" in out.columns and "volume" in out.columns:
        out["volume"] = out["volume"].combine_first(out["vol"])
    elif "vol" in out.columns:
        out["volume"] = out["vol"]
    elif "volume" in out.columns:
        out["vol"] = out["volume"]
    out = out.sort_values("date").drop_duplicates("trade_date", keep="last").reset_index(drop=True)
    return _order_columns(out)


def normalize_tushare_market_daily(
    df: pd.DataFrame,
    name_by_symbol: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Normalize a complete Tushare trade-date response without per-symbol loops."""

    out = df.copy().rename(columns={"pct_change": "pct_chg"})
    out = out.loc[:, ~out.columns.duplicated()].copy()
    if "ts_code" not in out.columns:
        raise ValueError("Tushare market daily data missing ts_code")
    if "trade_date" not in out.columns:
        if "date" not in out.columns:
            raise ValueError("Tushare market daily data missing trade_date/date")
        out["trade_date"] = _normalize_date(out["date"])
    else:
        out["trade_date"] = out["trade_date"].astype(str).str.replace("-", "", regex=False)
    out["ts_code"] = out["ts_code"].astype(str).map(normalize_ts_code)
    out["symbol"] = out["ts_code"]
    out["date"] = pd.to_datetime(out["trade_date"], format="%Y%m%d")
    if name_by_symbol is not None:
        out["name"] = out["ts_code"].map(name_by_symbol).fillna(out["ts_code"])
    if "vol" in out.columns and "volume" in out.columns:
        out["volume"] = out["volume"].combine_first(out["vol"])
    elif "vol" in out.columns:
        out["volume"] = out["vol"]
    elif "volume" in out.columns:
        out["vol"] = out["volume"]
    out = (
        out.sort_values(["trade_date", "ts_code"])
        .drop_duplicates(["ts_code", "trade_date"], keep="last")
        .reset_index(drop=True)
    )
    for column in TUSHARE_MARKET_DAILY_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    return out[TUSHARE_MARKET_DAILY_COLUMNS].copy()


def build_tushare_daily_audit(symbol: str, rows: int, merged_rows: int, status: str = "tushare_daily") -> DailyRefreshAudit:
    return DailyRefreshAudit(
        symbol=normalize_ts_code(symbol),
        source="tushare",
        rows=rows,
        merged_rows=merged_rows,
        status=status,
    )


def _order_columns(df: pd.DataFrame) -> pd.DataFrame:
    first = [col for col in TUSHARE_DAILY_COLUMNS if col in df.columns]
    rest = [col for col in df.columns if col not in first]
    return df[first + rest].copy()


def audits_to_frame(audits: Iterable[DailyRefreshAudit]) -> pd.DataFrame:
    rows = []
    for audit in audits:
        rows.append(
            {
                "symbol": audit.symbol,
                "source": audit.source,
                "status": audit.status,
                "rows": audit.rows,
                "merged_rows": audit.merged_rows,
                "error": audit.error,
                "attempts": audit.attempts,
            }
        )
    return pd.DataFrame(rows)
