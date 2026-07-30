from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from quant.data import MarketDataStore, MarketDataStoreConfig
from quant.features.variable_library import build_continuous_ohlc


@dataclass(frozen=True)
class ExitRule:
    name: str
    kind: str
    hold_days: int
    take_profit: float | None = None
    stop_loss: float | None = None
    trail_drawdown: float | None = None
    stop_trigger: str = "intraday"


def add_future_prices(
    candidates: pd.DataFrame,
    daily_dir: Path,
    max_hold_days: int,
) -> pd.DataFrame:
    """Attach adjusted T+1 entry and future OHLC bars to candidate rows."""

    if candidates.empty:
        return candidates.copy()
    prepared = candidates.copy()
    prepared["date"] = pd.to_datetime(prepared["date"])
    prepared["symbol"] = prepared["symbol"].astype(str)
    prepared["_candidate_order"] = np.arange(len(prepared))
    symbols = prepared["symbol"].dropna().unique().tolist()
    start_date = prepared["date"].min().strftime("%Y%m%d")
    end_date = (
        prepared["date"].max() + pd.Timedelta(days=max_hold_days * 3 + 10)
    ).strftime("%Y%m%d")
    store = MarketDataStore(MarketDataStoreConfig.from_env(root=daily_dir.parent))
    market = store.read_market_range(
        daily_dir.name,
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
        columns=[
            "ts_code",
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "pre_close",
        ],
    )
    if market.empty:
        raise RuntimeError(f"No canonical daily rows found under {daily_dir.parent}")
    market["ts_code"] = market["ts_code"].astype(str)
    market_by_symbol = {
        symbol: frame
        for symbol, frame in market.groupby("ts_code", sort=False)
    }
    candidate_by_symbol = {
        symbol: frame
        for symbol, frame in prepared.groupby("symbol", sort=False)
    }
    frames: list[pd.DataFrame] = []

    for count, symbol in enumerate(symbols, start=1):
        raw_daily = market_by_symbol.get(symbol)
        if raw_daily is None or raw_daily.empty:
            continue
        daily = raw_daily.copy()
        daily["date"] = pd.to_datetime(
            daily["trade_date"].astype(str),
            format="%Y%m%d",
            errors="coerce",
        )
        daily["symbol"] = symbol
        if "vol" in daily.columns and "volume" not in daily.columns:
            daily = daily.rename(columns={"vol": "volume"})
        daily = build_continuous_ohlc(
            daily.sort_values("date").reset_index(drop=True)
        )
        daily = daily[["symbol", "date", "open", "high", "low", "close"]].copy()
        future = daily[["symbol", "date"]].copy()
        future["_adjusted_signal_close"] = daily["close"]
        future["entry_open"] = daily["open"].shift(-1)
        for day in range(1, max_hold_days + 2):
            future[f"date_t{day}"] = daily["date"].shift(-day)
            future[f"open_t{day}"] = daily["open"].shift(-day)
            future[f"high_t{day}"] = daily["high"].shift(-day)
            future[f"low_t{day}"] = daily["low"].shift(-day)
            future[f"close_t{day}"] = daily["close"].shift(-day)
        enriched = candidate_by_symbol[symbol].merge(
            future,
            on=["symbol", "date"],
            how="left",
        )
        frames.append(enriched)
        if count % 500 == 0:
            print(f"  future prices: {count}/{len(symbols)} symbols")

    if not frames:
        raise RuntimeError(
            f"No candidate symbols matched canonical daily data under {daily_dir.parent}"
        )

    merged = pd.concat(frames, ignore_index=True).sort_values("_candidate_order")
    if "_adjusted_signal_close" in merged.columns:
        merged["close"] = merged["_adjusted_signal_close"].combine_first(
            merged.get("close")
        )
        merged = merged.drop(columns=["_adjusted_signal_close"])
    return (
        merged.drop(columns=["_candidate_order"])
        .dropna(subset=["entry_open"])
        .copy()
    )


def simulate_exit(df: pd.DataFrame, rule: ExitRule) -> pd.DataFrame:
    """Simulate one exit policy against attached future OHLC bars."""

    if rule.stop_trigger not in {"intraday", "close"}:
        raise ValueError(f"Unsupported stop_trigger: {rule.stop_trigger}")

    entry = df["entry_open"].to_numpy(dtype=float)
    count = len(df)
    returns = np.full(count, np.nan)
    exit_day = np.full(count, -1, dtype=int)
    exit_date = np.full(count, np.datetime64("NaT"), dtype="datetime64[ns]")
    exit_type = np.full(count, "unknown", dtype=object)
    active = np.zeros(count, dtype=bool)
    peak = np.zeros(count, dtype=float)
    open_mask = ~np.isnan(entry)

    for day in range(2, rule.hold_days + 2):
        unresolved = open_mask & np.isnan(returns)
        if not unresolved.any():
            break

        open_ = df[f"open_t{day}"].to_numpy(dtype=float)
        high = df[f"high_t{day}"].to_numpy(dtype=float)
        low = df[f"low_t{day}"].to_numpy(dtype=float)
        close = df[f"close_t{day}"].to_numpy(dtype=float)
        date_t = pd.to_datetime(df[f"date_t{day}"]).to_numpy(
            dtype="datetime64[ns]"
        )
        valid_day = (
            unresolved
            & ~np.isnan(open_)
            & ~np.isnan(high)
            & ~np.isnan(low)
            & ~np.isnan(close)
        )
        if not valid_day.any():
            continue

        if rule.kind == "expiry" and rule.stop_loss is None:
            if day == rule.hold_days + 1:
                returns[valid_day] = close[valid_day] / entry[valid_day] - 1
                exit_day[valid_day] = day
                exit_date[valid_day] = date_t[valid_day]
                exit_type[valid_day] = "expiry"
            continue

        stop_price = entry * (1 - (rule.stop_loss or 0))
        stop_hit = (
            valid_day & (close <= stop_price)
            if rule.stop_trigger == "close"
            else valid_day & (low <= stop_price)
        )
        if stop_hit.any():
            if rule.stop_trigger == "close":
                returns[stop_hit] = close[stop_hit] / entry[stop_hit] - 1
            else:
                gap_stop = stop_hit & (open_ <= stop_price)
                normal_stop = stop_hit & ~gap_stop
                returns[gap_stop] = open_[gap_stop] / entry[gap_stop] - 1
                returns[normal_stop] = (
                    stop_price[normal_stop] / entry[normal_stop] - 1
                )
            exit_day[stop_hit] = day
            exit_date[stop_hit] = date_t[stop_hit]
            exit_type[stop_hit] = "stop_loss"

        still_open = valid_day & np.isnan(returns)
        if not still_open.any():
            continue

        if rule.kind == "fixed":
            take_profit_hit = still_open & (
                high >= entry * (1 + (rule.take_profit or 0))
            )
            if take_profit_hit.any():
                returns[take_profit_hit] = rule.take_profit or 0
                exit_day[take_profit_hit] = day
                exit_date[take_profit_hit] = date_t[take_profit_hit]
                exit_type[take_profit_hit] = "take_profit"
        elif rule.kind == "trailing":
            peak[still_open] = np.maximum(peak[still_open], high[still_open])
            target_hit = still_open & (
                peak >= entry * (1 + (rule.take_profit or 0))
            )
            active |= target_hit
            trailing_price = peak * (1 - (rule.trail_drawdown or 0))
            trailing_hit = still_open & active & (low <= trailing_price)
            if trailing_hit.any():
                gap_trailing = trailing_hit & (open_ <= trailing_price)
                normal_trailing = trailing_hit & ~gap_trailing
                returns[gap_trailing] = (
                    open_[gap_trailing] / entry[gap_trailing] - 1
                )
                returns[normal_trailing] = (
                    trailing_price[normal_trailing] / entry[normal_trailing] - 1
                )
                exit_day[trailing_hit] = day
                exit_date[trailing_hit] = date_t[trailing_hit]
                exit_type[trailing_hit] = "trailing_stop"

        expiry = (
            valid_day
            & np.isnan(returns)
            & (day == rule.hold_days + 1)
        )
        if expiry.any():
            returns[expiry] = close[expiry] / entry[expiry] - 1
            exit_day[expiry] = day
            exit_date[expiry] = date_t[expiry]
            exit_type[expiry] = "expiry"

    result = df[["date", "symbol"]].copy()
    result["return_pct"] = returns * 100
    result["exit_day"] = exit_day
    result["exit_date"] = exit_date
    result["exit_type"] = exit_type
    return result.dropna(subset=["return_pct"])


def max_drawdown_from_daily_returns(daily_returns_pct: pd.Series) -> float:
    equity = (1 + daily_returns_pct / 100).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1
    return float(drawdown.min() * 100)


def summarize_returns(trades: pd.DataFrame) -> dict[str, float | int]:
    returns = trades["return_pct"].dropna()
    if returns.empty:
        return {}
    daily = trades.groupby("date")["return_pct"].mean().sort_index()
    wins = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    return {
        "trades": int(len(returns)),
        "days": int(daily.shape[0]),
        "avg_return_pct": float(returns.mean()),
        "median_return_pct": float(returns.median()),
        "win_rate": float((returns > 0).mean()),
        "p10_return_pct": float(returns.quantile(0.10)),
        "p90_return_pct": float(returns.quantile(0.90)),
        "daily_avg_pct": float(daily.mean()),
        "daily_sharpe": (
            float(np.sqrt(244) * daily.mean() / daily.std())
            if daily.std()
            else np.nan
        ),
        "max_drawdown_pct": max_drawdown_from_daily_returns(daily),
        "profit_factor": float(wins / losses) if losses > 0 else np.nan,
        "stop_rate": float((trades["exit_type"] == "stop_loss").mean()),
        "take_profit_rate": float(
            (trades["exit_type"] == "take_profit").mean()
        ),
        "trailing_stop_rate": float(
            (trades["exit_type"] == "trailing_stop").mean()
        ),
        "expiry_rate": float((trades["exit_type"] == "expiry").mean()),
    }
