from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant.data import TushareDataFetcher
from quant.strategies.convertible_bond.rotation import (
    ConvertibleBondRotationConfig,
    ConvertibleBondSelector,
)
from quant.strategies.convertible_bond.trend_enhanced import (
    ConvertibleBondTrendEnhancedConfig,
    ConvertibleBondTrendEnhancedSelector,
    add_trend_enhanced_features,
)


@dataclass(frozen=True)
class ConvertibleBondBacktestConfig:
    start_date: str = "20180101"
    end_date: str = "20260616"
    rebalance: str = "daily"
    commission_rate: float = 0.0002
    slippage_rate: float = 0.0002
    initial_cash: float = 1_000_000.0
    min_history_trade_dates: int = 20
    selector: ConvertibleBondRotationConfig = ConvertibleBondRotationConfig()


@dataclass(frozen=True)
class ConvertibleBondTrendEnhancedBacktestConfig:
    start_date: str = "20180101"
    end_date: str = "20260616"
    rebalance: str = "daily"
    commission_rate: float = 0.0002
    slippage_rate: float = 0.0002
    initial_cash: float = 1_000_000.0
    min_history_trade_dates: int = 20
    selector: ConvertibleBondTrendEnhancedConfig = ConvertibleBondTrendEnhancedConfig()


@dataclass(frozen=True)
class ConvertibleBondBacktestResult:
    summary: dict[str, Any]
    equity: pd.DataFrame
    targets: pd.DataFrame
    trades: pd.DataFrame
    position_trades: pd.DataFrame | None = None


def normalize_trade_date(value: str | int | pd.Timestamp) -> str:
    parsed = pd.to_datetime(str(value).replace("-", ""), format="%Y%m%d", errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid trade date: {value}")
    return parsed.strftime("%Y%m%d")


def collect_convertible_bond_history(
    start_date: str,
    end_date: str,
    output_dir: Path,
    fetcher: TushareDataFetcher | None = None,
    sleep_seconds: float = 0.25,
    force: bool = False,
) -> dict[str, Path]:
    """Fetch as much Tushare convertible-bond history as possible and cache it locally."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fetcher = fetcher or TushareDataFetcher(cache_dir=output_dir / "tushare_cache")
    start_date = normalize_trade_date(start_date)
    end_date = normalize_trade_date(end_date)

    basic_path = output_dir / "cb_basic_all.parquet"
    call_path = output_dir / f"cb_call_{start_date}_{end_date}.parquet"
    daily_path = output_dir / f"cb_daily_{start_date}_{end_date}.parquet"
    manifest_path = output_dir / f"cb_history_manifest_{start_date}_{end_date}.json"

    if force or not basic_path.exists():
        basic = fetcher.get_cb_basic(list_status="all")
        basic.to_parquet(basic_path, index=False)
    else:
        basic = pd.read_parquet(basic_path)

    if not basic.empty and "list_date" in basic.columns:
        listed_dates = basic["list_date"].dropna().astype(str)
        if not listed_dates.empty:
            start_date = min(start_date, listed_dates.min())

    if force or not call_path.exists():
        call = fetcher.get_cb_call(start_date=start_date, end_date=end_date)
        call.to_parquet(call_path, index=False)

    if force or not daily_path.exists():
        trade_cal = fetcher.get_trade_calendar(start_date=start_date, end_date=end_date)
        if trade_cal.empty or "cal_date" not in trade_cal.columns:
            raise ValueError("Tushare trade_cal returned no open dates")
        dates = trade_cal["cal_date"].dropna().astype(str).sort_values().tolist()
        frames: list[pd.DataFrame] = []
        failed_dates: list[str] = []
        for index, trade_date in enumerate(dates, start=1):
            try:
                frame = fetcher.get_cb_daily(trade_date=trade_date)
                if frame is not None and not frame.empty:
                    frames.append(frame)
            except Exception as exc:
                failed_dates.append(f"{trade_date}: {exc}")
            if sleep_seconds > 0 and index < len(dates):
                time.sleep(sleep_seconds)
        daily = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if not daily.empty:
            daily = daily.drop_duplicates(["ts_code", "trade_date"]).sort_values(
                ["trade_date", "ts_code"]
            )
        daily.to_parquet(daily_path, index=False)
        manifest = {
            "start_date": start_date,
            "end_date": end_date,
            "trade_dates": len(dates),
            "daily_rows": int(len(daily)),
            "failed_dates": failed_dates,
            "paths": {
                "basic": str(basic_path),
                "call": str(call_path),
                "daily": str(daily_path),
            },
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"basic": basic_path, "call": call_path, "daily": daily_path, "manifest": manifest_path}


def load_convertible_bond_history(data_dir: Path, start_date: str, end_date: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    start_date = normalize_trade_date(start_date)
    end_date = normalize_trade_date(end_date)
    daily_path = data_dir / f"cb_daily_{start_date}_{end_date}.parquet"
    call_path = data_dir / f"cb_call_{start_date}_{end_date}.parquet"
    basic_path = data_dir / "cb_basic_all.parquet"
    missing = [path for path in [daily_path, basic_path] if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing convertible-bond history files: {missing_text}")
    daily = pd.read_parquet(daily_path)
    basic = pd.read_parquet(basic_path)
    call = pd.read_parquet(call_path) if call_path.exists() else pd.DataFrame()
    return daily, basic, call


def backtest_convertible_bond_rotation(
    daily: pd.DataFrame,
    basic: pd.DataFrame,
    call: pd.DataFrame | None = None,
    config: ConvertibleBondBacktestConfig | None = None,
) -> ConvertibleBondBacktestResult:
    """Run a close-to-next-close daily rotation backtest using historical Tushare data."""
    config = config or ConvertibleBondBacktestConfig()
    selector = ConvertibleBondSelector(config.selector)
    return _backtest_convertible_bond_selector(
        daily=daily,
        basic=basic,
        call=call,
        config=config,
        selector=selector,
    )


def backtest_convertible_bond_trend_enhanced(
    daily: pd.DataFrame,
    basic: pd.DataFrame,
    call: pd.DataFrame | None = None,
    config: ConvertibleBondTrendEnhancedBacktestConfig | None = None,
) -> ConvertibleBondBacktestResult:
    """Run the trend-enhanced convertible-bond strategy on historical daily bars."""
    config = config or ConvertibleBondTrendEnhancedBacktestConfig()
    featured_daily = daily if _has_trend_enhanced_features(daily) else add_trend_enhanced_features(daily)
    selector = ConvertibleBondTrendEnhancedSelector(config.selector)
    return _backtest_convertible_bond_selector(
        daily=featured_daily,
        basic=basic,
        call=call,
        config=config,
        selector=selector,
    )


def _backtest_convertible_bond_selector(
    daily: pd.DataFrame,
    basic: pd.DataFrame,
    call: pd.DataFrame | None,
    config: ConvertibleBondBacktestConfig | ConvertibleBondTrendEnhancedBacktestConfig,
    selector: ConvertibleBondSelector | ConvertibleBondTrendEnhancedSelector,
) -> ConvertibleBondBacktestResult:
    prepared_daily = _prepare_daily(daily, config.start_date, config.end_date)
    prepared_basic = _prepare_basic(basic)
    prepared_call = _prepare_call(call if call is not None else pd.DataFrame())
    trade_dates = sorted(prepared_daily["trade_date"].dropna().astype(str).unique())
    trade_dates = trade_dates[config.min_history_trade_dates :]
    if len(trade_dates) < 2:
        raise ValueError("Not enough convertible-bond trade dates for backtest")

    weights: dict[str, float] = {}
    previous_close: dict[str, float] = {}
    equity = config.initial_cash
    equity_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []

    for index, trade_date in enumerate(trade_dates):
        day = prepared_daily[prepared_daily["trade_date"] == trade_date].copy()
        close_map = dict(zip(day["ts_code"], day["close"]))
        gross_return = _portfolio_return(weights, previous_close, close_map)
        equity *= 1.0 + gross_return
        active_basic = _active_basic(prepared_basic, trade_date)
        active_call = _active_call(prepared_call, trade_date)
        if _should_rebalance(index, config.rebalance):
            target = selector.target_portfolio(daily=day, basic=active_basic, call=active_call)
            target_weights = (
                dict(zip(target["ts_code"], target["target_weight"])) if not target.empty else {}
            )
            turnover = _turnover(weights, target_weights)
            cost = turnover * (config.commission_rate + config.slippage_rate)
            equity *= 1.0 - cost
            for ts_code in sorted(set(weights) | set(target_weights)):
                current_weight = float(weights.get(ts_code, 0.0))
                target_weight = float(target_weights.get(ts_code, 0.0))
                delta = target_weight - current_weight
                if abs(delta) >= config.selector.rebalance_threshold:
                    trade_rows.append(
                        {
                            "trade_date": trade_date,
                            "ts_code": ts_code,
                            "current_weight": current_weight,
                            "target_weight": target_weight,
                            "delta_weight": delta,
                            "turnover": abs(delta),
                        }
                    )
            if not target.empty:
                for row in target.to_dict(orient="records"):
                    row["trade_date"] = trade_date
                    target_rows.append(row)
            weights = target_weights
        else:
            turnover = 0.0
            cost = 0.0
        equity_rows.append(
            {
                "trade_date": trade_date,
                "equity": equity,
                "daily_return": gross_return - cost,
                "gross_return": gross_return,
                "cost": cost,
                "turnover": turnover,
                "positions": len(weights),
            }
        )
        previous_close = close_map

    equity_frame = pd.DataFrame(equity_rows)
    targets_frame = pd.DataFrame(target_rows)
    trades_frame = pd.DataFrame(trade_rows)
    summary = summarize_backtest(
        equity=equity_frame,
        trades=trades_frame,
        config=config,
        first_source_date=str(prepared_daily["trade_date"].min()),
        last_source_date=str(prepared_daily["trade_date"].max()),
    )
    return ConvertibleBondBacktestResult(
        summary=summary,
        equity=equity_frame,
        targets=targets_frame,
        trades=trades_frame,
    )


def summarize_backtest(
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    config: ConvertibleBondBacktestConfig,
    first_source_date: str,
    last_source_date: str,
) -> dict[str, Any]:
    if equity.empty:
        return {}
    returns = equity["equity"].pct_change().fillna(0.0)
    total_return = float(equity["equity"].iloc[-1] / config.initial_cash - 1.0)
    periods = max(len(equity), 1)
    annual_return = float((1.0 + total_return) ** (252.0 / periods) - 1.0)
    annual_volatility = float(returns.std(ddof=0) * np.sqrt(252))
    sharpe = annual_return / annual_volatility if annual_volatility > 0 else np.nan
    drawdown = equity["equity"] / equity["equity"].cummax() - 1.0
    win_rate = float((returns > 0).mean())
    return {
        "start_date": str(equity["trade_date"].iloc[0]),
        "end_date": str(equity["trade_date"].iloc[-1]),
        "source_start_date": first_source_date,
        "source_end_date": last_source_date,
        "trade_days": int(len(equity)),
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe": float(sharpe) if np.isfinite(sharpe) else None,
        "max_drawdown": float(drawdown.min()),
        "win_rate": win_rate,
        "final_equity": float(equity["equity"].iloc[-1]),
        "average_turnover": float(equity["turnover"].mean()),
        "trade_count": int(len(trades)),
        "config": asdict(config),
        "caveats": [
            "cb_basic rating and remain_size are current/static Tushare fields, not full point-in-time fundamentals.",
            "Signals are formed after each close and applied to the next close-to-close holding period.",
        ],
    }


def write_backtest_outputs(
    result: ConvertibleBondBacktestResult,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    equity_path = output_dir / "equity.csv"
    targets_path = output_dir / "targets.csv"
    trades_path = output_dir / "trades.csv"
    position_trades_path = output_dir / "position_trades.csv"
    summary_path.write_text(
        json.dumps(result.summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    result.equity.to_csv(equity_path, index=False)
    result.targets.to_csv(targets_path, index=False)
    result.trades.to_csv(trades_path, index=False)
    if result.position_trades is not None:
        result.position_trades.to_csv(position_trades_path, index=False)
    return {
        "summary": summary_path,
        "equity": equity_path,
        "targets": targets_path,
        "trades": trades_path,
        "position_trades": position_trades_path,
    }


def _prepare_daily(daily: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    frame = daily.copy()
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["ts_code"] = frame["ts_code"].astype(str).str.upper()
    for column in ["close", "amount", "pct_chg", "bond_over_rate", "premium_rate"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[(frame["trade_date"] >= start_date) & (frame["trade_date"] <= end_date)]
    frame = frame.dropna(subset=["ts_code", "trade_date", "close"])
    return frame.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def _prepare_basic(basic: pd.DataFrame) -> pd.DataFrame:
    frame = basic.copy()
    if frame.empty:
        return frame
    frame["ts_code"] = frame["ts_code"].astype(str).str.upper()
    for column in ["list_date", "delist_date", "conv_start_date", "maturity_date"]:
        if column in frame.columns:
            frame[column] = frame[column].fillna("").astype(str)
    return frame


def _prepare_call(call: pd.DataFrame) -> pd.DataFrame:
    frame = call.copy()
    if frame.empty:
        return frame
    frame["ts_code"] = frame["ts_code"].astype(str).str.upper()
    if "ann_date" in frame.columns:
        frame["ann_date"] = frame["ann_date"].fillna("").astype(str)
    else:
        frame["ann_date"] = ""
    return frame


def _active_basic(basic: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    if basic.empty:
        return basic
    frame = basic.copy()
    if "list_date" in frame.columns:
        frame = frame[(frame["list_date"] == "") | (frame["list_date"] <= trade_date)]
    if "delist_date" in frame.columns:
        frame = frame[(frame["delist_date"] == "") | (frame["delist_date"] > trade_date)]
    return frame


def _active_call(call: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    if call.empty:
        return call
    return call[(call["ann_date"] == "") | (call["ann_date"] <= trade_date)].copy()


def _portfolio_return(
    weights: dict[str, float],
    previous_close: dict[str, float],
    current_close: dict[str, float],
) -> float:
    if not weights or not previous_close:
        return 0.0
    result = 0.0
    for ts_code, weight in weights.items():
        prev = previous_close.get(ts_code)
        current = current_close.get(ts_code)
        if prev is None or current is None or prev <= 0:
            result += weight * 0.0
        else:
            result += weight * (current / prev - 1.0)
    return float(result)


def _turnover(current: dict[str, float], target: dict[str, float]) -> float:
    return float(sum(abs(target.get(code, 0.0) - current.get(code, 0.0)) for code in set(current) | set(target)))


def _should_rebalance(index: int, rebalance: str) -> bool:
    if rebalance == "daily":
        return True
    if rebalance == "weekly":
        return index % 5 == 0
    if rebalance == "monthly":
        return index % 21 == 0
    raise ValueError(f"Unsupported rebalance frequency: {rebalance}")


def _has_trend_enhanced_features(daily: pd.DataFrame) -> bool:
    required = {
        "trend_strength",
        "return_5d",
        "return_1d",
        "six_sword_daily",
        "consecutive_six_sword",
        "band_state",
        "price_position_60d",
        "market_median_double_low",
        "market_trend_20d",
        "market_trend_breadth",
    }
    return required.issubset(set(daily.columns))
