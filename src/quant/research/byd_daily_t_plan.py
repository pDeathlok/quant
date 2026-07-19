"""Daily, pre-market BYD T plan selected from sideways-era history.

The live plan never needs intraday updates. Completed five-minute bars are
used only to validate whether an entry happened before its target and to fit a
small, deliberately constrained decision tree. The selected positive-T rule
is trained from 2022 onward so the 2020-2021 BYD bull run cannot dominate it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from quant.research.byd_t_backtest import ChinaAStockFees


Direction = Literal["positive", "reverse"]
FEATURE_WINDOWS = (1, 2, 3, 5, 10, 20, 40, 60)
TREND_WINDOWS = (5, 10, 20, 40, 60)
FEATURE_COLUMNS = tuple(
    [f"ret{window}" for window in FEATURE_WINDOWS]
    + [
        feature
        for window in TREND_WINDOWS
        for feature in (f"gap{window}", f"slope{window}", f"pos{window}")
    ]
    + ["body", "range", "vol5", "atrp", "rsi"]
)


@dataclass(frozen=True)
class DailyTConfig:
    direction: Direction
    entry_deviation: float
    target_deviation: float
    shares: int = 500
    model_max_depth: int = 3
    model_min_samples_leaf: int = 20
    score_threshold: float = 60.0


POSITIVE_T_CONFIG = DailyTConfig(
    direction="positive",
    entry_deviation=0.008,
    target_deviation=0.010,
)


def normalize_daily_bars(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize completed daily bars without introducing future values."""
    required = {"open", "high", "low", "close"}
    missing = required.difference(raw.columns)
    if "date" not in raw.columns and "trade_date" not in raw.columns:
        missing.add("date")
    if missing:
        raise ValueError(f"daily bars missing columns: {sorted(missing)}")
    out = raw.copy()
    parsed_date = pd.to_datetime(
        out["date"] if "date" in out.columns else pd.Series(pd.NaT, index=out.index),
        errors="coerce",
    )
    if "trade_date" in out.columns:
        trade_date = pd.to_datetime(
            out["trade_date"].astype(str), format="%Y%m%d", errors="coerce"
        )
        parsed_date = parsed_date.fillna(trade_date)
    out["date"] = parsed_date.dt.normalize()
    for column in ["open", "high", "low", "close", "volume"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    if "volume" not in out.columns:
        out["volume"] = 1.0
    return (
        out.dropna(subset=["date", "open", "high", "low", "close"])
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .set_index("date")
    )


def daily_t_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Return features known after each completed daily session."""
    daily = normalize_daily_bars(raw)
    features = pd.DataFrame(index=daily.index)
    close = daily["close"]
    for window in FEATURE_WINDOWS:
        features[f"ret{window}"] = close.pct_change(window)
    for window in TREND_WINDOWS:
        average = close.rolling(window, min_periods=window).mean()
        rolling_low = daily["low"].rolling(window, min_periods=window).min()
        rolling_high = daily["high"].rolling(window, min_periods=window).max()
        features[f"gap{window}"] = close / average - 1
        features[f"slope{window}"] = average.pct_change(5)
        features[f"pos{window}"] = (
            (close - rolling_low) / (rolling_high - rolling_low).replace(0, np.nan)
        )
    features["body"] = close / daily["open"].replace(0, np.nan) - 1
    features["range"] = daily["high"] / daily["low"].replace(0, np.nan) - 1
    features["vol5"] = (
        daily["volume"] / daily["volume"].rolling(5, min_periods=5).mean() - 1
    )
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            daily["high"] - daily["low"],
            (daily["high"] - previous_close).abs(),
            (daily["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    features["atrp"] = true_range.rolling(14, min_periods=14).mean() / close
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    features["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    return features.loc[:, FEATURE_COLUMNS]


def daily_from_intraday(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate completed intraday bars into a feature-compatible frame."""
    required = {"datetime", "open", "high", "low", "close"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"intraday bars missing columns: {sorted(missing)}")
    bars = raw.copy()
    bars["datetime"] = pd.to_datetime(bars["datetime"], errors="coerce")
    bars = bars.dropna(subset=["datetime", "open", "high", "low", "close"])
    bars["date"] = bars["datetime"].dt.normalize()
    if "volume" not in bars.columns:
        bars["volume"] = 1.0
    return (
        bars.sort_values("datetime")
        .groupby("date", as_index=False)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
    )


def simulate_daily_t(
    raw: pd.DataFrame,
    config: DailyTConfig,
    fees: ChinaAStockFees = ChinaAStockFees(),
) -> pd.DataFrame:
    """Simulate one fixed pre-market T opportunity per completed session."""
    bars = raw.copy()
    bars["datetime"] = pd.to_datetime(bars["datetime"], errors="coerce")
    bars = bars.dropna(subset=["datetime", "open", "high", "low", "close"])
    bars["date"] = bars["datetime"].dt.normalize()
    bars = bars.sort_values("datetime")
    daily = daily_from_intraday(bars).set_index("date")
    previous_close = daily["close"].shift(1)
    grouped = {
        day: frame[["open", "high", "low", "close"]].to_numpy(dtype=float)
        for day, frame in bars.groupby("date", sort=True)
    }
    records: list[dict[str, Any]] = []
    for day, reference in previous_close.items():
        if not np.isfinite(reference):
            continue
        day_bars = grouped[day]
        if config.direction == "positive":
            entry_line = float(reference) * (1 - config.entry_deviation)
            entries = np.flatnonzero(day_bars[:, 2] <= entry_line)
        else:
            entry_line = float(reference) * (1 + config.entry_deviation)
            entries = np.flatnonzero(day_bars[:, 1] >= entry_line)
        if not len(entries):
            continue
        entry_index = int(entries[0])
        entry_open = float(day_bars[entry_index, 0])
        later_bars = day_bars[entry_index + 1 :]
        if config.direction == "positive":
            entry_raw = min(entry_open, entry_line)
            target_raw = entry_raw * (1 + config.target_deviation)
            target_hit = bool(len(later_bars) and np.any(later_bars[:, 1] >= target_raw))
            exit_raw = target_raw if target_hit else float(day_bars[-1, 3])
            buy_price = fees.execution_price(entry_raw, "BUY")
            sell_price = fees.execution_price(exit_raw, "SELL")
            pnl = (
                (sell_price - buy_price) * config.shares
                - fees.order_fee(buy_price, config.shares, "BUY")
                - fees.order_fee(sell_price, config.shares, "SELL")
            )
        else:
            entry_raw = max(entry_open, entry_line)
            target_raw = entry_raw * (1 - config.target_deviation)
            target_hit = bool(len(later_bars) and np.any(later_bars[:, 2] <= target_raw))
            exit_raw = target_raw if target_hit else float(day_bars[-1, 3])
            sell_price = fees.execution_price(entry_raw, "SELL")
            buy_price = fees.execution_price(exit_raw, "BUY")
            pnl = (
                (sell_price - buy_price) * config.shares
                - fees.order_fee(sell_price, config.shares, "SELL")
                - fees.order_fee(buy_price, config.shares, "BUY")
            )
        records.append(
            {
                "date": day,
                "pnl": float(pnl),
                "entry_price": entry_raw,
                "target_price": target_raw,
                "exit_price": exit_raw,
                "target_hit": target_hit,
            }
        )
    return pd.DataFrame.from_records(records).set_index("date")


def _metrics(samples: pd.DataFrame) -> dict[str, Any]:
    if samples.empty:
        return {"cycles": 0, "win_rate": 0.0, "profit_factor": 0.0, "net_pnl": 0.0}
    gains = float(samples.loc[samples["pnl"] > 0, "pnl"].sum())
    losses = float(-samples.loc[samples["pnl"] <= 0, "pnl"].sum())
    return {
        "cycles": int(len(samples)),
        "win_rate": round(float((samples["pnl"] > 0).mean()), 4),
        "target_hit_rate": round(float(samples["target_hit"].mean()), 4),
        "profit_factor": round(gains / losses, 4) if losses else None,
        "net_pnl": round(float(samples["pnl"].sum()), 2),
    }


def _fit_model(samples: pd.DataFrame, config: DailyTConfig) -> DecisionTreeRegressor:
    model = DecisionTreeRegressor(
        max_depth=config.model_max_depth,
        min_samples_leaf=config.model_min_samples_leaf,
        random_state=1,
    )
    return model.fit(samples.loc[:, FEATURE_COLUMNS], samples["pnl"])


def evaluate_positive_t(
    intraday: pd.DataFrame,
    current_daily: pd.DataFrame,
    config: DailyTConfig = POSITIVE_T_CONFIG,
) -> dict[str, Any]:
    """Evaluate the frozen positive-T rule and score the next session."""
    historical_daily = daily_from_intraday(intraday)
    historical_features = daily_t_features(historical_daily).shift(1)
    outcomes = simulate_daily_t(intraday, config)
    samples = outcomes.join(historical_features, how="inner").dropna(
        subset=list(FEATURE_COLUMNS)
    )
    selection_train = samples.loc["2022":"2023"]
    validation = samples.loc["2024":"2025"]
    selection_model = _fit_model(selection_train, config)
    train_selected = selection_train[
        selection_model.predict(selection_train.loc[:, FEATURE_COLUMNS])
        > config.score_threshold
    ]
    validation_selected = validation[
        selection_model.predict(validation.loc[:, FEATURE_COLUMNS])
        > config.score_threshold
    ]
    through_2025 = samples.loc["2022":"2025"]
    held_out = samples.loc["2026":]
    held_out_model = _fit_model(through_2025, config)
    held_out_selected = held_out[
        held_out_model.predict(held_out.loc[:, FEATURE_COLUMNS])
        > config.score_threshold
    ]
    latest_features = daily_t_features(current_daily).dropna().tail(1)
    live_training = samples.loc["2022":]
    live_model = _fit_model(live_training, config)
    score = float(live_model.predict(latest_features.loc[:, FEATURE_COLUMNS])[0])
    train_metrics = _metrics(train_selected)
    validation_metrics = _metrics(validation_selected)
    held_out_metrics = _metrics(held_out_selected)
    passed = (
        train_metrics["cycles"] >= 40
        and train_metrics["win_rate"] >= 0.60
        and (train_metrics["profit_factor"] or 0) >= 1.20
        and train_metrics["net_pnl"] > 0
        and validation_metrics["cycles"] >= 25
        and validation_metrics["win_rate"] >= 0.60
        and (validation_metrics["profit_factor"] or 0) >= 1.20
        and validation_metrics["net_pnl"] > 0
        and held_out_metrics["cycles"] >= 10
        and held_out_metrics["win_rate"] >= 0.55
        and (held_out_metrics["profit_factor"] or 0) >= 1.10
        and held_out_metrics["net_pnl"] > 0
    )
    return {
        "passed": passed,
        "next_session_gate": score > config.score_threshold,
        "score": round(score, 2),
        "score_threshold": config.score_threshold,
        "results": [
            {"name": "训练：横盘期正T", "period": "2022-2023", **train_metrics},
            {"name": "验证：横盘期正T", "period": "2024-2025", **validation_metrics},
            {"name": "留出：最新横盘期", "period": "2026", **held_out_metrics},
        ],
    }


def _round_lot(shares: int) -> int:
    return max(int(shares) // 100 * 100, 0)


def build_daily_t_plan(
    daily: pd.DataFrame,
    intraday: pd.DataFrame,
    shares: int,
    full_shares: int = 10000,
) -> dict[str, Any]:
    """Build the user-facing fixed next-session positive/reverse plan."""
    normalized = normalize_daily_bars(daily)
    latest = normalized.iloc[-1]
    signal_date = normalized.index[-1]
    close = float(latest["close"])
    evaluation = evaluate_positive_t(intraday, normalized.reset_index())
    positive_shares = min(
        POSITIVE_T_CONFIG.shares,
        _round_lot(max(full_shares + 2000 - int(shares), 0)),
        _round_lot(max(int(shares), 0)),
    )
    buy_price = close * (1 - POSITIVE_T_CONFIG.entry_deviation)
    target_price = buy_price * (1 + POSITIVE_T_CONFIG.target_deviation)
    positive_enabled = bool(
        evaluation["passed"]
        and evaluation["next_session_gate"]
        and positive_shares >= 100
    )
    reverse_capacity = min(500, _round_lot(max(int(shares) - 8000, 0)))
    if shares > full_shares:
        inventory_note = f"当前高于满仓 {shares - full_shares} 股；先完成T，再择反弹把收盘仓位降回10000股。"
    elif shares < 8000:
        inventory_note = f"当前低于合理仓下限 {8000 - shares} 股；停止反T，低位优先恢复库存。"
    else:
        inventory_note = "当前处于合理仓位；正T完成后保持原收盘股数。"
    return {
        "signal_date": signal_date.strftime("%Y-%m-%d"),
        "reference_close": round(close, 2),
        "basis": "盘前固定日线计划；历史5分钟线只用于验证成交先后，盘中不刷新计划",
        "priority": "positive",
        "positive": {
            "execution_enabled": positive_enabled,
            "strategy_passed": evaluation["passed"],
            "gate_passed": evaluation["next_session_gate"],
            "status": "正T可执行" if positive_enabled else "今日不挂正T单",
            "shares": positive_shares,
            "buy_price": round(buy_price, 2),
            "target_price": round(target_price, 2),
            "entry_rule": f"仅当价格到 {buy_price:.2f} 元或更低时买入 {positive_shares} 股；不到不买。",
            "exit_rule": f"买入后优先在 {target_price:.2f} 元卖出同等股数；目标不到，14:50后直接卖出完成T。",
            "no_fill_rule": "买点不到不买，当天不做正T，也不追价。",
            "score": evaluation["score"],
            "score_threshold": evaluation["score_threshold"],
        },
        "reverse": {
            "execution_enabled": False,
            "status": "反T暂停，仅观察",
            "shares": reverse_capacity,
            "sell_price": round(close * 1.008, 2),
            "buyback_price": round(close * 1.008 * (1 - 0.012), 2),
            "reason": "2022年后的横盘样本中，没有反T参数同时通过训练期和验证期盈利闸门；优先执行正T。",
        },
        "inventory": {
            "shares": int(shares),
            "full_shares": full_shares,
            "reasonable_min_shares": 8000,
            "intraday_max_shares": full_shares + 2000,
            "note": inventory_note,
        },
        "validation": {
            "status": "passed_positive_only" if evaluation["passed"] else "failed",
            "execution_enabled": evaluation["passed"],
            "label": "横盘期正T通过；反T暂停" if evaluation["passed"] else "正T验证未通过",
            "period": "2022-01-01 至最新（忽略早期大涨阶段）",
            "decision": (
                "固定买价、固定目标价、目标不到则尾盘退出；只在日线质量闸门通过的交易日挂正T买单。"
                "反T未通过相同标准，暂不执行。"
            ),
            "requirements": {
                "minimum_validation_cycles": 25,
                "minimum_validation_win_rate": 0.60,
                "minimum_held_out_win_rate": 0.55,
                "minimum_profit_factor": 1.10,
                "positive_net_pnl": True,
            },
            "held_out_results": evaluation["results"],
        },
    }
