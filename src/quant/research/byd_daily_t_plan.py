"""Cost-aware pre-market BYD T plans with quarterly walk-forward checks.

Completed five-minute bars are historical training and validation samples, not
same-day live features. The workspace records their maximum date and enforces a
configurable freshness SLA. Models train from 2022 onward, with each forecast
quarter excluded from its own training labels. A failed audit disables entries.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
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
    + ["body", "range", "vol5", "atrp", "rsi", "width60"]
)


@dataclass(frozen=True)
class DailyTConfig:
    direction: Direction
    entry_deviation: float
    target_deviation: float
    shares: int = 500
    model_max_depth: int = 3
    model_min_samples_leaf: int = 20
    score_threshold: float = 10.0
    latest_entry_minute: int = 14 * 60 + 30
    force_flat_minute: int = 14 * 60 + 50
    stop_loss: float | None = None

    def __post_init__(self) -> None:
        if self.shares <= 0 or self.shares % 100:
            raise ValueError("shares must be positive board lots")
        if not 0 < self.entry_deviation < 1 or not 0 < self.target_deviation < 1:
            raise ValueError("entry and target deviations must be between zero and one")
        if not 9 * 60 + 35 <= self.latest_entry_minute < self.force_flat_minute <= 15 * 60:
            raise ValueError("entry cutoff must precede the scheduled exit")
        if self.stop_loss is not None and not 0 < self.stop_loss < 1:
            raise ValueError("stop_loss must be between zero and one")


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
        out["volume"] = pd.to_numeric(out["vol"], errors="coerce") if "vol" in out else 1.0
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
    features["rsi"] = (100 - 100 / (1 + gain / loss.replace(0, np.nan))).where(
        loss.ne(0), np.where(gain.gt(0), 100.0, 50.0)
    )
    features["width60"] = (
        daily["high"].rolling(60).max() / daily["low"].rolling(60).min() - 1
    )
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


TRADE_COLUMNS = [
    "pnl", "return_bps", "entry_price", "target_price", "exit_price",
    "target_hit", "entry_time", "exit_time", "exit_reason", "fees_paid",
]


def simulate_daily_t(
    raw: pd.DataFrame,
    config: DailyTConfig,
    fees: ChinaAStockFees = ChinaAStockFees(),
) -> pd.DataFrame:
    """Replay fixed limit orders, bounded entry times and a scheduled close.

    Five-minute timestamps are bar ends. No target fill is credited on the
    entry bar. Stops take priority when a later bar touches both prices.
    A positive T sells an equal quantity of pre-existing settled inventory;
    the caller must reserve that inventory and the extra buy-side cash.
    """
    empty = pd.DataFrame(columns=TRADE_COLUMNS, index=pd.DatetimeIndex([], name="date"))
    if raw.empty:
        return empty
    required = {"datetime", "open", "high", "low", "close"}
    if not required.issubset(raw):
        raise ValueError(f"intraday bars missing columns: {sorted(required - set(raw))}")
    bars = raw.copy()
    bars["datetime"] = pd.to_datetime(bars["datetime"], errors="raise")
    if bars["datetime"].duplicated().any():
        raise ValueError("duplicate intraday timestamps")
    bars["date"] = bars["datetime"].dt.normalize()
    bars = bars.sort_values("datetime")
    daily = daily_from_intraday(bars).set_index("date")
    previous_close = daily["close"].shift(1)
    records: list[dict[str, Any]] = []
    for day, frame in bars.groupby("date", sort=True):
        reference = previous_close.loc[day]
        if not np.isfinite(reference):
            continue
        times = frame["datetime"].reset_index(drop=True)
        minutes = times.dt.hour * 60 + times.dt.minute
        exits = np.flatnonzero(minutes.to_numpy() >= config.force_flat_minute)
        if not len(exits):
            # A partial day cannot supply a scheduled liquidation label.
            continue
        exit_index = int(exits[0])
        values = frame[["open", "high", "low", "close"]].to_numpy(dtype=float)
        is_positive = config.direction == "positive"
        entry_line = round(float(reference) * (1 - config.entry_deviation if is_positive else 1 + config.entry_deviation), 2)
        touches = values[:, 2] <= entry_line if is_positive else values[:, 1] >= entry_line
        eligible = touches & minutes.between(9 * 60 + 35, config.latest_entry_minute).to_numpy()
        entries = np.flatnonzero(eligible & (np.arange(len(values)) < exit_index))
        if not len(entries):
            continue
        entry_index = int(entries[0])
        entry_open = float(values[entry_index, 0])
        entry_raw = min(entry_open, entry_line) if is_positive else max(entry_open, entry_line)
        # This is the same fixed target printed in the pre-market plan,
        # including on a gap-open entry with a better execution price.
        target_raw = round(entry_line * (1 + config.target_deviation if is_positive else 1 - config.target_deviation), 2)
        stop_raw = None if config.stop_loss is None else round(entry_raw * (
            1 - config.stop_loss if is_positive else 1 + config.stop_loss
        ), 2)
        exit_raw = float(values[exit_index, 3])
        reason = "SESSION_EXIT"
        for index in range(entry_index, exit_index + 1):
            opening, high, low, _ = values[index]
            stopped = stop_raw is not None and (low <= stop_raw if is_positive else high >= stop_raw)
            if stopped:
                exit_raw = min(float(opening), stop_raw) if is_positive else max(float(opening), stop_raw)
                reason, exit_index = "STOP", index
                break
            if index > entry_index and (high >= target_raw if is_positive else low <= target_raw):
                exit_raw, reason, exit_index = target_raw, "TARGET", index
                break
        buy_raw, sell_raw = (entry_raw, exit_raw) if is_positive else (exit_raw, entry_raw)
        buy_price = fees.execution_price(buy_raw, "BUY")
        sell_price = fees.execution_price(sell_raw, "SELL")
        order_cost = fees.order_fee(buy_price, config.shares, "BUY") + fees.order_fee(sell_price, config.shares, "SELL")
        pnl = (sell_price - buy_price) * config.shares - order_cost
        records.append({
            "date": day, "pnl": float(pnl),
            "return_bps": float(pnl / (entry_raw * config.shares) * 10000),
            "entry_price": entry_raw, "target_price": target_raw, "exit_price": exit_raw,
            "target_hit": reason == "TARGET", "entry_time": times.iloc[entry_index],
            "exit_time": times.iloc[exit_index], "exit_reason": reason,
            "fees_paid": float(order_cost),
        })
    return pd.DataFrame.from_records(records).set_index("date") if records else empty


def _metrics(samples: pd.DataFrame) -> dict[str, Any]:
    if samples.empty:
        return {"cycles": 0, "win_rate": 0.0, "profit_factor": 0.0, "net_pnl": 0.0,
                "cycle_drawdown_yuan": 0.0, "worst_cycle": 0.0}
    gains = float(samples.loc[samples["pnl"] > 0, "pnl"].sum())
    losses = float(-samples.loc[samples["pnl"] <= 0, "pnl"].sum())
    equity = np.r_[0.0, samples["pnl"].cumsum().to_numpy(dtype=float)]
    return {
        "cycles": int(len(samples)),
        "win_rate": round(float((samples["pnl"] > 0).mean()), 4),
        "target_hit_rate": round(float(samples["target_hit"].mean()), 4),
        "profit_factor": round(gains / losses, 4) if losses else None,
        "net_pnl": round(float(samples["pnl"].sum()), 2),
        "cycle_drawdown_yuan": round(float((np.maximum.accumulate(equity) - equity).max()), 2),
        "worst_cycle": round(float(samples["pnl"].min()), 2),
        "average_return_bps": round(float(samples["return_bps"].mean()), 2),
    }


def _fit_model(samples: pd.DataFrame, config: DailyTConfig) -> DecisionTreeRegressor:
    model = DecisionTreeRegressor(
        max_depth=config.model_max_depth,
        min_samples_leaf=config.model_min_samples_leaf,
        random_state=1,
    )
    return model.fit(samples.loc[:, FEATURE_COLUMNS], samples["return_bps"])


def sideways_feature_mask(features: pd.DataFrame) -> pd.Series:
    """The same completed-daily regime gate for historical and next-day orders."""
    ma_gap = (1 + features["gap60"]) / (1 + features["gap20"]) - 1
    return (
        features["ret60"].abs().le(.12)
        & features["slope20"].abs().le(.025)
        & ma_gap.abs().le(.08)
        & features["width60"].between(.08, .35)
        & features["atrp"].between(.01, .06)
    )


def walk_forward_scores(samples: pd.DataFrame, config: DailyTConfig) -> pd.DataFrame:
    """Predict each quarter with labels available strictly before its start."""
    pieces = []
    for quarter in samples.index.to_period("Q").unique().sort_values():
        if quarter.year < 2023:
            continue
        training = samples.loc[(samples.index >= "2022-01-01") & (samples.index < quarter.start_time)]
        test = samples.loc[samples.index.to_period("Q") == quarter]
        if len(training) < max(40, config.model_min_samples_leaf * 2) or test.empty:
            continue
        model = _fit_model(training, config)
        pieces.append(pd.DataFrame({
            "score": model.predict(test.loc[:, FEATURE_COLUMNS]),
            "training_end": training.index.max(),
        }, index=test.index))
    return pd.concat(pieces).sort_index() if pieces else pd.DataFrame(
        columns=["score", "training_end"], index=pd.DatetimeIndex([], name="date")
    )


def _segment_passed(metrics: dict[str, Any], cycles: int, win_rate: float, factor: float) -> bool:
    pf = metrics["profit_factor"]
    return bool(metrics["cycles"] >= cycles and metrics["win_rate"] >= win_rate
                and metrics["net_pnl"] > 0 and (pf is None or pf >= factor))


def evaluate_positive_t(
    intraday: pd.DataFrame,
    current_daily: pd.DataFrame,
    config: DailyTConfig = POSITIVE_T_CONFIG,
) -> dict[str, Any]:
    """Apply a quarterly frozen model, separating selection from 2026 monitoring."""
    result: dict[str, Any] = {
        "passed": False, "selection_passed": False, "monitoring_passed": False,
        "next_session_gate": False, "score": None, "score_unit": "bps",
        "score_threshold": config.score_threshold, "results": [],
        "status": "insufficient_samples", "model_training_end": None,
        "regime_passed": False,
    }
    current = normalize_daily_bars(current_daily)
    if current.empty:
        result["status"] = "latest_features_missing"
        return result
    date_column = "date" if "date" in current_daily else "trade_date"
    supplied_dates = pd.to_datetime(current_daily[date_column].astype(str), errors="coerce")
    if supplied_dates.max().normalize() != current.index.max():
        result["status"] = "latest_features_missing"
        return result
    features = daily_t_features(current.reset_index())
    latest = features.tail(1)
    if latest.empty or not np.isfinite(latest.to_numpy(dtype=float)).all():
        result["status"] = "latest_features_missing"
        return result
    if intraday.empty:
        return result
    # Current research must never train on sessions after the decision date.
    intraday = intraday.loc[pd.to_datetime(intraday["datetime"]).dt.normalize() <= current.index.max()]
    historical_daily = daily_from_intraday(intraday)
    historical_features = daily_t_features(historical_daily).shift(1)
    outcomes = simulate_daily_t(intraday, config)
    samples = outcomes.join(historical_features, how="inner").dropna(subset=list(FEATURE_COLUMNS))
    scores = walk_forward_scores(samples, config)
    scored = samples.join(scores, how="inner")
    selected = scored.loc[scored["score"].gt(config.score_threshold) & sideways_feature_mask(scored)]
    windows = [("开发：季度前推", "2023", "2023"),
               ("验证：季度前推", "2024", "2025"),
               ("冻结后监测", "2026", str(current.index.max().year))]
    segment_metrics = [_metrics(selected.loc[first:last]) for _, first, last in windows]
    result["results"] = [{"name": name, "period": first if first == last else f"{first}-{last}", **metrics}
                         for (name, first, last), metrics in zip(windows, segment_metrics)]
    selection_passed = _segment_passed(segment_metrics[0], 40, .60, 1.20) and _segment_passed(segment_metrics[1], 25, .60, 1.20)
    monitoring_passed = _segment_passed(segment_metrics[2], 10, .55, 1.10)
    stressed = simulate_daily_t(intraday, config, replace(ChinaAStockFees(), slippage_rate=.0002))
    stress_metrics = [_metrics(stressed.loc[stressed.index.intersection(selected.loc[first:last].index)])
                      for _, first, last in windows]
    stress_passed = all(item["net_pnl"] > 0 for item in stress_metrics)
    result.update(
        selection_passed=selection_passed, monitoring_passed=monitoring_passed,
        stress_passed=stress_passed, stress_results=stress_metrics,
        passed=bool(selection_passed and monitoring_passed and stress_passed),
        intraday_end=str(pd.to_datetime(intraday["datetime"]).max()),
        selection_period="2023 development; 2024-2025 validation",
        monitoring_period="2026 (previously inspected, not an untouched holdout)",
    )
    # Match the live model's cutoff to the quarter being forecast, rather than
    # refitting on its own monitored labels and reusing an unrelated pass flag.
    forecast_quarter = (current.index.max() + pd.offsets.BDay(1)).to_period("Q")
    training = samples.loc[(samples.index >= "2022-01-01") & (samples.index < forecast_quarter.start_time)]
    if len(training) < max(40, config.model_min_samples_leaf * 2):
        return result
    score = float(_fit_model(training, config).predict(latest.loc[:, FEATURE_COLUMNS])[0])
    regime_passed = bool(sideways_feature_mask(latest).iloc[0])
    result.update(score=round(score, 2), model_training_end=training.index.max().strftime("%Y-%m-%d"),
                  forecast_quarter=str(forecast_quarter), regime_passed=regime_passed,
                  next_session_gate=bool(score > config.score_threshold and regime_passed), status="evaluated")
    return result


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
    buy_price = round(close * (1 - POSITIVE_T_CONFIG.entry_deviation), 2)
    target_price = round(buy_price * (1 + POSITIVE_T_CONFIG.target_deviation), 2)
    positive_enabled = bool(
        evaluation["passed"]
        and evaluation["next_session_gate"]
        and positive_shares == POSITIVE_T_CONFIG.shares
    )
    reverse_capacity = min(500, _round_lot(max(int(shares) - 8000, 0)))
    if not evaluation["passed"]:
        disabled_reason = "规则尚未同时通过开发、验证和滚动监测，暂停新开正T仓。"
    elif not evaluation["next_session_gate"]:
        disabled_reason = "下一交易日评分或横盘条件未通过，不挂正T买单。"
    elif positive_shares < POSITIVE_T_CONFIG.shares:
        disabled_reason = "底仓可卖数量或盘中持仓容量不足，不挂正T买单。"
    else:
        disabled_reason = ""
    entry_cutoff = f"{POSITIVE_T_CONFIG.latest_entry_minute // 60:02}:{POSITIVE_T_CONFIG.latest_entry_minute % 60:02}"
    exit_clock = f"{POSITIVE_T_CONFIG.force_flat_minute // 60:02}:{POSITIVE_T_CONFIG.force_flat_minute % 60:02}"
    if shares > full_shares:
        inventory_note = f"当前高于满仓 {shares - full_shares} 股；先完成T，再择反弹把收盘仓位降回10000股。"
    elif shares < 8000:
        inventory_note = f"当前低于合理仓下限 {8000 - shares} 股；停止反T，低位优先恢复库存。"
    else:
        inventory_note = "当前处于合理仓位；正T完成后保持原收盘股数。"
    return {
        "signal_date": signal_date.strftime("%Y-%m-%d"),
        "reference_close": round(close, 2),
        "basis": (
            "盘前固定日线计划；历史5分钟线用于模型训练与成交先后验证，"
            "不是当日特征，盘中不刷新计划"
        ),
        "priority": "positive",
        "positive": {
            "execution_enabled": positive_enabled,
            "strategy_passed": evaluation["passed"],
            "gate_passed": evaluation["next_session_gate"],
            "status": "正T可执行" if positive_enabled else "今日不挂正T单",
            "disabled_reason": disabled_reason,
            "shares": positive_shares,
            "buy_price": round(buy_price, 2),
            "target_price": round(target_price, 2),
            "entry_rule": f"仅在{entry_cutoff}之前，价格到 {buy_price:.2f} 元或更低时买入 {positive_shares} 股；不到不买。",
            "exit_rule": f"买入后优先在固定目标 {target_price:.2f} 元卖出同等底仓；目标不到，{exit_clock}定时退出完成T。",
            "latest_entry_time": entry_cutoff,
            "scheduled_exit_time": exit_clock,
            "required_settled_shares": positive_shares,
            "required_cash_estimate": round(buy_price * positive_shares + ChinaAStockFees().order_fee(buy_price, positive_shares, "BUY"), 2) if positive_shares else 0.0,
            "no_fill_rule": "买点不到不买，当天不做正T，也不追价。",
            "score": evaluation["score"],
            "score_threshold": evaluation["score_threshold"],
            "score_unit": evaluation.get("score_unit", "bps"),
            "model_training_end": evaluation.get("model_training_end"),
            "regime_passed": evaluation.get("regime_passed", False),
        },
        "reverse": {
            "execution_enabled": False,
            "status": "反T暂停，仅观察",
            "shares": reverse_capacity,
            "sell_price": round(close * 1.008, 2),
            "buyback_price": round(close * 1.008 * (1 - 0.012), 2),
            "reason": "历史研究中没有反T参数同时通过盈利闸门；本轮未重新优化反T，继续暂停。",
        },
        "inventory": {
            "shares": int(shares),
            "full_shares": full_shares,
            "reasonable_min_shares": 8000,
            "intraday_max_shares": full_shares + 2000,
            "note": inventory_note + "须另行确认可卖底仓与可用现金；持仓股数不代表当日可卖股数。",
        },
        "validation": {
            "status": "passed_positive_only" if evaluation["passed"] else "failed",
            "execution_enabled": evaluation["passed"],
            "label": "横盘期正T通过；反T暂停" if evaluation["passed"] else "正T验证未通过",
            "period": "2022起训练；2023开发、2024-2025验证、2026滚动监测",
            "decision": (
                "固定买价和目标价，14:30停止开仓、14:50退出；模型按季度冻结，质量分为扣费收益基点估计。"
                + ("仅在当日质量与横盘闸门通过时启用。" if evaluation["passed"] else disabled_reason)
                + "2026已在历史研究中被查看，作为滚动监测，不声称全新留出。反T继续暂停。"
            ),
            "requirements": {
                "minimum_validation_cycles": 25,
                "minimum_validation_win_rate": 0.60,
                "minimum_held_out_win_rate": 0.55,
                "minimum_profit_factor": 1.10,
                "positive_net_pnl": True,
            },
            "held_out_results": evaluation["results"],
            "selection_passed": evaluation.get("selection_passed", False),
            "monitoring_passed": evaluation.get("monitoring_passed", False),
            "stress_passed": evaluation.get("stress_passed", False),
            "stress_results": evaluation.get("stress_results", []),
            "evaluation_status": evaluation.get("status"),
            "intraday_end": evaluation.get("intraday_end"),
        },
    }
