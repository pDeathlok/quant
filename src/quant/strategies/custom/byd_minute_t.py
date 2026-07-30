"""BYD single-stock inventory-aware T strategy.

This module is built for a personal 002594.SZ holding. It is not a general
stock selector. The objective is to keep the closing inventory inside a
reasonable band while using a validation-gated positive T to reduce cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SYMBOL = "002594.SZ"
NAME = "比亚迪"
LOT_SIZE = 100


@dataclass(frozen=True)
class BydHolding:
    shares: int = 10000
    cost: float = 110.6061
    full_shares: int = 10000


@dataclass(frozen=True)
class BydMinuteConfig:
    risk_floor_shares: int = 6000
    reasonable_min_shares: int = 8000
    preferred_shares: int = 9000
    max_intraday_extra_shares: int = 2000
    max_positive_t_buy_shares: int = 500
    max_positive_t_open_shares: int = 1500
    micro_sell_1: float = 0.008
    micro_sell_2: float = 0.015
    micro_sell_3: float = 0.022
    micro_buyback_discount: float = 0.008
    positive_t_entry_deviation: float = 0.008
    positive_t_previous_close_deviation: float = 0.004
    positive_t_stack_gap: float = 0.008
    positive_t_profit_target: float = 0.004
    positive_t_stop_loss: float = 0.020
    positive_t_max_holding_sessions: int = 3


BYD_T_VALIDATION: dict[str, Any] = {
    "status": "passed_limited",
    "execution_enabled": True,
    "allowed_new_entry_kinds": ["positive_t"],
    "label": "严格横盘正T通过；反T仍暂停",
    "asof": "2026-07-17",
    "bars": 76032,
    "sessions": 1584,
    "period": "2020-01-02 至 2026-07-17",
    "requirements": {
        "minimum_cycles_per_selection_segment": 12,
        "minimum_win_rate": 0.55,
        "minimum_profit_factor": 1.20,
        "positive_net_pnl": True,
        "t1_violations": 0,
    },
    "held_out_results": [
        {
            "name": "训练：横盘正T",
            "period": "2020-2023（强趋势期无开仓）",
            "cycles": 15,
            "win_rate": 0.9333,
            "profit_factor": 3.0372,
            "net_pnl": 1758.61,
        },
        {
            "name": "验证：横盘正T",
            "period": "2024-2025",
            "cycles": 12,
            "win_rate": 0.8333,
            "profit_factor": 1.3500,
            "net_pnl": 871.40,
        },
        {
            "name": "样本外：横盘正T",
            "period": "2026-01-05 至 2026-07-17",
            "cycles": 2,
            "win_rate": 1.0,
            "profit_factor": None,
            "net_pnl": 296.84,
        },
    ],
    "selected_rule": {
        "direction": "只做正T",
        "lot_shares": 500,
        "entry": "横盘且60日收益≤0；位于60日区间下半部；低于VWAP 0.8%并出现5分钟反转",
        "tail_entry": "14:35后需低于VWAP 1.1%",
        "target": "买入均价上方0.4%",
        "stop": "买入均价下方2.0%",
        "maximum_holding_sessions": 3,
    },
    "decision": (
        "只启用低频的严格横盘正T；反T因少数大亏会吞噬多数小盈利，继续暂停。"
        "2026样本外只有2次，仍属小样本，必须保留500股分档、止损和滚动复核。"
    ),
}


def round_lot(shares: float) -> int:
    return max(int(shares // LOT_SIZE) * LOT_SIZE, 0)


def capped_sell_delta(requested_shares: int, current_shares: int, floor_shares: int) -> int:
    """Return a board-lot sell delta without crossing the inventory floor."""
    capacity = round_lot(max(current_shares - floor_shares, 0))
    return -min(round_lot(abs(requested_shares)), capacity)


def weighted_price(parts: list[tuple[int, float | None]]) -> float | None:
    """Return the board-lot weighted average for valid quantity/price pairs."""
    valid = [(round_lot(shares), float(price)) for shares, price in parts if shares and price]
    valid = [(shares, price) for shares, price in valid if shares > 0 and price > 0]
    total_shares = sum(shares for shares, _ in valid)
    if total_shares <= 0:
        return None
    return sum(shares * price for shares, price in valid) / total_shares


def latest_qfq_cache(cache_dir: Path) -> Path:
    files = sorted(cache_dir.glob("tushare_002594.SZ_*_qfq.parquet"))
    files += sorted(cache_dir.glob("sz002594_*_qfq.parquet"))
    if not files:
        raise FileNotFoundError(f"No BYD qfq cache found under {cache_dir}")
    return max(files, key=lambda item: item.stat().st_mtime)


def load_daily_qfq(cache_dir: Path) -> pd.DataFrame:
    path = latest_qfq_cache(cache_dir)
    df = pd.read_parquet(path).copy()
    if "date" not in df.columns and "trade_date" in df.columns:
        df["date"] = pd.to_datetime(df["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    else:
        text = df["date"].astype(str).str.replace("-", "", regex=False)
        if text.str.fullmatch(r"\d{8}").all():
            df["date"] = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
        else:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "vol" in df.columns and "volume" not in df.columns:
        df["volume"] = df["vol"]
    return df.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date").reset_index(drop=True)


def normalize_minutes(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "trade_time" not in out.columns:
        for candidate in ["time", "datetime", "date"]:
            if candidate in out.columns:
                out = out.rename(columns={candidate: "trade_time"})
                break
    if "trade_time" in out.columns:
        out["trade_time"] = pd.to_datetime(out["trade_time"], errors="coerce")
    if "vol" in out.columns and "volume" not in out.columns:
        out["volume"] = out["vol"]
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    needed = [col for col in ["trade_time", "open", "high", "low", "close"] if col in out.columns]
    if len(needed) < 5:
        return pd.DataFrame()
    return out.dropna(subset=["trade_time", "close"]).sort_values("trade_time").reset_index(drop=True)


def daily_range_levels(daily: pd.DataFrame) -> dict[str, Any]:
    latest = daily.iloc[-1]
    hist = daily.tail(121).copy()
    range_low = float(hist["low"].tail(60).min())
    range_high = float(hist["high"].tail(60).max())
    low_20 = float(hist["low"].tail(20).min())
    high_20 = float(hist["high"].tail(20).max())
    close_120 = hist["close"].tail(120).dropna()
    q20 = float(close_120.quantile(0.20)) if not close_120.empty else range_low
    q50 = float(close_120.quantile(0.50)) if not close_120.empty else (range_low + range_high) / 2
    q80 = float(close_120.quantile(0.80)) if not close_120.empty else range_high
    width = range_high - range_low
    return {
        "daily_signal_date": latest["date"].strftime("%Y-%m-%d"),
        "daily_close": float(latest["close"]),
        "prev_close": float(daily["close"].iloc[-2]) if len(daily) >= 2 else float(latest["close"]),
        "range_low": range_low,
        "range_high": range_high,
        "low_20": low_20,
        "high_20": high_20,
        "q20_120": q20,
        "q50_120": q50,
        "q80_120": q80,
        "support_break": range_low * 0.985,
        "mid_trim": range_low + 0.50 * width,
        "weak_trim": range_low + 0.62 * width,
        "strong_trim": range_low + 0.78 * width,
        "range_width_pct": width / ((range_high + range_low) / 2) if width > 0 else np.nan,
    }


def daily_atr(daily: pd.DataFrame, window: int = 14) -> float:
    hist = daily.copy()
    prev_close = hist["close"].shift(1)
    tr = pd.concat([
        hist["high"] - hist["low"],
        (hist["high"] - prev_close).abs(),
        (hist["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    value = tr.tail(window).mean()
    return float(value) if pd.notna(value) and np.isfinite(float(value)) else 0.0


def daily_sideways_snapshot(daily: pd.DataFrame) -> dict[str, Any]:
    """Classify the next-session regime from completed daily bars only."""
    if len(daily) < 61:
        return {
            "available": False,
            "sideways": False,
            "positive_t_allowed": False,
            "reason": "日线不足61个交易日，无法计算横盘闸门。",
        }
    close = daily["close"].astype(float)
    ma20 = close.rolling(20, min_periods=20).mean()
    ma60 = close.rolling(60, min_periods=60).mean()
    return_60 = float(close.iloc[-1] / close.iloc[-61] - 1)
    ma20_slope_5 = float(ma20.iloc[-1] / ma20.iloc[-6] - 1)
    ma_gap = float(ma20.iloc[-1] / ma60.iloc[-1] - 1)
    high_60 = float(daily["high"].astype(float).tail(60).max())
    low_60 = float(daily["low"].astype(float).tail(60).min())
    range_width_60 = high_60 / low_60 - 1 if low_60 else np.nan
    atr_pct = daily_atr(daily) / float(close.iloc[-1]) if close.iloc[-1] else np.nan
    checks = {
        "abs_return_60_le_12pct": abs(return_60) <= 0.12,
        "abs_ma20_slope_5_le_2_5pct": abs(ma20_slope_5) <= 0.025,
        "abs_ma20_ma60_gap_le_8pct": abs(ma_gap) <= 0.08,
        "range_width_60_between_8_35pct": 0.08 <= range_width_60 <= 0.35,
        "atr_pct_between_1_6pct": 0.01 <= atr_pct <= 0.06,
    }
    sideways = all(checks.values())
    positive_allowed = sideways and return_60 <= 0
    if positive_allowed:
        reason = "横盘闸门通过且60日收益不为正，可以等待严格正T低吸信号。"
    elif not sideways:
        failed = [name for name, passed in checks.items() if not passed]
        reason = f"横盘闸门未通过：{', '.join(failed)}。"
    else:
        reason = "虽处横盘范围，但60日收益仍为正；历史验证显示此时正T尾部风险较高。"
    return {
        "available": True,
        "asof": pd.to_datetime(daily.iloc[-1]["date"]).strftime("%Y-%m-%d"),
        "sideways": sideways,
        "positive_t_allowed": positive_allowed,
        "reason": reason,
        "checks": checks,
        "return_60_pct": round(return_60 * 100, 3),
        "ma20_slope_5_pct": round(ma20_slope_5 * 100, 3),
        "ma20_ma60_gap_pct": round(ma_gap * 100, 3),
        "range_width_60_pct": round(range_width_60 * 100, 3),
        "atr14_pct": round(atr_pct * 100, 3),
        "range_low": round(low_60, 3),
        "range_high": round(high_60, 3),
    }


def price_band(low: float, high: float) -> dict[str, Any]:
    lo = min(float(low), float(high))
    hi = max(float(low), float(high))
    return {
        "low": round(lo, 2),
        "high": round(hi, 2),
        "label": f"{lo:.2f}-{hi:.2f}",
    }


def planned_t_ranges(
    daily: pd.DataFrame,
    levels: dict[str, Any],
    holding: BydHolding,
    config: BydMinuteConfig = BydMinuteConfig(),
) -> dict[str, Any]:
    close = float(levels["daily_close"])
    prev_close = float(levels["prev_close"])
    atr = daily_atr(daily)
    atr = atr if atr > 0 else close * 0.025
    atr_pct = atr / close * 100 if close else None
    low_guard = float(levels["range_low"])
    support_break = float(levels["support_break"])
    q20 = float(levels["q20_120"])
    q50 = float(levels["q50_120"])

    sell_1_low = max(close + atr * 0.25, close * 1.008, prev_close * 0.998)
    sell_1_high = max(sell_1_low * 1.003, close + atr * 0.42)
    sell_2_low = max(close + atr * 0.48, sell_1_high * 1.002)
    sell_2_high = max(sell_2_low * 1.003, close + atr * 0.70)
    sell_3_low = max(close + atr * 0.78, sell_2_high * 1.002, q20 * 0.998)
    sell_3_high = min(max(sell_3_low * 1.003, close + atr * 1.05), q50)
    sell_requests = (500, 800, 1200) if holding.shares >= config.preferred_shares else (300, 500, 800)
    sell_deltas = [
        capped_sell_delta(requested, holding.shares, config.reasonable_min_shares)
        for requested in sell_requests
    ]
    sell_zones = [
        {
            "key": "PLAN_SELL_1",
            "label": "反T观察一档",
            "range": price_band(sell_1_low, sell_1_high),
            "shares": sell_deltas[0],
            "condition": "反弹修复到前收盘附近或半个ATR以内，先卖一笔建立反T仓，但不跌破合理库存下限。",
        },
        {
            "key": "PLAN_SELL_2",
            "label": "反T观察二档",
            "range": price_band(sell_2_low, sell_2_high),
            "shares": sell_deltas[1],
            "condition": "反弹扩大到约0.5-0.7个ATR，分批高抛并等待低价买回。",
        },
        {
            "key": "PLAN_SELL_3",
            "label": "强反抽减仓档",
            "range": price_band(sell_3_low, sell_3_high),
            "shares": sell_deltas[2],
            "condition": "接近120日低位分位或1个ATR反抽，作为较强反T区，不机械永久降仓。",
        },
    ]
    buyback_zones = []
    for zone in sell_zones:
        sell_range = zone["range"]
        buy_high = float(sell_range["low"]) * (1 - 0.008)
        buy_low = float(sell_range["low"]) * (1 - 0.012)
        buyback_zones.append({
            "key": zone["key"].replace("SELL", "BUYBACK"),
            "label": zone["label"].replace("反T", "买回"),
            "sell_range": sell_range,
            "buy_range": price_band(buy_low, buy_high),
            "shares": abs(int(zone["shares"])),
            "validity": "当日有效，未成交则隔日继续以卖出价为锚点",
            "condition": "只买回对应卖出T仓；价格没有回落到买回区就不补。",
        })

    return {
        "basis": "前复权日线 + 近14日ATR + 20/60日箱体",
        "signal_date": levels["daily_signal_date"],
        "reference_close": round(close, 2),
        "prev_close": round(prev_close, 2),
        "atr14": round(atr, 2),
        "atr14_pct": round(atr_pct, 2) if atr_pct is not None else None,
        "stage_note": (
            f"合理收盘仓位 {config.reasonable_min_shares}-{holding.full_shares} 股；"
            f"盘中正T最多临时到 {holding.full_shares + config.max_intraday_extra_shares} 股。"
        ),
        "no_sell_zone": {
            "label": "低位不追卖区",
            "range": price_band(low_guard, min(sell_1_low * 0.995, close + atr * 0.18)),
            "action": "不主动卖出，除非跌破风控线",
        },
        "sell_zones": sell_zones,
        "buyback_zones": buyback_zones,
        "buyback_rule": {
            "label": "反T买回规则",
            "range": "卖出均价下方0.8%-1.2%",
            "action": "只买回已卖出的T仓；没跌回买回区就保留现金。",
        },
        "risk_zone": {
            "label": "下破风控区",
            "range": price_band(support_break * 0.995, support_break),
            "shares": -1500,
            "action": "跌破箱体下沿1.5%，这是风控减仓，不是做T。",
        },
    }


def intraday_reverse_t_levels(snap: dict[str, Any], levels: dict[str, Any], config: BydMinuteConfig) -> dict[str, float]:
    intraday_ref = float(snap.get("open") or levels["prev_close"] or levels["daily_close"])
    vwap = snap.get("vwap")
    vwap_ref = float(vwap) if vwap is not None and np.isfinite(float(vwap)) else intraday_ref
    return {
        "reference": intraday_ref,
        "vwap_reference": vwap_ref,
        "micro_sell_1": min(intraday_ref * (1 + config.micro_sell_1), vwap_ref * 1.006),
        "micro_sell_2": intraday_ref * (1 + config.micro_sell_2),
        "micro_sell_3": intraday_ref * (1 + config.micro_sell_3),
    }


def minute_snapshot(minutes: pd.DataFrame, daily_levels: dict[str, Any]) -> dict[str, Any]:
    if minutes.empty:
        last = float(daily_levels["daily_close"])
        return {
            "has_minute": False,
            "asof": daily_levels["daily_signal_date"],
            "last": last,
            "open": None,
            "high": None,
            "low": None,
            "vwap": None,
            "intraday_return_pct": None,
            "from_high_pct": None,
            "from_low_pct": None,
            "minute_count": 0,
        }
    last_row = minutes.iloc[-1]
    day_open = float(minutes["open"].dropna().iloc[0])
    high = float(minutes["high"].max())
    low = float(minutes["low"].min())
    amount = minutes.get("amount")
    volume = minutes.get("volume")
    if amount is not None and volume is not None and float(volume.sum() or 0) > 0:
        vwap = float(amount.sum() / volume.sum())
        if vwap > 1000:
            vwap = vwap / 1000
    else:
        vwap = float((minutes["close"] * minutes.get("volume", 1)).sum() / minutes.get("volume", pd.Series(1, index=minutes.index)).sum())
    last = float(last_row["close"])
    return {
        "has_minute": True,
        "asof": pd.to_datetime(last_row["trade_time"]).strftime("%Y-%m-%d %H:%M"),
        "last": last,
        "open": day_open,
        "high": high,
        "low": low,
        "vwap": vwap,
        "intraday_return_pct": (last / day_open - 1) * 100 if day_open else None,
        "from_high_pct": (last / high - 1) * 100 if high else None,
        "from_low_pct": (last / low - 1) * 100 if low else None,
        "minute_count": int(len(minutes)),
    }


def intraday_dynamic_zones(minutes: pd.DataFrame, snap: dict[str, Any]) -> dict[str, Any]:
    if minutes.empty or not snap.get("has_minute"):
        return {
            "available": False,
            "state": "无分钟数据",
            "current_position_pct": None,
            "low_zone": None,
            "high_zone": None,
            "rebound_sell_1": None,
            "rebound_sell_2": None,
            "rebound_sell_3": None,
            "pullback_buy": None,
        }
    close = minutes["close"].dropna()
    if close.empty:
        return {"available": False, "state": "分钟收盘价不足"}
    day_low = float(minutes["low"].min())
    day_high = float(minutes["high"].max())
    last = float(snap["last"])
    width = day_high - day_low
    if width <= 0:
        return {
            "available": True,
            "state": "窄幅横盘",
            "current_position_pct": 50,
            "day_low": day_low,
            "day_high": day_high,
            "low_zone": {"low": round(day_low, 2), "high": round(day_low, 2), "label": f"{day_low:.2f}"},
            "high_zone": {"low": round(day_high, 2), "high": round(day_high, 2), "label": f"{day_high:.2f}"},
            "rebound_sell_1": day_high,
            "rebound_sell_2": day_high,
            "rebound_sell_3": day_high,
            "pullback_buy": day_low,
        }
    q20 = float(close.quantile(0.20))
    q80 = float(close.quantile(0.80))
    low_zone_high = min(q20, day_low + width * 0.236)
    high_zone_low = max(q80, day_low + width * 0.764)
    rebound_sell_1 = day_low + width * 0.382
    rebound_sell_2 = day_low + width * 0.500
    rebound_sell_3 = day_low + width * 0.618
    pullback_buy = min(day_low + width * 0.236, float(snap["vwap"]) * 0.995 if snap.get("vwap") else day_low + width * 0.236)
    position = (last - day_low) / width * 100
    if position <= 25:
        state = "日内低位区"
    elif position >= 75:
        state = "日内高位区"
    elif last < (snap.get("vwap") or last):
        state = "低位反抽观察"
    else:
        state = "中位震荡"
    return {
        "available": True,
        "state": state,
        "day_low": round(day_low, 3),
        "day_high": round(day_high, 3),
        "current_position_pct": round(position, 1),
        "low_zone": {
            "low": round(day_low, 2),
            "high": round(low_zone_high, 2),
            "label": f"{day_low:.2f}-{low_zone_high:.2f}",
        },
        "high_zone": {
            "low": round(high_zone_low, 2),
            "high": round(day_high, 2),
            "label": f"{high_zone_low:.2f}-{day_high:.2f}",
        },
        "rebound_sell_1": round(rebound_sell_1, 3),
        "rebound_sell_2": round(rebound_sell_2, 3),
        "rebound_sell_3": round(rebound_sell_3, 3),
        "pullback_buy": round(pullback_buy, 3),
    }


def safe_float(value: Any, digits: int | None = None) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return round(numeric, digits) if digits is not None else numeric


def resample_minute_bars(minutes: pd.DataFrame, rule: str) -> pd.DataFrame:
    if minutes.empty:
        return pd.DataFrame()
    base = minutes.copy()
    if rule != "1min":
        agg = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        }
        if "volume" in base.columns:
            agg["volume"] = "sum"
        if "amount" in base.columns:
            agg["amount"] = "sum"
        base = (
            base.set_index("trade_time")
            .resample(rule, label="right", closed="right")
            .agg(agg)
            .dropna(subset=["open", "high", "low", "close"])
            .reset_index()
        )
    cols = [col for col in ["trade_time", "open", "high", "low", "close", "volume", "amount"] if col in base.columns]
    return base[cols].dropna(subset=["trade_time", "close"]).reset_index(drop=True)


def calc_kdj(bars: pd.DataFrame, n: int = 9) -> pd.DataFrame:
    if bars.empty:
        return bars.copy()
    out = bars.copy()
    low_n = out["low"].rolling(n, min_periods=3).min()
    high_n = out["high"].rolling(n, min_periods=3).max()
    spread = (high_n - low_n).replace(0, np.nan)
    rsv = ((out["close"] - low_n) / spread * 100).clip(lower=0, upper=100).fillna(50)
    out["k"] = rsv.ewm(com=2, adjust=False).mean()
    out["d"] = out["k"].ewm(com=2, adjust=False).mean()
    out["j"] = 3 * out["k"] - 2 * out["d"]
    return out


def timeframe_indicator_snapshot(minutes: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {
        "timeframes": {},
        "confirmation": {
            "sell_score": 0,
            "buy_score": 0,
            "sell_reasons": [],
            "buy_reasons": [],
            "breakout_guard": False,
        },
    }
    if minutes.empty:
        return result

    day_volume_mean = float(minutes["volume"].replace(0, np.nan).dropna().mean()) if "volume" in minutes.columns else np.nan
    for label, rule, vol_window in [
        ("1m", "1min", 20),
        ("5m", "5min", 12),
        ("15m", "15min", 8),
        ("30m", "30min", 6),
    ]:
        bars = calc_kdj(resample_minute_bars(minutes, rule))
        if bars.empty:
            result["timeframes"][label] = {"available": False}
            continue
        latest = bars.iloc[-1]
        previous = bars.iloc[-2] if len(bars) >= 2 else latest
        prev_vol = bars["volume"].shift(1).tail(vol_window).replace(0, np.nan).dropna() if "volume" in bars.columns else pd.Series(dtype=float)
        volume_ratio = float(latest["volume"] / prev_vol.mean()) if not prev_vol.empty and prev_vol.mean() else None
        j = safe_float(latest.get("j"))
        prev_j = safe_float(previous.get("j"))
        j_slope = safe_float((j or 0) - (prev_j or 0)) if j is not None and prev_j is not None else None
        price_change_pct = safe_float((latest["close"] / previous["close"] - 1) * 100) if previous["close"] else None
        k = safe_float(latest.get("k"))
        d = safe_float(latest.get("d"))
        above_vwap = None
        if "volume" in bars.columns:
            vol_sum = bars["volume"].sum()
            if vol_sum:
                vwap = float((bars["close"] * bars["volume"]).sum() / vol_sum)
                above_vwap = bool(latest["close"] >= vwap)

        if j is None:
            kdj_state = "无KDJ"
        elif j >= 90 and (j_slope or 0) <= 0:
            kdj_state = "高位钝化转弱"
        elif j >= 80:
            kdj_state = "高位强势"
        elif j <= 20 and (j_slope or 0) >= 0:
            kdj_state = "低位回勾"
        elif j <= 20:
            kdj_state = "低位弱势"
        elif (k or 0) > (d or 0) and (j_slope or 0) > 0:
            kdj_state = "上行确认"
        elif (k or 0) < (d or 0) and (j_slope or 0) < 0:
            kdj_state = "下行确认"
        else:
            kdj_state = "震荡"

        if volume_ratio is None:
            volume_state = "量比不足"
        elif volume_ratio >= 1.6:
            volume_state = "明显放量"
        elif volume_ratio >= 1.15:
            volume_state = "温和放量"
        elif volume_ratio <= 0.7:
            volume_state = "缩量"
        else:
            volume_state = "平量"

        price_volume_state = "价量中性"
        if price_change_pct is not None and volume_ratio is not None:
            if price_change_pct > 0.15 and volume_ratio >= 1.15:
                price_volume_state = "放量上冲"
            elif price_change_pct < -0.15 and volume_ratio >= 1.15:
                price_volume_state = "放量下压"
            elif price_change_pct > 0 and volume_ratio <= 0.8:
                price_volume_state = "缩量反抽"
            elif price_change_pct < 0 and volume_ratio <= 0.8:
                price_volume_state = "缩量回落"

        snapshot = {
            "available": True,
            "bars": int(len(bars)),
            "asof": pd.to_datetime(latest["trade_time"]).strftime("%H:%M"),
            "close": safe_float(latest["close"], 3),
            "k": safe_float(k, 2),
            "d": safe_float(d, 2),
            "j": safe_float(j, 2),
            "j_slope": safe_float(j_slope, 2),
            "kdj_state": kdj_state,
            "volume_ratio": safe_float(volume_ratio, 2),
            "volume_state": volume_state,
            "price_change_pct": safe_float(price_change_pct, 2),
            "price_volume_state": price_volume_state,
            "above_vwap": above_vwap,
        }
        result["timeframes"][label] = snapshot

    tf = result["timeframes"]
    conf = result["confirmation"]
    for label in ["1m", "5m"]:
        item = tf.get(label, {})
        if item.get("kdj_state") in {"高位钝化转弱", "下行确认"}:
            conf["sell_score"] += 1
            conf["sell_reasons"].append(f"{label} KDJ {item['kdj_state']}")
        if item.get("price_volume_state") in {"放量下压", "缩量反抽"}:
            conf["sell_score"] += 1
            conf["sell_reasons"].append(f"{label} {item['price_volume_state']}")
        if item.get("kdj_state") in {"低位回勾", "上行确认"} and (item.get("j") or 100) <= 45:
            conf["buy_score"] += 1
            conf["buy_reasons"].append(f"{label} KDJ {item['kdj_state']}")
        if item.get("price_volume_state") in {"缩量回落", "放量上冲"} and (item.get("j") or 100) <= 55:
            conf["buy_score"] += 1
            conf["buy_reasons"].append(f"{label} {item['price_volume_state']}")

    strong_rise = 0
    for label in ["15m", "30m"]:
        item = tf.get(label, {})
        if item.get("kdj_state") in {"高位强势", "上行确认"} and item.get("price_volume_state") == "放量上冲":
            strong_rise += 1
    conf["breakout_guard"] = strong_rise >= 2
    if not np.isfinite(day_volume_mean):
        conf["volume_note"] = "分钟成交量不足，量比只作弱参考"
    return result


def validated_positive_t_snapshot(
    minutes: pd.DataFrame,
    levels: dict[str, Any],
    regime: dict[str, Any],
    config: BydMinuteConfig,
) -> dict[str, Any]:
    """Evaluate the live leg of the historically selected positive-T rule."""
    previous_close = float(levels["daily_close"])
    planned_line = previous_close * (1 - config.positive_t_entry_deviation)
    if minutes.empty:
        return {
            "available": False,
            "signal": False,
            "price_line": round(planned_line, 3),
            "reason": "等待盘中5分钟数据验证VWAP偏离、RSI和止跌反转。",
        }
    bars = resample_minute_bars(minutes, "5min")
    if bars.empty:
        return {
            "available": False,
            "signal": False,
            "price_line": round(planned_line, 3),
            "reason": "5分钟数据不足。",
        }
    latest_date = pd.to_datetime(bars["trade_time"]).dt.normalize().iloc[-1]
    bars = bars[pd.to_datetime(bars["trade_time"]).dt.normalize().eq(latest_date)].copy()
    if len(bars) < 6:
        return {
            "available": False,
            "signal": False,
            "price_line": round(planned_line, 3),
            "reason": "至少需要6根5分钟K线计算反转确认。",
        }
    volume = bars.get("volume", pd.Series(1.0, index=bars.index)).astype(float).clip(lower=0)
    typical = (bars["high"] + bars["low"] + bars["close"]) / 3
    total_volume = float(volume.sum())
    vwap = float((typical * volume).sum() / total_volume) if total_volume else float(bars.iloc[-1]["close"])
    latest = bars.iloc[-1]
    current_price = float(latest["close"])
    current_minute = pd.to_datetime(latest["trade_time"]).hour * 60 + pd.to_datetime(
        latest["trade_time"]
    ).minute
    required_deviation = config.positive_t_entry_deviation
    if current_minute >= 14 * 60 + 35:
        required_deviation += 0.003
    prior_close_line = previous_close * (1 - config.positive_t_previous_close_deviation)
    vwap_line = vwap * (1 - required_deviation)
    range_low = float(regime.get("range_low") or levels["range_low"])
    range_high = float(regime.get("range_high") or levels["range_high"])
    range_position = (
        (current_price - range_low) / (range_high - range_low)
        if range_high > range_low
        else np.nan
    )
    closes = pd.concat(
        [pd.Series([previous_close]), bars["close"].astype(float).reset_index(drop=True)],
        ignore_index=True,
    )
    delta = closes.diff().dropna().tail(6)
    average_gain = float(delta.clip(lower=0).mean())
    average_loss = float((-delta.clip(upper=0)).mean())
    if average_loss == 0:
        rsi6 = 100.0 if average_gain > 0 else 50.0
    else:
        rs = average_gain / average_loss
        rsi6 = 100 - 100 / (1 + rs)
    previous_bar_close = float(bars.iloc[-2]["close"])
    midpoint = (float(latest["high"]) + float(latest["low"])) / 2
    turn_up = current_price > previous_bar_close and current_price >= midpoint
    time_allowed = 9 * 60 + 45 <= current_minute <= 14 * 60 + 50
    checks = {
        "daily_regime": bool(regime.get("positive_t_allowed")),
        "time_window": time_allowed,
        "below_vwap": current_price <= vwap_line,
        "below_previous_close": current_price <= prior_close_line,
        "range_lower_half": np.isfinite(range_position) and -0.03 <= range_position <= 0.50,
        "rsi6_le_40": rsi6 <= 40,
        "turn_up": turn_up,
    }
    signal = all(checks.values())
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "available": True,
        "signal": signal,
        "reason": "严格横盘正T信号通过。" if signal else f"等待条件：{', '.join(failed)}。",
        "checks": checks,
        "asof": pd.to_datetime(latest["trade_time"]).strftime("%Y-%m-%d %H:%M"),
        "price": round(current_price, 3),
        "price_line": round(min(vwap_line, prior_close_line), 3),
        "vwap": round(vwap, 3),
        "vwap_deviation_pct": round((current_price / vwap - 1) * 100, 3),
        "previous_close_deviation_pct": round(
            (current_price / previous_close - 1) * 100, 3
        ),
        "range_position_pct": round(range_position * 100, 2)
        if np.isfinite(range_position)
        else None,
        "rsi6": round(rsi6, 2),
        "required_vwap_deviation_pct": round(required_deviation * 100, 2),
    }


def holding_stage(
    shares: int,
    config: BydMinuteConfig,
    full_shares: int = 10000,
) -> dict[str, Any]:
    intraday_limit = full_shares + config.max_intraday_extra_shares
    common = {
        "closing_min_shares": config.reasonable_min_shares,
        "closing_max_shares": full_shares,
        "intraday_limit_shares": intraday_limit,
        "intraday_excess_shares": round_lot(max(shares - full_shares, 0)),
    }
    if shares > full_shares:
        return {
            **common,
            "key": "INTRADAY_OVERWEIGHT",
            "label": "日内超仓：等待正T卖出",
            "goal_shares": full_shares,
            "mode": "只允许已记录正T仓按0.8%间距再加一档；否则等待盈利卖出",
            "positive_t_buy_cap": 0,
            "reverse_t_sell_cap": round_lot(max(shares - config.reasonable_min_shares, 0)),
        }
    if shares >= config.preferred_shares:
        return {
            **common,
            "key": "FULL_T",
            "label": "充足仓：等待严格正T",
            "goal_shares": shares,
            "mode": "仅横盘正T通过验证；反T继续暂停",
            "positive_t_buy_cap": min(
                config.max_positive_t_buy_shares,
                round_lot(max(intraday_limit - shares, 0)),
            ),
            "reverse_t_sell_cap": round_lot(max(shares - config.reasonable_min_shares, 0)),
        }
    if shares >= config.reasonable_min_shares:
        return {
            **common,
            "key": "BALANCED_T",
            "label": "合理仓：成本优先",
            "goal_shares": shares,
            "mode": "保持合理库存；低于满仓时不把补仓伪装成已验证正T",
            "positive_t_buy_cap": min(
                config.max_positive_t_buy_shares,
                round_lot(max(intraday_limit - shares, 0)),
            ),
            "reverse_t_sell_cap": round_lot(max(shares - config.reasonable_min_shares, 0)),
        }
    return {
        **common,
        "key": "UNDERWEIGHT",
        "label": "低于合理仓：优先恢复库存",
        "goal_shares": config.reasonable_min_shares,
        "mode": "停止常规高抛，只在低位分批恢复到合理仓位",
        "positive_t_buy_cap": min(
            config.max_positive_t_buy_shares,
            round_lot(max(config.reasonable_min_shares - shares, 0)),
        ),
        "reverse_t_sell_cap": 0,
    }


def build_alerts(
    holding: BydHolding,
    config: BydMinuteConfig,
    levels: dict[str, Any],
    snap: dict[str, Any],
    indicators: dict[str, Any] | None = None,
    dynamic_zones: dict[str, Any] | None = None,
    plan: dict[str, Any] | None = None,
    regime: dict[str, Any] | None = None,
    validated_positive: dict[str, Any] | None = None,
    sold_today_shares: int = 0,
    sold_today_price: float | None = None,
    bought_today_shares: int = 0,
    bought_today_price: float | None = None,
    open_t_shares: int = 0,
    open_t_price: float | None = None,
    open_positive_shares: int = 0,
    open_positive_price: float | None = None,
) -> list[dict[str, Any]]:
    price = float(snap["last"])
    shares = holding.shares
    stage = holding_stage(shares, config, holding.full_shares)
    alerts: list[dict[str, Any]] = []

    def add(kind: str, priority: int, action: str, trigger: bool, shares_delta: int, price_line: float, title: str, detail: str) -> None:
        if shares_delta == 0:
            return
        line = float(price_line)
        if action == "SELL" and kind == "risk":
            price_range = {"low": round(line * 0.99, 2), "high": round(line, 2), "label": f"{line * 0.99:.2f}-{line:.2f}"}
        elif action == "SELL":
            price_range = {"low": round(line, 2), "high": round(line * 1.003, 2), "label": f"{line:.2f}-{line * 1.003:.2f}"}
        else:
            price_range = {"low": round(line * 0.997, 2), "high": round(line, 2), "label": f"{line * 0.997:.2f}-{line:.2f}"}
        alerts.append({
            "kind": kind,
            "priority": priority,
            "action": action,
            "triggered": bool(trigger),
            "shares_delta": int(shares_delta),
            "price_line": round(line, 2),
            "price_range": price_range,
            "title": title,
            "detail": detail,
        })

    micro = intraday_reverse_t_levels(snap, levels, config)
    intraday_ref = micro["reference"]
    vwap_ref = micro["vwap_reference"]
    micro_1 = micro["micro_sell_1"]
    micro_2 = micro["micro_sell_2"]
    micro_3 = micro["micro_sell_3"]
    confirmation = (indicators or {}).get("confirmation", {})
    sell_score = int(confirmation.get("sell_score") or 0)
    buy_score = int(confirmation.get("buy_score") or 0)
    breakout_guard = bool(confirmation.get("breakout_guard"))
    sell_reasons = confirmation.get("sell_reasons") or []
    buy_reasons = confirmation.get("buy_reasons") or []
    sell_confirmed = sell_score >= 1 or not snap.get("has_minute")
    buy_confirmed = buy_score >= 1 or not snap.get("has_minute")
    sell_note = f"分钟确认：{' / '.join(sell_reasons[:3])}" if sell_reasons else "分钟确认不足，按价格线观察。"
    buy_note = f"买回确认：{' / '.join(buy_reasons[:3])}" if buy_reasons else "买回确认不足，等 1/5 分钟 KDJ 回勾。"
    guard_note = "15/30 分钟同步放量上行，先缩小反T股数，避免卖在真突破。" if breakout_guard else sell_note
    sold_today_shares = round_lot(sold_today_shares)
    sold_today_price = float(sold_today_price) if sold_today_price and sold_today_price > 0 else None
    bought_today_shares = round_lot(bought_today_shares)
    bought_today_price = float(bought_today_price) if bought_today_price and bought_today_price > 0 else None
    open_t_shares = round_lot(open_t_shares)
    open_t_price = float(open_t_price) if open_t_price and open_t_price > 0 else None
    open_positive_shares = round_lot(open_positive_shares)
    open_positive_price = (
        float(open_positive_price) if open_positive_price and open_positive_price > 0 else None
    )
    buyback_shares = round_lot(sold_today_shares + open_t_shares)
    buyback_anchor = weighted_price(
        [
            (sold_today_shares, sold_today_price),
            (open_t_shares, open_t_price),
        ]
    )
    can_reverse_t = int(stage["reverse_t_sell_cap"]) > 0
    full_inventory_gap = round_lot(max(holding.full_shares - shares, 0))
    intraday_excess = int(stage["intraday_excess_shares"])
    plan = plan or {}
    dynamic_zones = dynamic_zones or {}
    dynamic_available = bool(dynamic_zones.get("available"))
    dynamic_sell_1 = dynamic_zones.get("rebound_sell_1")
    dynamic_sell_2 = dynamic_zones.get("rebound_sell_2")
    dynamic_sell_3 = dynamic_zones.get("rebound_sell_3")
    dynamic_low = dynamic_zones.get("day_low")
    dynamic_high = dynamic_zones.get("day_high")

    positive_open_shares = round_lot(bought_today_shares + open_positive_shares)
    positive_anchor = weighted_price(
        [
            (bought_today_shares, bought_today_price),
            (open_positive_shares, open_positive_price),
        ]
    )
    recorded_base_shares = shares - positive_open_shares
    positive_capacity = min(
        round_lot(max(config.max_positive_t_open_shares - positive_open_shares, 0)),
        round_lot(max(int(stage["intraday_limit_shares"]) - shares, 0)),
    )
    positive_buy_delta = min(config.max_positive_t_buy_shares, positive_capacity)
    can_positive_t = (
        recorded_base_shares == holding.full_shares
        and positive_buy_delta >= LOT_SIZE
        and bool(BYD_T_VALIDATION["execution_enabled"])
    )
    positive_exit_requested = max(positive_open_shares, intraday_excess)
    positive_exit_shares = min(
        positive_exit_requested,
        round_lot(max(shares - config.reasonable_min_shares, 0)),
    )
    if positive_exit_shares > 0:
        positive_exit_line = (
            positive_anchor * (1 + config.positive_t_profit_target)
            if positive_anchor is not None
            else micro_1
        )
        gross_profit = (
            (positive_exit_line - positive_anchor) * positive_exit_shares
            if positive_anchor is not None
            else None
        )
        profit_note = f"目标毛差约 {gross_profit:.0f} 元。" if gross_profit is not None else "请补录正T买入均价以计算目标差价。"
        add(
            "positive_t_exit",
            0,
            "SELL",
            price >= positive_exit_line and (sell_confirmed or price >= positive_exit_line * 1.002),
            -positive_exit_shares,
            positive_exit_line,
            "正T卖出：收盘回归基准仓",
            (
                f"卖出日内正T仓 {positive_exit_shares} 股，目标收盘不超过 {holding.full_shares} 股；"
                f"正常目标价差 {config.positive_t_profit_target:.1%}，{profit_note}"
                "允许跨日等待，但必须持续记录待卖出数量和买入均价。"
            ),
        )
        if positive_anchor is not None:
            positive_stop_line = positive_anchor * (1 - config.positive_t_stop_loss)
            add(
                "positive_t_stop",
                0,
                "SELL",
                price <= positive_stop_line,
                -positive_exit_shares,
                positive_stop_line,
                "正T止损：控制单次尾部损失",
                (
                    f"正T买入均价下方 {config.positive_t_stop_loss:.1%} 触发止损；"
                    "历史回测的盈利性依赖该止损，不能改成无限期摊平。"
                ),
            )

    add(
        "risk",
        1,
        "SELL",
        price <= levels["support_break"],
        capped_sell_delta(1500, shares, config.risk_floor_shares),
        levels["support_break"],
        "箱体下破降风险",
        f"跌破 60 日箱体下沿 1.5%，这是风险减仓；常规做T的 {config.reasonable_min_shares} 股下限在风控场景可被打破。",
    )
    for zone in plan.get("sell_zones", []):
        zone_range = zone.get("range") or {}
        line = float(zone_range.get("low") or 0)
        if line <= 0:
            continue
        add(
            "planned_reverse_t",
            1,
            "SELL",
            can_reverse_t and price >= line and (sell_confirmed or not snap.get("has_minute")),
            int(zone.get("shares") or 0),
            line,
            zone.get("label") or "计划反T卖出",
            zone.get("condition") or "提前计划区间触发，按计划卖出T仓。",
        )
    if dynamic_available and dynamic_sell_1 and dynamic_sell_2 and dynamic_sell_3:
        add(
            "intraday_rebound_t",
            1,
            "SELL",
            can_reverse_t and price >= dynamic_sell_1 and sell_confirmed and not breakout_guard,
            capped_sell_delta(400, shares, config.reasonable_min_shares),
            dynamic_sell_1,
            "日内低点反抽一档",
            f"按今天分钟线低点 {dynamic_low:.2f} 到高点 {dynamic_high:.2f} 计算，反抽到 38.2% 附近先卖一笔。{sell_note}",
        )
        add(
            "intraday_rebound_t",
            1,
            "SELL",
            can_reverse_t and price >= dynamic_sell_2 and sell_confirmed,
            capped_sell_delta(700, shares, config.reasonable_min_shares),
            dynamic_sell_2,
            "日内低点反抽二档",
            f"按今天日内振幅中位高抛，不等回到昨收；用于把下跌日里的反抽变成降仓机会。{guard_note}",
        )
        add(
            "intraday_rebound_t",
            1,
            "SELL",
            can_reverse_t and price >= dynamic_sell_3 and (sell_confirmed or price >= dynamic_sell_3 * 1.003),
            capped_sell_delta(600 if breakout_guard else 1000, shares, config.reasonable_min_shares),
            dynamic_sell_3,
            "日内低点反抽三档",
            f"反抽接近日内高位区，优先降低满仓压力；若随后回落，再按买回线处理。{guard_note}",
        )
    add(
        "micro_reverse_t",
        1,
        "SELL",
        can_reverse_t and price >= micro_1 and sell_confirmed and not breakout_guard,
        capped_sell_delta(500, shares, config.reasonable_min_shares),
        micro_1,
        "日内小反T第一笔",
        f"不等回本，较开盘/昨收/VWAP 小幅拉起先卖一笔，建立可买回的 T 仓。{sell_note}",
    )
    add(
        "micro_reverse_t",
        1,
        "SELL",
        can_reverse_t and price >= micro_2 and sell_confirmed,
        capped_sell_delta(500 if breakout_guard else 800, shares, config.reasonable_min_shares),
        micro_2,
        "日内小反T第二笔",
        f"盘中反弹扩大后再卖一笔，优先降低满仓压力。{guard_note}",
    )
    add(
        "micro_reverse_t",
        1,
        "SELL",
        can_reverse_t and price >= micro_3 and (sell_confirmed or price >= micro_3 * 1.003),
        capped_sell_delta(800 if breakout_guard else 1200, shares, config.reasonable_min_shares),
        micro_3,
        "日内小反T第三笔",
        f"日内涨幅超过约 2.2%，加大反T股数；后续只在明显回落时买回。{guard_note}",
    )
    add(
        "reverse_t",
        2,
        "SELL",
        can_reverse_t and price >= levels["mid_trim"],
        capped_sell_delta(500, shares, config.reasonable_min_shares),
        levels["mid_trim"],
        "反弹到箱体中位先减",
        "反弹到箱体中位先建立反T仓，只有回落形成足够价差才买回。",
    )
    add(
        "reverse_t",
        2,
        "SELL",
        can_reverse_t and price >= levels["weak_trim"],
        capped_sell_delta(700, shares, config.reasonable_min_shares),
        levels["weak_trim"],
        "中上沿动能转弱减仓",
        "到箱体 62% 分位附近，不等最高点，按纪律卖出一笔。",
    )
    add(
        "reverse_t",
        1,
        "SELL",
        can_reverse_t and price >= levels["strong_trim"],
        capped_sell_delta(1000, shares, config.reasonable_min_shares),
        levels["strong_trim"],
        "上沿强减仓",
        "接近箱体上沿，优先降低总仓位；若盘中放量冲高回落，执行更坚决。",
    )
    if buyback_shares > 0 and buyback_anchor is not None:
        buyback_line = min(buyback_anchor * (1 - config.micro_buyback_discount), intraday_ref * 0.996, vwap_ref * 0.996)
        deep_discount = price <= buyback_anchor * (1 - config.micro_buyback_discount * 1.5)
        buyback_delta = min(
            buyback_shares,
            full_inventory_gap,
        )
        add(
            "micro_buyback",
            2,
            "BUY",
            buyback_delta > 0 and price <= buyback_line and (buy_confirmed or deep_discount),
            buyback_delta,
            buyback_line,
            "反T买回线",
            f"只买回已卖出的T仓，买回后不超过 {holding.full_shares} 股；目标价差至少 {config.micro_buyback_discount:.1%}。{buy_note}",
        )

    regime = regime or {}
    validated_positive = validated_positive or {}
    positive_buy_line = float(
        validated_positive.get("price_line")
        or min(
            float(levels["daily_close"])
            * (1 - config.positive_t_previous_close_deviation),
            vwap_ref * (1 - config.positive_t_entry_deviation),
        )
    )
    stack_ok = (
        positive_anchor is None
        or price <= positive_anchor * (1 - config.positive_t_stack_gap)
    )
    add(
        "positive_t",
        3,
        "BUY",
        can_positive_t
        and bool(regime.get("positive_t_allowed"))
        and bool(validated_positive.get("signal"))
        and stack_ok,
        positive_buy_delta,
        positive_buy_line,
        "严格横盘正T低吸",
        (
            f"每档 {config.max_positive_t_buy_shares} 股，累计正T仓不超过 {config.max_positive_t_open_shares} 股；"
            f"需低于VWAP {config.positive_t_entry_deviation:.1%}、位于60日区间下半部且5分钟止跌。"
            f"反弹 {config.positive_t_profit_target:.1%} 止盈，回撤 {config.positive_t_stop_loss:.1%} 止损，"
            f"最多跨 {config.positive_t_max_holding_sessions} 个交易日。{regime.get('reason', '')}"
        ),
    )

    if stage["key"] == "UNDERWEIGHT":
        add(
            "inventory_recovery",
            2,
            "BUY",
            price <= positive_buy_line and buy_confirmed,
            int(stage["positive_t_buy_cap"]),
            positive_buy_line,
            "低于合理仓：分批恢复库存",
            f"当前低于 {config.reasonable_min_shares} 股，不再常规高抛；只在低位确认后分批恢复。{buy_note}",
        )
    management_kinds = {
        "risk",
        "micro_buyback",
        "positive_t_exit",
        "positive_t_stop",
    }
    validated_entry_kinds = set(BYD_T_VALIDATION.get("allowed_new_entry_kinds") or [])
    for alert in alerts:
        enabled = alert["kind"] in management_kinds or (
            BYD_T_VALIDATION["execution_enabled"]
            and alert["kind"] in validated_entry_kinds
        )
        alert["execution_enabled"] = enabled
        if not enabled:
            alert["research_triggered"] = bool(alert["triggered"])
            alert["triggered"] = False
            alert["detail"] = f"{alert['detail']} 当前仅作观察：该方向未通过历史验证。"
    return sorted(alerts, key=lambda item: (not item["triggered"], item["priority"]))


def build_minute_payload(
    daily: pd.DataFrame,
    minutes: pd.DataFrame,
    holding: BydHolding = BydHolding(),
    config: BydMinuteConfig = BydMinuteConfig(),
    data_status: str = "daily_fallback",
    sold_today_shares: int = 0,
    sold_today_price: float | None = None,
    bought_today_shares: int = 0,
    bought_today_price: float | None = None,
    open_t_shares: int = 0,
    open_t_price: float | None = None,
    open_positive_shares: int = 0,
    open_positive_price: float | None = None,
) -> dict[str, Any]:
    levels = daily_range_levels(daily)
    minutes = normalize_minutes(minutes)
    snap = minute_snapshot(minutes, levels)
    plan = planned_t_ranges(daily, levels, holding, config)
    dynamic_zones = intraday_dynamic_zones(minutes, snap)
    indicators = timeframe_indicator_snapshot(minutes)
    regime = daily_sideways_snapshot(daily)
    validated_positive = validated_positive_t_snapshot(minutes, levels, regime, config)
    stage = holding_stage(holding.shares, config, holding.full_shares)
    price = float(snap["last"])
    market_value = holding.shares * price
    cost_value = holding.shares * holding.cost
    pnl = market_value - cost_value
    pnl_pct = price / holding.cost - 1
    range_pos = (price - levels["range_low"]) / (levels["range_high"] - levels["range_low"]) if levels["range_high"] > levels["range_low"] else np.nan
    alerts = build_alerts(
        holding,
        config,
        levels,
        snap,
        indicators=indicators,
        dynamic_zones=dynamic_zones,
        plan=plan,
        regime=regime,
        validated_positive=validated_positive,
        sold_today_shares=sold_today_shares,
        sold_today_price=sold_today_price,
        bought_today_shares=bought_today_shares,
        bought_today_price=bought_today_price,
        open_t_shares=open_t_shares,
        open_t_price=open_t_price,
        open_positive_shares=open_positive_shares,
        open_positive_price=open_positive_price,
    )
    triggered = [item for item in alerts if item["triggered"]]
    if triggered:
        primary = triggered[0]
    elif stage["key"] == "INTRADAY_OVERWEIGHT":
        positive_anchor = weighted_price(
            [
                (bought_today_shares, bought_today_price),
                (open_positive_shares, open_positive_price),
            ]
        )
        positive_exit_line = (
            positive_anchor * (1 + config.positive_t_profit_target)
            if positive_anchor is not None
            else None
        )
        primary = {
            "action": "WAIT_RECORD_OR_EXIT",
            "title": "超出基准仓：记录待卖T仓并等待盈利退出",
            "detail": (
                f"当前超过基准满仓 {stage['intraday_excess_shares']} 股；"
                + (
                    f"已记录均价对应的观察退出线为 {positive_exit_line:.2f}。"
                    if positive_exit_line is not None
                    else "请补录待卖出股数和买入均价后再计算盈利退出线。"
                )
                + "允许跨日，不因时间强制亏损卖出。"
            ),
            "shares_delta": -int(stage["intraday_excess_shares"]),
        }
    else:
        if BYD_T_VALIDATION["execution_enabled"]:
            primary = {
                "action": "WAIT_STRICT_POSITIVE_T",
                "title": "等待严格横盘正T信号",
                "detail": (
                    f"{regime.get('reason', '')} {validated_positive.get('reason', '')}"
                    "反T继续暂停。"
                ).strip(),
                "shares_delta": 0,
            }
        else:
            primary = {
                "action": "WAIT_VALIDATION",
                "title": "暂停新开T仓：历史验证未同时满足胜率和盈利性",
                "detail": BYD_T_VALIDATION["decision"],
                "shares_delta": 0,
            }
    recent_minutes = []
    if not minutes.empty:
        keep = minutes.tail(80)
        recent_minutes = [
            {
                "time": pd.to_datetime(row["trade_time"]).strftime("%H:%M"),
                "close": float(row["close"]),
                "volume": float(row["volume"]) if "volume" in keep.columns and pd.notna(row.get("volume")) else None,
            }
            for _, row in keep.iterrows()
        ]
    return {
        "symbol": SYMBOL,
        "name": NAME,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_status": data_status,
        "holding": {
            "shares": holding.shares,
            "cost": holding.cost,
            "full_shares": holding.full_shares,
            "reasonable_min_shares": config.reasonable_min_shares,
            "intraday_limit_shares": holding.full_shares + config.max_intraday_extra_shares,
            "intraday_excess_shares": int(stage["intraday_excess_shares"]),
            "market_value": market_value,
            "cost_value": cost_value,
            "unrealized_pnl": pnl,
            "unrealized_pnl_pct": pnl_pct * 100,
            "inventory_ratio": holding.shares / holding.full_shares if holding.full_shares else None,
        },
        "stage": stage,
        "minute": snap,
        "planned_t": plan,
        "intraday_dynamic": dynamic_zones,
        "indicators": indicators,
        "regime": regime,
        "validated_positive_t": validated_positive,
        "daily_levels": levels,
        "range_position_pct": range_pos * 100 if np.isfinite(range_pos) else None,
        "primary_action": primary,
        "alerts": alerts,
        "today_t": {
            "sold_shares": round_lot(sold_today_shares),
            "sold_price": sold_today_price,
            "bought_shares": round_lot(bought_today_shares),
            "bought_price": bought_today_price,
            "open_t_shares": round_lot(open_t_shares),
            "open_t_price": open_t_price,
            "open_positive_shares": round_lot(open_positive_shares),
            "open_positive_price": open_positive_price,
            "buyback_enabled": (round_lot(sold_today_shares) > 0 and bool(sold_today_price))
            or (round_lot(open_t_shares) > 0 and bool(open_t_price)),
            "positive_exit_enabled": (
                (round_lot(bought_today_shares) > 0 and bool(bought_today_price))
                or (round_lot(open_positive_shares) > 0 and bool(open_positive_price))
            ),
        },
        "validation": BYD_T_VALIDATION,
        "intraday_levels": intraday_reverse_t_levels(snap, levels, config),
        "recent_minutes": recent_minutes,
        "playbook": [
            "成功率闸门优先：只启用通过训练、验证和样本外检查的严格横盘正T；反T继续暂停。",
            "2020-2021强上涨阶段由滞后横盘指标自动排除，不参与开仓，不是事后删除行情。",
            f"常规收盘仓位保持在 {config.reasonable_min_shares}-{holding.full_shares} 股；只有箱体破位风控才允许低于下限。",
            f"正T每次 {config.max_positive_t_buy_shares} 股，最多累计 {config.max_positive_t_open_shares} 股；相邻买入至少再下跌 {config.positive_t_stack_gap:.1%}。",
            f"正T反弹约 {config.positive_t_profit_target:.1%} 止盈、回撤 {config.positive_t_stop_loss:.1%} 止损，最多持有 {config.positive_t_max_holding_sessions} 个交易日。",
            f"若你已人工卖出反T仓，录入待买回数量和卖出均价；回落约 {config.micro_buyback_discount:.1%} 后，盈利买回提醒仍可执行。",
            "14:35后开仓门槛提高到低于VWAP 1.1%；没有完整5分钟信号时只展示计划价，不触发。",
        ],
    }
