"""BYD single-stock minute-level T strategy.

This module is built for a high-inventory holder of 002594.SZ. It is not a
general stock selector. The main objective is to reduce inventory first, then
switch gradually from reverse T to positive T when shares are lower.
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
    shares: int = 10500
    cost: float = 110.6061
    full_shares: int = 10500


@dataclass(frozen=True)
class BydMinuteConfig:
    core_shares: int = 6000
    positive_t_threshold: int = 6000
    transition_threshold: int = 7500
    overload_threshold: int = 9000
    first_goal_shares: int = 8500
    second_goal_shares: int = 7500
    final_goal_shares: int = 6000
    max_buyback_shares_high_inventory: int = 0
    max_buyback_shares_transition: int = 300
    max_positive_t_buy_shares: int = 500
    micro_sell_1: float = 0.008
    micro_sell_2: float = 0.015
    micro_sell_3: float = 0.022
    micro_buyback_discount: float = 0.008
    max_daily_reverse_t_shares: int = 2500


def round_lot(shares: float) -> int:
    return max(int(shares // LOT_SIZE) * LOT_SIZE, 0)


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
    hist = daily.tail(80).copy()
    range_low = float(hist["low"].shift(1).tail(60).min())
    range_high = float(hist["high"].shift(1).tail(60).max())
    low_20 = float(hist["low"].shift(1).tail(20).min())
    high_20 = float(hist["high"].shift(1).tail(20).max())
    close_120 = hist["close"].shift(1).tail(120).dropna()
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


def price_band(low: float, high: float) -> dict[str, Any]:
    lo = min(float(low), float(high))
    hi = max(float(low), float(high))
    return {
        "low": round(lo, 2),
        "high": round(hi, 2),
        "label": f"{lo:.2f}-{hi:.2f}",
    }


def planned_t_ranges(daily: pd.DataFrame, levels: dict[str, Any], holding: BydHolding) -> dict[str, Any]:
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
    sell_zones = [
        {
            "key": "PLAN_SELL_1",
            "label": "高确定性反T一档",
            "range": price_band(sell_1_low, sell_1_high),
            "shares": -500 if holding.shares >= 9000 else -300,
            "condition": "反弹修复到前收盘附近或半个ATR以内，先卖一笔建立T仓。",
        },
        {
            "key": "PLAN_SELL_2",
            "label": "高确定性反T二档",
            "range": price_band(sell_2_low, sell_2_high),
            "shares": -800 if holding.shares >= 9000 else -500,
            "condition": "反弹扩大到约0.5-0.7个ATR，优先降低满仓压力。",
        },
        {
            "key": "PLAN_SELL_3",
            "label": "强反抽减仓档",
            "range": price_band(sell_3_low, sell_3_high),
            "shares": -1200 if holding.shares >= 9000 else -800,
            "condition": "接近120日低位分位或1个ATR反抽，作为日内/隔日强减仓区。",
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
        "stage_note": "满仓阶段只做反T减仓，不新增底仓；买回只针对已卖出的T仓。",
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


def holding_stage(shares: int, config: BydMinuteConfig) -> dict[str, Any]:
    if shares >= config.overload_threshold:
        return {
            "key": "OVERLOADED_REVERSE_T",
            "label": "超载仓：反T降仓",
            "goal_shares": config.first_goal_shares,
            "mode": "先卖后买，买回从严",
            "buyback_cap": config.max_buyback_shares_high_inventory,
        }
    if shares >= config.transition_threshold:
        return {
            "key": "REDUCE_REVERSE_T",
            "label": "高仓：反T为主",
            "goal_shares": config.second_goal_shares,
            "mode": "反弹减仓，深回落少量买回",
            "buyback_cap": config.max_buyback_shares_transition,
        }
    if shares >= config.positive_t_threshold:
        return {
            "key": "TRANSITION",
            "label": "过渡仓：反T转正T",
            "goal_shares": config.final_goal_shares,
            "mode": "高抛低吸均可，但不增总仓",
            "buyback_cap": config.max_buyback_shares_transition,
        }
    return {
        "key": "POSITIVE_T",
        "label": "可控仓：正T为主",
        "goal_shares": max(config.core_shares, min(shares, config.final_goal_shares)),
        "mode": "低吸后高抛，保留核心仓",
        "buyback_cap": config.max_positive_t_buy_shares,
    }


def build_alerts(
    holding: BydHolding,
    config: BydMinuteConfig,
    levels: dict[str, Any],
    snap: dict[str, Any],
    indicators: dict[str, Any] | None = None,
    dynamic_zones: dict[str, Any] | None = None,
    plan: dict[str, Any] | None = None,
    sold_today_shares: int = 0,
    sold_today_price: float | None = None,
    open_t_shares: int = 0,
    open_t_price: float | None = None,
) -> list[dict[str, Any]]:
    price = float(snap["last"])
    shares = holding.shares
    stage = holding_stage(shares, config)
    alerts: list[dict[str, Any]] = []

    def add(kind: str, priority: int, action: str, trigger: bool, shares_delta: int, price_line: float, title: str, detail: str) -> None:
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

    overload = stage["key"] == "OVERLOADED_REVERSE_T"
    high_inventory = stage["key"] in {"OVERLOADED_REVERSE_T", "REDUCE_REVERSE_T"}
    can_positive_t = stage["key"] in {"TRANSITION", "POSITIVE_T"}
    can_buy_back = int(stage["buyback_cap"]) > 0
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
    open_t_shares = round_lot(open_t_shares)
    open_t_price = float(open_t_price) if open_t_price and open_t_price > 0 else None
    buyback_shares = round_lot(sold_today_shares + open_t_shares)
    buyback_anchor = sold_today_price or open_t_price
    plan = plan or {}
    dynamic_zones = dynamic_zones or {}
    dynamic_available = bool(dynamic_zones.get("available"))
    dynamic_sell_1 = dynamic_zones.get("rebound_sell_1")
    dynamic_sell_2 = dynamic_zones.get("rebound_sell_2")
    dynamic_sell_3 = dynamic_zones.get("rebound_sell_3")
    dynamic_low = dynamic_zones.get("day_low")
    dynamic_high = dynamic_zones.get("day_high")

    add(
        "risk",
        1,
        "SELL",
        price <= levels["support_break"],
        -min(round_lot(max(shares - config.core_shares, 0)), 1500),
        levels["support_break"],
        "箱体下破降风险",
        "跌破 60 日箱体下沿 1.5%，不做补仓幻想，先卖出 1000-1500 股。",
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
            high_inventory and price >= line and (sell_confirmed or not snap.get("has_minute")),
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
            high_inventory and price >= dynamic_sell_1 and sell_confirmed and not breakout_guard,
            -400 if overload else -300,
            dynamic_sell_1,
            "日内低点反抽一档",
            f"按今天分钟线低点 {dynamic_low:.2f} 到高点 {dynamic_high:.2f} 计算，反抽到 38.2% 附近先卖一笔。{sell_note}",
        )
        add(
            "intraday_rebound_t",
            1,
            "SELL",
            high_inventory and price >= dynamic_sell_2 and sell_confirmed,
            -700 if overload else -500,
            dynamic_sell_2,
            "日内低点反抽二档",
            f"按今天日内振幅中位高抛，不等回到昨收；用于把下跌日里的反抽变成降仓机会。{guard_note}",
        )
        add(
            "intraday_rebound_t",
            1,
            "SELL",
            high_inventory and price >= dynamic_sell_3 and (sell_confirmed or price >= dynamic_sell_3 * 1.003),
            -1000 if overload and not breakout_guard else -600,
            dynamic_sell_3,
            "日内低点反抽三档",
            f"反抽接近日内高位区，优先降低满仓压力；若随后回落，再按买回线处理。{guard_note}",
        )
    add(
        "micro_reverse_t",
        1,
        "SELL",
        high_inventory and price >= micro_1 and sell_confirmed and not breakout_guard,
        -500 if overload else -300,
        micro_1,
        "日内小反T第一笔",
        f"不等回本，较开盘/昨收/VWAP 小幅拉起先卖一笔，建立可买回的 T 仓。{sell_note}",
    )
    add(
        "micro_reverse_t",
        1,
        "SELL",
        high_inventory and price >= micro_2 and sell_confirmed,
        -500 if breakout_guard else (-800 if overload else -500),
        micro_2,
        "日内小反T第二笔",
        f"盘中反弹扩大后再卖一笔，优先降低满仓压力。{guard_note}",
    )
    add(
        "micro_reverse_t",
        1,
        "SELL",
        high_inventory and price >= micro_3 and (sell_confirmed or price >= micro_3 * 1.003),
        -800 if breakout_guard else (-1200 if overload else -800),
        micro_3,
        "日内小反T第三笔",
        f"日内涨幅超过约 2.2%，加大反T股数；后续只在明显回落时买回。{guard_note}",
    )
    add(
        "reverse_t",
        2,
        "SELL",
        high_inventory and price >= levels["mid_trim"],
        -500 if overload else -300,
        levels["mid_trim"],
        "反弹到箱体中位先减",
        "当前仓位很高，反弹到中位先卖一笔，把库存往阶段目标压。",
    )
    add(
        "reverse_t",
        2,
        "SELL",
        price >= levels["weak_trim"],
        -700 if high_inventory else -400,
        levels["weak_trim"],
        "中上沿动能转弱减仓",
        "到箱体 62% 分位附近，不等最高点，按纪律卖出一笔。",
    )
    add(
        "reverse_t",
        1,
        "SELL",
        price >= levels["strong_trim"],
        -1000 if high_inventory else -600,
        levels["strong_trim"],
        "上沿强减仓",
        "接近箱体上沿，优先降低总仓位；若盘中放量冲高回落，执行更坚决。",
    )
    add(
        "buyback",
        3,
        "BUY",
        can_buy_back and price <= levels["range_low"] * 1.01 and buy_confirmed,
        int(stage["buyback_cap"]),
        levels["range_low"] * 1.01,
        "下沿确认才买回",
        f"只在已经降过仓且低于买回上限时使用；满仓阶段此条禁用。{buy_note}",
    )
    if buyback_shares > 0 and buyback_anchor is not None:
        buyback_line = min(buyback_anchor * (1 - config.micro_buyback_discount), intraday_ref * 0.996, vwap_ref * 0.996)
        deep_discount = price <= buyback_anchor * (1 - config.micro_buyback_discount * 1.5)
        add(
            "micro_buyback",
            2,
            "BUY",
            price <= buyback_line and (buy_confirmed or deep_discount),
            min(buyback_shares, config.max_daily_reverse_t_shares),
            buyback_line,
            "反T买回线",
            f"买回今天或隔日待买回的 T 仓；买回价必须低于卖出锚点，不增加原始满仓。{buy_note}",
        )
    add(
        "positive_t",
        3,
        "BUY",
        can_positive_t and price <= levels["range_low"] * 1.008 and buy_confirmed,
        min(config.max_positive_t_buy_shares, max(config.final_goal_shares - shares, 0)),
        levels["range_low"] * 1.008,
        "正T低吸",
        f"仓位降到可控后才启用，低位买入的 T 仓必须在反弹时卖出。{buy_note}",
    )
    return sorted(alerts, key=lambda item: (not item["triggered"], item["priority"]))


def build_minute_payload(
    daily: pd.DataFrame,
    minutes: pd.DataFrame,
    holding: BydHolding = BydHolding(),
    config: BydMinuteConfig = BydMinuteConfig(),
    data_status: str = "daily_fallback",
    sold_today_shares: int = 0,
    sold_today_price: float | None = None,
    open_t_shares: int = 0,
    open_t_price: float | None = None,
) -> dict[str, Any]:
    levels = daily_range_levels(daily)
    minutes = normalize_minutes(minutes)
    snap = minute_snapshot(minutes, levels)
    plan = planned_t_ranges(daily, levels, holding)
    dynamic_zones = intraday_dynamic_zones(minutes, snap)
    indicators = timeframe_indicator_snapshot(minutes)
    stage = holding_stage(holding.shares, config)
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
        sold_today_shares=sold_today_shares,
        sold_today_price=sold_today_price,
        open_t_shares=open_t_shares,
        open_t_price=open_t_price,
    )
    triggered = [item for item in alerts if item["triggered"]]
    if triggered:
        primary = triggered[0]
    elif holding.shares >= config.overload_threshold and price < levels["mid_trim"]:
        micro = intraday_reverse_t_levels(snap, levels, config)
        if dynamic_zones.get("available"):
            primary = {
                "action": "WAIT_REBOUND",
                "title": "等待日内低位反抽",
                "detail": (
                    f"当前处于{dynamic_zones['state']}，今日低位区 {dynamic_zones['low_zone']['label']}；"
                    f"若反抽到 {dynamic_zones['rebound_sell_1']:.2f}/{dynamic_zones['rebound_sell_2']:.2f}/{dynamic_zones['rebound_sell_3']:.2f} 再分批卖。"
                ),
                "shares_delta": 0,
            }
        else:
            first_zone = (plan.get("sell_zones") or [{}])[0]
            second_zone = (plan.get("sell_zones") or [{}, {}])[1] if len(plan.get("sell_zones") or []) > 1 else {}
            third_zone = (plan.get("sell_zones") or [{}, {}, {}])[2] if len(plan.get("sell_zones") or []) > 2 else {}
            first_label = first_zone.get("range", {}).get("label") or f"{micro['micro_sell_1']:.2f}"
            second_label = second_zone.get("range", {}).get("label") or f"{micro['micro_sell_2']:.2f}"
            third_label = third_zone.get("range", {}).get("label") or f"{micro['micro_sell_3']:.2f}"
            primary = {
                "action": "WAIT_REBOUND",
                "title": "等待计划反T区间",
                "detail": (
                    f"分钟线不可用，按提前计划执行："
                    f"{first_label} / {second_label} / {third_label} 分批卖。"
                ),
                "shares_delta": 0,
            }
    else:
        primary = {
            "action": "HOLD",
            "title": "未触发交易",
            "detail": "价格未到计划线，继续等待。",
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
        "daily_levels": levels,
        "range_position_pct": range_pos * 100 if np.isfinite(range_pos) else None,
        "primary_action": primary,
        "alerts": alerts,
        "today_t": {
            "sold_shares": round_lot(sold_today_shares),
            "sold_price": sold_today_price,
            "open_t_shares": round_lot(open_t_shares),
            "open_t_price": open_t_price,
            "buyback_enabled": (round_lot(sold_today_shares) > 0 and bool(sold_today_price))
            or (round_lot(open_t_shares) > 0 and bool(open_t_price)),
        },
        "intraday_levels": intraday_reverse_t_levels(snap, levels, config),
        "recent_minutes": recent_minutes,
        "playbook": [
            "分钟线不可稳定获取时，主策略回退为提前计划区间：按前复权日线、ATR和箱体分位给出高确定性反T区。",
            "低位不追卖，只在计划卖出一档/二档/三档触发时分批卖 500/800/1200 股。",
            "反T卖出后必须设置买回锚点，回落到卖价下方约 0.8%-1.2% 才买回。",
            "买回可以当日完成，也可以隔日完成；没到买回线就保留现金，不强行回补。",
            "降到 7500 股附近，允许下沿确认后少量买回 300 股。",
            "降到 6000 股以下，正T成为主策略：低吸 300-500 股，反弹卖出 T 仓。",
        ],
    }
