"""Extended daily pattern signals.

This module ports multiple discretionary tactics into self-contained daily-data
rules for the selector. These are observation signals first: they use Tushare
daily OHLCV fields and avoid minute/L2 assumptions unless the signal text
explicitly says a later intraday confirmation is needed.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant.features.variable_library import build_continuous_ohlc


EXTENDED_STRATEGIES: list[dict[str, str]] = [
    {"key": "CHANGAN", "label": "长安战法", "status": "日线三日确认"},
    {"key": "PINGHANG", "label": "平行重炮", "status": "日线右侧确认"},
    {"key": "DOUBLE_GUN", "label": "双枪战法", "status": "日线右侧确认"},
    {"key": "YIDONG_DILIAN", "label": "异动地量", "status": "日线低吸观察"},
    {"key": "NANA", "label": "娜娜图形", "status": "日线回调观察"},
    {"key": "GOLDEN_BOWL", "label": "黄金碗", "status": "日线支撑观察"},
    {"key": "BREATHING", "label": "呼吸结构", "status": "日线节奏观察"},
    {"key": "KENGQI", "label": "坑里起好货", "status": "日线填坑观察"},
    {"key": "DUICHEN_VA", "label": "对称VA", "status": "日线企稳观察"},
    {"key": "ZAIHOU", "label": "灾后重建", "status": "日线回踩观察"},
    {"key": "YUEYUE", "label": "跃跃欲试", "status": "日线平台观察"},
    {"key": "KEY_K", "label": "关键K", "status": "日线关键位"},
    {"key": "VIOLENCE_K", "label": "暴力K", "status": "日线底部异动"},
]


PROJECT_ROOT = Path(__file__).resolve().parents[4]
STOCK_BASIC_PATHS = [
    PROJECT_ROOT / "data/cache/tushare_stock_basic_all.parquet",
    PROJECT_ROOT / "data/raw/stock_basic.parquet",
    PROJECT_ROOT / "data/cache/source_merge/tushare/tushare_stock_basic_all.parquet",
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None or pd.isna(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "nat"} else text


@lru_cache(maxsize=1)
def _stock_basic_map() -> dict[str, dict[str, str]]:
    for path in STOCK_BASIC_PATHS:
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path)
        except Exception:
            continue
        if "ts_code" not in df.columns:
            continue
        result: dict[str, dict[str, str]] = {}
        for _, row in df.iterrows():
            symbol = _clean_text(row.get("ts_code"))
            if not symbol:
                continue
            result[symbol] = {
                "name": _clean_text(row.get("name")),
                "industry": _clean_text(row.get("industry")),
            }
        if result:
            return result
    return {}


def _normalize_daily(path: Path, signal_date: str | None) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if df.empty:
        return df
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if "trade_date" in out.columns:
        trade_date = pd.to_datetime(out["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
        if "date" not in out.columns:
            out["date"] = trade_date
        else:
            out["date"] = out["date"].fillna(trade_date)
    if "volume" not in out.columns and "vol" in out.columns:
        out["volume"] = out["vol"]
    elif "volume" in out.columns and "vol" in out.columns:
        out["volume"] = out["volume"].fillna(out["vol"])
    if "ts_code" not in out.columns:
        out["ts_code"] = path.stem
    out["symbol"] = out["ts_code"].fillna(path.stem)
    if "name" not in out.columns:
        out["name"] = ""
    basic = _stock_basic_map().get(str(out["ts_code"].dropna().iloc[-1]) if out["ts_code"].notna().any() else path.stem, {})
    fallback_name = basic.get("name", "")
    fallback_industry = basic.get("industry", "")
    out["name"] = out["name"].map(_clean_text)
    if fallback_name:
        out["name"] = out["name"].replace("", fallback_name)
    if "industry" not in out.columns:
        out["industry"] = fallback_industry
    else:
        out["industry"] = out["industry"].map(_clean_text)
        if fallback_industry:
            out["industry"] = out["industry"].replace("", fallback_industry)
    name_series = out["name"].fillna("").astype(str)
    st_mask = name_series.str.upper().str.contains("ST") | name_series.str.contains("退")
    out = out[~st_mask].copy()
    needed = ["date", "open", "high", "low", "close", "volume"]
    if not set(needed) <= set(out.columns):
        return pd.DataFrame()
    out = out.dropna(subset=needed).sort_values("date")
    if signal_date:
        cutoff = pd.to_datetime(signal_date, errors="coerce")
        if pd.notna(cutoff):
            out = out[out["date"] <= cutoff]
    out = out.tail(180).reset_index(drop=True)
    if len(out) < 10:
        return pd.DataFrame()
    if signal_date:
        cutoff = pd.to_datetime(signal_date, errors="coerce")
        if pd.notna(cutoff) and out["date"].max() < cutoff - pd.Timedelta(days=14):
            return pd.DataFrame()
    out["pre_close"] = out.get("pre_close", out["close"].shift(1))
    out["pre_close"] = out["pre_close"].replace(0, np.nan).fillna(out["close"].shift(1))
    if "pct_chg" not in out.columns:
        out["pct_chg"] = out["close"].pct_change() * 100
    out["pct_chg"] = out["pct_chg"].fillna(out["close"].pct_change() * 100).fillna(0)
    price = build_continuous_ohlc(out)
    out[["open", "high", "low", "close"]] = price[["open", "high", "low", "close"]]
    out["pre_close"] = out["close"].shift(1).replace(0, np.nan).fillna(out["pre_close"])
    out["amplitude"] = (out["high"] - out["low"]) / out["pre_close"].replace(0, np.nan) * 100
    out["close_pos"] = (out["close"] - out["low"]) / (out["high"] - out["low"]).replace(0, np.nan)
    out["vol_ratio_5"] = out["volume"] / out["volume"].shift(1).rolling(5, min_periods=1).mean()
    out["vol_ratio_prev"] = out["volume"] / out["volume"].shift(1).replace(0, np.nan)
    out["vol_ma10"] = out["volume"].rolling(10, min_periods=3).mean()
    out["vol_ma20"] = out["volume"].rolling(20, min_periods=5).mean()
    out["is_rise"] = out["close"] > out["open"]
    out["is_shrink"] = out["volume"] < out["volume"].shift(1) * 0.75
    out["is_beidou"] = (out["pct_chg"] >= 3) & (out["vol_ratio_5"] >= 1.5)
    out["is_big_yin"] = (out["close"] < out["open"]) & (out["vol_ratio_5"] >= 1.5) & (out["pct_chg"] <= -2)
    out["ma3"] = out["close"].rolling(3, min_periods=1).mean()
    out["ma6"] = out["close"].rolling(6, min_periods=2).mean()
    out["ma12"] = out["close"].rolling(12, min_periods=4).mean()
    out["ma24"] = out["close"].rolling(24, min_periods=8).mean()
    out["bbi"] = (out["ma3"] + out["ma6"] + out["ma12"] + out["ma24"]) / 4
    out["zg_white"] = out["close"].ewm(span=10, adjust=False).mean().ewm(span=10, adjust=False).mean()
    out["dg_yellow"] = (
        out["close"].rolling(14, min_periods=8).mean()
        + out["close"].rolling(28, min_periods=14).mean()
        + out["close"].rolling(57, min_periods=28).mean()
        + out["close"].rolling(114, min_periods=60).mean()
    ) / 4
    out["kdj_j"] = _calculate_kdj_j(out)
    return out


def _calculate_kdj_j(df: pd.DataFrame) -> pd.Series:
    low9 = df["low"].rolling(9, min_periods=3).min()
    high9 = df["high"].rolling(9, min_periods=3).max()
    rsv = (df["close"] - low9) / (high9 - low9).replace(0, np.nan) * 100
    rsv = rsv.fillna(50)
    k = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    d = k.ewm(alpha=1 / 3, adjust=False).mean()
    return 3 * k - 2 * d


def _signal(
    key: str,
    name: str,
    logic: str,
    reason: str,
    buy_plan: str,
    sell_plan: str,
    strength: float,
    timeframe: str = "日线级，收盘确认，T+1 开盘观察",
) -> dict[str, Any]:
    return {
        "strategy_key": key,
        "strategy_family": key,
        "strategy_name": name,
        "timeframe": timeframe,
        "logic": logic,
        "reason": reason,
        "buy_plan": buy_plan,
        "sell_plan": sell_plan,
        "metrics": None,
        "metrics_text": "暂无本项目正式回测，先作为扩展策略日线观察信号",
        "strength_score": round(float(strength), 3),
    }


def _detect_changan(df: pd.DataFrame) -> dict[str, Any] | None:
    if len(df) < 12:
        return None
    d1, d2, d3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    half_vol = _safe_float(d3["volume"]) <= _safe_float(d2["volume"]) * 0.55
    if (
        _safe_float(d1["kdj_j"]) < -13
        and _safe_float(d2["pct_chg"]) >= 4
        and bool(d2["is_rise"])
        and _safe_float(d2["vol_ratio_5"]) >= 1.4
        and _safe_float(d2["kdj_j"]) > _safe_float(d1["kdj_j"])
        and 0 < _safe_float(d3["pct_chg"]) < 2.2
        and _safe_float(d3["amplitude"]) < 7
        and half_vol
    ):
        return _signal(
            "CHANGAN",
            "长安战法",
            "第一天 J<-13 形成 B1 痕迹，第二天放量长阳且 J 拐头，第三天小阳分歧转一致并缩半量。",
            f"J1={d1['kdj_j']:.1f}，Day2涨幅={d2['pct_chg']:.1f}%、量比={d2['vol_ratio_5']:.1f}，Day3缩量={d3['volume']/d2['volume']:.0%}",
            "T+1 若开盘高低开在 -1.5% 到 +2.5% 内，且不跌破确认日低点，可小仓观察；高开过大放弃追价。",
            "确认日低点作为风控线；若 3-5 个交易日无法继续放量上攻，优先退出观察。",
            1.1 + min(_safe_float(d2["pct_chg"]) / 10, 0.8),
        )
    return None


def _detect_pinghang(df: pd.DataFrame) -> dict[str, Any] | None:
    if len(df) < 12:
        return None
    scan = df.tail(8).copy()
    yang = scan.index[(scan["is_rise"]) & (scan["is_beidou"])].tolist()
    if len(yang) < 2:
        return None
    y1, y2 = yang[-2], yang[-1]
    if y2 != df.index[-1] or y2 - y1 - 1 < 2:
        return None
    middle = df.loc[y1 + 1 : y2 - 1]
    yin_count = int((middle["close"] <= middle["open"]).sum())
    max_mid_vol = _safe_float(middle["volume"].max())
    row2 = df.loc[y2]
    if (
        yin_count >= len(middle) * 0.5
        and _safe_float(df.loc[y1, "volume"]) >= max_mid_vol * 1.15
        and _safe_float(row2["volume"]) >= max_mid_vol * 1.15
        and _safe_float(row2["volume"]) >= _safe_float(df.loc[y1, "volume"]) * 0.9
        and _safe_float(row2["pct_chg"]) >= 4
        and _safe_float(row2["kdj_j"]) < 55
    ):
        return _signal(
            "PINGHANG",
            "平行重炮",
            "两根放量阳线夹至少两根中间K线，中间以阴线/缩量为主，第二根阳线涨幅>=4%、量能不弱于第一根。",
            f"第二炮涨幅={row2['pct_chg']:.1f}%、J={row2['kdj_j']:.1f}，中间{len(middle)}天阴线{yin_count}天",
            "T+1 开盘不高于确认日收盘 3% 时观察；若回踩第二炮实体上半区不破，可作为更优买点。",
            "跌破第二炮低点或放量长阴失败退出；强势上攻可按 3%-6% 分批止盈。",
            1.0 + min(_safe_float(row2["pct_chg"]) / 10, 0.9),
        )
    return None


def _detect_double_gun(df: pd.DataFrame) -> dict[str, Any] | None:
    if len(df) < 18:
        return None
    gun2 = None
    for idx in range(len(df) - 2, max(0, len(df) - 16), -1):
        row = df.iloc[idx]
        if _safe_float(row["pct_chg"]) >= 3 and bool(row["is_rise"]) and _safe_float(row["vol_ratio_prev"]) >= 1.8:
            gun2 = idx
            break
    if gun2 is None or len(df) - 1 - gun2 > 4:
        return None
    gun1 = None
    for idx in range(gun2 - 3, max(0, gun2 - 12), -1):
        row = df.iloc[idx]
        if _safe_float(row["pct_chg"]) >= 3 and bool(row["is_rise"]) and _safe_float(row["vol_ratio_prev"]) >= 1.8:
            gun1 = idx
            break
    if gun1 is None:
        return None
    middle = df.iloc[gun1 + 1 : gun2]
    avg_mid_ratio = _safe_float(middle["vol_ratio_prev"].mean())
    j_before = _safe_float(df.iloc[gun2 - 1]["kdj_j"]) if gun2 > 0 else 99
    today = df.iloc[-1]
    if avg_mid_ratio < 1.2 and j_before < 20 and 3 <= gun2 - gun1 <= 10 and _safe_float(today["close"]) >= _safe_float(df.iloc[gun2]["low"]):
        return _signal(
            "DOUBLE_GUN",
            "双枪战法",
            "两根放量阳线中间夹缩量整理，第二枪前一日保留 B1 低位痕迹。",
            f"两枪间隔{gun2-gun1}天，量比={df.iloc[gun1]['vol_ratio_prev']:.1f}/{df.iloc[gun2]['vol_ratio_prev']:.1f}，枪前J={j_before:.1f}",
            "第二枪后 1-4 日内仍站稳第二枪低点时观察；开盘高于第二枪收盘 3% 以上不追。",
            "跌破第二枪低点退出；突破第二枪高点后可用 5 日线或 3%-6% 目标管理。",
            0.9 + min(_safe_float(df.iloc[gun2]["vol_ratio_prev"]) / 4, 0.8),
        )
    return None


def _detect_yidong_dilian(df: pd.DataFrame) -> dict[str, Any] | None:
    if len(df) < 16:
        return None
    today = df.iloc[-1]
    yidong_idx = None
    for idx in range(len(df) - 2, max(0, len(df) - 12), -1):
        row = df.iloc[idx]
        if bool(row["is_rise"]) and _safe_float(row["pct_chg"]) >= 2.5 and _safe_float(row["vol_ratio_5"]) >= 1.8:
            yidong_idx = idx
            break
    if yidong_idx is None or len(df) - 1 - yidong_idx < 2:
        return None
    after = df.iloc[yidong_idx + 1 :]
    shrink_ok = _safe_float(today["volume"]) <= _safe_float(df.iloc[yidong_idx]["volume"]) * 0.75
    ground_ok = _safe_float(today["volume"]) <= _safe_float(df["volume"].tail(60).quantile(0.25))
    pullback_ok = _safe_float(today["close"]) >= _safe_float(df.iloc[yidong_idx]["low"]) * 0.96
    low_absorb = -3 <= _safe_float(today["pct_chg"]) <= 1.5 and _safe_float(today["close"]) <= _safe_float(today["bbi"]) * 1.03
    if (shrink_ok or ground_ok) and pullback_ok and low_absorb and _safe_float(today["kdj_j"]) < 35:
        return _signal(
            "YIDONG_DILIAN",
            "异动地量",
            "前期突然放量上涨，随后缩量回调到地量区域，优先寻找异动后的低吸点。",
            f"异动后{len(after)-1}天，今日量/异动量={today['volume']/df.iloc[yidong_idx]['volume']:.0%}，J={today['kdj_j']:.1f}",
            "T+1 低开不破最近回调低点时观察；若高开超过 2% 且无量，不追。",
            "跌破异动日低点或回调低点退出；若放量反包，可转右侧持有。",
            0.8 + max(0, (35 - _safe_float(today["kdj_j"])) / 60),
        )
    return None


def _detect_nana(df: pd.DataFrame) -> dict[str, Any] | None:
    if len(df) < 24:
        return None
    recent = df.tail(15)
    build = recent.iloc[:8]
    pullback = recent.iloc[8:]
    rise_count = int(((build["is_rise"]) & (build["volume"] > build["volume"].shift(1))).sum())
    shrink_count = int((pullback["volume"] < pullback["volume"].shift(1)).sum())
    has_big_yin = bool(recent["is_big_yin"].any())
    today = df.iloc[-1]
    low_pullback = _safe_float(today["pct_chg"]) <= 1.5 and _safe_float(today["close"]) <= _safe_float(today["bbi"]) * 1.03
    if rise_count >= 3 and shrink_count >= 2 and not has_big_yin and low_pullback and _safe_float(today["kdj_j"]) < 15:
        return _signal(
            "NANA",
            "娜娜图形",
            "先连续放量上涨，顶部没有巨量阴线，随后连续缩量回调，J 回到低位。",
            f"放量上涨{rise_count}天，缩量回调{shrink_count}天，J={today['kdj_j']:.1f}",
            "T+1 靠近 5/10 日均线或前一日低点不破时观察；急拉超过 3% 先等待回落。",
            "跌破缩量回调低点退出；重新放量阳线后可用 5 日线跟踪。",
            0.7 + max(0, (10 - _safe_float(today["kdj_j"])) / 50),
        )
    return None


def _detect_golden_bowl(df: pd.DataFrame) -> dict[str, Any] | None:
    if len(df) < 120:
        return None
    today = df.iloc[-1]
    white = _safe_float(today["zg_white"])
    yellow = _safe_float(today["dg_yellow"])
    close = _safe_float(today["close"])
    near_support = yellow <= close <= (white + yellow) / 2 and (close - yellow) / yellow <= 0.04 if yellow > 0 else False
    if white > yellow * 1.005 and near_support and _safe_float(today["kdj_j"]) < 80:
        return _signal(
            "GOLDEN_BOWL",
            "黄金碗",
            "Z哥白线位于大哥黄线之上，股价回落到两线之间，属于多头结构内的支撑观察。",
            f"收盘={close:.2f}，白线={white:.2f}，黄线={yellow:.2f}",
            "T+1 不跌破黄线且开盘偏离白线不超过 2% 时观察；若直接跌破黄线，放弃。",
            "收盘跌破黄线退出；若重新站上白线且放量，可按右侧趋势持有。",
            0.55 + min((white - yellow) / max(yellow, 1) * 10, 0.6),
        )
    return None


def _detect_breathing(df: pd.DataFrame) -> dict[str, Any] | None:
    if len(df) < 16:
        return None
    recent = df.tail(7)
    phases = []
    for _, row in recent.iterrows():
        if _safe_float(row["pct_chg"]) > 0 and _safe_float(row["vol_ratio_prev"]) > 1:
            phases.append("放量涨")
        elif _safe_float(row["pct_chg"]) < 0 and _safe_float(row["vol_ratio_prev"]) < 1:
            phases.append("缩量跌")
        else:
            phases.append("整理")
    lows = df["low"].tail(10).iloc[[0, 3, 6, 9]].tolist()
    n_type = len(lows) >= 4 and lows[-1] > min(lows[-3:-1]) * 0.98
    today = df.iloc[-1]
    if (
        phases[-1] == "放量涨"
        and phases.count("放量涨") >= 2
        and phases.count("缩量跌") >= 2
        and n_type
        and _safe_float(today["pct_chg"]) >= 1
        and _safe_float(today["vol_ratio_prev"]) >= 1.2
        and _safe_float(today["close_pos"]) >= 0.6
    ):
        return _signal(
            "BREATHING",
            "呼吸结构",
            "近期呈现放量涨、缩量跌、再放量涨的 N 型节奏，低点逐步抬高。",
            f"近7日节奏={'/'.join(phases)}，今日量比={today['vol_ratio_prev']:.1f}",
            "T+1 回踩不破最新抬高低点时观察；若高开过大且量能不能延续，等待下一次缩量回踩。",
            "跌破最近 N 型低点退出；放量突破后用 5 日线跟踪。",
            0.7 + min(_safe_float(today["vol_ratio_prev"]) / 5, 0.5),
        )
    return None


def _detect_kengqi(df: pd.DataFrame) -> dict[str, Any] | None:
    if len(df) < 25:
        return None
    window = df.tail(18).reset_index(drop=True)
    low_idx = int(window["low"].idxmin())
    if low_idx < 5:
        return None
    pre_high = _safe_float(window.loc[low_idx - 5 : low_idx - 1, "high"].max())
    keng_low = _safe_float(window.loc[low_idx, "low"])
    depth = (pre_high - keng_low) / pre_high if pre_high > 0 else 0
    today = window.iloc[-1]
    fill = (_safe_float(today["close"]) - keng_low) / (pre_high - keng_low) if pre_high > keng_low else 0
    post_vol = _safe_float(window.loc[low_idx + 1 : min(low_idx + 5, len(window) - 1), "volume"].mean())
    pre_vol = _safe_float(window.loc[low_idx - 5 : low_idx - 1, "volume"].mean())
    pit_day = window.loc[low_idx]
    if (
        depth >= 0.12
        and bool(pit_day["close"] < pit_day["open"])
        and _safe_float(pit_day["vol_ratio_prev"]) >= 1.25
        and 0.78 <= fill <= 1.12
        and post_vol < pre_vol * 0.8
        and _safe_float(today["pct_chg"]) <= 3
    ):
        target = 2 * pre_high - keng_low
        return _signal(
            "KENGQI",
            "坑里起好货",
            "放量挖坑后缩量填坑，当前回到坑沿附近，使用祖冲之法估算上方目标。",
            f"坑深={depth*100:.0f}%、填坑={fill*100:.0f}%、目标={target:.2f}",
            "T+1 不跌回坑内中位线时观察；接近坑沿但高开过大时等待回踩。",
            "跌破坑底或填坑失败退出；目标价附近分批止盈。",
            0.7 + min(fill, 1.0) * 0.5,
        )
    return None


def _detect_duichen_va(df: pd.DataFrame) -> dict[str, Any] | None:
    if len(df) < 30:
        return None
    window = df.tail(22).reset_index(drop=True)
    peak_idx = int(window["high"].idxmax())
    trough_idx = int(window.loc[:peak_idx, "low"].idxmin()) if peak_idx > 2 else -1
    if not (0 <= trough_idx < peak_idx < len(window) - 2):
        return None
    trough = _safe_float(window.loc[trough_idx, "low"])
    peak = _safe_float(window.loc[peak_idx, "high"])
    close = _safe_float(window.iloc[-1]["close"])
    up_days = peak_idx - trough_idx
    down_days = len(window) - 1 - peak_idx
    up_pct = (peak - trough) / trough if trough > 0 else 0
    down_pct = (peak - close) / peak if peak > 0 else 0
    time_sym = down_days / up_days if up_days > 0 else 0
    space_sym = down_pct / up_pct if up_pct > 0 else 0
    today = window.iloc[-1]
    if 0.5 <= time_sym <= 2.0 and 0.4 <= space_sym <= 1.1 and _safe_float(today["vol_ratio_prev"]) < 0.75 and _safe_float(today["kdj_j"]) < 25:
        return _signal(
            "DUICHEN_VA",
            "对称VA",
            "上涨波段后的回落在时间和空间上接近对称，当前缩量低位企稳，观察守恒被破坏后的反弹。",
            f"时间对称={time_sym:.1f}，空间对称={space_sym:.1f}，J={today['kdj_j']:.1f}",
            "T+1 不破对称完成低点时观察；若继续缩量横住，可等待第一根放量阳线确认。",
            "跌破对称低点退出；反弹至前高压力区前逐步止盈。",
            0.6 + max(0, (25 - _safe_float(today["kdj_j"])) / 70),
        )
    return None


def _detect_zaihou(df: pd.DataFrame) -> dict[str, Any] | None:
    if len(df) < 60:
        return None
    recent = df.tail(15).reset_index(drop=True)
    fangliang_idx = None
    for idx in range(5, len(recent) - 1):
        if _safe_float(recent.loc[idx, "pct_chg"]) > 5 and _safe_float(recent.loc[idx, "volume"]) > _safe_float(recent.loc[idx - 5 : idx - 1, "volume"].mean()) * 1.5:
            fangliang_idx = idx
            break
    if fangliang_idx is None:
        return None
    today = df.iloc[-1]
    yellow = _safe_float(today["bbi"])
    yellow_up = _safe_float(df.iloc[-1]["bbi"]) > _safe_float(df.iloc[-6]["bbi"])
    near_yellow = abs(_safe_float(today["close"]) - yellow) / yellow < 0.025 if yellow > 0 else False
    shrink = _safe_float(today["volume"]) < _safe_float(recent.loc[fangliang_idx, "volume"]) * 0.6
    if yellow_up and near_yellow and shrink:
        return _signal(
            "ZAIHOU",
            "灾后重建",
            "前期放量大阳后，当前缩量回踩上行黄线/BBI，属于最后震仓观察位。",
            f"放量日涨幅={recent.loc[fangliang_idx, 'pct_chg']:.1f}%；当前距BBI={(_safe_float(today['close'])/yellow-1)*100:.1f}%",
            "T+1 不跌破 BBI 且开盘偏离不超过 2% 时观察；破线放弃。",
            "收盘跌破 BBI 或放量阴线退出；再次放量上攻后跟踪 5 日线。",
            0.65 + min(_safe_float(recent.loc[fangliang_idx, "pct_chg"]) / 20, 0.4),
        )
    return None


def _detect_yueyue(df: pd.DataFrame) -> dict[str, Any] | None:
    if len(df) < 35:
        return None
    recent = df.tail(20)
    amplitude = (_safe_float(recent["high"].max()) - _safe_float(recent["low"].min())) / max(_safe_float(recent["low"].min()), 1)
    if amplitude > 0.16:
        return None
    huge = recent["volume"] > df["volume"].tail(30).rolling(10, min_periods=5).mean().tail(20) * 2
    count = int(huge.sum())
    yang_count = int((huge & (recent["close"] > recent["open"])).sum())
    if count >= 2 and yang_count / max(count, 1) >= 0.5:
        today = df.iloc[-1]
        return _signal(
            "YUEYUE",
            "跃跃欲试",
            "横盘平台内多次巨量试盘，且巨量日以阳线为主，观察平台突破前的蓄势。",
            f"20日振幅={amplitude*100:.1f}%、巨量{count}次、阳线占比={yang_count/max(count,1):.0%}",
            "T+1 仍在平台内低吸只做观察；真正买点优先等待放量突破平台上沿。",
            "跌破平台下沿退出；突破失败并出现放量阴线时退出。",
            0.55 + min(count / 10, 0.5) + min(_safe_float(today["close_pos"]) / 5, 0.2),
        )
    return None


def _detect_key_k(df: pd.DataFrame) -> dict[str, Any] | None:
    if len(df) < 20:
        return None
    today = df.iloc[-1]
    body_pct = abs(_safe_float(today["close"]) - _safe_float(today["open"])) / max(_safe_float(today["pre_close"]), 1) * 100
    vol_threshold = 1.1 if body_pct >= 7 else 1.3
    high20 = _safe_float(df["high"].tail(21).iloc[:-1].max())
    low20 = _safe_float(df["low"].tail(21).iloc[:-1].min())
    at_key = (_safe_float(today["high"]) >= high20 * 0.98) or (_safe_float(today["low"]) <= low20 * 1.02)
    if body_pct >= 3 and bool(today["is_rise"]) and _safe_float(today["close_pos"]) >= 0.75 and _safe_float(today["pct_chg"]) >= 2 and _safe_float(today["vol_ratio_5"]) >= vol_threshold and at_key:
        key_type = "反转/突破" if bool(today["is_rise"]) else "衰竭/风险"
        return _signal(
            "KEY_K",
            "关键K",
            "关键位置出现放量长实体K线，可能成为后续走势的管理K线。",
            f"{key_type}，实体={body_pct:.1f}%、5日量比={today['vol_ratio_5']:.1f}",
            "若为放量阳线，T+1 不破关键K实体中位观察；若为阴线，优先作为风险提示，不追买。",
            "关键K低点/高点作为多空分界；跌破阳线关键K低点退出。",
            0.45 + min(body_pct / 10, 0.8),
        )
    return None


def _detect_violence_k(df: pd.DataFrame) -> dict[str, Any] | None:
    if len(df) < 20:
        return None
    today = df.iloc[-1]
    body_pct = abs(_safe_float(today["close"]) - _safe_float(today["open"])) / max(_safe_float(today["pre_close"]), 1) * 100
    prev_bodies = (df["close"].tail(7).iloc[:-1] - df["open"].tail(7).iloc[:-1]).abs() / df["pre_close"].tail(7).iloc[:-1].replace(0, np.nan) * 100
    at_bottom = _safe_float(today["low"]) <= _safe_float(df["low"].tail(21).iloc[:-1].min()) * 1.05
    if (
        at_bottom
        and bool(today["is_rise"])
        and _safe_float(today["pct_chg"]) > 0
        and _safe_float(today["close_pos"]) >= 0.7
        and body_pct >= 5
        and body_pct > _safe_float(prev_bodies.mean()) * 2
        and _safe_float(today["vol_ratio_5"]) >= 2
    ):
        return _signal(
            "VIOLENCE_K",
            "暴力K",
            "底部区域突然出现倍量大实体K线，是关键K的更强版本，代表资金强行改变节奏。",
            f"实体={body_pct:.1f}%、5日量比={today['vol_ratio_5']:.1f}、收盘位置={today['close_pos']:.0%}",
            "T+1 不高开超过 4% 且不跌破暴力K实体中位时观察；最好等回踩确认。",
            "跌破暴力K低点退出；若连续放量冲高，可按短线目标分批止盈。",
            0.8 + min(_safe_float(today["vol_ratio_5"]) / 5, 0.7),
        )
    return None


DETECTORS = [
    _detect_changan,
    _detect_pinghang,
    _detect_double_gun,
    _detect_yidong_dilian,
    _detect_nana,
    _detect_golden_bowl,
    _detect_breathing,
    _detect_kengqi,
    _detect_duichen_va,
    _detect_zaihou,
    _detect_yueyue,
    _detect_key_k,
    _detect_violence_k,
]


def _build_one(path: Path, signal_date: str | None) -> tuple[str, dict[str, Any]] | None:
    try:
        df = _normalize_daily(path, signal_date)
        if df.empty:
            return None
        signals = [signal for detector in DETECTORS if (signal := detector(df)) is not None]
        if not signals:
            return None
        latest = df.iloc[-1]
        symbol = str(latest.get("ts_code") or latest.get("symbol") or path.stem)
        return symbol, {
            "symbol": symbol,
            "name": _clean_text(latest.get("name")),
            "date": latest["date"].strftime("%Y-%m-%d") if pd.notna(latest.get("date")) else None,
            "close": _safe_float(latest.get("close"), np.nan),
            "industry": _clean_text(latest.get("industry")),
            "signals": signals,
        }
    except Exception:
        return None


def build_z_skill_daily_signals(
    daily_dir: Path,
    signal_date: str | None = None,
    max_workers: int = 24,
) -> dict[str, dict[str, Any]]:
    """Scan raw daily files and return latest extended pattern hits by symbol."""
    suffixes = (".SZ.parquet", ".SH.parquet", ".BJ.parquet")
    files = sorted(path for path in Path(daily_dir).glob("*.parquet") if path.name.endswith(suffixes))
    if not files:
        return {}
    workers = max(1, min(max_workers, len(files)))
    results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_build_one, path, signal_date) for path in files]
        for future in as_completed(futures):
            item = future.result()
            if item is None:
                continue
            symbol, payload = item
            results[symbol] = payload
    return results


def build_extended_daily_signals(
    daily_dir: Path,
    signal_date: str | None = None,
    max_workers: int = 24,
) -> dict[str, dict[str, Any]]:
    """Compatibility-friendly public name for selector strategy scans."""
    return build_z_skill_daily_signals(daily_dir, signal_date=signal_date, max_workers=max_workers)


Z_SKILL_STRATEGIES = EXTENDED_STRATEGIES
