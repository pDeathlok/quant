"""Triple-volume shrink-consolidation breakout signal.

This module translates the Tongdaxin stock-picking formula:

三倍量缩量盘整突破

into vector-friendly pandas columns for research and backtesting. The signal is
meant for daily bars and assumes T+0 signal, T+1 open entry in research scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from quant.features.variable_library import build_continuous_ohlc


@dataclass(frozen=True)
class TripleVolumeVariant:
    id: str
    name: str
    volume_multiple: float
    signal_mode: str
    tier: str
    base_score: float
    buy_plan: str
    sell_plan: str
    metrics: dict[str, float | int]
    description: str


TRIPLE_VOLUME_VARIANTS = [
    TripleVolumeVariant(
        id="tvb_conservative_v3_bull_no60",
        name="三倍量突破 保守主策略",
        volume_multiple=3.0,
        signal_mode="avg_pre_shrink_bull_no60",
        tier="conservative",
        base_score=92.0,
        buy_plan="T+1 开盘观察；若高开过大或跌破信号日低点则放弃，优先小仓试错。",
        sell_plan="固定止盈 5%；盘中硬止损 2.5%；最长持有到 T+9；若提前跌破信号日突破结构则减仓或退出。",
        metrics={
            "trades": 18,
            "avg_return_pct": 2.396637,
            "win_rate": 0.722222,
            "max_drawdown_pct": -7.423568,
            "profit_factor": 4.160687,
        },
        description="3倍量锚点；突破日前平均缩量；MA5>MA10>MA20 且 MA20 上行；不强制 MA60。",
    ),
    TripleVolumeVariant(
        id="tvb_expanded_v25_bull_no60",
        name="三倍量突破 扩展候选池",
        volume_multiple=2.5,
        signal_mode="avg_pre_shrink_bull_no60",
        tier="expanded",
        base_score=78.0,
        buy_plan="T+1 开盘观察；只作为扩展候选，需结合市场环境、成交额和板块强度二次确认。",
        sell_plan="优先沿用固定止盈 5%、硬止损 2.5%、最长 T+9；弱于保守层，仓位应低一档。",
        metrics={
            "trades": 35,
            "avg_return_pct": 1.845164,
            "win_rate": 0.628571,
            "max_drawdown_pct": -14.194666,
            "profit_factor": 3.014726,
        },
        description="2.5倍量锚点；覆盖保守主策略，增加样本但回撤更高，需降仓和过滤。",
    ),
]


def _normalize_daily_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "trade_date" in out.columns:
        out["date"] = pd.to_datetime(out["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    elif "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    else:
        raise ValueError("daily frame must contain date or trade_date")

    if "volume" not in out.columns and "vol" in out.columns:
        out = out.rename(columns={"vol": "volume"})
    if "symbol" not in out.columns and "ts_code" in out.columns:
        out["symbol"] = out["ts_code"].astype(str)

    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"daily frame missing required columns: {sorted(missing)}")

    out = out.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    return out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def _is_risk_name(name: pd.Series) -> pd.Series:
    text = name.fillna("").astype(str).str.upper()
    return text.str.contains("ST", regex=False) | text.str.contains("*", regex=False) | text.str.contains("退", regex=False)


def add_triple_volume_breakout_signals(
    df: pd.DataFrame,
    volume_multiple: float = 3.0,
) -> pd.DataFrame:
    """Add signal columns for the triple-volume shrink-consolidation breakout.

    Output highlights:
    - ``signal_triple_volume_breakout``: final stock-picking signal.
    - ``days_since_triple_volume``: Tongdaxin ``BARSLAST(三倍量)`` equivalent.
    - ``triple_volume_price``: close of the most recent triple-volume bar.
    - ``candidate_score``: interpretable ranking score for crowded signal days.
    """
    out = _normalize_daily_frame(df)
    price = build_continuous_ohlc(out)
    open_ = price["open"].astype(float)
    high = price["high"].astype(float)
    low = price["low"].astype(float)
    close = price["close"].astype(float)
    volume = out["volume"].astype(float)

    ma5 = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    volume_ma5 = volume.rolling(5).mean()

    triple_volume = volume.shift(1) >= volume.shift(2) * volume_multiple
    triple_pos = np.where(triple_volume.fillna(False).to_numpy(), np.arange(len(out)), np.nan)
    last_triple_pos = pd.Series(triple_pos, index=out.index).ffill()
    pos = pd.Series(np.arange(len(out)), index=out.index)
    days_since = pos - last_triple_pos
    days_since[last_triple_pos.isna()] = np.nan

    triple_price = pd.Series(np.nan, index=out.index, dtype=float)
    valid_last = last_triple_pos.notna()
    if valid_last.any():
        triple_price.loc[valid_last] = close.iloc[last_triple_pos.loc[valid_last].astype(int)].to_numpy()

    shrink_consolidation = pd.Series(False, index=out.index)
    consolidation_range = pd.Series(np.nan, index=out.index, dtype=float)
    for idx in out.index[valid_last]:
        start = int(last_triple_pos.loc[idx]) + 1
        end = int(idx)
        if start > end:
            continue
        window = slice(start, end + 1)
        below_volume_ma5 = (volume.iloc[window] < volume_ma5.iloc[window]).all()
        below_anchor_pre_volume = volume.iloc[end] < volume.iloc[int(last_triple_pos.loc[idx]) - 1] if int(last_triple_pos.loc[idx]) >= 1 else False
        shrink_consolidation.loc[idx] = bool(below_volume_ma5 and below_anchor_pre_volume)
        window_high = high.iloc[window].max()
        window_low = low.iloc[window].min()
        consolidation_range.loc[idx] = window_high / window_low if window_low else np.nan

    right_side_bull = (ma5 > ma10) & (ma10 > ma20) & (ma20 > ma60) & (ma20 > ma20.shift(1))
    breakout = (close > triple_price) & (close > open_)
    range_ok = consolidation_range < 1.15
    if "name" in out.columns:
        risk_name = _is_risk_name(out["name"])
    else:
        risk_name = pd.Series(False, index=out.index)
    tradable = open_.notna() & (open_ > 0) & ~risk_name

    signal = (
        (days_since > 0)
        & right_side_bull
        & shrink_consolidation
        & breakout
        & range_ok
        & tradable
    )

    out["triple_volume_anchor"] = triple_volume.astype(int)
    out["days_since_triple_volume"] = days_since
    out["triple_volume_price"] = triple_price
    out["right_side_bull"] = right_side_bull.astype(int)
    out["shrink_consolidation"] = shrink_consolidation.astype(int)
    out["consolidation_range"] = consolidation_range
    out["breakout_over_triple_price"] = breakout.astype(int)
    out["signal_triple_volume_breakout"] = signal.fillna(False).astype(int)

    breakout_pct = close / triple_price.replace(0, np.nan) - 1
    volume_dryness = 1 - volume / volume_ma5.replace(0, np.nan)
    range_tightness = 1.15 - consolidation_range
    trend_slope = ma20 / ma20.shift(5).replace(0, np.nan) - 1
    out["candidate_score"] = (
        0.35 * breakout_pct.rank(pct=True)
        + 0.25 * volume_dryness.rank(pct=True)
        + 0.25 * range_tightness.rank(pct=True)
        + 0.15 * trend_slope.rank(pct=True)
    )
    out["breakout_pct"] = breakout_pct
    out["volume_dryness"] = volume_dryness
    out["ma20_slope_5d"] = trend_slope
    return out


def add_triple_volume_research_signals(
    df: pd.DataFrame,
    volume_multiple: float = 3.0,
) -> pd.DataFrame:
    """Add strict and relaxed research signal variants for one volume multiple."""
    out = add_triple_volume_breakout_signals(df, volume_multiple=volume_multiple)
    price = build_continuous_ohlc(out)
    close = price["close"].astype(float)
    volume = out["volume"].astype(float)
    ma5 = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    volume_ma5 = volume.rolling(5).mean()

    strict_bull = (ma5 > ma10) & (ma10 > ma20) & (ma20 > ma60) & (ma20 > ma20.shift(1))
    bull_no60 = (ma5 > ma10) & (ma10 > ma20) & (ma20 > ma20.shift(1))
    close_ma20_up = (close > ma20) & (ma20 > ma20.shift(1))
    common = (
        (out["days_since_triple_volume"] > 0)
        & (out["breakout_over_triple_price"] == 1)
        & (out["consolidation_range"] < 1.15)
    )

    idx = pd.Series(np.arange(len(out)), index=out.index)
    last_anchor = idx - out["days_since_triple_volume"]
    pre_shrink = pd.Series(False, index=out.index)
    avg_pre_shrink = pd.Series(False, index=out.index)
    soft_shrink = pd.Series(False, index=out.index)
    for i in out.index:
        if pd.isna(last_anchor.loc[i]) or out.loc[i, "days_since_triple_volume"] <= 0:
            continue
        anchor = int(last_anchor.loc[i])
        start = anchor + 1
        end = int(i)
        pre_end = max(start, end - 1)
        if anchor < 1:
            continue
        current_below_pre_anchor = volume.iloc[end] < volume.iloc[anchor - 1]
        pre_window = slice(start, pre_end + 1)
        full_window = slice(start, end + 1)
        pre_shrink.loc[i] = bool(
            current_below_pre_anchor and (volume.iloc[pre_window] < volume_ma5.iloc[pre_window]).all()
        )
        avg_pre_shrink.loc[i] = bool(
            current_below_pre_anchor and volume.iloc[pre_window].mean() < volume_ma5.iloc[pre_window].mean()
        )
        soft_shrink.loc[i] = bool(
            current_below_pre_anchor and (volume.iloc[full_window] < volume_ma5.iloc[full_window] * 1.15).all()
        )

    out["signal_strict"] = out["signal_triple_volume_breakout"].astype(int)
    out["signal_pre_shrink_strict_bull"] = (common & strict_bull & pre_shrink).astype(int)
    out["signal_soft_shrink_strict_bull"] = (common & strict_bull & soft_shrink).astype(int)
    out["signal_avg_pre_shrink_strict_bull"] = (common & strict_bull & avg_pre_shrink).astype(int)
    out["signal_pre_shrink_bull_no60"] = (common & bull_no60 & pre_shrink).astype(int)
    out["signal_avg_pre_shrink_bull_no60"] = (common & bull_no60 & avg_pre_shrink).astype(int)
    out["signal_pre_shrink_close_ma20"] = (common & close_ma20_up & pre_shrink).astype(int)
    out["signal_avg_pre_shrink_close_ma20"] = (common & close_ma20_up & avg_pre_shrink).astype(int)
    out["volume_recovery"] = volume / volume_ma5.replace(0, np.nan)
    return out


def add_triple_volume_strategy_pool_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Add the merged conservative + expanded short-term strategy signal.

    The expanded 2.5x layer is intentionally a superset candidate pool around
    the conservative 3x layer. When both fire, the conservative plan wins.
    """
    frames = {
        variant.id: add_triple_volume_research_signals(df, volume_multiple=variant.volume_multiple)
        for variant in TRIPLE_VOLUME_VARIANTS
    }
    base = frames[TRIPLE_VOLUME_VARIANTS[0].id].copy()
    base["signal_tvb_conservative"] = 0
    base["signal_tvb_expanded"] = 0
    base["signal_tvb_merged"] = 0
    base["tvb_variant_id"] = ""
    base["tvb_variant_name"] = ""
    base["tvb_tier"] = ""
    base["tvb_score"] = np.nan
    base["tvb_volume_multiple"] = np.nan
    base["tvb_buy_plan"] = ""
    base["tvb_sell_plan"] = ""
    base["tvb_description"] = ""
    base["tvb_metrics"] = ""

    for variant in reversed(TRIPLE_VOLUME_VARIANTS):
        frame = frames[variant.id]
        signal_col = f"signal_{variant.signal_mode}"
        hit = frame[signal_col].fillna(0).astype(bool) if signal_col in frame.columns else pd.Series(False, index=frame.index)
        score = variant.base_score + _candidate_score_adjustment(frame)
        if variant.tier == "conservative":
            base.loc[hit, "signal_tvb_conservative"] = 1
        else:
            base.loc[hit, "signal_tvb_expanded"] = 1
        update = hit & (base["tvb_variant_id"].eq("") | (variant.tier == "conservative"))
        base.loc[update, "tvb_variant_id"] = variant.id
        base.loc[update, "tvb_variant_name"] = variant.name
        base.loc[update, "tvb_tier"] = variant.tier
        base.loc[update, "tvb_score"] = score.loc[update]
        base.loc[update, "tvb_volume_multiple"] = variant.volume_multiple
        base.loc[update, "tvb_buy_plan"] = variant.buy_plan
        base.loc[update, "tvb_sell_plan"] = variant.sell_plan
        base.loc[update, "tvb_description"] = variant.description
        base.loc[update, "tvb_metrics"] = str(variant.metrics)

    base["signal_tvb_merged"] = ((base["signal_tvb_conservative"] == 1) | (base["signal_tvb_expanded"] == 1)).astype(int)
    return base


def _candidate_score_adjustment(frame: pd.DataFrame) -> pd.Series:
    raw = frame.get("candidate_score", pd.Series(0.5, index=frame.index)).fillna(0.5)
    breakout = frame.get("breakout_pct", pd.Series(0.0, index=frame.index)).fillna(0.0)
    dryness = frame.get("volume_dryness", pd.Series(0.0, index=frame.index)).fillna(0.0)
    recovery = frame.get("volume_recovery", pd.Series(1.0, index=frame.index)).fillna(1.0)
    recovery_penalty = (recovery - 1.4).clip(lower=0) * 8.0
    return (raw - 0.5) * 10.0 + breakout.clip(-0.02, 0.08) * 80.0 + dryness.clip(-0.5, 0.8) * 4.0 - recovery_penalty


def summarize_triple_volume_breakout(df: pd.DataFrame) -> dict[str, float | int]:
    out = add_triple_volume_breakout_signals(df)
    signal = out["signal_triple_volume_breakout"] == 1
    return {
        "rows": int(len(out)),
        "signals": int(signal.sum()),
        "first_signal": out.loc[signal, "date"].min() if signal.any() else pd.NaT,
        "last_signal": out.loc[signal, "date"].max() if signal.any() else pd.NaT,
    }
