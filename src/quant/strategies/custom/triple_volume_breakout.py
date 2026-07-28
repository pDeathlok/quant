"""Triple-volume shrink-consolidation breakout signal.

This module translates the Tongdaxin stock-picking formula:

三倍量缩量盘整突破

into vector-friendly pandas columns for research and backtesting. The signal is
meant for daily bars and assumes T+0 signal, T+1 open entry in research scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

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


PROJECT_ROOT = Path(__file__).resolve().parents[4]
TRIPLE_VOLUME_CONFIG_PATH = (
    PROJECT_ROOT / "configs/strategies/triple_volume_breakout.yaml"
)


def load_triple_volume_variants(
    path: Path = TRIPLE_VOLUME_CONFIG_PATH,
) -> list[TripleVolumeVariant]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    strategy = payload.get("strategy") or {}
    if not strategy.get("enabled", True):
        raise ValueError(f"triple-volume strategy is disabled in {path}")
    variants: list[TripleVolumeVariant] = []
    for raw in strategy.get("variants") or []:
        metrics = raw.get("backtest_2024") or {}
        variants.append(
            TripleVolumeVariant(
                id=str(raw["id"]),
                name=str(raw["name"]),
                volume_multiple=float(raw["volume_multiple"]),
                signal_mode=str(raw["signal_mode"]),
                tier=str(raw["tier"]),
                base_score=float(raw["base_score"]),
                buy_plan=str(raw["buy_plan"]),
                sell_plan=str(raw["sell_plan"]),
                metrics={
                    key: float(value) if key != "trades" else int(value)
                    for key, value in metrics.items()
                },
                description=str(raw.get("description", "")),
            )
        )
    if {variant.tier for variant in variants} != {"conservative", "expanded"}:
        raise ValueError(
            f"{path} must define exactly the conservative and expanded tiers"
        )
    return sorted(
        variants,
        key=lambda variant: 0 if variant.tier == "conservative" else 1,
    )


TRIPLE_VOLUME_VARIANTS = load_triple_volume_variants()


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


def _anchor_window_metrics(
    volume: np.ndarray,
    volume_ma5: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    anchor_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Calculate anchor-relative windows in one forward pass."""

    size = len(volume)
    anchor_positions = np.full(size, -1, dtype=np.int64)
    days_since = np.full(size, np.nan, dtype=float)
    shrink_consolidation = np.zeros(size, dtype=bool)
    consolidation_range = np.full(size, np.nan, dtype=float)
    pre_shrink = np.zeros(size, dtype=bool)
    avg_pre_shrink = np.zeros(size, dtype=bool)
    soft_shrink = np.zeros(size, dtype=bool)

    last_anchor = -1
    all_below_ma5 = True
    all_soft_below_ma5 = True
    prior_below_ma5 = True
    running_high = np.nan
    running_low = np.nan
    prior_volume_sum = 0.0
    prior_volume_count = 0
    prior_ma5_sum = 0.0
    prior_ma5_count = 0

    for index in range(size):
        if anchor_mask[index]:
            last_anchor = index
            anchor_positions[index] = index
            days_since[index] = 0.0
            all_below_ma5 = True
            all_soft_below_ma5 = True
            prior_below_ma5 = True
            running_high = np.nan
            running_low = np.nan
            prior_volume_sum = 0.0
            prior_volume_count = 0
            prior_ma5_sum = 0.0
            prior_ma5_count = 0
            continue
        if last_anchor < 0:
            continue

        anchor_positions[index] = last_anchor
        days_since[index] = float(index - last_anchor)
        current_volume = volume[index]
        current_ma5 = volume_ma5[index]
        below_ma5 = bool(current_volume < current_ma5)
        below_soft_ma5 = bool(current_volume < current_ma5 * 1.15)
        all_below_ma5 = all_below_ma5 and below_ma5
        all_soft_below_ma5 = all_soft_below_ma5 and below_soft_ma5
        running_high = (
            high[index]
            if pd.isna(running_high)
            else np.fmax(running_high, high[index])
        )
        running_low = (
            low[index]
            if pd.isna(running_low)
            else np.fmin(running_low, low[index])
        )
        consolidation_range[index] = (
            running_high / running_low
            if pd.notna(running_low) and running_low
            else np.nan
        )

        if last_anchor >= 1:
            below_pre_anchor = bool(
                current_volume < volume[last_anchor - 1]
            )
            shrink_consolidation[index] = (
                below_pre_anchor and all_below_ma5
            )
            soft_shrink[index] = (
                below_pre_anchor and all_soft_below_ma5
            )
            if index == last_anchor + 1:
                pre_window_below = below_ma5
                pre_volume_mean = current_volume
                pre_ma5_mean = current_ma5
            else:
                pre_window_below = prior_below_ma5
                pre_volume_mean = (
                    prior_volume_sum / prior_volume_count
                    if prior_volume_count
                    else np.nan
                )
                pre_ma5_mean = (
                    prior_ma5_sum / prior_ma5_count
                    if prior_ma5_count
                    else np.nan
                )
            pre_shrink[index] = (
                below_pre_anchor and pre_window_below
            )
            avg_pre_shrink[index] = bool(
                below_pre_anchor
                and pd.notna(pre_volume_mean)
                and pd.notna(pre_ma5_mean)
                and pre_volume_mean < pre_ma5_mean
            )

        prior_below_ma5 = prior_below_ma5 and below_ma5
        if pd.notna(current_volume):
            prior_volume_sum += float(current_volume)
            prior_volume_count += 1
        if pd.notna(current_ma5):
            prior_ma5_sum += float(current_ma5)
            prior_ma5_count += 1

    return {
        "anchor_positions": anchor_positions,
        "days_since": days_since,
        "shrink_consolidation": shrink_consolidation,
        "consolidation_range": consolidation_range,
        "pre_shrink": pre_shrink,
        "avg_pre_shrink": avg_pre_shrink,
        "soft_shrink": soft_shrink,
    }


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
    anchor_metrics = _anchor_window_metrics(
        volume.to_numpy(dtype=float),
        volume_ma5.to_numpy(dtype=float),
        high.to_numpy(dtype=float),
        low.to_numpy(dtype=float),
        triple_volume.fillna(False).to_numpy(dtype=bool),
    )
    last_triple_pos = anchor_metrics["anchor_positions"]
    days_since = pd.Series(
        anchor_metrics["days_since"],
        index=out.index,
    )

    triple_price = pd.Series(np.nan, index=out.index, dtype=float)
    valid_last = last_triple_pos >= 0
    if valid_last.any():
        triple_price.loc[valid_last] = close.iloc[
            last_triple_pos[valid_last]
        ].to_numpy()

    shrink_consolidation = pd.Series(
        anchor_metrics["shrink_consolidation"],
        index=out.index,
    )
    consolidation_range = pd.Series(
        anchor_metrics["consolidation_range"],
        index=out.index,
    )

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

    anchor_metrics = _anchor_window_metrics(
        volume.to_numpy(dtype=float),
        volume_ma5.to_numpy(dtype=float),
        price["high"].to_numpy(dtype=float),
        price["low"].to_numpy(dtype=float),
        out["triple_volume_anchor"].fillna(0).to_numpy(dtype=bool),
    )
    pre_shrink = pd.Series(
        anchor_metrics["pre_shrink"],
        index=out.index,
    )
    avg_pre_shrink = pd.Series(
        anchor_metrics["avg_pre_shrink"],
        index=out.index,
    )
    soft_shrink = pd.Series(
        anchor_metrics["soft_shrink"],
        index=out.index,
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
