"""Causal rule features for the unified right-side strategy research pool.

The module deliberately separates rule primitives from signal generation.  It
materializes the continuous margins and historical state used by the fourteen
right-side/mixed members so a pooled model can retain strategy identity without
having to reverse-engineer inline boolean predicates.

All calculations are single-symbol and causal: a row only depends on that row
and earlier rows after sorting by date.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence
import hashlib

import numpy as np
import pandas as pd

from quant.features.variable_library import build_continuous_ohlc


RIGHT_SIDE_SIGNALS: tuple[str, ...] = (
    "B2",
    "B3",
    "KEY_K",
    "VIOLENCE_K",
    "PINGHANG",
    "DOUBLE_GUN",
    "CHANGAN",
    "KENGQI",
    "VEGAS",
    "TRIPLE_VOLUME_BREAKOUT",
    "GOLDEN_BOWL",
    "ZAIHOU",
    "BREATHING",
    "YUEYUE",
)


SIGNAL_FEATURE_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "B2": (
        "rs_pct_chg_1d",
        "rs_vol_ratio_5_inclusive",
        "rs_vol_ratio_20_inclusive",
        "rs_close_pos",
        "rs_family_kdj_j",
        "rs_recent_yin_count_4",
        "rs_close_to_ma60_pct",
        "rs_b1_support_ok",
        "rs_pre_oversold_prev5",
        "rs_b1_like_prev3",
        "rs_family_bbi_distance_pct",
        "rs_bbi_reclaim",
        "rs_upper_shadow_pct",
        "rs_b2_from_b1_pchg3",
        "rs_b2_from_b1_pchg4",
        "rs_b2_from_b1_pchg5",
        "rs_b2_any",
        "rs_b2_oversold",
        "rs_b2_bbi_reclaim",
    ),
    "B3": (
        "rs_b2_recent_prev3",
        "rs_b2_broad_recent_prev3",
        "rs_days_since_b2",
        "rs_pct_chg_1d",
        "rs_amplitude_pct",
        "rs_close_pos",
        "rs_vol_ratio_5_inclusive",
        "rs_b3_small_pos_amp7",
        "rs_b3_broad_small_pos",
        "rs_b3_broad_calm_pullback",
    ),
    "KEY_K": (
        "rs_body_abs_pct",
        "rs_is_rise",
        "rs_close_pos",
        "rs_pct_chg_1d",
        "rs_vol_ratio_prev5",
        "rs_high_to_prev20_high_pct",
        "rs_low_to_prev20_low_pct",
        "rs_at_key_20d",
    ),
    "VIOLENCE_K": (
        "rs_low_to_prev20_low_pct",
        "rs_at_bottom_20d",
        "rs_is_rise",
        "rs_pct_chg_1d",
        "rs_close_pos",
        "rs_body_abs_pct",
        "rs_body_vs_prev6",
        "rs_vol_ratio_prev5",
    ),
    "PINGHANG": (
        "rs_strong_yang",
        "rs_strong_yang_count_8d",
        "rs_two_yang_gap_days",
        "rs_mid_bar_count",
        "rs_mid_yin_share",
        "rs_first_vol_to_mid_max",
        "rs_second_vol_to_mid_max",
        "rs_second_to_first_vol",
        "rs_second_yang_pct",
        "rs_kdj_j",
    ),
    "DOUBLE_GUN": (
        "rs_double_gun_gap_days",
        "rs_double_gun_mid_avg_vol_ratio_prev",
        "rs_double_gun_pre_second_kdj_j",
        "rs_double_gun_first_vol_ratio_prev",
        "rs_double_gun_second_vol_ratio_prev",
        "rs_double_gun_days_since_second",
        "rs_double_gun_close_to_second_low_pct",
        "rs_double_gun_active",
    ),
    "GOLDEN_BOWL": (
        "rs_zg_white",
        "rs_dg_yellow",
        "rs_white_yellow_spread_pct",
        "rs_close_to_yellow_pct",
        "rs_close_bowl_position",
        "rs_kdj_j",
    ),
    "ZAIHOU": (
        "rs_fangliang_recent_3_12d",
        "rs_days_since_fangliang",
        "rs_fangliang_ref_volume_15d",
        "rs_volume_to_fangliang_ref",
        "rs_bbi_slope_5d_pct",
        "rs_bbi_distance_pct",
        "rs_pct_chg_1d",
    ),
    "BREATHING": (
        "rs_phase_exhale",
        "rs_phase_inhale",
        "rs_exhale_count_7d",
        "rs_inhale_count_7d",
        "rs_higher_low_ratio",
        "rs_pct_chg_1d",
        "rs_vol_ratio_prev",
        "rs_close_pos",
    ),
    "YUEYUE": (
        "rs_platform_range_20d",
        "rs_huge_volume_count_20d",
        "rs_huge_yang_share_20d",
        "rs_close_to_platform_high_pct",
        "rs_pct_chg_1d",
        "rs_close_pos",
    ),
    "CHANGAN": (
        "rs_kdj_j_lag2",
        "rs_pct_chg_lag1",
        "rs_is_rise_lag1",
        "rs_vol_ratio_prev5_lag1",
        "rs_kdj_j_lag1_minus_lag2",
        "rs_pct_chg_1d",
        "rs_amplitude_pct",
        "rs_vol_ratio_prev",
    ),
    "KENGQI": (
        "rs_pit_depth_18d",
        "rs_pit_recent_3_14d",
        "rs_days_since_pit",
        "rs_pit_fill_ratio",
        "rs_post_to_pre_volume_ratio",
        "rs_last_pit_vol_ratio_prev",
        "rs_pct_chg_1d",
    ),
    "VEGAS": (
        "rs_ema10",
        "rs_ema20",
        "rs_ema144",
        "rs_ema169",
        "rs_vegas_tunnel_upper",
        "rs_vegas_tunnel_distance",
        "rs_vegas_tunnel_slope_20d",
        "rs_vegas_fast_spread",
        "rs_vegas_recent_pullback_8d",
        "rs_vegas_trend_stack",
        "rs_vegas_tunnel_up",
        "rs_vegas_right_side_rebound",
        "rs_vegas_volume_strength",
        "rs_vegas_volume_confirm",
        "rs_vegas_not_overheated",
        "rs_vegas_history_ok",
        "rs_vegas_tradable",
        "rs_vegas_signal",
    ),
    "TRIPLE_VOLUME_BREAKOUT": (
        "rs_tvb_anchor_volume_multiple",
        "rs_tvb_days_since_anchor_25",
        "rs_tvb_days_since_anchor_30",
        "rs_tvb_anchor_price_25",
        "rs_tvb_anchor_price_30",
        "rs_tvb_consolidation_range_25",
        "rs_tvb_consolidation_range_30",
        "rs_tvb_avg_pre_shrink_25",
        "rs_tvb_avg_pre_shrink_30",
        "rs_tvb_breakout_pct_25",
        "rs_tvb_breakout_pct_30",
        "rs_tvb_bull_no60",
        "rs_ma20_slope_5d_pct",
        "rs_tvb_candidate_25",
        "rs_tvb_candidate_30",
        "rs_tvb_merged",
    ),
}


RULE_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    dict.fromkeys(
        feature
        for signal in RIGHT_SIDE_SIGNALS
        for feature in SIGNAL_FEATURE_REQUIREMENTS[signal]
    )
)


# Freeze the exact rule-factor contracts used by the first fair 105 vs 118
# ablation.  Keep the added-column tuple explicit: deriving the legacy schema
# from an old parquet file would make model behaviour depend on mutable data.
RULE_FEATURE_SCHEMA_VERSION = "right_side_rule_features_v2_118_20260813"
LEGACY_RULE_FEATURE_SCHEMA_VERSION_V1 = "right_side_rule_features_v1_105_20260812"
ADDED_RULE_FEATURE_COLUMNS_V2: tuple[str, ...] = (
    "rs_vol_ratio_20_inclusive",
    "rs_family_kdj_j",
    "rs_recent_yin_count_4",
    "rs_close_to_ma60_pct",
    "rs_b1_support_ok",
    "rs_family_bbi_distance_pct",
    "rs_b3_small_pos_amp7",
    "rs_b3_broad_small_pos",
    "rs_b3_broad_calm_pullback",
    "rs_vegas_history_ok",
    "rs_vegas_tradable",
    "rs_vegas_signal",
    "rs_tvb_merged",
)
LEGACY_RULE_FEATURE_COLUMNS_V1: tuple[str, ...] = tuple(
    column
    for column in RULE_FEATURE_COLUMNS
    if column not in frozenset(ADDED_RULE_FEATURE_COLUMNS_V2)
)


def rule_feature_columns_sha256(columns: Sequence[str]) -> str:
    """Return a stable, order-sensitive digest for a rule-factor contract."""

    payload = "\n".join(str(column) for column in columns).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


RULE_FEATURE_COLUMNS_SHA256 = rule_feature_columns_sha256(RULE_FEATURE_COLUMNS)
LEGACY_RULE_FEATURE_COLUMNS_SHA256_V1 = rule_feature_columns_sha256(
    LEGACY_RULE_FEATURE_COLUMNS_V1
)


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return pd.to_numeric(numerator, errors="coerce").div(
        pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
    )


def _days_since(mask: pd.Series) -> pd.Series:
    positions = pd.Series(np.arange(len(mask), dtype=float), index=mask.index)
    last_position = positions.where(mask.fillna(False)).ffill()
    return (positions - last_position).where(last_position.notna())


def _kengqi_event_state(frame: pd.DataFrame, vol_ratio_prev: pd.Series) -> dict[str, pd.Series]:
    """Materialize the exact trailing-18-bar state used by live KENGQI."""

    index = frame.index
    output = {
        "depth": np.full(len(frame), np.nan, dtype=float),
        "recent": np.zeros(len(frame), dtype=bool),
        "days_since": np.full(len(frame), np.nan, dtype=float),
        "fill": np.full(len(frame), np.nan, dtype=float),
        "post_to_pre_volume": np.full(len(frame), np.nan, dtype=float),
        "pit_vol_ratio_prev": np.full(len(frame), np.nan, dtype=float),
    }
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    open_ = frame["open"].to_numpy(dtype=float)
    close = frame["close"].to_numpy(dtype=float)
    volume = frame["volume"].to_numpy(dtype=float)
    ratio = vol_ratio_prev.to_numpy(dtype=float)
    for position in range(17, len(frame)):
        start = position - 17
        relative_low = int(np.nanargmin(low[start : position + 1]))
        pit = start + relative_low
        if relative_low < 5:
            continue
        pre_slice = slice(pit - 5, pit)
        pre_high = float(np.nanmax(high[pre_slice]))
        pit_low = float(low[pit])
        if not np.isfinite(pre_high) or pre_high <= 0 or pre_high <= pit_low:
            continue
        post_end = min(pit + 5, position) + 1
        post_slice = slice(pit + 1, post_end)
        pre_volume = float(np.nanmean(volume[pre_slice]))
        post_volume = float(np.nanmean(volume[post_slice])) if post_end > pit + 1 else np.nan
        output["depth"][position] = (pre_high - pit_low) / pre_high
        output["recent"][position] = bool(
            close[pit] < open_[pit] and ratio[pit] >= 1.25
        )
        output["days_since"][position] = float(position - pit)
        output["fill"][position] = (close[position] - pit_low) / (pre_high - pit_low)
        output["post_to_pre_volume"][position] = (
            post_volume / pre_volume if pre_volume > 0 else np.nan
        )
        output["pit_vol_ratio_prev"][position] = ratio[pit]
    return {name: pd.Series(values, index=index) for name, values in output.items()}


def _zaihou_event_state(
    frame: pd.DataFrame,
    pct_chg: pd.Series,
) -> dict[str, pd.Series]:
    """Materialize the first qualifying live ZAIHOU anchor in the last 15 bars."""

    index = frame.index
    output = {
        "recent": np.zeros(len(frame), dtype=bool),
        "days_since": np.full(len(frame), np.nan, dtype=float),
        "reference_volume": np.full(len(frame), np.nan, dtype=float),
    }
    volume = frame["volume"].to_numpy(dtype=float)
    pct = pct_chg.to_numpy(dtype=float)
    prior_mean = (
        pd.Series(volume)
        .shift(1)
        .rolling(5, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )
    qualifying = (pct > 5.0) & (volume > prior_mean * 1.5)
    active_anchors: deque[int] = deque()
    for position in range(len(frame)):
        previous = position - 1
        if previous >= 5 and qualifying[previous]:
            active_anchors.append(previous)
        lower_bound = max(5, position - 9)
        while active_anchors and active_anchors[0] < lower_bound:
            active_anchors.popleft()
        # Live detector scans the 15-bar window from oldest to newest, excludes
        # the current bar, and requires five prior bars inside the same window.
        if not active_anchors:
            continue
        anchor = active_anchors[0]
        output["recent"][position] = True
        output["days_since"][position] = float(position - anchor)
        output["reference_volume"][position] = volume[anchor]
    return {name: pd.Series(values, index=index) for name, values in output.items()}


def _parse_dates(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.replace(r"\.0$", "", regex=True)
    compact = text.str.fullmatch(r"\d{8}").fillna(False)
    parsed = pd.to_datetime(values, errors="coerce")
    if compact.any():
        parsed.loc[compact] = pd.to_datetime(
            text.loc[compact], format="%Y%m%d", errors="coerce"
        )
    return parsed


def _prepare_daily(daily: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(daily, pd.DataFrame):
        raise TypeError("daily must be a pandas DataFrame")

    date_column = "date" if "date" in daily.columns else "trade_date"
    volume_column = "volume" if "volume" in daily.columns else "vol"
    missing = [
        column
        for column in (date_column, "open", "high", "low", "close", volume_column)
        if column not in daily.columns
    ]
    if missing:
        raise ValueError(
            "right-side rule features require date/trade_date and OHLCV; "
            f"missing columns: {missing}"
        )

    out = daily.copy()
    out["date"] = _parse_dates(out[date_column])
    out["volume"] = pd.to_numeric(out[volume_column], errors="coerce")
    for column in ("open", "high", "low", "close", "pre_close"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    if out["date"].isna().any():
        raise ValueError("right-side rule features received invalid dates")
    if out["date"].duplicated().any():
        raise ValueError("right-side rule features require one row per trade date")

    for symbol_column in ("symbol", "ts_code"):
        if symbol_column in out.columns and out[symbol_column].dropna().nunique() > 1:
            raise ValueError("right-side rule features require a single-symbol frame")

    out = out.sort_values("date", kind="stable").copy()
    price = build_continuous_ohlc(out)
    for column in ("open", "high", "low", "close"):
        out[column] = pd.to_numeric(price[column], errors="coerce")
    out["pre_close"] = out["close"].shift(1)
    return out


def _two_yang_state(
    frame: pd.DataFrame,
    strong_yang: pd.Series,
    vol_ratio_prev: pd.Series,
    kdj_j: pd.Series,
    pct_chg: pd.Series,
) -> dict[str, pd.Series]:
    index = frame.index
    output = {
        name: np.full(len(frame), np.nan, dtype=float)
        for name in (
            "rs_two_yang_gap_days",
            "rs_mid_bar_count",
            "rs_mid_yin_share",
            "rs_first_vol_to_mid_max",
            "rs_second_vol_to_mid_max",
            "rs_second_to_first_vol",
            "rs_second_yang_pct",
            "rs_mid_avg_vol_ratio_prev",
            "rs_pre_second_kdj_j",
            "rs_second_vol_ratio_prev",
            "rs_days_since_second_yang",
            "rs_close_to_second_low_pct",
        )
    }
    event_positions: list[int] = []
    volume = frame["volume"].to_numpy(dtype=float)
    open_ = frame["open"].to_numpy(dtype=float)
    close = frame["close"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    ratio = vol_ratio_prev.to_numpy(dtype=float)
    j_values = kdj_j.to_numpy(dtype=float)
    pct_values = pct_chg.to_numpy(dtype=float)
    events = strong_yang.fillna(False).to_numpy(dtype=bool)
    cached_pair: tuple[int, int] | None = None
    pair_values: dict[str, float] = {}

    for position in range(len(frame)):
        if events[position]:
            event_positions.append(position)
        if len(event_positions) < 2:
            continue
        first, second = event_positions[-2:]
        pair = (first, second)
        if pair != cached_pair:
            middle = slice(first + 1, second)
            middle_count = second - first - 1
            first_volume = volume[first]
            second_volume = volume[second]
            pair_values = {
                "gap": float(second - first),
                "middle_count": float(middle_count),
                "second_pct": pct_values[second],
                "second_ratio": ratio[second],
                "pre_second_j": (
                    j_values[second - 1] if second > 0 else np.nan
                ),
                "second_to_first": (
                    second_volume / first_volume if first_volume else np.nan
                ),
                "mid_yin_share": np.nan,
                "first_to_mid_max": np.nan,
                "second_to_mid_max": np.nan,
                "mid_avg_ratio": np.nan,
            }
            if middle_count > 0:
                middle_volume = volume[middle]
                middle_max = float(np.nanmax(middle_volume))
                pair_values.update(
                    {
                        "mid_yin_share": float(
                            np.mean(close[middle] <= open_[middle])
                        ),
                        "first_to_mid_max": (
                            first_volume / middle_max if middle_max else np.nan
                        ),
                        "second_to_mid_max": (
                            second_volume / middle_max if middle_max else np.nan
                        ),
                        "mid_avg_ratio": float(np.nanmean(ratio[middle])),
                    }
                )
            cached_pair = pair

        output["rs_two_yang_gap_days"][position] = pair_values["gap"]
        output["rs_mid_bar_count"][position] = pair_values["middle_count"]
        output["rs_second_yang_pct"][position] = pair_values["second_pct"]
        output["rs_second_vol_ratio_prev"][position] = pair_values["second_ratio"]
        output["rs_days_since_second_yang"][position] = position - second
        output["rs_pre_second_kdj_j"][position] = pair_values["pre_second_j"]
        output["rs_second_to_first_vol"][position] = pair_values[
            "second_to_first"
        ]
        output["rs_close_to_second_low_pct"][position] = (
            (close[position] / low[second] - 1.0) * 100.0
            if low[second]
            else np.nan
        )
        output["rs_mid_yin_share"][position] = pair_values["mid_yin_share"]
        output["rs_first_vol_to_mid_max"][position] = pair_values[
            "first_to_mid_max"
        ]
        output["rs_second_vol_to_mid_max"][position] = pair_values[
            "second_to_mid_max"
        ]
        output["rs_mid_avg_vol_ratio_prev"][position] = pair_values[
            "mid_avg_ratio"
        ]
    return {name: pd.Series(values, index=index) for name, values in output.items()}


def _double_gun_state(
    frame: pd.DataFrame,
    vol_ratio_prev: pd.Series,
    kdj_j: pd.Series,
    pct_chg: pd.Series,
) -> dict[str, pd.Series]:
    """Materialize the exact event anchors used by the live DOUBLE_GUN detector.

    Unlike PINGHANG, a gun is defined by previous-day volume ratio >= 1.8 and
    the signal is observed one to four sessions after the second gun.  Keeping
    a separate state prevents the two visually similar strategies from sharing
    the wrong anchors.
    """

    index = frame.index
    names = (
        "rs_double_gun_gap_days",
        "rs_double_gun_mid_avg_vol_ratio_prev",
        "rs_double_gun_pre_second_kdj_j",
        "rs_double_gun_first_vol_ratio_prev",
        "rs_double_gun_second_vol_ratio_prev",
        "rs_double_gun_days_since_second",
        "rs_double_gun_close_to_second_low_pct",
    )
    output = {
        name: np.full(len(frame), np.nan, dtype=float)
        for name in names
    }
    output["rs_double_gun_active"] = np.zeros(len(frame), dtype=bool)

    open_ = frame["open"].to_numpy(dtype=float)
    close = frame["close"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    ratio = vol_ratio_prev.to_numpy(dtype=float)
    j_values = kdj_j.to_numpy(dtype=float)
    pct_values = pct_chg.to_numpy(dtype=float)
    gun = (close > open_) & (pct_values >= 3.0) & (ratio >= 1.8)

    for position in range(17, len(frame)):
        lower_second = max(0, position - 15)
        second = next(
            (idx for idx in range(position - 1, lower_second, -1) if gun[idx]),
            None,
        )
        if second is None or position - second > 4:
            continue
        lower_first = max(0, second - 12)
        first = next(
            (idx for idx in range(second - 3, lower_first, -1) if gun[idx]),
            None,
        )
        if first is None:
            continue
        gap = second - first
        middle_ratio = ratio[first + 1 : second]
        mid_average = float(np.nanmean(middle_ratio)) if len(middle_ratio) else np.nan
        pre_second_j = j_values[second - 1] if second > 0 else np.nan
        close_to_second_low = (
            (close[position] / low[second] - 1.0) * 100.0 if low[second] else np.nan
        )
        output["rs_double_gun_gap_days"][position] = gap
        output["rs_double_gun_mid_avg_vol_ratio_prev"][position] = mid_average
        output["rs_double_gun_pre_second_kdj_j"][position] = pre_second_j
        output["rs_double_gun_first_vol_ratio_prev"][position] = ratio[first]
        output["rs_double_gun_second_vol_ratio_prev"][position] = ratio[second]
        output["rs_double_gun_days_since_second"][position] = position - second
        output["rs_double_gun_close_to_second_low_pct"][position] = close_to_second_low
        output["rs_double_gun_active"][position] = bool(
            np.isfinite(mid_average)
            and mid_average < 1.2
            and np.isfinite(pre_second_j)
            and pre_second_j < 20.0
            and 3 <= gap <= 10
            and close[position] >= low[second]
        )
    return {name: pd.Series(values, index=index) for name, values in output.items()}


def _tvb_anchor_state(
    frame: pd.DataFrame,
    anchor: pd.Series,
    volume_ma5: pd.Series,
) -> dict[str, pd.Series]:
    index = frame.index
    days_since = np.full(len(frame), np.nan, dtype=float)
    anchor_price = np.full(len(frame), np.nan, dtype=float)
    consolidation_range = np.full(len(frame), np.nan, dtype=float)
    avg_pre_shrink = np.zeros(len(frame), dtype=bool)

    close = frame["close"].to_numpy(dtype=float)
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    volume = frame["volume"].to_numpy(dtype=float)
    volume_ma = volume_ma5.to_numpy(dtype=float)
    anchor_values = anchor.fillna(False).to_numpy(dtype=bool)
    volume_valid = np.isfinite(volume)
    volume_ma_valid = np.isfinite(volume_ma)
    volume_sum = np.concatenate(
        ([0.0], np.cumsum(np.where(volume_valid, volume, 0.0)))
    )
    volume_count = np.concatenate(([0], np.cumsum(volume_valid)))
    volume_ma_sum = np.concatenate(
        ([0.0], np.cumsum(np.where(volume_ma_valid, volume_ma, 0.0)))
    )
    volume_ma_count = np.concatenate(([0], np.cumsum(volume_ma_valid)))
    last_anchor = -1
    running_high = np.nan
    running_low = np.nan

    for position in range(len(frame)):
        if anchor_values[position]:
            last_anchor = position
            running_high = np.nan
            running_low = np.nan
        if last_anchor < 0:
            continue
        distance = position - last_anchor
        days_since[position] = float(distance)
        anchor_price[position] = close[last_anchor]
        if distance <= 0:
            continue
        running_high = (
            high[position]
            if np.isnan(running_high)
            else max(running_high, high[position])
        )
        running_low = low[position] if np.isnan(running_low) else min(running_low, low[position])
        consolidation_range[position] = (
            running_high / running_low if running_low else np.nan
        )
        if last_anchor < 1:
            continue
        prior_start = last_anchor + 1
        prior_end = position if position > prior_start else position + 1
        prior_volume_count = volume_count[prior_end] - volume_count[prior_start]
        prior_ma_count = volume_ma_count[prior_end] - volume_ma_count[prior_start]
        prior_volume_mean = (
            (volume_sum[prior_end] - volume_sum[prior_start])
            / prior_volume_count
            if prior_volume_count
            else np.nan
        )
        prior_ma_mean = (
            (volume_ma_sum[prior_end] - volume_ma_sum[prior_start])
            / prior_ma_count
            if prior_ma_count
            else np.nan
        )
        avg_pre_shrink[position] = bool(
            volume[position] < volume[last_anchor - 1]
            and prior_end > prior_start
            and prior_volume_count
            and prior_ma_count
            and prior_volume_mean < prior_ma_mean
        )
    return {
        "days_since": pd.Series(days_since, index=index),
        "anchor_price": pd.Series(anchor_price, index=index),
        "consolidation_range": pd.Series(consolidation_range, index=index),
        "avg_pre_shrink": pd.Series(avg_pre_shrink, index=index),
    }


def compute_right_side_rule_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Return causal rule primitives for one stock, ordered by trade date.

    The result contains exactly :data:`RULE_FEATURE_COLUMNS` and keeps the
    sorted input row index so callers can join it back to the source frame.
    """

    frame = _prepare_daily(daily)
    index = frame.index
    open_ = frame["open"]
    high = frame["high"]
    low = frame["low"]
    close = frame["close"]
    pre_close = frame["pre_close"]
    volume = frame["volume"]

    pct_chg = (_safe_div(close, pre_close) - 1.0) * 100.0
    amplitude = _safe_div(high - low, pre_close) * 100.0
    close_pos = _safe_div(close - low, high - low)
    signed_body = _safe_div(close - open_, pre_close) * 100.0
    body_abs = signed_body.abs()
    candle_top = pd.concat([open_, close], axis=1).max(axis=1)
    upper_shadow = _safe_div(high - candle_top, pre_close) * 100.0
    is_rise = close > open_
    is_yin = close < open_

    vol_ratio_prev = _safe_div(volume, volume.shift(1))
    vol_ratio_prev5 = _safe_div(volume, volume.shift(1).rolling(5, min_periods=2).mean())
    # Family B1/B2/B3 uses the project daily-factor layer's inclusive rolling
    # means, which require the complete 5/20-bar window.
    vol_ratio_5_inclusive = _safe_div(volume, volume.rolling(5).mean())
    vol_ratio_20_inclusive = _safe_div(volume, volume.rolling(20).mean())

    low9 = low.rolling(9, min_periods=3).min()
    high9 = high.rolling(9, min_periods=3).max()
    rsv = (_safe_div(close - low9, high9 - low9) * 100.0).fillna(50.0)
    kdj_k = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    kdj_d = kdj_k.ewm(alpha=1 / 3, adjust=False).mean()
    kdj_j = 3.0 * kdj_k - 2.0 * kdj_d

    # Family B2/B3 consumes the project KDJ implementation, which starts only
    # after a complete nine-bar RSV window.  The live Z detector intentionally
    # uses the partial-history/fill-50 KDJ above.
    family_low9 = low.rolling(9).min()
    family_high9 = high.rolling(9).max()
    family_rsv = _safe_div(close - family_low9, family_high9 - family_low9) * 100.0
    family_k = family_rsv.ewm(alpha=1 / 3, adjust=False).mean()
    family_d = family_k.ewm(alpha=1 / 3, adjust=False).mean()
    family_kdj_j = 3.0 * family_k - 2.0 * family_d

    ma3 = close.rolling(3, min_periods=1).mean()
    ma6 = close.rolling(6, min_periods=2).mean()
    ma12 = close.rolling(12, min_periods=4).mean()
    ma24 = close.rolling(24, min_periods=8).mean()
    bbi = (ma3 + ma6 + ma12 + ma24) / 4.0
    bbi_distance = (_safe_div(close, bbi) - 1.0) * 100.0
    bbi_slope_5d = (_safe_div(bbi, bbi.shift(5)) - 1.0) * 100.0

    ma5 = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    ma20_slope_5d = (_safe_div(ma20, ma20.shift(5)) - 1.0) * 100.0

    # The Web B2 generator consumes the project BBI, whose four moving
    # averages all require their complete windows.  Z-skill detectors use the
    # partial-history BBI above, so keep a separate family value rather than
    # silently mixing the two min-period contracts.
    family_bbi = (
        close.rolling(3).mean()
        + close.rolling(6).mean()
        + close.rolling(12).mean()
        + close.rolling(24).mean()
    ) / 4.0
    family_bbi_distance = (_safe_div(close, family_bbi) - 1.0) * 100.0

    recent_yin_count = is_yin.rolling(4, min_periods=1).sum()
    support_ok = (close >= family_bbi * 0.97) | (close >= ma60 * 0.97)
    b1_like = (
        (family_kdj_j <= -10.0)
        & (vol_ratio_20_inclusive < 1.0)
        & (recent_yin_count < 4.0)
        & support_ok
    )
    b1_like_prev3 = b1_like.shift(1, fill_value=False).rolling(3, min_periods=1).max().astype(bool)
    pre_oversold_prev5 = (
        family_kdj_j.lt(0)
        .shift(1, fill_value=False)
        .rolling(5, min_periods=1)
        .max()
        .astype(bool)
    )
    bbi_reclaim = (close.shift(1) <= family_bbi.shift(1) * 1.01) & (
        close > family_bbi
    )
    b2_from_b1_pchg3 = (
        b1_like_prev3
        & (pct_chg >= 3.0)
        & (vol_ratio_5_inclusive >= 1.2)
        & (close_pos >= 0.70)
        & (family_kdj_j < 60.0)
    )
    b2_from_b1_pchg4 = (
        b1_like_prev3
        & (pct_chg >= 4.0)
        & (vol_ratio_5_inclusive >= 1.5)
        & (high <= close * 1.01)
        & (family_kdj_j < 55.0)
    )
    b2_from_b1_pchg5 = (
        b1_like_prev3
        & (pct_chg >= 5.0)
        & (vol_ratio_5_inclusive >= 1.5)
        & (close_pos >= 0.75)
        & (family_kdj_j < 60.0)
    )
    b2_any = (
        (pct_chg >= 4.0)
        & (vol_ratio_5_inclusive >= 1.5)
        & (close_pos >= 0.75)
        & (family_kdj_j < 80.0)
    )
    b2_oversold = (
        pre_oversold_prev5
        & (pct_chg >= 3.0)
        & (vol_ratio_5_inclusive >= 1.2)
        & (close_pos >= 0.70)
        & (family_kdj_j < 80.0)
    )
    b2_bbi_reclaim = (
        bbi_reclaim
        & (pct_chg >= 3.0)
        & (vol_ratio_5_inclusive >= 1.2)
        & (close_pos >= 0.65)
        & (family_kdj_j < 80.0)
    )
    b2_reference = b2_from_b1_pchg3 | b2_from_b1_pchg4 | b2_from_b1_pchg5
    b2_broad = b2_reference | b2_any | b2_oversold | b2_bbi_reclaim
    b2_recent_prev3 = (
        b2_reference.shift(1, fill_value=False)
        .rolling(3, min_periods=1)
        .max()
        .astype(bool)
    )
    b2_broad_recent_prev3 = (
        b2_broad.shift(1, fill_value=False)
        .rolling(3, min_periods=1)
        .max()
        .astype(bool)
    )
    b3_small_pos_amp7 = (
        b2_recent_prev3
        & pct_chg.gt(0.0)
        & pct_chg.lt(2.0)
        & amplitude.lt(7.0)
    )
    b3_broad_small_pos = (
        b2_broad_recent_prev3
        & pct_chg.gt(0.0)
        & pct_chg.lt(2.0)
        & amplitude.lt(7.0)
        & close_pos.ge(0.50)
    )
    b3_broad_calm_pullback = (
        b2_broad_recent_prev3
        & pct_chg.ge(-1.0)
        & pct_chg.lt(2.0)
        & amplitude.lt(7.0)
        & vol_ratio_5_inclusive.le(1.3)
    )

    strong_yang = is_rise & (pct_chg >= 3.0) & (vol_ratio_prev5 >= 1.5)
    two_yang = _two_yang_state(frame, strong_yang, vol_ratio_prev, kdj_j, pct_chg)
    double_gun = _double_gun_state(frame, vol_ratio_prev, kdj_j, pct_chg)

    zg_white = close.ewm(span=10, adjust=False).mean().ewm(span=10, adjust=False).mean()
    dg_yellow = (
        close.rolling(14, min_periods=8).mean()
        + close.rolling(28, min_periods=14).mean()
        + close.rolling(57, min_periods=28).mean()
        + close.rolling(114, min_periods=60).mean()
    ) / 4.0
    white_yellow_spread = (_safe_div(zg_white, dg_yellow) - 1.0) * 100.0
    close_to_yellow = (_safe_div(close, dg_yellow) - 1.0) * 100.0
    close_bowl_position = _safe_div(close - dg_yellow, zg_white - dg_yellow)

    phase_exhale = (pct_chg > 0.0) & (vol_ratio_prev > 1.0)
    phase_inhale = (pct_chg < 0.0) & (vol_ratio_prev < 1.0)
    higher_low_reference = pd.concat([low.shift(3), low.shift(6)], axis=1).min(axis=1)
    higher_low_ratio = _safe_div(low, higher_low_reference) - 1.0

    kengqi = _kengqi_event_state(frame, vol_ratio_prev)

    fangliang = (pct_chg > 5.0) & (
        volume > volume.shift(1).rolling(5, min_periods=3).mean() * 1.5
    )
    zaihou = _zaihou_event_state(frame, pct_chg)

    platform_high = high.rolling(20, min_periods=15).max()
    platform_low = low.rolling(20, min_periods=15).min()
    # Live YUEYUE protects sub-one adjusted prices with max(platform_low, 1).
    platform_range = _safe_div(platform_high - platform_low, platform_low.clip(lower=1.0))
    huge_volume = volume > volume.rolling(10, min_periods=5).mean() * 2.0
    huge_volume_count = huge_volume.rolling(20, min_periods=10).sum()
    huge_yang_count = (huge_volume & is_rise).rolling(20, min_periods=10).sum()

    prev20_high = high.shift(1).rolling(20, min_periods=10).max()
    prev20_low = low.shift(1).rolling(20, min_periods=10).min()
    high_to_prev20_high = (_safe_div(high, prev20_high) - 1.0) * 100.0
    low_to_prev20_low = (_safe_div(low, prev20_low) - 1.0) * 100.0
    at_key = (high >= prev20_high * 0.98) | (low <= prev20_low * 1.02)
    at_bottom = low <= prev20_low * 1.05
    previous_body_mean = body_abs.shift(1).rolling(6, min_periods=2).mean()

    ema10 = close.ewm(span=10, adjust=False, min_periods=10).mean()
    ema20 = close.ewm(span=20, adjust=False, min_periods=20).mean()
    ema144 = close.ewm(span=144, adjust=False, min_periods=144).mean()
    ema169 = close.ewm(span=169, adjust=False, min_periods=169).mean()
    tunnel_upper = pd.concat([ema144, ema169], axis=1).max(axis=1)
    tunnel_mid = (ema144 + ema169) / 2.0
    recent_pullback = (
        (low <= tunnel_upper * 1.025).rolling(8, min_periods=1).max().fillna(False).astype(bool)
    )
    vegas_trend_stack = (close > tunnel_upper) & (ema10 > ema20) & (ema20 > tunnel_upper)
    vegas_tunnel_up = (tunnel_mid > tunnel_mid.shift(5)) & (ema144 > ema144.shift(20))
    vegas_rebound = (close > ema10) & is_rise & (close > close.shift(1))
    vegas_volume_strength = _safe_div(volume, volume.rolling(20).mean()) - 1.0
    vegas_volume_ratio = _safe_div(volume, volume.rolling(20).mean())
    history_position = pd.Series(np.arange(len(frame)), index=index)
    risk_name = (
        frame["name"].fillna("").astype(str).str.upper()
        if "name" in frame.columns
        else pd.Series("", index=index, dtype=str)
    )
    risk_name = (
        risk_name.str.contains("ST", regex=False)
        | risk_name.str.contains("*", regex=False)
        | risk_name.str.contains("退", regex=False)
    )
    rule_tradable = open_.notna() & open_.gt(0.0) & ~risk_name
    vegas_history_ok = history_position.ge(180)
    vegas_signal = (
        vegas_history_ok
        & vegas_trend_stack
        & vegas_tunnel_up
        & recent_pullback
        & vegas_rebound
        & vegas_volume_ratio.gt(1.05)
        & vegas_volume_ratio.lt(3.0)
        & (_safe_div(close, tunnel_upper) <= 1.18)
        & rule_tradable
    )

    tvb_anchor_multiple = _safe_div(volume.shift(1), volume.shift(2))
    tvb_25 = _tvb_anchor_state(frame, tvb_anchor_multiple >= 2.5, volume.rolling(5).mean())
    tvb_30 = _tvb_anchor_state(frame, tvb_anchor_multiple >= 3.0, volume.rolling(5).mean())
    tvb_bull_no60 = (ma5 > ma10) & (ma10 > ma20) & (ma20 > ma20.shift(1))
    tvb_breakout_25 = (_safe_div(close, tvb_25["anchor_price"]) - 1.0) * 100.0
    tvb_breakout_30 = (_safe_div(close, tvb_30["anchor_price"]) - 1.0) * 100.0
    tvb_candidate_25 = (
        (tvb_25["days_since"] > 0)
        & tvb_bull_no60
        & tvb_25["avg_pre_shrink"]
        & (tvb_25["consolidation_range"] < 1.15)
        & (close > tvb_25["anchor_price"])
        & is_rise
    )
    tvb_candidate_30 = (
        (tvb_30["days_since"] > 0)
        & tvb_bull_no60
        & tvb_30["avg_pre_shrink"]
        & (tvb_30["consolidation_range"] < 1.15)
        & (close > tvb_30["anchor_price"])
        & is_rise
    )
    # The two production YAML variants are merged directly.  Unlike Vegas,
    # add_triple_volume_strategy_pool_signals does not add a name tradability
    # predicate to the relaxed avg_pre_shrink_bull_no60 variants.
    tvb_merged = tvb_candidate_25 | tvb_candidate_30

    features: dict[str, pd.Series] = {
        "rs_pct_chg_1d": pct_chg,
        "rs_pct_chg_lag1": pct_chg.shift(1),
        "rs_amplitude_pct": amplitude,
        "rs_close_pos": close_pos,
        "rs_body_abs_pct": body_abs,
        "rs_upper_shadow_pct": upper_shadow,
        "rs_is_rise": is_rise,
        "rs_is_rise_lag1": is_rise.shift(1, fill_value=False),
        "rs_vol_ratio_prev": vol_ratio_prev,
        "rs_vol_ratio_prev5": vol_ratio_prev5,
        "rs_vol_ratio_prev5_lag1": vol_ratio_prev5.shift(1),
        "rs_vol_ratio_5_inclusive": vol_ratio_5_inclusive,
        "rs_vol_ratio_20_inclusive": vol_ratio_20_inclusive,
        "rs_kdj_j": kdj_j,
        "rs_family_kdj_j": family_kdj_j,
        "rs_kdj_j_lag1": kdj_j.shift(1),
        "rs_kdj_j_lag2": kdj_j.shift(2),
        "rs_kdj_j_lag1_minus_lag2": kdj_j.shift(1) - kdj_j.shift(2),
        "rs_bbi_distance_pct": bbi_distance,
        "rs_family_bbi_distance_pct": family_bbi_distance,
        "rs_recent_yin_count_4": recent_yin_count,
        "rs_close_to_ma60_pct": (_safe_div(close, ma60) - 1.0) * 100.0,
        "rs_b1_support_ok": support_ok,
        "rs_bbi_slope_5d_pct": bbi_slope_5d,
        "rs_pre_oversold_prev5": pre_oversold_prev5,
        "rs_b1_like_prev3": b1_like_prev3,
        "rs_bbi_reclaim": bbi_reclaim,
        "rs_b2_from_b1_pchg3": b2_from_b1_pchg3,
        "rs_b2_from_b1_pchg4": b2_from_b1_pchg4,
        "rs_b2_from_b1_pchg5": b2_from_b1_pchg5,
        "rs_b2_any": b2_any,
        "rs_b2_oversold": b2_oversold,
        "rs_b2_bbi_reclaim": b2_bbi_reclaim,
        "rs_b2_recent_prev3": b2_recent_prev3,
        "rs_b2_broad_recent_prev3": b2_broad_recent_prev3,
        "rs_days_since_b2": _days_since(b2_broad),
        "rs_b3_small_pos_amp7": b3_small_pos_amp7,
        "rs_b3_broad_small_pos": b3_broad_small_pos,
        "rs_b3_broad_calm_pullback": b3_broad_calm_pullback,
        "rs_high_to_prev20_high_pct": high_to_prev20_high,
        "rs_low_to_prev20_low_pct": low_to_prev20_low,
        "rs_at_key_20d": at_key,
        "rs_at_bottom_20d": at_bottom,
        "rs_body_vs_prev6": _safe_div(body_abs, previous_body_mean),
        "rs_strong_yang": strong_yang,
        "rs_strong_yang_count_8d": strong_yang.rolling(8, min_periods=1).sum(),
        **two_yang,
        **double_gun,
        "rs_zg_white": zg_white,
        "rs_dg_yellow": dg_yellow,
        "rs_white_yellow_spread_pct": white_yellow_spread,
        "rs_close_to_yellow_pct": close_to_yellow,
        "rs_close_bowl_position": close_bowl_position,
        "rs_fangliang_recent_3_12d": zaihou["recent"],
        "rs_days_since_fangliang": zaihou["days_since"],
        "rs_fangliang_ref_volume_15d": zaihou["reference_volume"],
        "rs_volume_to_fangliang_ref": _safe_div(volume, zaihou["reference_volume"]),
        "rs_phase_exhale": phase_exhale,
        "rs_phase_inhale": phase_inhale,
        "rs_exhale_count_7d": phase_exhale.rolling(7, min_periods=1).sum(),
        "rs_inhale_count_7d": phase_inhale.rolling(7, min_periods=1).sum(),
        "rs_higher_low_ratio": higher_low_ratio,
        "rs_platform_range_20d": platform_range,
        "rs_huge_volume_count_20d": huge_volume_count,
        "rs_huge_yang_share_20d": _safe_div(huge_yang_count, huge_volume_count),
        "rs_close_to_platform_high_pct": (_safe_div(close, platform_high) - 1.0) * 100.0,
        "rs_pit_depth_18d": kengqi["depth"],
        "rs_pit_recent_3_14d": kengqi["recent"],
        "rs_days_since_pit": kengqi["days_since"],
        "rs_pit_fill_ratio": kengqi["fill"],
        "rs_post_to_pre_volume_ratio": kengqi["post_to_pre_volume"],
        "rs_last_pit_vol_ratio_prev": kengqi["pit_vol_ratio_prev"],
        "rs_ema10": ema10,
        "rs_ema20": ema20,
        "rs_ema144": ema144,
        "rs_ema169": ema169,
        "rs_vegas_tunnel_upper": tunnel_upper,
        "rs_vegas_tunnel_distance": _safe_div(close, tunnel_upper) - 1.0,
        "rs_vegas_tunnel_slope_20d": _safe_div(tunnel_mid, tunnel_mid.shift(20)) - 1.0,
        "rs_vegas_fast_spread": _safe_div(ema10, ema20) - 1.0,
        "rs_vegas_recent_pullback_8d": recent_pullback,
        "rs_vegas_trend_stack": vegas_trend_stack,
        "rs_vegas_tunnel_up": vegas_tunnel_up,
        "rs_vegas_right_side_rebound": vegas_rebound,
        "rs_vegas_volume_strength": vegas_volume_strength,
        "rs_vegas_volume_confirm": (vegas_volume_ratio > 1.05) & (vegas_volume_ratio < 3.0),
        "rs_vegas_not_overheated": _safe_div(close, tunnel_upper) <= 1.18,
        "rs_vegas_history_ok": vegas_history_ok,
        "rs_vegas_tradable": rule_tradable,
        "rs_vegas_signal": vegas_signal,
        "rs_tvb_anchor_volume_multiple": tvb_anchor_multiple,
        "rs_tvb_days_since_anchor_25": tvb_25["days_since"],
        "rs_tvb_days_since_anchor_30": tvb_30["days_since"],
        "rs_tvb_anchor_price_25": tvb_25["anchor_price"],
        "rs_tvb_anchor_price_30": tvb_30["anchor_price"],
        "rs_tvb_consolidation_range_25": tvb_25["consolidation_range"],
        "rs_tvb_consolidation_range_30": tvb_30["consolidation_range"],
        "rs_tvb_avg_pre_shrink_25": tvb_25["avg_pre_shrink"],
        "rs_tvb_avg_pre_shrink_30": tvb_30["avg_pre_shrink"],
        "rs_tvb_breakout_pct_25": tvb_breakout_25,
        "rs_tvb_breakout_pct_30": tvb_breakout_30,
        "rs_tvb_bull_no60": tvb_bull_no60,
        "rs_ma20_slope_5d_pct": ma20_slope_5d,
        "rs_tvb_candidate_25": tvb_candidate_25,
        "rs_tvb_candidate_30": tvb_candidate_30,
        "rs_tvb_merged": tvb_merged,
    }
    result = pd.DataFrame(features, index=index).loc[:, RULE_FEATURE_COLUMNS]
    validate_signal_factor_contract(result.columns)
    return result


def audit_factor_coverage(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    """Report availability and basic usability for each requested factor."""

    row_count = len(frame)
    rows: list[dict[str, object]] = []
    for feature in feature_columns:
        if feature not in frame.columns:
            rows.append(
                {
                    "feature": feature,
                    "non_null": 0,
                    "coverage": 0.0,
                    "unique": 0,
                    "status": "missing",
                }
            )
            continue
        values = frame[feature]
        non_null = int(values.notna().sum())
        coverage = float(non_null / row_count) if row_count else 0.0
        unique = int(values.nunique(dropna=True))
        if row_count == 0:
            status = "empty_frame"
        elif non_null == 0:
            status = "all_null"
        elif unique <= 1:
            status = "constant"
        elif coverage < 0.80:
            status = "sparse"
        else:
            status = "ok"
        rows.append(
            {
                "feature": feature,
                "non_null": non_null,
                "coverage": coverage,
                "unique": unique,
                "status": status,
            }
        )
    return pd.DataFrame(
        rows,
        columns=["feature", "non_null", "coverage", "unique", "status"],
    )


def validate_signal_factor_contract(columns: Iterable[str]) -> None:
    """Raise a grouped error when any signal-required feature is absent."""

    available = {str(column) for column in columns}
    missing_by_signal = {
        signal: [
            feature
            for feature in SIGNAL_FEATURE_REQUIREMENTS[signal]
            if feature not in available
        ]
        for signal in RIGHT_SIDE_SIGNALS
    }
    missing_by_signal = {
        signal: missing
        for signal, missing in missing_by_signal.items()
        if missing
    }
    if not missing_by_signal:
        return
    details = "; ".join(
        f"{signal} missing {missing}"
        for signal, missing in missing_by_signal.items()
    )
    raise ValueError(f"right-side signal factor contract is incomplete: {details}")


__all__ = [
    "RIGHT_SIDE_SIGNALS",
    "SIGNAL_FEATURE_REQUIREMENTS",
    "RULE_FEATURE_COLUMNS",
    "RULE_FEATURE_SCHEMA_VERSION",
    "RULE_FEATURE_COLUMNS_SHA256",
    "LEGACY_RULE_FEATURE_SCHEMA_VERSION_V1",
    "LEGACY_RULE_FEATURE_COLUMNS_V1",
    "LEGACY_RULE_FEATURE_COLUMNS_SHA256_V1",
    "ADDED_RULE_FEATURE_COLUMNS_V2",
    "rule_feature_columns_sha256",
    "compute_right_side_rule_features",
    "audit_factor_coverage",
    "validate_signal_factor_contract",
]
