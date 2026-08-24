"""Canonical rule-factor contract for the unified left-side ranker.

The three remaining left-side Z-skill strategies share one model but retain
their strategy identity as task one-hots.  This module materializes every
distinct value used by their screening predicates.  Values already owned by
the project factor layer are declared as requirements and never emitted a
second time.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from quant.features.canonical_factor_names import (
    assert_no_forbidden_factor_names,
    stable_canonical_feature_union,
)
from quant.features.variable_library import build_continuous_ohlc
from quant.research.short_side_groups import LEFT_GROUPS


LEFT_SIDE_SIGNALS: tuple[str, ...] = (
    *LEFT_GROUPS,
)
LEFT_SIDE_RAW_LOW_PULLBACK_SIGNALS: tuple[str, ...] = (
    "DUICHEN_VA",
    "NANA",
    "YIDONG_DILIAN",
)
LEFT_SIDE_SIGNAL_SCHEMA_VERSION = "left_side_group_signal_v2_4_20260824"
LEFT_SIDE_RULE_FEATURE_SCHEMA_VERSION = "left_side_rule_features_v2_27_20260824"

LEFT_SIDE_PROJECT_FACTOR_REQUIREMENTS: tuple[str, ...] = (
    "close",
    "pct_chg",
    "bbi",
    "kdj_d_j",
    "ma_60",
    "amplitude_1",
    "volume_relative_5d",
    "volume_relative_20d",
)

LEFT_SIDE_SHARED_RULE_REQUIREMENTS: tuple[str, ...] = (
    "rs_is_rise",
    "rs_close_pos",
)

LEFT_SIDE_SIGNAL_FEATURE_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "B1": (
        "pct_chg",
        "amplitude_1",
        "bbi",
        "ma_60",
        "kdj_d_j",
    ),
    "SB1": (
        "volume_relative_5d",
        "kdj_d_j",
        "rs_is_rise",
        "ls_range3_pct",
        "ls_low_to_prev3_low_pct",
    ),
    "SUPER_B1": (
        "pct_chg",
        "amplitude_1",
        "volume_relative_5d",
        "volume_relative_20d",
        "kdj_d_j",
        "rs_is_rise",
        "rs_close_pos",
        "ls_recent_washout12_3d",
        "ls_recent_washout15_3d",
        "ls_small_reversal_loose",
        "ls_small_reversal_tight",
    ),
    "LOW_PULLBACK": (
        "pct_chg",
        "bbi",
        "kdj_d_j",
        "volume_relative_5d",
        "ls_days_since_yidong",
        "ls_yidong_pct_chg",
        "ls_yidong_volume_relative_prev5",
        "ls_volume_to_yidong",
        "ls_volume_to_ground_q25_60d",
        "ls_close_to_yidong_low",
        "ls_yidong_shrink_ok",
        "ls_yidong_ground_ok",
        "ls_yidong_pullback_ok",
        "ls_low_absorb_bbi",
        "ls_nana_rise_volume_count_8d",
        "ls_nana_shrink_count_7d",
        "ls_nana_big_yin_count_15d",
        "ls_close_to_bbi",
        "close",
        "ls_va_up_days",
        "ls_va_down_days",
        "ls_va_up_pct",
        "ls_va_down_pct",
        "ls_va_time_symmetry",
        "ls_va_space_symmetry",
        "ls_volume_ratio_prev",
    ),
}

LEFT_SIDE_RULE_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    dict.fromkeys(
        feature
        for signal in LEFT_SIDE_SIGNALS
        for feature in LEFT_SIDE_SIGNAL_FEATURE_REQUIREMENTS[signal]
        if feature not in frozenset(
            (*LEFT_SIDE_PROJECT_FACTOR_REQUIREMENTS, *LEFT_SIDE_SHARED_RULE_REQUIREMENTS)
        )
    )
)


def left_side_rule_columns_sha256(columns: Sequence[str]) -> str:
    payload = "\n".join(str(column) for column in columns).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


LEFT_SIDE_RULE_FEATURE_COLUMNS_SHA256 = left_side_rule_columns_sha256(
    LEFT_SIDE_RULE_FEATURE_COLUMNS
)


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.astype(float).div(denominator.astype(float).replace(0.0, np.nan))


def _prepare_daily(daily: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(daily, pd.DataFrame):
        raise TypeError("left-side factors require a pandas DataFrame")
    date_column = "date" if "date" in daily.columns else "trade_date"
    volume_column = "volume" if "volume" in daily.columns else "vol"
    required = {date_column, "open", "high", "low", "close", volume_column}
    missing = required - set(daily.columns)
    if missing:
        raise ValueError(f"left-side factors missing daily columns: {sorted(missing)}")
    out = daily.copy()
    date_text = out[date_column].astype(str).str.replace(r"\.0$", "", regex=True)
    compact = date_text.str.fullmatch(r"\d{8}").fillna(False)
    out["date"] = pd.to_datetime(out[date_column], errors="coerce")
    if compact.any():
        out.loc[compact, "date"] = pd.to_datetime(
            date_text.loc[compact], format="%Y%m%d", errors="coerce"
        )
    out["volume"] = pd.to_numeric(out[volume_column], errors="coerce")
    for column in ("open", "high", "low", "close", "pre_close", "pct_chg"):
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    out = out.sort_values("date", kind="stable").reset_index(drop=True)
    if out["date"].duplicated().any():
        raise ValueError("left-side factors require one row per trade date")
    continuous = build_continuous_ohlc(out)
    out[["open", "high", "low", "close"]] = continuous[
        ["open", "high", "low", "close"]
    ]
    out["pre_close"] = out["close"].shift(1)
    out["pct_chg"] = out["close"].pct_change() * 100.0
    return out


def _base_state(frame: pd.DataFrame) -> pd.DataFrame:
    volume = frame["volume"]
    close = frame["close"]
    low9 = frame["low"].rolling(9, min_periods=3).min()
    high9 = frame["high"].rolling(9, min_periods=3).max()
    rsv = (_safe_div(close - low9, high9 - low9) * 100.0).fillna(50.0)
    k = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    d = k.ewm(alpha=1 / 3, adjust=False).mean()
    ma3 = close.rolling(3, min_periods=1).mean()
    ma6 = close.rolling(6, min_periods=2).mean()
    ma12 = close.rolling(12, min_periods=4).mean()
    ma24 = close.rolling(24, min_periods=8).mean()
    return pd.DataFrame(
        {
            "pct_chg": frame["pct_chg"],
            "kdj_d_j": 3.0 * k - 2.0 * d,
            "bbi": (ma3 + ma6 + ma12 + ma24) / 4.0,
            "volume_relative_prev5": _safe_div(
                volume, volume.shift(1).rolling(5, min_periods=1).mean()
            ),
            "volume_ratio_prev": _safe_div(volume, volume.shift(1)),
            "is_rise": frame["close"] > frame["open"],
        },
        index=frame.index,
    )


def compute_left_side_rule_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Materialize the unique, causal predicate state for all left signals."""

    frame = _prepare_daily(daily)
    state = _base_state(frame)
    row_count = len(frame)
    values = {
        column: np.full(row_count, np.nan, dtype=float)
        for column in LEFT_SIDE_RULE_FEATURE_COLUMNS
    }
    boolean_columns = {
        "ls_yidong_shrink_ok",
        "ls_yidong_ground_ok",
        "ls_yidong_pullback_ok",
        "ls_low_absorb_bbi",
    }

    for position in range(row_count):
        today_close = float(frame.at[position, "close"])
        today_volume = float(frame.at[position, "volume"])
        today_pct = float(state.at[position, "pct_chg"])
        today_bbi = float(state.at[position, "bbi"])
        values["ls_close_to_bbi"][position] = (
            today_close / today_bbi - 1.0 if np.isfinite(today_bbi) and today_bbi else np.nan
        )
        low_absorb = bool(
            np.isfinite(today_pct)
            and -3.0 <= today_pct <= 1.5
            and np.isfinite(today_bbi)
            and today_close <= today_bbi * 1.03
        )
        values["ls_low_absorb_bbi"][position] = float(low_absorb)
        values["ls_volume_ratio_prev"][position] = state.at[
            position, "volume_ratio_prev"
        ]

        if position >= 15:
            yidong_position: int | None = None
            for candidate in range(position - 1, max(0, position - 11), -1):
                if (
                    bool(state.at[candidate, "is_rise"])
                    and float(state.at[candidate, "pct_chg"]) >= 2.5
                    and float(state.at[candidate, "volume_relative_prev5"]) >= 1.8
                ):
                    yidong_position = candidate
                    break
            if yidong_position is not None and position - yidong_position >= 2:
                yidong_volume = float(frame.at[yidong_position, "volume"])
                yidong_low = float(frame.at[yidong_position, "low"])
                ground_q25 = float(
                    frame.loc[max(0, position - 59) : position, "volume"].quantile(0.25)
                )
                shrink_ok = today_volume <= yidong_volume * 0.75
                ground_ok = today_volume <= ground_q25
                pullback_ok = today_close >= yidong_low * 0.96
                values["ls_days_since_yidong"][position] = position - yidong_position
                values["ls_yidong_pct_chg"][position] = state.at[
                    yidong_position, "pct_chg"
                ]
                values["ls_yidong_volume_relative_prev5"][position] = state.at[
                    yidong_position, "volume_relative_prev5"
                ]
                values["ls_volume_to_yidong"][position] = (
                    today_volume / yidong_volume if yidong_volume else np.nan
                )
                values["ls_volume_to_ground_q25_60d"][position] = (
                    today_volume / ground_q25 if ground_q25 else np.nan
                )
                values["ls_close_to_yidong_low"][position] = (
                    today_close / yidong_low - 1.0 if yidong_low else np.nan
                )
                values["ls_yidong_shrink_ok"][position] = float(shrink_ok)
                values["ls_yidong_ground_ok"][position] = float(ground_ok)
                values["ls_yidong_pullback_ok"][position] = float(pullback_ok)

        if position >= 23:
            recent = frame.iloc[position - 14 : position + 1]
            recent_state = state.iloc[position - 14 : position + 1]
            build = recent.iloc[:8]
            build_state = recent_state.iloc[:8]
            pullback = recent.iloc[8:]
            rise_count = int(
                (
                    build_state["is_rise"].astype(bool)
                    & build["volume"].gt(build["volume"].shift(1))
                ).sum()
            )
            shrink_count = int(
                pullback["volume"].lt(pullback["volume"].shift(1)).sum()
            )
            big_yin = (
                (recent["close"] < recent["open"])
                & recent_state["volume_relative_prev5"].ge(1.5)
                & recent_state["pct_chg"].le(-2.0)
            )
            values["ls_nana_rise_volume_count_8d"][position] = rise_count
            values["ls_nana_shrink_count_7d"][position] = shrink_count
            values["ls_nana_big_yin_count_15d"][position] = int(big_yin.sum())

        if position >= 29:
            window = frame.iloc[position - 21 : position + 1].reset_index(drop=True)
            peak = int(window["high"].idxmax())
            trough = int(window.loc[:peak, "low"].idxmin()) if peak > 2 else -1
            if 0 <= trough < peak < len(window) - 2:
                trough_price = float(window.at[trough, "low"])
                peak_price = float(window.at[peak, "high"])
                up_days = peak - trough
                down_days = len(window) - 1 - peak
                up_pct = (
                    (peak_price - trough_price) / trough_price
                    if trough_price > 0.0
                    else np.nan
                )
                down_pct = (
                    (peak_price - today_close) / peak_price
                    if peak_price > 0.0
                    else np.nan
                )
                values["ls_va_up_days"][position] = up_days
                values["ls_va_down_days"][position] = down_days
                values["ls_va_up_pct"][position] = up_pct
                values["ls_va_down_pct"][position] = down_pct
                values["ls_va_time_symmetry"][position] = down_days / up_days
                values["ls_va_space_symmetry"][position] = (
                    down_pct / up_pct if np.isfinite(up_pct) and up_pct > 0.0 else np.nan
                )

    result = pd.DataFrame(values, index=frame.index)
    previous_low3 = frame["low"].shift(1).rolling(3, min_periods=3).min()
    previous_range3 = (
        frame["high"].shift(1).rolling(3, min_periods=3).max()
        / previous_low3.replace(0.0, np.nan)
        - 1.0
    )
    volume_relative_5d = state["volume_relative_prev5"]
    volume_relative_20d = _safe_div(
        frame["volume"], frame["volume"].rolling(20, min_periods=5).mean()
    )
    amplitude = _safe_div(
        frame["high"] - frame["low"], frame["low"]
    ) * 100.0
    is_yin = ~state["is_rise"].astype(bool)
    washout12 = (
        is_yin
        & state["pct_chg"].gt(-9.5)
        & volume_relative_5d.ge(1.2)
        & frame["low"].lt(
            frame["low"].shift(1).rolling(10, min_periods=5).min()
        )
    )
    washout15 = washout12 & volume_relative_5d.ge(1.5)
    result["ls_range3_pct"] = previous_range3
    result["ls_low_to_prev3_low_pct"] = _safe_div(
        frame["low"], previous_low3
    ) - 1.0
    result["ls_recent_washout12_3d"] = (
        washout12.shift(1).rolling(3, min_periods=1).max().fillna(0.0) > 0
    )
    result["ls_recent_washout15_3d"] = (
        washout15.shift(1).rolling(3, min_periods=1).max().fillna(0.0) > 0
    )
    result["ls_small_reversal_loose"] = (
        amplitude.lt(8.0)
        & state["pct_chg"].gt(-2.5)
        & state["pct_chg"].lt(3.0)
        & volume_relative_20d.lt(1.0)
    )
    result["ls_small_reversal_tight"] = (
        amplitude.lt(7.0)
        & state["pct_chg"].gt(-2.0)
        & state["pct_chg"].lt(2.5)
        & volume_relative_20d.lt(0.9)
    )
    for column in boolean_columns:
        result[column] = result[column].fillna(0.0).astype(bool)
    result = result.loc[:, LEFT_SIDE_RULE_FEATURE_COLUMNS]
    validate_left_side_factor_contract(
        (
            *result.columns,
            *LEFT_SIDE_PROJECT_FACTOR_REQUIREMENTS,
            *LEFT_SIDE_SHARED_RULE_REQUIREMENTS,
        )
    )
    return result


def compute_left_side_signal_flags(
    daily: pd.DataFrame,
    *,
    rule_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Rebuild the four left-side strategy-group predicates without cooldown."""

    frame = _prepare_daily(daily)
    state = _base_state(frame)
    rules = (
        compute_left_side_rule_features(frame).reset_index(drop=True)
        if rule_features is None
        else rule_features.reset_index(drop=True).copy()
    )
    if len(rules) != len(frame):
        raise ValueError("left-side rule features must align one-to-one with daily rows")
    raw_low_flags = pd.DataFrame(
        False,
        index=frame.index,
        columns=LEFT_SIDE_RAW_LOW_PULLBACK_SIGNALS,
    )
    for position in range(len(frame)):
        j = float(state.at[position, "kdj_d_j"])
        if position >= 15 and np.isfinite(rules.at[position, "ls_days_since_yidong"]):
            raw_low_flags.at[position, "YIDONG_DILIAN"] = bool(
                (
                    bool(rules.at[position, "ls_yidong_shrink_ok"])
                    or bool(rules.at[position, "ls_yidong_ground_ok"])
                )
                and bool(rules.at[position, "ls_yidong_pullback_ok"])
                and bool(rules.at[position, "ls_low_absorb_bbi"])
                and j < 35.0
            )
        if position >= 23:
            raw_low_flags.at[position, "NANA"] = bool(
                rules.at[position, "ls_nana_rise_volume_count_8d"] >= 3
                and rules.at[position, "ls_nana_shrink_count_7d"] >= 2
                and rules.at[position, "ls_nana_big_yin_count_15d"] == 0
                and bool(rules.at[position, "ls_low_absorb_bbi"])
                and j < 15.0
            )
        if position >= 29:
            raw_low_flags.at[position, "DUICHEN_VA"] = bool(
                0.5 <= rules.at[position, "ls_va_time_symmetry"] <= 2.0
                and 0.4 <= rules.at[position, "ls_va_space_symmetry"] <= 1.1
                and rules.at[position, "ls_volume_ratio_prev"] < 0.75
                and j < 25.0
            )
    pct = state["pct_chg"]
    amplitude = _safe_div(frame["high"] - frame["low"], frame["low"]) * 100.0
    ma60 = frame["close"].rolling(60, min_periods=20).mean()
    close_pos = _safe_div(frame["close"] - frame["low"], frame["high"] - frame["low"])
    volume_relative_5d = state["volume_relative_prev5"]
    flags = pd.DataFrame(False, index=frame.index, columns=LEFT_SIDE_SIGNALS)
    flags["B1"] = (
        pct.between(-2.0, 2.0, inclusive="both")
        & amplitude.lt(7.0)
        & state["bbi"].gt(ma60)
        & state["kdj_d_j"].lt(0.0)
    )
    flags["SB1"] = (
        rules["ls_range3_pct"].lt(0.10)
        & ~state["is_rise"].astype(bool)
        & volume_relative_5d.ge(1.2)
        & rules["ls_low_to_prev3_low_pct"].lt(0.0)
        & state["kdj_d_j"].lt(0.0)
    )
    flags["SUPER_B1"] = (
        (
            rules["ls_recent_washout12_3d"].astype(bool)
            & rules["ls_small_reversal_loose"].astype(bool)
            & state["kdj_d_j"].lt(0.0)
        )
        | (
            rules["ls_recent_washout15_3d"].astype(bool)
            & rules["ls_small_reversal_tight"].astype(bool)
            & state["kdj_d_j"].lt(0.0)
        )
        | (
            rules["ls_recent_washout12_3d"].astype(bool)
            & amplitude.lt(8.0)
            & pct.gt(-2.5)
            & pct.lt(3.0)
            & state["kdj_d_j"].lt(-10.0)
            & close_pos.ge(0.40)
        )
    )
    flags["LOW_PULLBACK"] = raw_low_flags.any(axis=1)
    for column in flags:
        flags[column] = flags[column].fillna(False).astype(bool)

    output = frame[["date"]].copy()
    for symbol_column in ("symbol", "ts_code"):
        if symbol_column in frame:
            output["symbol"] = frame[symbol_column].astype(str)
            break
    if "symbol" not in output:
        output["symbol"] = ""
    return pd.concat([output[["symbol", "date"]], flags], axis=1)


def validate_left_side_factor_contract(columns: Iterable[object]) -> None:
    available = {str(column) for column in columns}
    assert_no_forbidden_factor_names(
        available,
        context="left-side unified factor contract",
    )
    missing = {
        signal: sorted(set(required) - available)
        for signal, required in LEFT_SIDE_SIGNAL_FEATURE_REQUIREMENTS.items()
        if set(required) - available
    }
    if missing:
        raise ValueError(f"left-side signal factor contract is incomplete: {missing}")
    stable_canonical_feature_union(
        LEFT_SIDE_PROJECT_FACTOR_REQUIREMENTS,
        LEFT_SIDE_RULE_FEATURE_COLUMNS,
    )


if len(LEFT_SIDE_RULE_FEATURE_COLUMNS) != 27:
    raise RuntimeError(
        "left-side rule schema version/count drifted: "
        f"{len(LEFT_SIDE_RULE_FEATURE_COLUMNS)}"
    )


__all__ = [
    "LEFT_SIDE_PROJECT_FACTOR_REQUIREMENTS",
    "LEFT_SIDE_RAW_LOW_PULLBACK_SIGNALS",
    "LEFT_SIDE_RULE_FEATURE_COLUMNS",
    "LEFT_SIDE_RULE_FEATURE_COLUMNS_SHA256",
    "LEFT_SIDE_RULE_FEATURE_SCHEMA_VERSION",
    "LEFT_SIDE_SIGNAL_FEATURE_REQUIREMENTS",
    "LEFT_SIDE_SHARED_RULE_REQUIREMENTS",
    "LEFT_SIDE_SIGNAL_SCHEMA_VERSION",
    "LEFT_SIDE_SIGNALS",
    "compute_left_side_rule_features",
    "compute_left_side_signal_flags",
    "left_side_rule_columns_sha256",
    "validate_left_side_factor_contract",
]
