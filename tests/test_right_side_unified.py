from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.research.right_side_unified import (
    DEFAULT_YEAR_FOLDS,
    RIGHT_SIDE_SIGNALS,
    aggregate_independent_predictions,
    balanced_sample_weights,
    binary_metrics,
    daily_top_k_trading_metrics,
    load_signal_universe,
    split_by_year_fold,
)
from quant.research.right_side_unified_features import (
    RIGHT_SIDE_SIGNALS as FEATURE_SIGNALS,
    RIGHT_SIDE_PROJECT_FACTOR_REQUIREMENTS,
    RULE_FEATURE_COLUMNS,
    SIGNAL_FEATURE_REQUIREMENTS,
    compute_right_side_rule_features,
    validate_signal_factor_contract,
)
from quant.research.right_side_unified_signals import (
    CANONICAL_RIGHT_SIDE_SIGNALS,
    CANONICAL_Z_SIGNALS,
    SIGNAL_CONTRACT_NOTES,
    compute_canonical_z_signal_flags,
    merge_canonical_signal_flags,
)
from quant.strategies.custom.z_skill_patterns import (
    _detect_breathing,
    _detect_changan,
    _detect_double_gun,
    _detect_golden_bowl,
    _detect_kengqi,
    _detect_key_k,
    _detect_pinghang,
    _detect_violence_k,
    _detect_yueyue,
    _detect_zaihou,
)
from quant.research.right_side_unified_labels import (
    build_right_side_unified_labels,
    detect_next_locked_limit_up,
    map_signals_to_next_market_date,
)


def _signal_frame(rows: int) -> pd.DataFrame:
    frame = pd.DataFrame(False, index=range(rows), columns=list(RIGHT_SIDE_SIGNALS))
    frame.loc[:, "B2"] = True
    return frame


def test_right_side_contract_contains_all_current_members() -> None:
    assert RIGHT_SIDE_SIGNALS == (
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


def test_balanced_weights_do_not_duplicate_multi_hit_rows() -> None:
    frame = _signal_frame(4)
    frame["date"] = pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"])
    frame.loc[0, "B3"] = True
    weights = balanced_sample_weights(frame)

    assert len(weights) == len(frame)
    assert weights.index.equals(frame.index)
    assert weights.mean() == pytest.approx(1.0)
    assert np.isfinite(weights).all()


def test_year_split_purges_train_and_validation_label_overlap() -> None:
    frame = _signal_frame(6)
    frame["date"] = pd.to_datetime(
        ["2022-12-20", "2022-12-29", "2023-06-01", "2023-12-29", "2024-02-01", "2025-02-01"]
    )
    frame["label_end_date"] = pd.to_datetime(
        ["2022-12-27", "2023-01-05", "2023-06-08", "2024-01-08", "2024-02-08", "2025-02-08"]
    )

    split = split_by_year_fold(frame, DEFAULT_YEAR_FOLDS[0])

    assert split.train["date"].tolist() == [pd.Timestamp("2022-12-20")]
    assert split.validation["date"].tolist() == [pd.Timestamp("2023-06-01")]
    assert split.test["date"].tolist() == [pd.Timestamp("2024-02-01")]


def test_independent_prediction_aggregation_uses_only_active_models() -> None:
    frame = _signal_frame(3)
    frame.loc[1:, "B2"] = False
    frame.loc[1:, "B3"] = True
    frame.loc[2, "KEY_K"] = True
    predictions = {
        "B2": np.array([0.3, 0.99, 0.99]),
        "B3": np.array([0.99, 0.4, 0.5]),
        "KEY_K": np.array([0.99, 0.99, 0.8]),
    }

    combined = aggregate_independent_predictions(frame, predictions)

    assert combined.tolist() == pytest.approx([0.3, 0.4, 0.8])


def test_binary_metrics_reports_top_lift() -> None:
    metrics = binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.9, 0.8], top_fraction=0.5)

    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["top_precision"] == pytest.approx(1.0)
    assert metrics["top_lift"] == pytest.approx(2.0)


def test_signal_cache_merge_ors_overlapping_members(tmp_path) -> None:
    z = pd.DataFrame(
        {
            "symbol": ["A"],
            "date": ["2024-01-02"],
            **{
                signal: [signal == "KEY_K"]
                for signal in RIGHT_SIDE_SIGNALS
                if signal not in {"B2", "B3", "VEGAS", "TRIPLE_VOLUME_BREAKOUT"}
            },
        }
    )
    family = pd.DataFrame(
        {
            "symbol": ["A"],
            "date": ["2024-01-02"],
            "b2_any_pchg4_vol15": [True],
            "b2_oversold_pchg3_vol12": [False],
            "b2_bbi_reclaim_vol12": [False],
            "b2_pchg4_vol15": [False],
            "b3_broad_small_pos": [False],
            "b3_broad_calm_pullback": [False],
            "b3_small_pos_amp7": [False],
            "signal_vegas_tunnel": [False],
            "signal_tvb_merged": [False],
        }
    )
    z_path = tmp_path / "z.parquet"
    family_path = tmp_path / "family.parquet"
    z.to_parquet(z_path, index=False)
    family.to_parquet(family_path, index=False)

    merged = load_signal_universe(z_path, family_path)

    assert len(merged) == 1
    assert merged.loc[0, "B2"]
    assert merged.loc[0, "KEY_K"]
    assert merged.loc[0, "signal_count"] == 2


def test_daily_top_k_trading_metrics_use_fixed_capacity_and_cost() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"] * 3 + ["2024-01-03"] * 3),
            "terminal_return": [0.03, 0.02, -0.10, 0.04, -0.02, 0.01],
        }
    )
    metrics = daily_top_k_trading_metrics(
        frame,
        [0.9, 0.8, 0.1, 0.9, 0.8, 0.1],
        top_k=2,
        round_trip_cost_bps=10,
    )

    assert metrics["trades"] == 4
    assert metrics["trading_days"] == 2
    assert metrics["average_net_return"] == pytest.approx((0.03 + 0.02 + 0.04 - 0.02) / 4 - 0.001)


def test_exact_and_proxy_locked_limit_detection() -> None:
    next_day = pd.DataFrame(
        {
            "open": [11.0, 5.25, 11.0, 10.5],
            "high": [11.0, 5.25, 11.1, 10.5],
            "low": [11.0, 5.25, 11.0, 10.5],
            "close": [11.0, 5.25, 11.05, 10.5],
            "pre_close": [10.0, 5.0, 10.0, 10.0],
            "up_limit": [11.0, np.nan, 11.0, 11.0],
        }
    )

    result = detect_next_locked_limit_up(next_day)

    assert result["locked_limit_up"].tolist() == [True, True, False, False]
    assert result["locked_limit_source"].tolist() == [
        "exact_up_limit",
        "ohlc_4p8_proxy",
        "exact_up_limit",
        "exact_up_limit",
    ]


def test_market_calendar_mapping_does_not_jump_by_stock_rows() -> None:
    signals = pd.DataFrame({"symbol": ["A"], "date": ["2024-01-02"]})
    mapped = map_signals_to_next_market_date(
        signals,
        pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )

    assert mapped.loc[0, "entry_date"] == pd.Timestamp("2024-01-03")


def _daily_path(rows: int = 13) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=rows)
    close = 10.0 + np.arange(rows) * 0.1
    return pd.DataFrame(
        {
            "ts_code": "A",
            "trade_date": dates.strftime("%Y%m%d"),
            "open": close - 0.02,
            "high": close + 0.20,
            "low": close - 0.10,
            "close": close,
            "pre_close": np.r_[10.0, close[:-1]],
            "volume": 1000.0 + np.arange(rows),
        }
    )


def test_open_and_close_labels_start_outcomes_after_entry_and_tail_is_na() -> None:
    daily = _daily_path()
    calendar = pd.to_datetime(daily["trade_date"], format="%Y%m%d")
    signals = pd.DataFrame(
        {
            "symbol": ["A", "A"],
            "date": [calendar.iloc[0], calendar.iloc[-1]],
            "B2": [True, True],
        }
    )

    labels = build_right_side_unified_labels(
        signals,
        daily,
        calendar,
        horizons=(3,),
        entry_modes=("next_open", "next_close"),
    )

    mature = labels[labels["date"].eq(calendar.iloc[0])]
    tail = labels[labels["date"].eq(calendar.iloc[-1])]
    assert mature["mature"].all()
    assert mature["entry_date"].eq(calendar.iloc[1]).all()
    assert mature["label_end_date"].eq(calendar.iloc[4]).all()
    assert mature.set_index("entry_mode").loc["next_open", "entry_price"] != mature.set_index(
        "entry_mode"
    ).loc["next_close", "entry_price"]
    assert not tail["mature"].any()
    assert tail[["mfe", "mae", "terminal_return"]].isna().all().all()
    assert tail[["hit_up3", "hit_up5", "hit_up8", "hit_down3", "good_path5"]].isna().all().all()


def test_locked_limit_entry_is_excluded_from_labels() -> None:
    daily = _daily_path(8)
    daily.loc[1, ["open", "high", "low", "close"]] = 11.0
    daily.loc[1, "pre_close"] = 10.0
    calendar = pd.to_datetime(daily["trade_date"], format="%Y%m%d")
    signals = pd.DataFrame({"symbol": ["A"], "date": [calendar[0]], "B2": [True]})
    tradability = pd.DataFrame(
        {
            "ts_code": ["A"],
            "trade_date": [daily.loc[1, "trade_date"]],
            "up_limit": [11.0],
            "is_suspended": [False],
        }
    )

    labels = build_right_side_unified_labels(
        signals,
        daily,
        calendar,
        tradability,
        horizons=(3,),
        entry_modes=("next_open",),
    )

    assert labels.loc[0, "locked_limit_up"]
    assert not labels.loc[0, "entry_executable"]
    assert labels.loc[0, "maturity_reason"] == "locked_limit_up"
    assert pd.isna(labels.loc[0, "good_path5"])


def test_next_open_mae_includes_unavoidable_entry_day_low() -> None:
    daily = _daily_path(8)
    daily.loc[1, ["open", "high", "low", "close"]] = [10.0, 10.2, 9.0, 9.8]
    daily.loc[1, "pre_close"] = 10.0
    calendar = pd.to_datetime(daily["trade_date"], format="%Y%m%d")
    signals = pd.DataFrame({"symbol": ["A"], "date": [calendar.iloc[0]], "B2": [True]})

    labels = build_right_side_unified_labels(
        signals,
        daily,
        calendar,
        horizons=(3,),
        entry_modes=("next_open", "next_close"),
    ).set_index("entry_mode")

    assert labels.loc["next_open", "mae"] == pytest.approx(-0.10)
    assert labels.loc["next_open", "hit_down3"]
    assert labels.loc["next_close", "mae"] > labels.loc["next_open", "mae"]


def test_every_signal_has_materialized_rule_features() -> None:
    assert set(FEATURE_SIGNALS) == set(RIGHT_SIDE_SIGNALS)
    assert set(SIGNAL_FEATURE_REQUIREMENTS) == set(RIGHT_SIDE_SIGNALS)
    assert len(RULE_FEATURE_COLUMNS) >= 90
    validate_signal_factor_contract(
        (*RULE_FEATURE_COLUMNS, *RIGHT_SIDE_PROJECT_FACTOR_REQUIREMENTS)
    )


def test_rule_features_are_prefix_causal() -> None:
    daily = _daily_path(220)
    full = compute_right_side_rule_features(daily)
    prefix = compute_right_side_rule_features(daily.iloc[:180])

    pd.testing.assert_frame_equal(
        full.loc[prefix.index, list(RULE_FEATURE_COLUMNS)],
        prefix,
        check_dtype=False,
    )


def test_kengqi_event_features_match_live_detector() -> None:
    daily = _daily_path(30)
    daily["open"] = daily["close"] - 0.02
    daily["high"] = daily["close"] + 0.10
    daily["low"] = daily["close"] - 0.10
    daily["volume"] = 1000.0
    pit = 21
    daily.loc[pit - 5 : pit - 1, "high"] = 12.0
    daily.loc[pit - 5 : pit - 1, "volume"] = 2000.0
    daily.loc[pit, ["open", "close", "low", "volume"]] = [10.0, 9.7, 9.5, 2600.0]
    daily.loc[pit + 1 : pit + 5, "volume"] = 800.0
    daily.loc[29, "close"] = 11.6
    live = daily.copy()
    live["date"] = pd.to_datetime(live["trade_date"], format="%Y%m%d")
    live["pct_chg"] = live["close"].pct_change() * 100
    live["vol_ratio_prev"] = live["volume"] / live["volume"].shift(1)
    live["bbi"] = live["close"].rolling(5, min_periods=1).mean()

    features = compute_right_side_rule_features(daily)

    assert _detect_kengqi(live) is not None
    assert features.iloc[-1]["rs_pit_recent_3_14d"]
    assert features.iloc[-1]["rs_days_since_pit"] == 8
    assert features.iloc[-1]["rs_pit_depth_18d"] == pytest.approx((12.0 - 9.5) / 12.0)


def test_zaihou_anchor_features_use_first_live_window_event() -> None:
    daily = _daily_path(65)
    daily["volume"] = 1000.0
    daily.loc[55, ["open", "close", "high", "low", "volume"]] = [15.0, 17.0, 17.1, 14.9, 2000.0]
    daily.loc[59, ["open", "close", "high", "low", "volume"]] = [15.5, 17.5, 17.6, 15.4, 3000.0]
    daily.loc[64, "volume"] = 500.0
    live = daily.copy()
    live["date"] = pd.to_datetime(live["trade_date"], format="%Y%m%d")
    live["pct_chg"] = live["close"].pct_change() * 100
    live["vol_ratio_prev"] = live["volume"] / live["volume"].shift(1)
    live["bbi"] = live["close"].rolling(5, min_periods=1).mean()
    live.loc[59:, "bbi"] = np.linspace(16.0, 16.4, 6)

    features = compute_right_side_rule_features(daily)

    assert features.iloc[-1]["rs_fangliang_recent_3_12d"]
    assert features.iloc[-1]["rs_days_since_fangliang"] == 9
    assert features.iloc[-1]["rs_fangliang_ref_volume_15d"] == pytest.approx(2000.0)
    assert _detect_zaihou(live) is not None


LIVE_Z_DETECTORS = {
    "CHANGAN": _detect_changan,
    "PINGHANG": _detect_pinghang,
    "DOUBLE_GUN": _detect_double_gun,
    "GOLDEN_BOWL": _detect_golden_bowl,
    "BREATHING": _detect_breathing,
    "KENGQI": _detect_kengqi,
    "ZAIHOU": _detect_zaihou,
    "YUEYUE": _detect_yueyue,
    "KEY_K": _detect_key_k,
    "VIOLENCE_K": _detect_violence_k,
}


def _live_z_frame(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame["trade_date"], format="%Y%m%d")
    frame["pre_close"] = frame["close"].shift(1)
    frame["pct_chg"] = frame["close"].pct_change().mul(100).fillna(0)
    frame["amplitude"] = (
        (frame["high"] - frame["low"]) / frame["pre_close"].replace(0, np.nan) * 100
    )
    frame["close_pos"] = (frame["close"] - frame["low"]) / (
        frame["high"] - frame["low"]
    ).replace(0, np.nan)
    frame["vol_ratio_prev"] = frame["volume"] / frame["volume"].shift(1)
    frame["vol_ratio_5"] = frame["volume"] / frame["volume"].shift(1).rolling(
        5,
        min_periods=1,
    ).mean()
    frame["is_rise"] = frame["close"] > frame["open"]
    frame["is_beidou"] = (frame["pct_chg"] >= 3) & (frame["vol_ratio_5"] >= 1.5)
    for window, min_periods in ((3, 1), (6, 2), (12, 4), (24, 8)):
        frame[f"ma{window}"] = frame["close"].rolling(window, min_periods=min_periods).mean()
    frame["bbi"] = (frame["ma3"] + frame["ma6"] + frame["ma12"] + frame["ma24"]) / 4
    frame["zg_white"] = frame["close"].ewm(span=10, adjust=False).mean().ewm(
        span=10,
        adjust=False,
    ).mean()
    frame["dg_yellow"] = (
        frame["close"].rolling(14, min_periods=8).mean()
        + frame["close"].rolling(28, min_periods=14).mean()
        + frame["close"].rolling(57, min_periods=28).mean()
        + frame["close"].rolling(114, min_periods=60).mean()
    ) / 4
    low9 = frame["low"].rolling(9, min_periods=3).min()
    high9 = frame["high"].rolling(9, min_periods=3).max()
    rsv = ((frame["close"] - low9) / (high9 - low9).replace(0, np.nan) * 100).fillna(50)
    k = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    d = k.ewm(alpha=1 / 3, adjust=False).mean()
    frame["kdj_j"] = 3 * k - 2 * d
    return frame


def test_canonical_signal_contract_is_complete() -> None:
    assert set(LIVE_Z_DETECTORS) == set(CANONICAL_Z_SIGNALS)
    assert set(SIGNAL_CONTRACT_NOTES) == set(CANONICAL_RIGHT_SIDE_SIGNALS)


def test_canonical_z_flags_match_live_detectors_for_every_prefix_row() -> None:
    daily = _daily_path(145)
    rng = np.random.default_rng(20260813)
    returns = rng.normal(0.001, 0.035, len(daily))
    close = 10.0 * np.cumprod(1.0 + returns)
    open_ = close * (1.0 + rng.normal(0.0, 0.012, len(daily)))
    daily["open"] = open_
    daily["close"] = close
    daily["high"] = np.maximum(open_, close) * (1.0 + rng.uniform(0.001, 0.03, len(daily)))
    daily["low"] = np.minimum(open_, close) * (1.0 - rng.uniform(0.001, 0.03, len(daily)))
    daily["pre_close"] = np.r_[close[0], close[:-1]]
    daily["volume"] = rng.lognormal(np.log(1000), 0.7, len(daily))
    live = _live_z_frame(daily)
    canonical = compute_canonical_z_signal_flags(daily)

    for position in range(len(daily)):
        prefix = live.iloc[: position + 1]
        for signal, detector in LIVE_Z_DETECTORS.items():
            expected = detector(prefix) is not None
            assert bool(canonical.loc[position, signal]) is expected, (
                signal,
                position,
                canonical.loc[position, "date"],
            )


def test_merge_canonical_signal_flags_ors_web_family_columns() -> None:
    z = pd.DataFrame(
        {
            "symbol": ["A", "A"],
            "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            **{
                signal: [signal == "KEY_K", signal == "YUEYUE"]
                for signal in CANONICAL_Z_SIGNALS
            },
        }
    )
    family = pd.DataFrame(
        {
            "symbol": ["A", "A"],
            "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "b2_any_pchg4_vol15": [True, False],
            "b3_broad_calm_pullback": [False, True],
            "signal_vegas_tunnel": [True, False],
            "signal_tvb_merged": [False, True],
        }
    )

    merged = merge_canonical_signal_flags(z, family)

    assert len(merged) == 1
    expected_hits = {
        "B2",
        "B3",
        "VEGAS",
        "TRIPLE_VOLUME_BREAKOUT",
        "KEY_K",
        "YUEYUE",
    }
    actual_hits = {
        signal
        for signal in CANONICAL_RIGHT_SIDE_SIGNALS
        if bool(merged.loc[0, signal])
    }
    assert actual_hits == expected_hits
