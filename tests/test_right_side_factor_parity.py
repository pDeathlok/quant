from __future__ import annotations

import ast
import inspect
from pathlib import Path
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_SCRIPTS = PROJECT_ROOT / "scripts/research"
if str(RESEARCH_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(RESEARCH_SCRIPTS))

from analyze_b1_family_rule_backtest import compute_signal_flags
from quant.research.right_side_factor_parity import (
    CONTRACT_BY_SIGNAL,
    FAMILY_CACHE_TO_RULE_FACTOR,
    PREDICATE_FACTOR_CONTRACTS,
    TRIPLE_VOLUME_CONFIG_SHA256,
    VEGAS_OPTIMIZED_PARAMS_SHA256,
    audit_signal_factor_slice,
    contract_factor_audit,
    predicate_contract_summary,
    reconstruct_web_family_flags,
    triple_volume_config_fingerprint,
    validate_generator_fingerprints,
    validate_predicate_factor_contracts,
    validate_signal_factor_slice,
    vegas_optimized_params_fingerprint,
)
from quant.research.right_side_unified_features import (
    ADDED_RULE_FEATURE_COLUMNS_V2,
    LEGACY_RULE_FEATURE_COLUMNS_SHA256_V1,
    LEGACY_RULE_FEATURE_COLUMNS_V1,
    RIGHT_SIDE_SIGNALS,
    RULE_FEATURE_COLUMNS_SHA256,
    RULE_FEATURE_SCHEMA_VERSION,
    RULE_FEATURE_COLUMNS,
    SIGNAL_FEATURE_REQUIREMENTS,
    compute_right_side_rule_features,
)
from quant.research.right_side_unified_signals import (
    B2_FAMILY_SOURCE_COLUMNS,
    B3_FAMILY_SOURCE_COLUMNS,
    CANONICAL_Z_SIGNALS,
    FAMILY_DIRECT_SOURCE_COLUMNS,
)
from quant.strategies.custom.triple_volume_breakout import (
    add_triple_volume_strategy_pool_signals,
)
from quant.strategies.custom.vegas_tunnel import (
    OPTIMIZED_VEGAS_TUNNEL_PARAMS,
    add_vegas_tunnel_signals,
)
from scripts.research.validate_unified_right_side_models import train_models


def _random_daily(seed: int = 42, rows: int = 360) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 10.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.025, rows))
    open_ = close * (1.0 + rng.normal(0.0, 0.01, rows))
    high = np.maximum(close, open_) * (1.0 + rng.uniform(0.001, 0.03, rows))
    low = np.minimum(close, open_) * (1.0 - rng.uniform(0.001, 0.03, rows))
    volume = rng.lognormal(np.log(1_000_000), 0.6, rows)
    dates = pd.bdate_range("2023-01-02", periods=rows)
    frame = pd.DataFrame(
        {
            "ts_code": "TEST.SZ",
            "symbol": "TEST.SZ",
            "trade_date": dates.strftime("%Y%m%d"),
            "date": dates,
            "name": "测试股份",
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
    frame["pre_close"] = frame["close"].shift(1)
    frame["pct_chg"] = frame["close"].pct_change() * 100.0
    return frame


def _vegas_daily(name: str = "测试股份") -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=260, freq="D")
    close = 10.0 + np.arange(len(dates)) * 0.035
    open_ = close - 0.04
    high = close + 0.10
    low = close - 0.10
    volume = np.full(len(dates), 1_000.0)

    probe = pd.Series(close)
    ema144 = probe.ewm(span=144, adjust=False, min_periods=144).mean()
    ema169 = probe.ewm(span=169, adjust=False, min_periods=169).mean()
    tunnel_upper = pd.concat([ema144, ema169], axis=1).max(axis=1)
    pullback = 246
    signal = 250
    low[pullback] = tunnel_upper.iloc[pullback] * 1.01
    close[pullback] = max(
        tunnel_upper.iloc[pullback] * 1.035,
        close[pullback] * 0.98,
    )
    open_[pullback] = close[pullback] + 0.03
    high[pullback] = close[pullback] + 0.08
    volume[pullback] = 950.0
    close[signal] = close[signal - 1] * 1.012
    open_[signal] = close[signal] * 0.985
    high[signal] = close[signal] * 1.01
    low[signal] = open_[signal] * 0.995
    volume[signal] = 1_250.0
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": "TEST.SZ",
            "name": name,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "pre_close": pd.Series(close).shift(1).fillna(close[0]),
            "volume": volume,
        }
    )


def _tvb_daily(volume_multiple: float) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=90, freq="D")
    close = np.linspace(10.0, 14.0, len(dates))
    open_ = close - 0.03
    high = close + 0.08
    low = close - 0.08
    volume = np.full(len(dates), 1_000.0)
    anchor = 78
    volume[anchor - 1] = 1_000.0
    volume[anchor] = volume_multiple * 1_000.0 + 100.0
    close[anchor] = 13.20
    open_[anchor] = 13.12
    high[anchor] = 13.28
    low[anchor] = 13.08
    for idx, price in zip(
        range(anchor + 1, anchor + 4),
        [13.24, 13.26, 13.25],
        strict=True,
    ):
        close[idx] = price
        open_[idx] = price - 0.02
        high[idx] = price + 0.04
        low[idx] = price - 0.04
        volume[idx] = 850.0
    signal = anchor + 4
    close[signal] = 13.55
    open_[signal] = 13.32
    high[signal] = 13.62
    low[signal] = 13.28
    volume[signal] = 830.0
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": "TEST.SZ",
            "name": "测试股份",
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "pre_close": pd.Series(close).shift(1).fillna(close[0]),
            "volume": volume,
        }
    )


def test_versioned_contract_covers_every_signal_and_every_rule_factor() -> None:
    validate_predicate_factor_contracts()
    assert tuple(contract.signal for contract in PREDICATE_FACTOR_CONTRACTS) == RIGHT_SIDE_SIGNALS
    assert set(CONTRACT_BY_SIGNAL) == set(RIGHT_SIDE_SIGNALS)
    assert len(RULE_FEATURE_COLUMNS) == 118
    assert set(RULE_FEATURE_COLUMNS) == {
        factor
        for contract in PREDICATE_FACTOR_CONTRACTS
        for predicate in contract.predicates
        for factor in predicate.factors
    }
    audit = contract_factor_audit()
    assert audit["status"].eq("ok").all()
    assert not audit["predicate_id"].duplicated().any()


def test_rule_feature_version_contract_freezes_105_plus_13_equals_118() -> None:
    assert RULE_FEATURE_SCHEMA_VERSION == "right_side_rule_features_v2_118_20260813"
    assert len(LEGACY_RULE_FEATURE_COLUMNS_V1) == 105
    assert len(ADDED_RULE_FEATURE_COLUMNS_V2) == 13
    assert not set(LEGACY_RULE_FEATURE_COLUMNS_V1).intersection(
        ADDED_RULE_FEATURE_COLUMNS_V2
    )
    assert tuple(
        column
        for column in RULE_FEATURE_COLUMNS
        if column not in set(ADDED_RULE_FEATURE_COLUMNS_V2)
    ) == LEGACY_RULE_FEATURE_COLUMNS_V1
    assert len(RULE_FEATURE_COLUMNS_SHA256) == 64
    assert len(LEGACY_RULE_FEATURE_COLUMNS_SHA256_V1) == 64
    assert RULE_FEATURE_COLUMNS_SHA256 != LEGACY_RULE_FEATURE_COLUMNS_SHA256_V1


def test_contract_distinguishes_live_exact_from_cache_reconstruction() -> None:
    summary = predicate_contract_summary().set_index("signal")
    assert set(summary[summary["authority"].eq("live_detector")].index) == set(
        CANONICAL_Z_SIGNALS
    )
    assert set(summary[summary["authority"].eq("web_family_cache")].index) == {
        "B2",
        "B3",
        "VEGAS",
        "TRIPLE_VOLUME_BREAKOUT",
    }
    assert summary["proxy_predicates"].sum() == 0
    assert summary.loc["B2", "cache_reconstruction_predicates"] >= 1
    assert summary.loc["VEGAS", "cache_reconstruction_predicates"] >= 1


def test_family_cache_columns_have_explicit_model_factor_reconstruction() -> None:
    expected_sources = {
        *B2_FAMILY_SOURCE_COLUMNS,
        *B3_FAMILY_SOURCE_COLUMNS,
        *FAMILY_DIRECT_SOURCE_COLUMNS.values(),
    }
    assert set(FAMILY_CACHE_TO_RULE_FACTOR) == expected_sources
    assert set(FAMILY_CACHE_TO_RULE_FACTOR.values()) <= set(RULE_FEATURE_COLUMNS)

    rule = pd.DataFrame(
        False,
        index=range(len(expected_sources)),
        columns=list(RULE_FEATURE_COLUMNS),
    )
    for row, factor in enumerate(FAMILY_CACHE_TO_RULE_FACTOR.values()):
        rule.loc[row, factor] = True
    reconstructed = reconstruct_web_family_flags(rule)
    for row, source in enumerate(FAMILY_CACHE_TO_RULE_FACTOR):
        assert bool(reconstructed.loc[row, source])
    assert reconstructed["B2"].sum() == len(B2_FAMILY_SOURCE_COLUMNS)
    assert reconstructed["B3"].sum() == len(B3_FAMILY_SOURCE_COLUMNS)


def test_b2_b3_rule_flags_match_current_production_generator_for_every_row() -> None:
    daily = _random_daily()
    production = compute_signal_flags(daily)
    rule = compute_right_side_rule_features(daily)
    reconstructed = reconstruct_web_family_flags(rule)
    np.testing.assert_allclose(
        rule["rs_family_kdj_j"],
        production["kdj_d_j"],
        equal_nan=True,
    )
    sources = (*B2_FAMILY_SOURCE_COLUMNS, *B3_FAMILY_SOURCE_COLUMNS)
    assert production[list(sources)].any(axis=None)
    for source in sources:
        pd.testing.assert_series_equal(
            reconstructed[source].reset_index(drop=True),
            production[source].fillna(False).astype(bool).reset_index(drop=True),
            check_names=False,
        )


@pytest.mark.parametrize("name", ["测试股份", "*ST测试"])
def test_vegas_rule_flag_matches_optimized_production_generator(name: str) -> None:
    daily = _vegas_daily(name)
    production = add_vegas_tunnel_signals(
        daily,
        **OPTIMIZED_VEGAS_TUNNEL_PARAMS,
    )
    rule = compute_right_side_rule_features(daily).reset_index(drop=True)
    pd.testing.assert_series_equal(
        rule["rs_vegas_signal"].astype(bool),
        production["signal_vegas_tunnel"].astype(bool),
        check_names=False,
    )
    assert rule["rs_vegas_history_ok"].iloc[179] == np.bool_(False)
    assert rule["rs_vegas_history_ok"].iloc[180] == np.bool_(True)
    assert bool(rule["rs_vegas_tradable"].all()) is (name == "测试股份")
    if name == "测试股份":
        assert production["signal_vegas_tunnel"].any()


@pytest.mark.parametrize("volume_multiple", [2.5, 3.0])
def test_tvb_rule_variants_and_merge_match_yaml_production_generator(
    volume_multiple: float,
) -> None:
    daily = _tvb_daily(volume_multiple)
    production = add_triple_volume_strategy_pool_signals(daily)
    rule = compute_right_side_rule_features(daily).reset_index(drop=True)
    expected = {
        "rs_tvb_candidate_25": "signal_tvb_expanded",
        "rs_tvb_candidate_30": "signal_tvb_conservative",
        "rs_tvb_merged": "signal_tvb_merged",
    }
    for factor, source in expected.items():
        pd.testing.assert_series_equal(
            rule[factor].astype(bool),
            production[source].astype(bool),
            check_names=False,
        )
    assert production["signal_tvb_merged"].any()


def test_production_generator_fingerprints_are_frozen() -> None:
    assert vegas_optimized_params_fingerprint() == VEGAS_OPTIMIZED_PARAMS_SHA256
    assert triple_volume_config_fingerprint() == TRIPLE_VOLUME_CONFIG_SHA256
    validate_generator_fingerprints()


def test_yueyue_platform_margin_uses_exact_live_sub_one_denominator() -> None:
    rows = 40
    dates = pd.bdate_range("2024-01-02", periods=rows)
    daily = pd.DataFrame(
        {
            "date": dates,
            "open": np.full(rows, 0.54),
            "high": np.full(rows, 0.60),
            "low": np.full(rows, 0.50),
            "close": np.full(rows, 0.55),
            "volume": np.linspace(1_000, 1_100, rows),
        }
    )
    rule = compute_right_side_rule_features(daily)
    # Live detector divides by max(platform_low, 1), not platform_low.
    assert rule["rs_platform_range_20d"].iloc[-1] == pytest.approx(0.10)


def test_new_rule_columns_remain_prefix_causal() -> None:
    daily = _random_daily(rows=260)
    full = compute_right_side_rule_features(daily)
    prefix = compute_right_side_rule_features(daily.iloc[:220])
    pd.testing.assert_frame_equal(
        full.loc[prefix.index, list(RULE_FEATURE_COLUMNS)],
        prefix,
        check_dtype=False,
    )


def test_active_signal_slice_audit_rejects_empty_missing_and_infinite() -> None:
    frame = pd.DataFrame(1.0, index=range(len(RIGHT_SIDE_SIGNALS)), columns=RULE_FEATURE_COLUMNS)
    for signal in RIGHT_SIDE_SIGNALS:
        frame[signal] = False
    for row, signal in enumerate(RIGHT_SIDE_SIGNALS):
        frame.loc[row, signal] = True
        validate_signal_factor_slice(frame, signal)

    broken = frame.copy()
    broken.loc[0, SIGNAL_FEATURE_REQUIREMENTS["B2"][0]] = np.inf
    audit = audit_signal_factor_slice(broken, "B2")
    assert audit.loc[
        audit["factor"].eq(SIGNAL_FEATURE_REQUIREMENTS["B2"][0]),
        "status",
    ].item() == "non_finite"
    with pytest.raises(ValueError, match="non_finite"):
        validate_signal_factor_slice(broken, "B2")

    empty = frame.copy()
    empty["B2"] = False
    with pytest.raises(ValueError, match="empty_signal"):
        validate_signal_factor_slice(empty, "B2")


def test_actual_training_code_keeps_every_rule_factor_in_both_unified_arms() -> None:
    tree = ast.parse(textwrap.dedent(inspect.getsource(train_models)))
    assignments: dict[str, str] = {}
    experiment_tuple = ""
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                assignments[target.id] = ast.unparse(node.value)
        if isinstance(node, ast.For) and isinstance(node.target, ast.Tuple):
            names = [item.id for item in node.target.elts if isinstance(item, ast.Name)]
            if names == ["experiment", "features", "balanced"]:
                experiment_tuple = ast.unparse(node.iter)

    assert "RULE_FEATURE_COLUMNS" in assignments["fixed_features"]
    assert "RULE_FEATURE_COLUMNS" in assignments["common_features"]
    assert assignments["conditioned_features"].startswith("[*common_features")
    assert "('unified_without_signal_id', common_features, False)" in experiment_tuple
    assert "('unified_with_signal_id', conditioned_features, False)" in experiment_tuple

    common = ["project_factor", *RULE_FEATURE_COLUMNS]
    conditioned = [
        *common,
        *RIGHT_SIDE_SIGNALS,
        "signal_count",
        "has_right_signal",
        "has_mixed_signal",
    ]
    assert set(RULE_FEATURE_COLUMNS) <= set(common) <= set(conditioned)
    assert not set(RIGHT_SIDE_SIGNALS) & set(common)
