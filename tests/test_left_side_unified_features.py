from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quant.features.canonical_factor_names import FORBIDDEN_COMPATIBILITY_ALIASES
from quant.features.left_side_factor_contract import (
    LEFT_SIDE_FACTOR_COLUMNS,
    LEFT_SIDE_MODEL_INPUT_COLUMNS,
    LEFT_SIDE_SCORING_INPUT_COLUMNS,
    LEFT_SIDE_TASK_FEATURE_COLUMNS,
    left_side_contract_payload,
)
from quant.features.variable_library import PROJECT_FACTOR_COLUMNS
from quant.research.left_side_unified_features import (
    LEFT_SIDE_PROJECT_FACTOR_REQUIREMENTS,
    LEFT_SIDE_RULE_FEATURE_COLUMNS,
    LEFT_SIDE_RULE_FEATURE_SCHEMA_VERSION,
    LEFT_SIDE_SIGNAL_FEATURE_REQUIREMENTS,
    LEFT_SIDE_SIGNALS,
    LEFT_SIDE_SHARED_RULE_REQUIREMENTS,
    compute_left_side_rule_features,
    compute_left_side_signal_flags,
    validate_left_side_factor_contract,
)
from quant.strategies.custom.z_skill_patterns import (
    _detect_duichen_va,
    _detect_nana,
    _detect_yidong_dilian,
    _normalize_daily,
)
from scripts.research.validate_unified_left_side_models import _process_symbol


def _daily(rows: int = 220) -> pd.DataFrame:
    position = np.arange(rows)
    close = 10.0 + np.sin(position / 8.0) + position * 0.01
    return pd.DataFrame(
        {
            "ts_code": "000001.SZ",
            "symbol": "000001.SZ",
            "name": "",
            "date": pd.bdate_range("2025-01-02", periods=rows),
            "open": close * 0.995,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "pre_close": pd.Series(close).shift(1),
            "pct_chg": pd.Series(close).pct_change() * 100.0,
            "volume": 1_000_000 * (1.0 + 0.2 * np.sin(position / 5.0)),
        }
    )


def test_left_side_contract_contains_all_predicate_state_without_aliases() -> None:
    assert LEFT_SIDE_SIGNALS == ("B1", "SB1", "SUPER_B1", "LOW_PULLBACK")
    assert LEFT_SIDE_RULE_FEATURE_SCHEMA_VERSION == (
        "left_side_rule_features_v2_27_20260824"
    )
    assert len(LEFT_SIDE_RULE_FEATURE_COLUMNS) == 27
    assert len(set(LEFT_SIDE_RULE_FEATURE_COLUMNS)) == 27
    assert FORBIDDEN_COMPATIBILITY_ALIASES.isdisjoint(
        LEFT_SIDE_RULE_FEATURE_COLUMNS
    )
    assert not set(LEFT_SIDE_PROJECT_FACTOR_REQUIREMENTS).intersection(
        LEFT_SIDE_RULE_FEATURE_COLUMNS
    )
    assert set(LEFT_SIDE_SIGNAL_FEATURE_REQUIREMENTS) == set(LEFT_SIDE_SIGNALS)
    validate_left_side_factor_contract(
        (
            *LEFT_SIDE_PROJECT_FACTOR_REQUIREMENTS,
            *LEFT_SIDE_SHARED_RULE_REQUIREMENTS,
            *LEFT_SIDE_RULE_FEATURE_COLUMNS,
        )
    )
    payload = left_side_contract_payload()
    assert len(LEFT_SIDE_FACTOR_COLUMNS) == (
        len(PROJECT_FACTOR_COLUMNS) + len(LEFT_SIDE_RULE_FEATURE_COLUMNS)
        + len(LEFT_SIDE_SHARED_RULE_REQUIREMENTS)
    )
    assert len(LEFT_SIDE_TASK_FEATURE_COLUMNS) == 4
    assert payload["factor_count"] == len(LEFT_SIDE_FACTOR_COLUMNS)
    assert payload["model_input_count"] == len(LEFT_SIDE_MODEL_INPUT_COLUMNS)
    assert len(LEFT_SIDE_MODEL_INPUT_COLUMNS) == len(LEFT_SIDE_FACTOR_COLUMNS) + 4
    assert len(LEFT_SIDE_SCORING_INPUT_COLUMNS) == len(LEFT_SIDE_FACTOR_COLUMNS) + 4
    assert set(LEFT_SIDE_SCORING_INPUT_COLUMNS[-4:]) == set(LEFT_SIDE_SIGNALS)
    assert FORBIDDEN_COMPATIBILITY_ALIASES.isdisjoint(
        LEFT_SIDE_MODEL_INPUT_COLUMNS
    )
    assert FORBIDDEN_COMPATIBILITY_ALIASES.isdisjoint(
        LEFT_SIDE_SCORING_INPUT_COLUMNS
    )


def test_left_side_rule_features_are_prefix_causal() -> None:
    daily = _daily()
    full = compute_left_side_rule_features(daily)
    prefix = compute_left_side_rule_features(daily.iloc[:190])

    pd.testing.assert_frame_equal(
        full.iloc[:190].reset_index(drop=True),
        prefix.reset_index(drop=True),
        check_dtype=False,
    )


def test_left_side_signal_rebuild_matches_live_detectors_on_recent_prefixes() -> None:
    daily = _daily()
    rules = compute_left_side_rule_features(daily)
    actual = compute_left_side_signal_flags(daily, rule_features=rules)
    detectors = {
        "DUICHEN_VA": _detect_duichen_va,
        "NANA": _detect_nana,
        "YIDONG_DILIAN": _detect_yidong_dilian,
    }

    for position in range(180, len(daily)):
        prefix = daily.iloc[: position + 1]
        normalized = _normalize_daily(
            Path("000001.SZ"),
            str(prefix["date"].iloc[-1].date()),
            source_frame=prefix,
        )
        expected_low_pullback = any(
            detector(normalized) is not None for detector in detectors.values()
        )
        assert bool(actual.at[position, "LOW_PULLBACK"]) == expected_low_pullback


def test_left_side_dataset_worker_emits_canonical_events_and_tradeable_labels() -> None:
    daily = _daily(260).rename(columns={"volume": "vol"})
    daily["trade_date"] = daily["date"].dt.strftime("%Y%m%d")
    daily["amount"] = daily["close"] * daily["vol"]

    events, labels, error = _process_symbol(
        (
            "000001.SZ",
            daily,
            pd.DataFrame(),
            daily["date"].iloc[70],
            daily["date"].iloc[90],
            pd.DatetimeIndex(daily["date"]),
            (5,),
            ("next_close",),
        )
    )

    assert error is None
    assert not events.empty
    assert len(events) == len(labels)
    assert FORBIDDEN_COMPATIBILITY_ALIASES.isdisjoint(events.columns)
    assert events["factor_schema_version"].eq(
        "project-v5-canonical-alias-free"
    ).all()
    assert labels["locked_limit_up"].eq(False).all()
