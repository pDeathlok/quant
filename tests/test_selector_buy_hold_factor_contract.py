from __future__ import annotations

import pandas as pd
import pytest

from quant.features.canonical_factor_names import FORBIDDEN_COMPATIBILITY_ALIASES
from quant.features.factor_registry import FACTOR_REGISTRY
from quant.features.long_weekly_factors import (
    MONTHLY_VALUATION_HISTORY_COLUMNS,
    add_monthly_valuation_history,
)
from quant.features.selector_buy_hold_factor_contract import (
    SELECTOR_BUY_HOLD_ARTIFACT_SCHEMA_VERSION,
    SELECTOR_BUY_HOLD_CANDIDATE_FACTOR_COLUMNS,
    SELECTOR_BUY_HOLD_FACTOR_CONTRACT_SHA256,
    SELECTOR_BUY_HOLD_FEATURE_SCHEMA_VERSION,
    selector_buy_hold_model_input_sha256,
    validate_selector_buy_hold_artifact,
)
from quant.features.selector_buy_hold_materialization import (
    audit_selector_buy_hold_training_materialization,
    require_complete_selector_buy_hold_training_materialization,
    selected_production_feature_union,
    selector_buy_hold_training_materialization_plan,
)


def _artifact(features: tuple[str, ...]) -> dict[str, object]:
    return {
        "schema_version": SELECTOR_BUY_HOLD_ARTIFACT_SCHEMA_VERSION,
        "feature_schema_version": SELECTOR_BUY_HOLD_FEATURE_SCHEMA_VERSION,
        "factor_contract_sha256": SELECTOR_BUY_HOLD_FACTOR_CONTRACT_SHA256,
        "candidate_features": list(SELECTOR_BUY_HOLD_CANDIDATE_FACTOR_COLUMNS),
        "features": list(features),
        "model_input_contract_sha256": selector_buy_hold_model_input_sha256(features),
    }


def test_selector_buy_hold_candidate_contract_is_complete_canonical_registry() -> None:
    registry_features = tuple(
        definition.name for definition in FACTOR_REGISTRY if definition.role == "feature"
    )

    assert SELECTOR_BUY_HOLD_CANDIDATE_FACTOR_COLUMNS == registry_features
    assert len(SELECTOR_BUY_HOLD_CANDIDATE_FACTOR_COLUMNS) >= 550
    assert len(SELECTOR_BUY_HOLD_CANDIDATE_FACTOR_COLUMNS) == len(
        set(SELECTOR_BUY_HOLD_CANDIDATE_FACTOR_COLUMNS)
    )
    assert not (
        set(SELECTOR_BUY_HOLD_CANDIDATE_FACTOR_COLUMNS)
        & set(FORBIDDEN_COMPATIBILITY_ALIASES)
    )
    assert len(SELECTOR_BUY_HOLD_FACTOR_CONTRACT_SHA256) == 64


def test_selector_buy_hold_artifact_rejects_unknown_or_reordered_features() -> None:
    features = SELECTOR_BUY_HOLD_CANDIDATE_FACTOR_COLUMNS[:3]
    assert validate_selector_buy_hold_artifact(_artifact(features)) == features

    unknown = _artifact(features)
    unknown["features"] = [*features, "not_registered"]
    with pytest.raises(ValueError, match="outside the registry contract"):
        validate_selector_buy_hold_artifact(unknown)

    reordered = _artifact(features)
    reordered["candidate_features"] = list(reversed(SELECTOR_BUY_HOLD_CANDIDATE_FACTOR_COLUMNS))
    with pytest.raises(ValueError, match="candidate factor universe drifted"):
        validate_selector_buy_hold_artifact(reordered)


def test_selector_training_materialization_plan_covers_every_candidate_factor() -> None:
    plan = selector_buy_hold_training_materialization_plan()
    planned = [
        factor
        for calculator in plan["calculators"]
        for factor in calculator["factors"]
    ]

    assert plan["required_factor_count"] == len(
        SELECTOR_BUY_HOLD_CANDIDATE_FACTOR_COLUMNS
    )
    assert tuple(planned) != ()
    assert set(planned) == set(SELECTOR_BUY_HOLD_CANDIDATE_FACTOR_COLUMNS)
    assert len(planned) == len(set(planned))
    assert {
        calculator["training_execution_mode"] for calculator in plan["calculators"]
    } == {"full_history_on_demand"}


def test_selector_training_rejects_partial_registry_instead_of_pruning_it() -> None:
    coverage = pd.DataFrame(
        {
            "factor": SELECTOR_BUY_HOLD_CANDIDATE_FACTOR_COLUMNS,
            "non_null": 100,
            "unique_values": 10,
        }
    )
    missing_factor = SELECTOR_BUY_HOLD_CANDIDATE_FACTOR_COLUMNS[-1]
    coverage.loc[coverage["factor"].eq(missing_factor), ["non_null", "unique_values"]] = 0

    audit = audit_selector_buy_hold_training_materialization(coverage)

    assert audit.complete is False
    assert audit.usable_factor_count == len(coverage) - 1
    assert audit.issues[0].factor == missing_factor
    assert audit.issues[0].status == "insufficient_materialized_history"
    with pytest.raises(RuntimeError, match=r"usable=609/610"):
        require_complete_selector_buy_hold_training_materialization(coverage)


def test_selector_training_keeps_materialized_constant_factors_for_discovery() -> None:
    coverage = pd.DataFrame(
        {
            "factor": SELECTOR_BUY_HOLD_CANDIDATE_FACTOR_COLUMNS,
            "non_null": 100,
            "unique_values": 10,
        }
    )
    constant_factor = SELECTOR_BUY_HOLD_CANDIDATE_FACTOR_COLUMNS[-1]
    coverage.loc[coverage["factor"].eq(constant_factor), "unique_values"] = 1

    audit = require_complete_selector_buy_hold_training_materialization(coverage)

    assert audit.complete is True
    assert audit.usable_factor_count == len(coverage)
    assert audit.informative_factor_count == len(coverage) - 1
    assert audit.issues == ()
    assert audit.warnings[0].factor == constant_factor
    assert audit.warnings[0].status == "constant_materialized_history"


def test_daily_union_uses_only_final_artifacts_and_strategy_consumers() -> None:
    candidates = SELECTOR_BUY_HOLD_CANDIDATE_FACTOR_COLUMNS
    artifacts = (
        {"features": [candidates[0], candidates[2]]},
        {"features": [candidates[2], candidates[4]]},
    )

    selected = selected_production_feature_union(
        artifacts,
        strategy_features=(candidates[6],),
    )

    assert selected == (candidates[0], candidates[2], candidates[4], candidates[6])
    assert candidates[1] not in selected


def test_monthly_valuation_history_materializes_unsuffixed_contract() -> None:
    dates = pd.date_range("2020-01-31", periods=30, freq="ME")
    frame = pd.DataFrame(
        {
            "date": dates,
            "ts_code": "000001.SZ",
            "industry": "银行",
            "roe": range(10, 40),
            "pe_ttm": range(20, 50),
            "pb": [1.0 + index / 100 for index in range(30)],
        }
    )

    materialized = add_monthly_valuation_history(frame)

    assert set(MONTHLY_VALUATION_HISTORY_COLUMNS) <= set(materialized.columns)
    assert pd.isna(materialized.loc[22, "pe_hist_percentile"])
    assert materialized.loc[23, "pe_hist_percentile"] == pytest.approx(100.0)
    assert materialized.loc[23, "valuation_history_points"] == 24
    assert materialized.loc[23, "roe_history_points"] == 24
