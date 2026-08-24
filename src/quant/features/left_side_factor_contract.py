"""Versioned canonical model-input contract for the unified left-side ranker."""

from __future__ import annotations

from quant.features.canonical_factor_names import (
    assert_no_forbidden_factor_names,
    stable_canonical_feature_union,
)
from quant.features.project_factor_layer import PROJECT_FACTOR_SCHEMA_VERSION
from quant.features.right_side_factor_contract import factor_contract_sha256
from quant.features.variable_library import PROJECT_FACTOR_COLUMNS
from quant.research.left_side_unified_features import (
    LEFT_SIDE_RULE_FEATURE_COLUMNS,
    LEFT_SIDE_RULE_FEATURE_COLUMNS_SHA256,
    LEFT_SIDE_RULE_FEATURE_SCHEMA_VERSION,
    LEFT_SIDE_SIGNAL_SCHEMA_VERSION,
    LEFT_SIDE_SIGNALS,
    LEFT_SIDE_SHARED_RULE_REQUIREMENTS,
)


LEFT_SIDE_ARTIFACT_SCHEMA_VERSION = (
    "left-side-unified-ranking-production-v4-group4-canonical-alias-free"
)
LEFT_SIDE_SCORE_SCHEMA_VERSION = "left-side-unified-ranking-score-v4-group4"
LEFT_SIDE_TASK_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    f"task_{signal}" for signal in LEFT_SIDE_SIGNALS
)
LEFT_SIDE_FACTOR_COLUMNS: tuple[str, ...] = stable_canonical_feature_union(
    PROJECT_FACTOR_COLUMNS,
    LEFT_SIDE_SHARED_RULE_REQUIREMENTS,
    LEFT_SIDE_RULE_FEATURE_COLUMNS,
)
LEFT_SIDE_FEATURE_SCHEMA_VERSION = (
    "left-side-features-v2-project-v5-shared-right2-rule-v2-canonical-"
    f"{len(LEFT_SIDE_FACTOR_COLUMNS)}"
)
LEFT_SIDE_MODEL_INPUT_COLUMNS: tuple[str, ...] = stable_canonical_feature_union(
    LEFT_SIDE_FACTOR_COLUMNS,
    LEFT_SIDE_TASK_FEATURE_COLUMNS,
)
LEFT_SIDE_SCORING_INPUT_COLUMNS: tuple[str, ...] = stable_canonical_feature_union(
    LEFT_SIDE_FACTOR_COLUMNS,
    LEFT_SIDE_SIGNALS,
)
LEFT_SIDE_FACTOR_CONTRACT_SHA256 = factor_contract_sha256(
    LEFT_SIDE_FACTOR_COLUMNS,
    schema_version=LEFT_SIDE_FEATURE_SCHEMA_VERSION,
)
LEFT_SIDE_MODEL_INPUT_CONTRACT_SHA256 = factor_contract_sha256(
    LEFT_SIDE_MODEL_INPUT_COLUMNS,
    schema_version=LEFT_SIDE_ARTIFACT_SCHEMA_VERSION,
)
LEFT_SIDE_SCORING_INPUT_CONTRACT_SHA256 = factor_contract_sha256(
    LEFT_SIDE_SCORING_INPUT_COLUMNS,
    schema_version=LEFT_SIDE_SCORE_SCHEMA_VERSION,
)


def validate_left_side_model_contract() -> None:
    assert_no_forbidden_factor_names(
        LEFT_SIDE_MODEL_INPUT_COLUMNS,
        context="left-side unified model input contract",
    )
    if len(LEFT_SIDE_FACTOR_COLUMNS) != (
        len(PROJECT_FACTOR_COLUMNS)
        + len(LEFT_SIDE_SHARED_RULE_REQUIREMENTS)
        + len(LEFT_SIDE_RULE_FEATURE_COLUMNS)
    ):
        overlap = sorted(
            set(PROJECT_FACTOR_COLUMNS)
            & set((*LEFT_SIDE_SHARED_RULE_REQUIREMENTS, *LEFT_SIDE_RULE_FEATURE_COLUMNS))
        )
        raise ValueError(f"left-side factor overlap: {overlap}")
    if len(LEFT_SIDE_MODEL_INPUT_COLUMNS) != len(
        set(LEFT_SIDE_MODEL_INPUT_COLUMNS)
    ):
        raise ValueError("left-side model input contract contains duplicates")
    if len(LEFT_SIDE_SCORING_INPUT_COLUMNS) != len(
        set(LEFT_SIDE_SCORING_INPUT_COLUMNS)
    ):
        raise ValueError("left-side scoring input contract contains duplicates")


def left_side_contract_payload() -> dict[str, object]:
    validate_left_side_model_contract()
    return {
        "artifact_schema_version": LEFT_SIDE_ARTIFACT_SCHEMA_VERSION,
        "feature_schema_version": LEFT_SIDE_FEATURE_SCHEMA_VERSION,
        "score_schema_version": LEFT_SIDE_SCORE_SCHEMA_VERSION,
        "project_factor_schema_version": PROJECT_FACTOR_SCHEMA_VERSION,
        "project_factor_count": len(PROJECT_FACTOR_COLUMNS),
        "rule_factor_schema_version": LEFT_SIDE_RULE_FEATURE_SCHEMA_VERSION,
        "shared_right_rule_factor_count": len(LEFT_SIDE_SHARED_RULE_REQUIREMENTS),
        "shared_right_rule_factors": list(LEFT_SIDE_SHARED_RULE_REQUIREMENTS),
        "rule_factor_count": len(LEFT_SIDE_RULE_FEATURE_COLUMNS),
        "rule_factor_columns_sha256": LEFT_SIDE_RULE_FEATURE_COLUMNS_SHA256,
        "factor_count": len(LEFT_SIDE_FACTOR_COLUMNS),
        "factor_contract_sha256": LEFT_SIDE_FACTOR_CONTRACT_SHA256,
        "signal_schema_version": LEFT_SIDE_SIGNAL_SCHEMA_VERSION,
        "signal_identity_count": len(LEFT_SIDE_SIGNALS),
        "task_feature_count": len(LEFT_SIDE_TASK_FEATURE_COLUMNS),
        "model_input_count": len(LEFT_SIDE_MODEL_INPUT_COLUMNS),
        "model_input_contract_sha256": LEFT_SIDE_MODEL_INPUT_CONTRACT_SHA256,
        "scoring_input_count": len(LEFT_SIDE_SCORING_INPUT_COLUMNS),
        "scoring_input_contract_sha256": LEFT_SIDE_SCORING_INPUT_CONTRACT_SHA256,
    }


__all__ = [
    "LEFT_SIDE_ARTIFACT_SCHEMA_VERSION",
    "LEFT_SIDE_FACTOR_COLUMNS",
    "LEFT_SIDE_FACTOR_CONTRACT_SHA256",
    "LEFT_SIDE_FEATURE_SCHEMA_VERSION",
    "LEFT_SIDE_MODEL_INPUT_COLUMNS",
    "LEFT_SIDE_MODEL_INPUT_CONTRACT_SHA256",
    "LEFT_SIDE_SCORE_SCHEMA_VERSION",
    "LEFT_SIDE_SCORING_INPUT_COLUMNS",
    "LEFT_SIDE_SCORING_INPUT_CONTRACT_SHA256",
    "LEFT_SIDE_TASK_FEATURE_COLUMNS",
    "left_side_contract_payload",
    "validate_left_side_model_contract",
]
