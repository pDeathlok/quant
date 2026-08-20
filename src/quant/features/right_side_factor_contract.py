"""Versioned factor contract for the unified right-side shadow model.

The project-factor and rule-factor calculators live in different modules, but
the released ranking model consumes one ordered union.  This module is the
single registry-facing contract for that union and its stable hashes.
"""

from __future__ import annotations

import hashlib
import json
from typing import Sequence

from quant.features.project_factor_layer import PROJECT_FACTOR_SCHEMA_VERSION
from quant.features.variable_library import PROJECT_FACTOR_COLUMNS
from quant.research.right_side_unified_features import (
    RIGHT_SIDE_SIGNALS,
    RULE_FEATURE_COLUMNS,
    RULE_FEATURE_COLUMNS_SHA256,
    RULE_FEATURE_SCHEMA_VERSION,
)


CANONICAL_SIGNAL_SCHEMA_VERSION = "right_side_unified_signal_v1_live_z_20260813"


RIGHT_SIDE_SHADOW_ARTIFACT_SCHEMA_VERSION = (
    "right-side-unified-ranking-shadow-v1"
)
RIGHT_SIDE_SHADOW_FEATURE_SCHEMA_VERSION = (
    "right-side-shadow-features-v1-project-v4-rule-v2-118"
)

RIGHT_SIDE_SHADOW_FACTOR_COLUMNS: tuple[str, ...] = (
    *PROJECT_FACTOR_COLUMNS,
    *RULE_FEATURE_COLUMNS,
)
RIGHT_SIDE_SHADOW_IDENTITY_COLUMNS: tuple[str, ...] = tuple(RIGHT_SIDE_SIGNALS)
RIGHT_SIDE_SHADOW_MODEL_INPUT_COLUMNS: tuple[str, ...] = (
    *RIGHT_SIDE_SHADOW_FACTOR_COLUMNS,
    *RIGHT_SIDE_SHADOW_IDENTITY_COLUMNS,
)


def factor_contract_sha256(
    columns: Sequence[str],
    *,
    schema_version: str,
) -> str:
    """Return an order-sensitive hash for one model-facing factor schema."""

    payload = {
        "schema_version": str(schema_version),
        "columns": [str(column) for column in columns],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


RIGHT_SIDE_PROJECT_FACTOR_COLUMNS_SHA256 = factor_contract_sha256(
    PROJECT_FACTOR_COLUMNS,
    schema_version=PROJECT_FACTOR_SCHEMA_VERSION,
)
RIGHT_SIDE_SHADOW_FACTOR_CONTRACT_SHA256 = factor_contract_sha256(
    RIGHT_SIDE_SHADOW_FACTOR_COLUMNS,
    schema_version=RIGHT_SIDE_SHADOW_FEATURE_SCHEMA_VERSION,
)
RIGHT_SIDE_SHADOW_MODEL_INPUT_CONTRACT_SHA256 = factor_contract_sha256(
    RIGHT_SIDE_SHADOW_MODEL_INPUT_COLUMNS,
    schema_version=RIGHT_SIDE_SHADOW_ARTIFACT_SCHEMA_VERSION,
)


def validate_right_side_shadow_factor_contract() -> None:
    """Fail closed when any frozen count, ordering, or namespace drifts."""

    if len(PROJECT_FACTOR_COLUMNS) != 147:
        raise ValueError(
            "right-side shadow project-factor contract must contain 147 columns"
        )
    if len(RULE_FEATURE_COLUMNS) != 118:
        raise ValueError(
            "right-side shadow rule-factor contract must contain 118 columns"
        )
    if len(RIGHT_SIDE_SHADOW_IDENTITY_COLUMNS) != 14:
        raise ValueError(
            "right-side shadow identity contract must contain 14 signals"
        )
    duplicates = sorted(
        {
            column
            for column in RIGHT_SIDE_SHADOW_MODEL_INPUT_COLUMNS
            if RIGHT_SIDE_SHADOW_MODEL_INPUT_COLUMNS.count(column) > 1
        }
    )
    if duplicates:
        raise ValueError(
            f"right-side shadow input contract contains duplicates: {duplicates}"
        )
    if not RULE_FEATURE_COLUMNS_SHA256 or len(RULE_FEATURE_COLUMNS_SHA256) != 64:
        raise ValueError("right-side rule-factor contract has no valid SHA256")


def right_side_shadow_contract_payload() -> dict[str, object]:
    """Return the JSON-ready factor contract embedded in every manifest."""

    validate_right_side_shadow_factor_contract()
    return {
        "feature_schema_version": RIGHT_SIDE_SHADOW_FEATURE_SCHEMA_VERSION,
        "artifact_schema_version": RIGHT_SIDE_SHADOW_ARTIFACT_SCHEMA_VERSION,
        "project_factor_schema_version": PROJECT_FACTOR_SCHEMA_VERSION,
        "project_factor_count": len(PROJECT_FACTOR_COLUMNS),
        "project_factor_columns_sha256": RIGHT_SIDE_PROJECT_FACTOR_COLUMNS_SHA256,
        "rule_factor_schema_version": RULE_FEATURE_SCHEMA_VERSION,
        "rule_factor_count": len(RULE_FEATURE_COLUMNS),
        "rule_factor_columns_sha256": RULE_FEATURE_COLUMNS_SHA256,
        "factor_count": len(RIGHT_SIDE_SHADOW_FACTOR_COLUMNS),
        "factor_contract_sha256": RIGHT_SIDE_SHADOW_FACTOR_CONTRACT_SHA256,
        "signal_schema_version": CANONICAL_SIGNAL_SCHEMA_VERSION,
        "signal_identity_count": len(RIGHT_SIDE_SHADOW_IDENTITY_COLUMNS),
        "model_input_count": len(RIGHT_SIDE_SHADOW_MODEL_INPUT_COLUMNS),
        "model_input_contract_sha256": (
            RIGHT_SIDE_SHADOW_MODEL_INPUT_CONTRACT_SHA256
        ),
    }


__all__ = [
    "RIGHT_SIDE_PROJECT_FACTOR_COLUMNS_SHA256",
    "RIGHT_SIDE_SHADOW_ARTIFACT_SCHEMA_VERSION",
    "RIGHT_SIDE_SHADOW_FACTOR_COLUMNS",
    "RIGHT_SIDE_SHADOW_FACTOR_CONTRACT_SHA256",
    "RIGHT_SIDE_SHADOW_FEATURE_SCHEMA_VERSION",
    "RIGHT_SIDE_SHADOW_IDENTITY_COLUMNS",
    "RIGHT_SIDE_SHADOW_MODEL_INPUT_COLUMNS",
    "RIGHT_SIDE_SHADOW_MODEL_INPUT_CONTRACT_SHA256",
    "factor_contract_sha256",
    "right_side_shadow_contract_payload",
    "validate_right_side_shadow_factor_contract",
]
