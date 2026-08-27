"""Canonical registry contract for selector buy/hold return models.

The discovery universe intentionally follows the complete governed factor
registry.  Training must materialize every candidate factor before discovery;
only the frozen post-discovery model input may be smaller, so production can
refresh the final model and strategy consumer union instead of all candidates.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from quant.features.canonical_factor_names import (
    assert_no_forbidden_factor_names,
    stable_canonical_feature_union,
)
from quant.features.factor_registry import FACTOR_REGISTRY
from quant.features.right_side_factor_contract import factor_contract_sha256


SELECTOR_BUY_HOLD_ARTIFACT_SCHEMA_VERSION = (
    "selector-buy-hold-return-model-v3-production-materialized"
)
SELECTOR_BUY_HOLD_FEATURE_SCHEMA_VERSION = (
    "selector-buy-hold-features-v2-factor-registry-canonical"
)
SELECTOR_BUY_HOLD_MANIFEST_SCHEMA_VERSION = (
    "selector-buy-hold-production-manifest-v3-materialized"
)
SELECTOR_BUY_HOLD_RELEASE_ID = "selector-buy-hold-registry-v3-20260827"

# A factor may participate in the 610-factor discovery pass before it is safe
# to become a daily model input.  Chan currently publishes only its own signal
# candidates (not the selector candidate union), while the research-only
# calculators have no production freshness node.  Keep those factors in
# discovery, but freeze the released model to calculators whose exact-date
# cross-section is already produced for every selector consumer.
SELECTOR_BUY_HOLD_PRODUCTION_CALCULATORS = frozenset(
    {
        "project_daily",
        "right_side_rule",
        "left_side_rule",
        "selector_live",
        "long_snapshot",
    }
)


SELECTOR_BUY_HOLD_CANDIDATE_FACTOR_COLUMNS: tuple[str, ...] = (
    stable_canonical_feature_union(
        tuple(
            definition.name
            for definition in FACTOR_REGISTRY
            if definition.role == "feature"
        )
    )
)
SELECTOR_BUY_HOLD_FACTOR_CONTRACT_SHA256 = factor_contract_sha256(
    SELECTOR_BUY_HOLD_CANDIDATE_FACTOR_COLUMNS,
    schema_version=SELECTOR_BUY_HOLD_FEATURE_SCHEMA_VERSION,
)
SELECTOR_BUY_HOLD_PRODUCTION_FACTOR_COLUMNS: tuple[str, ...] = (
    stable_canonical_feature_union(
        tuple(
            definition.name
            for definition in FACTOR_REGISTRY
            if definition.role == "feature"
            and definition.calculator_id
            in SELECTOR_BUY_HOLD_PRODUCTION_CALCULATORS
        )
    )
)


def selector_buy_hold_model_input_sha256(features: Sequence[str]) -> str:
    """Return the stable hash for the effective model input columns."""

    canonical = stable_canonical_feature_union(tuple(features))
    if tuple(str(value) for value in features) != canonical:
        raise ValueError("selector buy/hold model features are not stable and unique")
    unknown = sorted(set(canonical) - set(SELECTOR_BUY_HOLD_CANDIDATE_FACTOR_COLUMNS))
    if unknown:
        raise ValueError(
            "selector buy/hold model features are outside the registry contract: "
            f"{unknown}"
        )
    return factor_contract_sha256(
        canonical,
        schema_version=SELECTOR_BUY_HOLD_ARTIFACT_SCHEMA_VERSION,
    )


def validate_selector_buy_hold_artifact(
    artifact: Mapping[str, Any],
) -> tuple[str, ...]:
    """Fail closed on schema, registry, alias, or model-input drift."""

    if artifact.get("schema_version") != SELECTOR_BUY_HOLD_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("selector buy/hold artifact schema version drifted")
    if artifact.get("feature_schema_version") != SELECTOR_BUY_HOLD_FEATURE_SCHEMA_VERSION:
        raise ValueError("selector buy/hold feature schema version drifted")
    if artifact.get("factor_contract_sha256") != SELECTOR_BUY_HOLD_FACTOR_CONTRACT_SHA256:
        raise ValueError("selector buy/hold factor contract hash drifted")
    candidate_features = tuple(str(value) for value in artifact.get("candidate_features") or ())
    if candidate_features != SELECTOR_BUY_HOLD_CANDIDATE_FACTOR_COLUMNS:
        raise ValueError("selector buy/hold candidate factor universe drifted")
    features = tuple(str(value) for value in artifact.get("features") or ())
    if not features:
        raise ValueError("selector buy/hold artifact declares no model features")
    assert_no_forbidden_factor_names(
        (*candidate_features, *features),
        context="selector buy/hold artifact",
    )
    expected_hash = selector_buy_hold_model_input_sha256(features)
    if artifact.get("model_input_contract_sha256") != expected_hash:
        raise ValueError("selector buy/hold model input contract hash drifted")
    feature_names: tuple[str, ...] = ()
    model = artifact.get("model")
    if model is not None:
        feature_names = tuple(str(value) for value in getattr(model, "feature_names_in_", ()))
    elif isinstance(artifact.get("models"), Mapping):
        for component in artifact["models"].values():
            current = tuple(
                str(value) for value in getattr(component, "feature_names_in_", ())
            )
            if current and current != features:
                raise ValueError(
                    "selector buy/hold component feature_names_in_ drifted"
                )
            if current:
                feature_names = current
    if feature_names and feature_names != features:
        raise ValueError("selector buy/hold model feature_names_in_ drifted")
    return features


def selector_buy_hold_factor_contract_payload() -> dict[str, Any]:
    """Return the registry contract embedded in training/release manifests."""

    assert_no_forbidden_factor_names(
        SELECTOR_BUY_HOLD_CANDIDATE_FACTOR_COLUMNS,
        context="selector buy/hold registry candidate contract",
    )
    return {
        "feature_schema_version": SELECTOR_BUY_HOLD_FEATURE_SCHEMA_VERSION,
        "artifact_schema_version": SELECTOR_BUY_HOLD_ARTIFACT_SCHEMA_VERSION,
        "candidate_factor_count": len(SELECTOR_BUY_HOLD_CANDIDATE_FACTOR_COLUMNS),
        "candidate_factor_columns": list(SELECTOR_BUY_HOLD_CANDIDATE_FACTOR_COLUMNS),
        "factor_contract_sha256": SELECTOR_BUY_HOLD_FACTOR_CONTRACT_SHA256,
    }


__all__ = [
    "SELECTOR_BUY_HOLD_ARTIFACT_SCHEMA_VERSION",
    "SELECTOR_BUY_HOLD_CANDIDATE_FACTOR_COLUMNS",
    "SELECTOR_BUY_HOLD_FACTOR_CONTRACT_SHA256",
    "SELECTOR_BUY_HOLD_FEATURE_SCHEMA_VERSION",
    "SELECTOR_BUY_HOLD_MANIFEST_SCHEMA_VERSION",
    "SELECTOR_BUY_HOLD_PRODUCTION_CALCULATORS",
    "SELECTOR_BUY_HOLD_PRODUCTION_FACTOR_COLUMNS",
    "SELECTOR_BUY_HOLD_RELEASE_ID",
    "selector_buy_hold_factor_contract_payload",
    "selector_buy_hold_model_input_sha256",
    "validate_selector_buy_hold_artifact",
]
