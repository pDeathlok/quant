"""Training-time full-registry and production-time selected-factor contracts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from quant.features.canonical_factor_names import (
    assert_no_forbidden_factor_names,
    stable_canonical_feature_union,
)
from quant.features.factor_execution import build_factor_execution_plan
from quant.features.factor_registry import FACTOR_REGISTRY
from quant.features.selector_buy_hold_factor_contract import (
    SELECTOR_BUY_HOLD_CANDIDATE_FACTOR_COLUMNS,
    SELECTOR_BUY_HOLD_FACTOR_CONTRACT_SHA256,
)


SELECTOR_BUY_HOLD_MATERIALIZATION_SCHEMA_VERSION = (
    "selector-buy-hold-full-registry-materialization-v1"
)


@dataclass(frozen=True)
class FactorMaterializationIssue:
    factor: str
    calculator_id: str
    status: str
    non_null: int
    unique_values: int


@dataclass(frozen=True)
class TrainingMaterializationAudit:
    schema_version: str
    factor_contract_sha256: str
    required_factor_count: int
    usable_factor_count: int
    informative_factor_count: int
    complete: bool
    issues: tuple[FactorMaterializationIssue, ...]
    warnings: tuple[FactorMaterializationIssue, ...]


def selector_buy_hold_training_materialization_plan() -> dict[str, Any]:
    """Return a complete on-demand calculator plan for all registry factors."""

    required = SELECTOR_BUY_HOLD_CANDIDATE_FACTOR_COLUMNS
    definitions = {
        definition.name: definition
        for definition in FACTOR_REGISTRY
        if definition.role == "feature"
    }
    if set(required) != set(definitions):
        raise RuntimeError("selector training factors drifted from the feature registry")
    calculators = build_factor_execution_plan(required)
    planned: set[str] = set()
    calculator_rows: list[dict[str, Any]] = []
    for calculator in calculators:
        factors = tuple(column for column in calculator.produces if column in definitions)
        planned.update(factors)
        calculator_rows.append(
            {
                "calculator_id": calculator.calculator_id,
                "entrypoint": calculator.entrypoint,
                "dependencies": list(calculator.dependencies),
                "materialization": calculator.materialization,
                "training_execution_mode": "full_history_on_demand",
                "factor_count": len(factors),
                "factors": list(factors),
            }
        )
    missing = sorted(set(required) - planned)
    if missing:
        raise RuntimeError(
            f"selector training materialization plan misses factors: {missing}"
        )
    return {
        "schema_version": SELECTOR_BUY_HOLD_MATERIALIZATION_SCHEMA_VERSION,
        "factor_contract_sha256": SELECTOR_BUY_HOLD_FACTOR_CONTRACT_SHA256,
        "required_factor_count": len(required),
        "calculator_count": len(calculator_rows),
        "calculators": calculator_rows,
    }


def audit_selector_buy_hold_training_materialization(
    coverage: pd.DataFrame,
    *,
    minimum_observations: int = 2,
) -> TrainingMaterializationAudit:
    """Require every registered candidate factor to be genuinely trainable."""

    required_columns = {"factor", "non_null", "unique_values"}
    missing_columns = sorted(required_columns - set(coverage.columns))
    if missing_columns:
        raise ValueError(
            f"selector training coverage misses audit columns: {missing_columns}"
        )
    names = coverage["factor"].astype(str)
    duplicates = sorted(names[names.duplicated()].unique())
    if duplicates:
        raise ValueError(
            f"selector training coverage contains duplicate factors: {duplicates}"
        )
    unknown = sorted(set(names) - set(SELECTOR_BUY_HOLD_CANDIDATE_FACTOR_COLUMNS))
    if unknown:
        raise ValueError(
            f"selector training coverage contains unregistered factors: {unknown}"
        )
    rows = coverage.set_index(names, drop=False)
    definitions = {definition.name: definition for definition in FACTOR_REGISTRY}
    issues: list[FactorMaterializationIssue] = []
    warnings: list[FactorMaterializationIssue] = []
    usable = 0
    informative = 0
    minimum = max(2, int(minimum_observations))
    for factor in SELECTOR_BUY_HOLD_CANDIDATE_FACTOR_COLUMNS:
        if factor not in rows.index:
            issues.append(
                FactorMaterializationIssue(
                    factor=factor,
                    calculator_id=definitions[factor].calculator_id,
                    status="missing_coverage_audit",
                    non_null=0,
                    unique_values=0,
                )
            )
            continue
        row = rows.loc[factor]
        non_null = int(row["non_null"])
        unique = int(row["unique_values"])
        if non_null < minimum:
            issues.append(
                FactorMaterializationIssue(
                    factor=factor,
                    calculator_id=definitions[factor].calculator_id,
                    status="insufficient_materialized_history",
                    non_null=non_null,
                    unique_values=unique,
                )
            )
            continue
        usable += 1
        if unique >= 2:
            informative += 1
        else:
            warnings.append(
                FactorMaterializationIssue(
                    factor=factor,
                    calculator_id=definitions[factor].calculator_id,
                    status="constant_materialized_history",
                    non_null=non_null,
                    unique_values=unique,
                )
            )
    return TrainingMaterializationAudit(
        schema_version=SELECTOR_BUY_HOLD_MATERIALIZATION_SCHEMA_VERSION,
        factor_contract_sha256=SELECTOR_BUY_HOLD_FACTOR_CONTRACT_SHA256,
        required_factor_count=len(SELECTOR_BUY_HOLD_CANDIDATE_FACTOR_COLUMNS),
        usable_factor_count=usable,
        informative_factor_count=informative,
        complete=not issues,
        issues=tuple(issues),
        warnings=tuple(warnings),
    )


def require_complete_selector_buy_hold_training_materialization(
    coverage: pd.DataFrame,
    *,
    minimum_observations: int = 2,
) -> TrainingMaterializationAudit:
    """Fail closed instead of silently training a partial-registry model."""

    audit = audit_selector_buy_hold_training_materialization(
        coverage,
        minimum_observations=minimum_observations,
    )
    if not audit.complete:
        by_calculator: dict[str, int] = {}
        for issue in audit.issues:
            by_calculator[issue.calculator_id] = (
                by_calculator.get(issue.calculator_id, 0) + 1
            )
        raise RuntimeError(
            "selector buy/hold full-registry materialization is incomplete: "
            f"usable={audit.usable_factor_count}/{audit.required_factor_count}; "
            f"issues_by_calculator={dict(sorted(by_calculator.items()))}"
        )
    return audit


def selected_production_feature_union(
    artifacts: Iterable[Mapping[str, Any]],
    strategy_features: Sequence[str] = (),
) -> tuple[str, ...]:
    """Return only final model inputs plus explicit strategy consumers."""

    selected = stable_canonical_feature_union(
        tuple(
            str(feature)
            for artifact in artifacts
            for feature in artifact.get("features") or ()
        ),
        tuple(strategy_features),
    )
    registered = {
        definition.name
        for definition in FACTOR_REGISTRY
        if definition.role in {"feature", "strategy_identity"}
    }
    unknown = sorted(set(selected) - registered)
    if unknown:
        raise ValueError(
            f"production selector feature union contains unregistered factors: {unknown}"
        )
    assert_no_forbidden_factor_names(
        selected,
        context="selector production selected feature union",
    )
    return selected


def training_materialization_audit_payload(
    audit: TrainingMaterializationAudit,
) -> dict[str, Any]:
    """Serialize the strict audit into a training report or manifest."""

    payload = asdict(audit)
    payload["issues"] = [asdict(issue) for issue in audit.issues]
    payload["warnings"] = [asdict(issue) for issue in audit.warnings]
    return payload


__all__ = [
    "FactorMaterializationIssue",
    "SELECTOR_BUY_HOLD_MATERIALIZATION_SCHEMA_VERSION",
    "TrainingMaterializationAudit",
    "audit_selector_buy_hold_training_materialization",
    "require_complete_selector_buy_hold_training_materialization",
    "selected_production_feature_union",
    "selector_buy_hold_training_materialization_plan",
    "training_materialization_audit_payload",
]
