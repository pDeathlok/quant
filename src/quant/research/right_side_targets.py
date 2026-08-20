"""Immutable training-target contracts for unified right-side research.

Derived targets live here rather than in the dataset builder so a new model
experiment cannot silently change the canonical forward-label parquet.  The
cost-aligned target below is intentionally fixed at 15 bps: changing the
trading-cost assumption requires a new target name and a new experiment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


TERMINAL_NET_POSITIVE_15BPS = "terminal_net_positive_15bps"
TERMINAL_NET_POSITIVE_COST_BPS = 15.0
TERMINAL_NET_POSITIVE_THRESHOLD_RETURN = (
    TERMINAL_NET_POSITIVE_COST_BPS / 10_000.0
)

PRECOMPUTED_BINARY_TARGETS: tuple[str, ...] = (
    "hit_up3",
    "hit_up5",
    "hit_up8",
    "good_path5",
)
SUPPORTED_TRAINING_TARGETS: tuple[str, ...] = (
    *PRECOMPUTED_BINARY_TARGETS,
    TERMINAL_NET_POSITIVE_15BPS,
)


@dataclass(frozen=True)
class TargetContract:
    """One immutable binary-target definition persisted with model outputs."""

    name: str
    kind: str
    source_column: str
    operator: str | None
    threshold_return: float | None
    cost_bps: float | None
    definition: str

    def metadata(self) -> dict[str, Any]:
        values = asdict(self)
        return {
            "target_kind": values["kind"],
            "target_source": values["source_column"],
            "target_operator": values["operator"],
            "target_threshold_return": values["threshold_return"],
            "target_cost_bps": values["cost_bps"],
            "target_definition": values["definition"],
        }


_PRECOMPUTED_DEFINITIONS: dict[str, str] = {
    "hit_up3": "canonical mature forward path reaches +3% MFE",
    "hit_up5": "canonical mature forward path reaches +5% MFE",
    "hit_up8": "canonical mature forward path reaches +8% MFE",
    "good_path5": "canonical mature forward path reaches +5% MFE without -3% MAE",
}

TARGET_CONTRACTS: dict[str, TargetContract] = {
    target: TargetContract(
        name=target,
        kind="precomputed_binary",
        source_column=target,
        operator=None,
        threshold_return=None,
        cost_bps=None,
        definition=definition,
    )
    for target, definition in _PRECOMPUTED_DEFINITIONS.items()
}
TARGET_CONTRACTS[TERMINAL_NET_POSITIVE_15BPS] = TargetContract(
    name=TERMINAL_NET_POSITIVE_15BPS,
    kind="derived_binary",
    source_column="terminal_return",
    operator=">",
    threshold_return=TERMINAL_NET_POSITIVE_THRESHOLD_RETURN,
    cost_bps=TERMINAL_NET_POSITIVE_COST_BPS,
    definition=(
        "terminal_return > 0.0015 (gross terminal return strictly exceeds "
        "the fixed 15 bps round-trip cost; equality is negative)"
    ),
)


def target_contract(name: str) -> TargetContract:
    """Return the registered target contract or fail closed."""

    try:
        return TARGET_CONTRACTS[str(name)]
    except KeyError as exc:
        raise ValueError(f"unsupported right-side training target: {name}") from exc


def target_source_columns(name: str) -> tuple[str, ...]:
    """Return canonical parquet columns needed to materialize ``name``."""

    return (target_contract(name).source_column,)


def validate_target_cost(name: str, round_trip_cost_bps: float) -> None:
    """Prevent a cost-derived target and trading metrics from drifting apart."""

    contract = target_contract(name)
    if contract.cost_bps is None:
        return
    cost = float(round_trip_cost_bps)
    if not np.isfinite(cost) or not np.isclose(cost, contract.cost_bps, atol=1e-12):
        raise ValueError(
            f"{name} is fixed to {contract.cost_bps:g} bps; "
            f"round_trip_cost_bps={round_trip_cost_bps!r} would misalign the target and return metrics"
        )


def materialize_training_target(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """Materialize one registered target while preserving missing outcomes.

    Precomputed targets are returned unchanged after column validation.  The
    derived target uses a strict comparison, so a gross return of exactly
    15 bps has zero net return and belongs to the negative class.
    """

    contract = target_contract(name)
    if contract.source_column not in frame.columns:
        raise ValueError(
            f"target source column missing for {name}: {contract.source_column}"
        )
    # The full research label slice can contain millions of rows.  A shallow
    # structural copy is sufficient because this helper only adds a target
    # column and never mutates an existing source column.
    out = frame.copy(deep=False)
    if contract.kind != "derived_binary":
        return out

    source = pd.to_numeric(out[contract.source_column], errors="coerce")
    finite = pd.Series(np.isfinite(source.to_numpy(dtype=float)), index=out.index)
    target = pd.Series(pd.NA, index=out.index, dtype="boolean")
    assert contract.threshold_return is not None
    target.loc[finite] = source.loc[finite].gt(contract.threshold_return).to_numpy()
    out[name] = target
    return out


def target_metadata(name: str) -> dict[str, Any]:
    """Return stable flat metadata suitable for CSV and Parquet artifacts."""

    return target_contract(name).metadata()


def validate_persisted_target_metadata(
    frame: pd.DataFrame,
    name: str,
) -> None:
    """Fail closed if a derived-target artifact lacks its immutable contract."""

    contract = target_contract(name)
    if contract.kind != "derived_binary":
        return
    expected = target_metadata(name)
    missing = set(expected) - set(frame.columns)
    if missing:
        raise ValueError(
            f"{name} artifact missing target metadata: {sorted(missing)}"
        )
    if frame.empty:
        raise ValueError(f"{name} artifact has no rows to validate target metadata")
    for column, expected_value in expected.items():
        values = frame[column]
        if expected_value is None:
            matched = values.isna()
        elif isinstance(expected_value, (float, int)):
            numeric = pd.to_numeric(values, errors="coerce")
            matched = numeric.notna() & np.isclose(
                numeric.to_numpy(dtype=float),
                float(expected_value),
                atol=1e-12,
            )
        else:
            matched = values.astype("string").eq(str(expected_value)).fillna(False)
        if not bool(np.asarray(matched).all()):
            raise ValueError(
                f"{name} artifact target metadata mismatch for {column}; "
                f"expected={expected_value!r}"
            )


__all__ = [
    "PRECOMPUTED_BINARY_TARGETS",
    "SUPPORTED_TRAINING_TARGETS",
    "TARGET_CONTRACTS",
    "TERMINAL_NET_POSITIVE_15BPS",
    "TERMINAL_NET_POSITIVE_COST_BPS",
    "TERMINAL_NET_POSITIVE_THRESHOLD_RETURN",
    "TargetContract",
    "materialize_training_target",
    "target_contract",
    "target_metadata",
    "target_source_columns",
    "validate_persisted_target_metadata",
    "validate_target_cost",
]
