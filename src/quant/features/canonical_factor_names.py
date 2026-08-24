"""Canonical-only model input policy and fail-closed legacy cache migration."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


FORBIDDEN_COMPATIBILITY_ALIASES: frozenset[str] = frozenset(
    {
        "price_level",
        "bb_middle",
        "rs_pct_chg_1d",
        "rs_amplitude_pct",
        "rs_vol_ratio_5_inclusive",
    }
)
LEGACY_TO_CANONICAL_FACTOR_NAMES: Mapping[str, str] = {
    "price_level": "close",
    "bb_middle": "ma_20",
    "rs_pct_chg_1d": "pct_chg",
    "rs_amplitude_pct": "amplitude_1",
    "rs_vol_ratio_5_inclusive": "volume_relative_5d",
}


def assert_no_forbidden_factor_names(
    columns: Iterable[object],
    *,
    context: str,
) -> None:
    names = {str(column) for column in columns}
    forbidden = sorted(names & FORBIDDEN_COMPATIBILITY_ALIASES)
    if forbidden:
        raise ValueError(f"{context} contains forbidden compatibility aliases: {forbidden}")


def stable_canonical_feature_union(*groups: Sequence[object]) -> tuple[str, ...]:
    """Return a stable canonical union without duplicate model columns."""

    output: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for value in group:
            name = str(value)
            assert_no_forbidden_factor_names((name,), context="feature union")
            if name not in seen:
                output.append(name)
                seen.add(name)
    return tuple(output)


def _assert_series_values_identical(
    alias_values: pd.Series,
    canonical_values: pd.Series,
    *,
    alias: str,
    canonical: str,
    context: str,
) -> None:
    alias_na = alias_values.isna().to_numpy()
    canonical_na = canonical_values.isna().to_numpy()
    if not np.array_equal(alias_na, canonical_na):
        mismatch = int(np.flatnonzero(alias_na != canonical_na)[0])
        raise ValueError(
            f"{context} alias migration mismatch {alias}->{canonical}: "
            f"NaN position differs at row {mismatch}"
        )
    valid = ~alias_na
    if not valid.any():
        return
    left = alias_values.to_numpy()[valid]
    right = canonical_values.to_numpy()[valid]
    try:
        equal = np.asarray(np.equal(left, right), dtype=bool)
    except TypeError:
        equal = np.asarray(
            [left_value == right_value for left_value, right_value in zip(left, right)],
            dtype=bool,
        )
    if not bool(equal.all()):
        local = int(np.flatnonzero(~equal)[0])
        mismatch = int(np.flatnonzero(valid)[local])
        raise ValueError(
            f"{context} alias migration mismatch {alias}->{canonical}: "
            f"value differs at row {mismatch}"
        )


def migrate_legacy_factor_columns(
    frame: pd.DataFrame,
    *,
    context: str,
    copy: bool = True,
) -> pd.DataFrame:
    """Migrate legacy cache columns once and remove every legacy name."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{context} alias migration requires a DataFrame")
    out = frame.copy() if copy else frame
    for alias, canonical in LEGACY_TO_CANONICAL_FACTOR_NAMES.items():
        if alias not in out:
            continue
        if canonical in out:
            _assert_series_values_identical(
                out[alias],
                out[canonical],
                alias=alias,
                canonical=canonical,
                context=context,
            )
            out = out.drop(columns=alias)
        else:
            out = out.rename(columns={alias: canonical})
    if out.columns.duplicated().any():
        duplicates = sorted(set(out.columns[out.columns.duplicated()].astype(str)))
        raise ValueError(f"{context} canonical migration produced duplicates: {duplicates}")
    assert_no_forbidden_factor_names(out.columns, context=context)
    return out


def find_forbidden_aliases_in_payload(payload: Any) -> tuple[str, ...]:
    """Recursively find forbidden aliases in manifest/config feature values."""

    found: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, str):
            if value in FORBIDDEN_COMPATIBILITY_ALIASES:
                found.add(value)
        elif isinstance(value, Mapping):
            for key, item in value.items():
                visit(key)
                visit(item)
        elif isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                visit(item)

    visit(payload)
    return tuple(sorted(found))
