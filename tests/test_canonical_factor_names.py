from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.features.canonical_factor_names import (
    FORBIDDEN_COMPATIBILITY_ALIASES,
    assert_no_forbidden_factor_names,
    find_forbidden_aliases_in_payload,
    migrate_legacy_factor_columns,
    stable_canonical_feature_union,
)


def test_migration_renames_alias_only_columns_and_removes_old_names() -> None:
    source = pd.DataFrame(
        {
            "price_level": [10.0, np.nan],
            "bb_middle": [9.5, 9.7],
            "keep": [1, 2],
        }
    )

    migrated = migrate_legacy_factor_columns(source, context="legacy cache")

    assert list(migrated.columns) == ["close", "ma_20", "keep"]
    assert migrated["close"].isna().tolist() == [False, True]
    assert FORBIDDEN_COMPATIBILITY_ALIASES.isdisjoint(migrated.columns)
    assert list(source.columns) == ["price_level", "bb_middle", "keep"]


def test_migration_validates_equal_alias_and_canonical_values_then_drops_alias() -> None:
    source = pd.DataFrame(
        {
            "rs_pct_chg_1d": [1.0, np.nan, -2.0],
            "pct_chg": [1.0, np.nan, -2.0],
        }
    )

    migrated = migrate_legacy_factor_columns(source, context="paired cache")

    assert list(migrated.columns) == ["pct_chg"]


@pytest.mark.parametrize(
    "alias_values,canonical_values,match",
    [
        ([1.0, np.nan], [1.0, 2.0], "NaN position differs"),
        ([1.0, 2.0], [1.0, 3.0], "value differs"),
    ],
)
def test_migration_fails_closed_when_alias_values_drift(
    alias_values: list[float],
    canonical_values: list[float],
    match: str,
) -> None:
    source = pd.DataFrame(
        {
            "rs_amplitude_pct": alias_values,
            "amplitude_1": canonical_values,
        }
    )

    with pytest.raises(ValueError, match=match):
        migrate_legacy_factor_columns(source, context="drifted cache")


def test_stable_union_rejects_aliases_and_deduplicates_canonical_names() -> None:
    assert stable_canonical_feature_union(
        ("close", "ma_20"),
        ("ma_20", "pct_chg"),
    ) == ("close", "ma_20", "pct_chg")

    with pytest.raises(ValueError, match="forbidden compatibility aliases"):
        stable_canonical_feature_union(("close",), ("price_level",))


def test_manifest_payload_scan_and_direct_assertion_find_every_alias() -> None:
    payload = {
        "features": ["close", "bb_middle"],
        "nested": {"selected": ["rs_vol_ratio_5_inclusive"]},
    }

    assert find_forbidden_aliases_in_payload(payload) == (
        "bb_middle",
        "rs_vol_ratio_5_inclusive",
    )
    with pytest.raises(ValueError, match="price_level"):
        assert_no_forbidden_factor_names(
            ("close", "price_level"),
            context="production manifest",
        )
