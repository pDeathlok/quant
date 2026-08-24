from __future__ import annotations

import pandas as pd

from quant.research.short_side_groups import (
    ALL_SHORT_GROUPS,
    GROUP_MEMBERS,
    LEFT_GROUPS,
    MIXED_GROUPS,
    RIGHT_GROUPS,
    aggregate_strategy_group_flags,
)


def test_short_side_group_partition_matches_strategy_contract() -> None:
    assert LEFT_GROUPS == ("B1", "SB1", "SUPER_B1", "LOW_PULLBACK")
    assert RIGHT_GROUPS == (
        "B2",
        "B3",
        "STRONG_K",
        "DOUBLE_YANG",
        "CHANGAN",
        "KENGQI",
        "VEGAS",
        "TRIPLE_VOLUME_BREAKOUT",
    )
    assert MIXED_GROUPS == ("SUPPORT_PULLBACK", "RHYTHM_PLATFORM")
    assert len(ALL_SHORT_GROUPS) == 14
    assert len(set(ALL_SHORT_GROUPS)) == 14


def test_group_aggregation_ors_raw_strategy_members() -> None:
    raw_columns = sorted({member for members in GROUP_MEMBERS.values() for member in members})
    frame = pd.DataFrame(False, index=range(2), columns=raw_columns)
    frame.loc[0, "KEY_K"] = True
    frame.loc[1, "VIOLENCE_K"] = True
    frame.loc[1, "GOLDEN_BOWL"] = True

    grouped = aggregate_strategy_group_flags(frame)

    assert grouped["STRONG_K"].tolist() == [True, True]
    assert grouped["SUPPORT_PULLBACK"].tolist() == [False, True]
    assert not grouped.drop(columns=["STRONG_K", "SUPPORT_PULLBACK"]).to_numpy().any()
