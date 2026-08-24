"""Stable strategy-group ownership for unified short-horizon rankers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd


LEFT_GROUP_MEMBERS: Mapping[str, tuple[str, ...]] = {
    "B1": ("B1",),
    "SB1": ("SB1",),
    "SUPER_B1": ("SUPER_B1",),
    "LOW_PULLBACK": ("YIDONG_DILIAN", "NANA", "DUICHEN_VA"),
}

RIGHT_GROUP_MEMBERS: Mapping[str, tuple[str, ...]] = {
    "B2": ("B2",),
    "B3": ("B3",),
    "STRONG_K": ("KEY_K", "VIOLENCE_K"),
    "DOUBLE_YANG": ("PINGHANG", "DOUBLE_GUN"),
    "CHANGAN": ("CHANGAN",),
    "KENGQI": ("KENGQI",),
    "VEGAS": ("VEGAS",),
    "TRIPLE_VOLUME_BREAKOUT": ("TRIPLE_VOLUME_BREAKOUT",),
}

MIXED_GROUP_MEMBERS: Mapping[str, tuple[str, ...]] = {
    "SUPPORT_PULLBACK": ("GOLDEN_BOWL", "ZAIHOU"),
    "RHYTHM_PLATFORM": ("BREATHING", "YUEYUE"),
}

LEFT_GROUPS: tuple[str, ...] = tuple(LEFT_GROUP_MEMBERS)
RIGHT_GROUPS: tuple[str, ...] = tuple(RIGHT_GROUP_MEMBERS)
MIXED_GROUPS: tuple[str, ...] = tuple(MIXED_GROUP_MEMBERS)
RIGHT_WITH_MIXED_GROUPS: tuple[str, ...] = (*RIGHT_GROUPS, *MIXED_GROUPS)
LEFT_WITH_MIXED_GROUPS: tuple[str, ...] = (*LEFT_GROUPS, *MIXED_GROUPS)
ALL_SHORT_GROUPS: tuple[str, ...] = (
    *LEFT_GROUPS,
    *RIGHT_GROUPS,
    *MIXED_GROUPS,
)

GROUP_MEMBERS: Mapping[str, tuple[str, ...]] = {
    **LEFT_GROUP_MEMBERS,
    **RIGHT_GROUP_MEMBERS,
    **MIXED_GROUP_MEMBERS,
}

GROUP_LABELS: Mapping[str, str] = {
    "B1": "B1",
    "SB1": "SB1",
    "SUPER_B1": "超级B1",
    "LOW_PULLBACK": "缩量回调低吸",
    "B2": "B2",
    "B3": "B3",
    "STRONG_K": "强K/突破",
    "DOUBLE_YANG": "双阳结构",
    "CHANGAN": "长安战法",
    "KENGQI": "坑里起好货",
    "VEGAS": "维加斯隧道",
    "TRIPLE_VOLUME_BREAKOUT": "三倍量突破",
    "SUPPORT_PULLBACK": "支撑回踩",
    "RHYTHM_PLATFORM": "节奏/平台",
}

GROUP_SIDE: Mapping[str, str] = {
    **{group: "left" for group in LEFT_GROUPS},
    **{group: "right" for group in RIGHT_GROUPS},
    **{group: "mixed" for group in MIXED_GROUPS},
}


def aggregate_strategy_group_flags(
    raw_flags: pd.DataFrame,
    *,
    groups: Sequence[str] = ALL_SHORT_GROUPS,
    require_all_members: bool = True,
) -> pd.DataFrame:
    """OR raw detector members into stable group flags."""

    output = pd.DataFrame(index=raw_flags.index)
    for group in groups:
        if group not in GROUP_MEMBERS:
            raise ValueError(f"unknown short strategy group: {group}")
        members = GROUP_MEMBERS[group]
        missing = [member for member in members if member not in raw_flags]
        if missing and require_all_members:
            raise ValueError(f"strategy group {group} missing raw members: {missing}")
        present = [member for member in members if member in raw_flags]
        output[group] = (
            raw_flags[present].fillna(False).astype(bool).any(axis=1)
            if present
            else False
        )
    return output


def validate_short_group_contract() -> None:
    if len(ALL_SHORT_GROUPS) != 14 or len(set(ALL_SHORT_GROUPS)) != 14:
        raise RuntimeError("short strategy group contract must contain 14 unique groups")
    if len(LEFT_GROUPS) != 4 or len(RIGHT_GROUPS) != 8 or len(MIXED_GROUPS) != 2:
        raise RuntimeError("short strategy side contract must be left=4/right=8/mixed=2")
    if set(GROUP_LABELS) != set(ALL_SHORT_GROUPS):
        raise RuntimeError("short strategy group labels drifted")
    if set(GROUP_SIDE) != set(ALL_SHORT_GROUPS):
        raise RuntimeError("short strategy side ownership drifted")


validate_short_group_contract()


__all__ = [
    "ALL_SHORT_GROUPS",
    "GROUP_LABELS",
    "GROUP_MEMBERS",
    "GROUP_SIDE",
    "LEFT_GROUPS",
    "LEFT_GROUP_MEMBERS",
    "LEFT_WITH_MIXED_GROUPS",
    "MIXED_GROUPS",
    "MIXED_GROUP_MEMBERS",
    "RIGHT_GROUPS",
    "RIGHT_GROUP_MEMBERS",
    "RIGHT_WITH_MIXED_GROUPS",
    "aggregate_strategy_group_flags",
    "validate_short_group_contract",
]
