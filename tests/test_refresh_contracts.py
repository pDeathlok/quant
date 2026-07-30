from __future__ import annotations

import pytest

from quant.application.refresh_contracts import (
    build_progress_steps,
    normalize_refresh_scope,
)


@pytest.mark.parametrize(
    ("raw_scope", "expected"),
    [
        (None, "all"),
        ("", "all"),
        ("short", "short"),
        ("cb-allotment", "cbAllotment"),
        ("allotment", "cbAllotment"),
        ("convertible_bond", "cb"),
        ("convertible-bond", "cb"),
    ],
)
def test_normalize_refresh_scope_supports_public_aliases(
    raw_scope: str | None,
    expected: str,
) -> None:
    assert normalize_refresh_scope(raw_scope) == expected


def test_normalize_refresh_scope_rejects_unknown_scope() -> None:
    with pytest.raises(ValueError, match="未知刷新范围"):
        normalize_refresh_scope("unknown")


def test_build_progress_steps_returns_fresh_mutable_records() -> None:
    first = build_progress_steps("chan")
    second = build_progress_steps("chan")

    assert [step["key"] for step in first] == ["refresh_data", "chan_model_strategy"]
    assert all(step["status"] == "pending" for step in first)
    first[0]["status"] = "success"
    assert second[0]["status"] == "pending"
