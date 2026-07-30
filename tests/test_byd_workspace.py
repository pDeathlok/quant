from __future__ import annotations

import pandas as pd

from quant.application.workspaces.byd import (
    BydWorkspaceDependencies,
    build_byd_daily_strategy,
)


def test_byd_workspace_returns_cached_payload_without_loading_market_data() -> None:
    cached = {"generated_at": "2026-07-30T08:00:00", "planned_t": {"signal_date": "2026-07-29"}}
    dependencies = BydWorkspaceDependencies(
        read_snapshot=lambda *args, **kwargs: cached,
        write_snapshot=lambda *args, **kwargs: None,
        load_daily=lambda: (_ for _ in ()).throw(AssertionError("daily data should not load")),
        load_intraday=lambda: (_ for _ in ()).throw(AssertionError("intraday data should not load")),
    )

    result = build_byd_daily_strategy(dependencies=dependencies)

    assert result is cached


def test_byd_workspace_builds_and_persists_daily_plan() -> None:
    writes: list[dict] = []

    def build_minute_payload(**kwargs):
        return {
            "generated_at": "2026-07-30T08:00:00",
            "today_t": {"obsolete": True},
            "holding": {"shares": kwargs["holding"].shares},
        }

    def build_daily_plan(**kwargs):
        return {
            "signal_date": "2026-07-29",
            "positive": {
                "execution_enabled": True,
                "shares": 500,
                "buy_price": 108.2,
                "entry_rule": "回落到计划价再买",
                "exit_rule": "成交后按目标价卖出",
                "no_fill_rule": "不到价不成交",
            },
            "reverse": {"reason": "反T未通过闸门"},
            "inventory": {"note": "保持基础仓"},
            "validation": {"status": "valid"},
        }

    dependencies = BydWorkspaceDependencies(
        read_snapshot=lambda *args, **kwargs: None,
        write_snapshot=lambda workspace, snapshot_date, payload, **kwargs: writes.append(
            {
                "workspace": workspace,
                "snapshot_date": snapshot_date,
                "payload": payload,
                **kwargs,
            }
        ),
        load_daily=lambda: pd.DataFrame({"close": [100.0]}),
        load_intraday=pd.DataFrame,
        build_minute_payload=build_minute_payload,
        build_daily_plan=build_daily_plan,
    )

    result = build_byd_daily_strategy(
        shares=10_500,
        cost=108.0,
        refresh=True,
        dependencies=dependencies,
    )

    assert result["primary_action"]["action"] == "BUY_LIMIT"
    assert result["stage"]["key"] == "OVERWEIGHT"
    assert result["planned_t"]["signal_date"] == "2026-07-29"
    assert "today_t" not in result
    assert writes == [
        {
            "workspace": "byd_daily_plan",
            "snapshot_date": "2026-07-29",
            "payload": result,
            "params": {"plan_version": 3, "shares": 10_500, "cost": 108.0},
            "write_sql": True,
        }
    ]
