from __future__ import annotations

import pandas as pd
import pytest

import quant.application.workspaces.byd as byd_workspace
from quant.application.workspaces.byd import (
    BydWorkspaceDependencies,
    build_byd_daily_strategy,
    load_byd_daily_frame,
)


def _daily_frame(date: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [pd.Timestamp(date)],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.0],
            "volume": [1_000_000.0],
        }
    )


def _intraday_frame(date: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": [pd.Timestamp(date) + pd.Timedelta(hours=15)],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.0],
            "volume": [10_000.0],
        }
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


def test_byd_workspace_rebuilds_stale_parameterized_snapshot() -> None:
    cached = {
        "generated_at": "2026-07-20T09:08:38",
        "planned_t": {"signal_date": "2026-07-17"},
    }
    writes: list[dict] = []

    dependencies = BydWorkspaceDependencies(
        read_snapshot=lambda *args, **kwargs: cached,
        write_snapshot=lambda workspace, snapshot_date, payload, **kwargs: writes.append(
            {
                "workspace": workspace,
                "snapshot_date": snapshot_date,
                "payload": payload,
                **kwargs,
            }
        ),
        is_snapshot_current=lambda payload: False,
        load_daily=lambda: _daily_frame("2026-08-07"),
        load_intraday=lambda: _intraday_frame("2026-07-17"),
        build_minute_payload=lambda **kwargs: {
            "generated_at": "2026-08-07T18:00:00",
            "holding": {"shares": kwargs["holding"].shares},
        },
        build_daily_plan=lambda **kwargs: {
            "signal_date": "2026-08-07",
            "positive": {
                "execution_enabled": False,
                "shares": 0,
                "buy_price": 0.0,
                "entry_rule": "等待",
                "exit_rule": "等待",
                "no_fill_rule": "不到价不成交",
            },
            "reverse": {"reason": "暂停"},
            "inventory": {"note": "保持基础仓"},
            "validation": {"status": "valid"},
        },
    )

    result = build_byd_daily_strategy(
        shares=10_500,
        cost=110.6061,
        dependencies=dependencies,
    )

    assert result is not cached
    assert result["planned_t"]["signal_date"] == "2026-08-07"
    assert writes[0]["params"] == {
        "plan_version": 4,
        "shares": 10_500,
        "cost": 110.6061,
        "intraday_training_max_staleness_days": 60,
    }
    assert writes[0]["write_sql"] is False


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
        load_daily=lambda: _daily_frame("2026-07-29"),
        load_intraday=lambda: _intraday_frame("2026-07-17"),
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
    assert result["planned_t"]["feature_freshness"] == {
        "daily_feature_date": "2026-07-29",
        "expected_daily_feature_date": "2026-07-29",
        "daily_feature_current": True,
        "daily_feature_source": "provided_dependency",
        "intraday_training_max_date": "2026-07-17",
        "intraday_training_role": "historical_model_training_and_validation_samples",
        "intraday_is_current_feature": False,
        "intraday_training_staleness_days": 12,
        "intraday_training_max_staleness_days": 60,
        "intraday_training_within_sla": True,
        "staleness_unit": "calendar_days",
    }
    assert "today_t" not in result
    assert writes == [
        {
            "workspace": "byd_daily_plan",
            "snapshot_date": "2026-07-29",
            "payload": result,
            "params": {
                "plan_version": 4,
                "shares": 10_500,
                "cost": 108.0,
                "intraday_training_max_staleness_days": 60,
            },
            "write_sql": True,
        }
    ]


def test_load_byd_daily_frame_rejects_unverified_stale_fallback(monkeypatch, tmp_path) -> None:
    class FakeStore:
        def read_frame(self, *args, **kwargs):
            raise RuntimeError("canonical unavailable")

        def latest_dataset_trade_date(self, *args, **kwargs):
            return pd.Timestamp("2026-08-12")

    monkeypatch.setattr(byd_workspace, "MarketDataStore", lambda *args, **kwargs: FakeStore())
    monkeypatch.setattr(
        byd_workspace,
        "load_daily_qfq",
        lambda _cache_dir: _daily_frame("2026-06-16"),
    )

    with pytest.raises(RuntimeError, match="expected=2026-08-12 actual=2026-06-16"):
        load_byd_daily_frame(daily_dir=tmp_path / "daily", cache_dir=tmp_path / "cache")


def test_byd_workspace_rejects_daily_feature_date_different_from_expected() -> None:
    dependencies = BydWorkspaceDependencies(
        read_snapshot=lambda *args, **kwargs: None,
        write_snapshot=lambda *args, **kwargs: None,
        load_daily=lambda: _daily_frame("2026-08-11"),
        load_intraday=lambda: _intraday_frame("2026-07-17"),
        build_minute_payload=lambda **kwargs: {},
        build_daily_plan=lambda **kwargs: pytest.fail("stale daily features must fail first"),
    )

    with pytest.raises(RuntimeError, match="daily feature date mismatch"):
        build_byd_daily_strategy(
            expected_trade_date="2026-08-12",
            dependencies=dependencies,
        )


def test_byd_workspace_rejects_intraday_training_history_beyond_configured_sla() -> None:
    dependencies = BydWorkspaceDependencies(
        read_snapshot=lambda *args, **kwargs: None,
        write_snapshot=lambda *args, **kwargs: None,
        load_daily=lambda: _daily_frame("2026-08-12"),
        load_intraday=lambda: _intraday_frame("2026-05-01"),
        build_minute_payload=lambda **kwargs: {},
        build_daily_plan=lambda **kwargs: pytest.fail("stale training history must fail first"),
    )

    with pytest.raises(RuntimeError, match="intraday training history exceeds freshness SLA"):
        build_byd_daily_strategy(
            expected_trade_date="2026-08-12",
            intraday_training_max_staleness_days=30,
            dependencies=dependencies,
        )
