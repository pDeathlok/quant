import numpy as np
import pandas as pd

from quant.research import byd_daily_t_plan as daily_plan_module
from quant.research.byd_daily_t_plan import (
    POSITIVE_T_CONFIG,
    build_daily_t_plan,
    daily_t_features,
    simulate_daily_t,
)


def _daily(periods: int = 90) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=periods)
    close = 100 + np.sin(np.arange(periods) / 7) * 2
    return pd.DataFrame(
        {
            "date": dates,
            "open": close - 0.2,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1_000_000 + np.arange(periods) * 100,
        }
    )


def test_daily_features_do_not_change_past_rows_when_future_changes() -> None:
    original = _daily()
    changed = original.copy()
    changed.loc[changed.index[-1], ["high", "low", "close", "volume"]] = [180, 50, 160, 9_000_000]

    original_features = daily_t_features(original)
    changed_features = daily_t_features(changed)

    pd.testing.assert_frame_equal(original_features.iloc[:-1], changed_features.iloc[:-1])


def test_positive_t_does_not_count_same_five_minute_bar_target() -> None:
    bars = pd.DataFrame(
        [
            ("2026-01-05 09:35", 100.0, 100.2, 99.8, 100.0),
            ("2026-01-05 15:00", 100.0, 100.1, 99.9, 100.0),
            # Entry and target both appear inside this bar; their order is unknown.
            ("2026-01-06 09:35", 100.0, 101.0, 99.0, 99.5),
            ("2026-01-06 09:40", 99.5, 99.7, 99.2, 99.4),
            ("2026-01-06 15:00", 99.4, 99.5, 99.0, 99.2),
        ],
        columns=["datetime", "open", "high", "low", "close"],
    )

    result = simulate_daily_t(bars, POSITIVE_T_CONFIG)

    assert len(result) == 1
    assert bool(result.iloc[0]["target_hit"]) is False
    assert result.iloc[0]["exit_price"] == 99.2


def test_user_plan_separates_positive_and_reverse_and_keeps_reverse_disabled(monkeypatch) -> None:
    monkeypatch.setattr(
        daily_plan_module,
        "evaluate_positive_t",
        lambda *args, **kwargs: {
            "passed": True,
            "next_session_gate": True,
            "score": 88.0,
            "score_threshold": 60.0,
            "results": [],
        },
    )
    daily = _daily()

    plan = build_daily_t_plan(daily, pd.DataFrame(), shares=10500)

    assert plan["positive"]["execution_enabled"] is True
    assert plan["positive"]["shares"] == 500
    assert plan["positive"]["target_price"] > plan["positive"]["buy_price"]
    assert "不到不买" in plan["positive"]["no_fill_rule"]
    assert "14:50" in plan["positive"]["exit_rule"]
    assert plan["reverse"]["execution_enabled"] is False
    assert "没有反T参数" in plan["reverse"]["reason"]
    assert "高于满仓 500 股" in plan["inventory"]["note"]


def _session(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(
        [("2026-01-05 15:00", 100, 100, 100, 100), *rows],
        columns=["datetime", "open", "high", "low", "close"],
    )


def test_entry_cutoff_and_scheduled_exit_ignore_later_prices() -> None:
    bars = _session([
        ("2026-01-06 10:00", 100, 100, 99, 99.1),
        ("2026-01-06 14:50", 99.1, 99.4, 99, 99.3),
        ("2026-01-06 15:00", 99.3, 120, 99, 119),
        ("2026-01-07 14:55", 100, 100, 90, 95),
        ("2026-01-07 15:00", 95, 100, 94, 98),
    ])
    trades = simulate_daily_t(bars, POSITIVE_T_CONFIG)

    assert len(trades) == 1
    assert trades.iloc[0]["exit_price"] == 99.3
    assert trades.iloc[0]["exit_reason"] == "SESSION_EXIT"
    assert trades.iloc[0]["target_hit"] == False


def test_gap_entry_keeps_the_fixed_pre_market_target() -> None:
    bars = _session([
        ("2026-01-06 09:35", 98, 98.4, 97.9, 98.2),
        ("2026-01-06 09:40", 98.2, 99.1, 98, 98.8),
        ("2026-01-06 14:50", 98.8, 99, 98.6, 98.9),
    ])
    trade = simulate_daily_t(bars, POSITIVE_T_CONFIG).iloc[0]

    assert trade["target_price"] == 100.19
    assert not trade["target_hit"]


def test_stop_is_conservative_when_target_and_stop_share_a_bar() -> None:
    from dataclasses import replace

    config = replace(POSITIVE_T_CONFIG, stop_loss=.02)
    bars = _session([
        ("2026-01-06 09:35", 100, 100, 99, 99.3),
        ("2026-01-06 09:40", 99.3, 102, 96, 100),
        ("2026-01-06 14:50", 100, 101, 99, 100),
    ])
    trade = simulate_daily_t(bars, config).iloc[0]

    assert trade["exit_reason"] == "STOP"
    assert trade["exit_price"] == round(trade["entry_price"] * .98, 2)
    assert trade["pnl"] < 0


def test_no_fill_returns_a_typed_empty_trade_frame() -> None:
    bars = _session([("2026-01-06 14:50", 100, 100.1, 99.9, 100)])
    trades = simulate_daily_t(bars, POSITIVE_T_CONFIG)

    assert trades.empty
    assert trades.index.name == "date"
    assert {"pnl", "return_bps", "entry_time", "exit_time"} <= set(trades.columns)


def test_quarterly_scores_do_not_use_current_or_future_quarter_labels() -> None:
    dates = pd.bdate_range("2022-01-03", "2026-09-30")
    samples = pd.DataFrame(0.1, index=dates, columns=daily_plan_module.FEATURE_COLUMNS)
    samples["return_bps"] = np.sin(np.arange(len(samples))) * 100
    before = daily_plan_module.walk_forward_scores(samples, POSITIVE_T_CONFIG)
    changed = samples.copy()
    changed.loc["2026-07-01":, "return_bps"] = 100000
    after = daily_plan_module.walk_forward_scores(changed, POSITIVE_T_CONFIG)

    pd.testing.assert_frame_equal(before.loc[:"2026-09-30"], after.loc[:"2026-09-30"])
    assert (pd.to_datetime(before["training_end"]) < before.index).all()


def test_missing_latest_features_disables_plan_instead_of_using_an_older_row() -> None:
    daily = _daily()
    daily.loc[daily.index[-1], "volume"] = np.nan
    result = daily_plan_module.evaluate_positive_t(pd.DataFrame(), daily)

    assert result["passed"] is False
    assert result["next_session_gate"] is False
    assert result["status"] == "latest_features_missing"


def test_invalid_latest_close_cannot_fall_back_to_previous_session() -> None:
    daily = _daily()
    daily.loc[daily.index[-1], "close"] = np.nan

    result = daily_plan_module.evaluate_positive_t(pd.DataFrame(), daily)

    assert result["status"] == "latest_features_missing"
    assert result["next_session_gate"] is False


def test_plan_disables_unvalidated_smaller_lots_and_explains_capacity(monkeypatch) -> None:
    monkeypatch.setattr(daily_plan_module, "evaluate_positive_t", lambda *args: {
        "passed": True, "next_session_gate": True, "score": 30.,
        "score_threshold": 10., "results": [],
    })

    plan = build_daily_t_plan(_daily(), pd.DataFrame(), shares=11900)

    assert plan["positive"]["shares"] == 100
    assert plan["positive"]["execution_enabled"] is False
    assert "容量不足" in plan["positive"]["disabled_reason"]
