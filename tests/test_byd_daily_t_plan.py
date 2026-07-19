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
