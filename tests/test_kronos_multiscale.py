# ruff: noqa: E402
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from quant.research.kronos_multiscale import (
    SCALES,
    KronosEOS,
    asof_windows,
    context_and_target,
    period_bars,
    validate_bars,
)
from quant.research.sequence_strategy_validation import (
    capital_backtest,
    choose_rule,
    purged_split,
    wilson_lower,
)

ROOT = Path(__file__).resolve().parents[1]


def bars(n=1500):
    dates = pd.bdate_range("2015-01-01", periods=n)
    close = 10 + np.arange(n) * 0.01
    return pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close + 0.1,
            "low": close - 0.1,
            "close": close,
            "volume": np.ones(n) * 100,
        }
    )


def test_future_append_cannot_change_any_historical_scale():
    frame = validate_bars(bars())
    origin = frame.date.iloc[1400]
    lengths = {"daily": 128, "weekly": 104, "monthly": 60}
    prior = frame[frame.date <= origin]
    first = asof_windows({s: period_bars(prior, s) for s in SCALES}, origin, lengths)
    changed = frame.copy()
    changed.loc[changed.date > origin, ["open", "high", "low", "close", "volume"]] *= 100
    second = asof_windows({s: period_bars(changed, s) for s in SCALES}, origin, lengths)
    assert first is not None and second is not None
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


def test_incomplete_week_month_are_unavailable_even_when_full_source_is_present():
    frame = validate_bars(bars(100))
    week = period_bars(frame, "weekly")
    month = period_bars(frame, "monthly")
    date = pd.Timestamp("2015-03-11")
    assert week[week.available_at <= date].date.max() < pd.Timestamp("2015-03-09")
    assert month[month.available_at <= date].date.max() < pd.Timestamp("2015-03-01")


def test_pretraining_target_never_changes_context_normalization():
    frame = period_bars(validate_bars(bars(200)), "daily")
    x1, y1 = context_and_target(frame, 128, 128)
    frame.loc[129, ["open", "high", "low", "close", "volume"]] *= 2
    x2, y2 = context_and_target(frame, 128, 128)
    np.testing.assert_array_equal(x1, x2)
    assert not np.array_equal(y1, y2)


def test_eos_is_trained_and_all_three_branches_contribute():
    torch.manual_seed(42)
    model = KronosEOS(
        ROOT / "models/research/kronos/source", hidden=32, layers=1, heads=4, dropout=0
    )
    windows = [torch.randn(3, size, 5) for size in (10, 8, 6)]
    hidden = model.concatenate(windows)
    assert hidden.shape == (3, 96)
    loss = sum(model.next_bar(model(windows[s], s)).square().mean() for s in range(3))
    loss.backward()
    assert (model.eos.grad.abs().sum(dim=(1, 2)) > 0).all()
    assert model.project.weight.grad.abs().sum() > 0
    original = model(windows[0], 0).detach()
    windows[0][:, 0] += 100
    assert not torch.allclose(original, model(windows[0], 0))


def test_label_end_purge_and_feedback_selection_are_disjoint():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-12-30", "2025-06-27", "2025-07-01", "2026-01-02"]),
            "entry_date": pd.to_datetime(["2024-12-31", "2025-06-30", "2025-07-02", "2026-01-05"]),
            "exit_date": pd.to_datetime(["2025-01-08", "2025-07-07", "2025-07-09", "2026-01-12"]),
        }
    )
    split = purged_split(frame)
    assert split["train"].empty and split["feedback"].empty
    assert split["selection"].index.tolist() == [2]
    assert split["test"].index.tolist() == [3]


def test_all_cash_is_valid_when_no_leaf_has_positive_evidence():
    frame = pd.DataFrame(
        {
            "x": np.arange(100),
            "baseline_score": np.linspace(0, 1, 100),
            "net_return": np.full(100, -0.01),
        }
    )
    _, leaves, records = choose_rule(frame, frame, ["x"], supervised=False)
    assert leaves == [] and not any(row["eligible"] for row in records)
    assert wilson_lower(0, 0) == 0


def test_capital_cost_limits_no_double_holding_and_delayed_exit():
    dates = pd.bdate_range("2026-01-01", periods=5)
    history = pd.DataFrame(
        {
            "date": dates,
            "close": [10, 10, 11, 12, 12],
            "buy_ok": [False, True, True, True, True],
            "sell_ok": [True, True, False, True, True],
        }
    )
    events = pd.DataFrame(
        {
            "symbol": ["X", "X"],
            "date": [dates[0], dates[1]],
            "entry_date": [dates[1], dates[2]],
            "exit_date": [dates[2], dates[3]],
            "execution_score": [1.0, 1.0],
        }
    )
    curve, trades, metrics = capital_backtest(events, {"X": history}, dates, slots=1, cost=0.002)
    assert len(trades) == 1 and trades.exit_date.iloc[0] == dates[3]
    assert metrics["rejected_orders"] == 1 and curve.positions.max() == 1
    assert metrics["total_return"] == pytest.approx(1.2 * 0.999 / 1.001 - 1)
    assert metrics["max_drawdown"] < 0  # entry fee against initial cash peak
    empty = events.iloc[:0]
    curve, trades, metrics = capital_backtest(empty, {"X": history}, dates, slots=1)
    assert metrics["total_return"] == 0 and metrics["cash_day_ratio"] == 1
    assert metrics["win_rate"] is None


def test_confidence_gate_keeps_cash_when_high_probability_has_no_evidence():
    from quant.research.sequence_strategy_validation import choose_confidence_gate

    frame = pd.DataFrame(
        {
            "symbol": ["X"] * 100,
            "date": pd.bdate_range("2025-01-01", periods=100),
            "score": np.full(100, 0.99),
            "net_return": np.full(100, -0.01),
        }
    )
    threshold, audit = choose_confidence_gate(frame, "score")
    assert threshold is None and not any(row["eligible"] for row in audit)


def test_paired_blocks_compare_identical_dates_and_identical_strategies_have_zero_gain():
    from quant.research.sequence_strategy_validation import paired_block_gain

    dates = pd.bdate_range("2026-01-01", periods=30)
    curve = pd.DataFrame({"date": dates, "equity": np.cumprod(1 + np.sin(np.arange(30)) * 0.01)})
    result = paired_block_gain(curve, curve, repetitions=50)
    assert result["gain"] == 0 and result["paired_block_95ci"] == [0.0, 0.0]
    assert not result["positive_gain_supported"]
