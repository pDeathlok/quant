import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/research/audit_overnight_opportunity.py"
SPEC = importlib.util.spec_from_file_location("overnight_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

DIRECT_SPEC = importlib.util.spec_from_file_location(
    "overnight_direct", SCRIPT.with_name("train_overnight_direct.py")
)
DIRECT = importlib.util.module_from_spec(DIRECT_SPEC)
DIRECT_SPEC.loader.exec_module(DIRECT)


def test_exit_rules_preserve_missing_and_opening_sensitivity():
    frame = pd.DataFrame({"y_open": [.03, 0, 0, np.nan],
                          "y_high": [.04, .0101, .005, np.nan],
                          "y_close": [-.04, -.03, -.01, np.nan]})
    result = MODULE.exit_returns(frame, cost_bps=20, penetration_bps=0, opening_improvement=True)
    assert result[:3].tolist() == pytest.approx([.028, .008, -.012])
    assert pd.isna(result.iloc[3])
    stress = MODULE.exit_returns(frame, cost_bps=40, penetration_bps=5, opening_improvement=False)
    assert stress[:3].tolist() == pytest.approx([.006, -.034, -.014])


def test_unknown_outcomes_are_not_cash_or_silently_removed():
    frame = pd.DataFrame({"date": pd.to_datetime(["2025-01-02"] * 2 + ["2025-01-03"]),
                          "y_high": [.02, np.nan, .03],
                          "corporate_action_gap": [False, True, False],
                          "exit_open_executable": [True, False, True]})
    result = MODULE.summarize(frame, pd.Series([.008, np.nan, .01]))
    assert result["signals"] == 3
    assert result["unresolved_outcomes"] == 1
    assert result["net_win_known"] == 1
    assert result["net_win_missing_as_failure"] == pytest.approx(2 / 3)
    assert result["complete_dates"] == 1
    assert result["equal_date_mean_complete_only"] == .01


def test_direct_selection_is_outcome_independent_and_respects_daily_slots():
    frame = pd.DataFrame({"symbol": ["A", "B", "C"],
                          "date": pd.to_datetime(["2025-01-02"] * 3),
                          "probability": [.9, .85, .99], "expected_net": [.005, .006, -.01],
                          "net": [-.1, .1, .2]})
    assert DIRECT.select(frame, .8, 0, 1).symbol.tolist() == ["A"]
    frame["net"] *= -1
    assert DIRECT.select(frame, .8, 0, 1).symbol.tolist() == ["A"]


def test_direct_metrics_reserve_cash_and_penalize_unknown_without_claiming_return():
    frame = pd.DataFrame({"date": pd.to_datetime(["2025-01-02"]), "net": [np.nan]})
    days = pd.date_range("2025-01-02", periods=2)
    result = DIRECT.metrics(frame, days, 5)
    assert result["unknown"] == 1
    assert result["allocation"] == .1
    assert result["daily_penalized_mean"] == pytest.approx(-.01)


def test_exact_limit_gate_rejects_limit_and_unknown_not_arbitrary_daily_return():
    close = pd.Series([10., 9.99, 9., 10., np.nan, 0.])
    upper = pd.Series([10., 10., np.nan, np.inf, 11., 1.])
    assert DIRECT.below_upper_limit(close, upper).tolist() == [False, True, False, False, False, False]


def test_risk_gate_uses_predictions_not_realized_losses():
    frame = pd.DataFrame({"symbol": ["A", "B", "C"],
                          "date": pd.to_datetime(["2025-01-02"] * 3),
                          "probability": [.9, .85, .99], "expected_net": [.005, .006, .01],
                          "risk": [.2, .04, np.nan], "net": [.1, -.1, .2]})
    assert DIRECT.select(frame, .8, 0, 1, maximum_risk=.05).symbol.tolist() == ["B"]
    frame["net"] *= -1
    assert DIRECT.select(frame, .8, 0, 1, maximum_risk=.05).symbol.tolist() == ["B"]


def test_barrier_exit_gap_priority_ambiguity_and_missing():
    frame = pd.DataFrame({"y_open": [.02, -.03, 0, 0, 0, 0, np.nan],
                          "y_high": [.03, .02, .02, .02, .003, .003, np.nan],
                          "y_low": [-.05, -.04, -.02, -.005, -.02, -.005, np.nan],
                          "y_close": [-.03, .01, .005, .01, -.005, .001, np.nan]})
    net, ambiguous = DIRECT.barrier_returns(frame, .01, .01)
    assert net.iloc[:6].tolist() == pytest.approx([.018, -.032, -.012, .008, -.012, -.001])
    assert pd.isna(net.iloc[6])
    assert ambiguous.tolist() == [False, False, True, False, False, False, False]
