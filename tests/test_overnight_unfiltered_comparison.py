import importlib
from pathlib import Path

import numpy as np
import pandas as pd


def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts/research"))
    return importlib.import_module("train_overnight_unfiltered_comparison")


def test_boundary_labels_are_purged_but_unknown_test_outcomes_remain(monkeypatch):
    m = module(monkeypatch)
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2025-06-27", "2025-06-30", "2025-07-01", "2025-09-30", "2025-10-09", "2026-09-03", "2026-09-04"]),
        "exit_date": pd.to_datetime(["2025-06-30", "2025-07-01", "2025-07-02", "2025-10-09", "2025-10-10", "2026-09-04", None]),
        "limit05": [0.003, 0.003, -0.01, 0.003, np.nan, np.nan, np.nan],
    })
    train, cal, evaluation = m.partition_masks(frame)
    assert train.tolist() == [True, False, False, False, False, False, False]
    assert cal.tolist() == [False, False, True, False, False, False, False]
    assert evaluation.tolist() == [False, False, False, False, True, True, False]


def test_date_weights_give_equal_total_mass_per_session(monkeypatch):
    m = module(monkeypatch)
    frame = pd.DataFrame({"date": ["a", "b", "b", "b"]})
    weights = m.date_weights(frame)
    np.testing.assert_allclose(weights.mean(), 1, rtol=1e-14, atol=0)
    np.testing.assert_allclose(weights.groupby(frame.date).sum(), [2, 2])


def test_early_calibration_separates_fit_calibration_and_evaluation(monkeypatch):
    m = module(monkeypatch)
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2024-12-30", "2024-12-31", "2025-01-02", "2025-06-30", "2025-07-01", "2025-09-30", "2025-10-09"]),
        "exit_date": pd.to_datetime(["2024-12-31", "2025-01-02", "2025-01-03", "2025-07-01", "2025-07-02", "2025-10-09", "2025-10-10"]),
    })
    train, cal, evaluation = m.partition_masks(frame, early_calibration=True)
    assert train.tolist() == [True, False, False, False, False, False, False]
    assert cal.tolist() == [False, False, True, False, False, False, False]
    assert evaluation.tolist() == [False, False, False, False, True, False, True]


def test_q4_evaluation_matches_original_year_end_boundary(monkeypatch):
    m = module(monkeypatch)
    frame = pd.DataFrame({"date": pd.to_datetime(["2025-12-30", "2025-12-31", "2026-01-05"]),
                          "exit_date": pd.to_datetime(["2025-12-31", "2026-01-05", "2026-01-06"])})
    _, _, evaluation = m.partition_masks(frame)
    assert evaluation.tolist() == [True, False, True]


def test_selection_does_not_use_future_exit_flags_or_returns(monkeypatch):
    m = module(monkeypatch)
    frame = pd.DataFrame({"symbol": ["a", "b"], "date": pd.to_datetime(["2026-01-05"] * 2),
                          "probability": [.9, .8], "expected_net": [.003, .004], "risk": [.05, .06],
                          "net": [np.nan, .003], "limit05_exit_uncertain": [True, False]})
    original = m.select(frame, .7, -np.inf, 1, .1)
    changed = m.select(frame.assign(net=[.5, -.2], limit05_exit_uncertain=[False, True]), .7, -np.inf, 1, .1)
    assert original.symbol.tolist() == changed.symbol.tolist() == ["a"]


def test_direct_exit_audit_is_target_specific_and_preserves_unknowns(monkeypatch):
    m = module(monkeypatch)
    frame = pd.DataFrame({"next_open": [9., 9.1, 9.1, 9.1, np.nan],
                          "next_close": [9.2, 9., 9.1, 9.1, np.nan],
                          "net_open": [-.102, -.092, -.092, -.092, np.nan],
                          "net_close": [-.082, -.102, -.092, -.092, np.nan],
                          "down_limit": [9., 9., np.nan, 9., 9.],
                          "price_limit_inconsistent": [False, False, False, True, False]})
    opened = m.direct_exit_flags(frame, "net_open")
    closed = m.direct_exit_flags(frame, "net_close")
    assert opened.net_open_exit_uncertain.tolist() == [True, False, True, True, True]
    assert closed.net_close_exit_uncertain.tolist() == [False, True, True, True, True]
    assert opened.net_open_net_after_exit_audit.notna().tolist() == [False, True, False, False, False]
    assert frame.net_open.iloc[0] == -.102
