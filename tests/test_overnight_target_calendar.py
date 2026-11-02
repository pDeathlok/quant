import importlib
from pathlib import Path

import pandas as pd
import pytest


def test_calendar_month_boundary_and_weekend_gap(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts/research"))
    fn = importlib.import_module("build_overnight_target_calendar").target_calendar
    dates = pd.to_datetime(["2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02", "2024-02-05", "2024-02-06"])
    f = fn(dates, dates[[0, 1, 3]])
    assert f.calendar_target_last_month_session.tolist() == [True, False, False]
    assert f.calendar_target_first_month_session.tolist() == [False, True, False]
    assert f.calendar_target_gap_days.tolist() == [1, 1, 3]
    assert f.calendar_target_weekday.tolist() == [2, 3, 0]
    with pytest.raises(ValueError, match="two known"):
        fn(dates, dates[-1:])
    with pytest.raises(ValueError, match="market sessions"):
        fn(dates, pd.to_datetime(["2024-02-03"]))


@pytest.mark.parametrize("sessions", [
    ["2024-03-28", "2024-03-29", "2024-04-01", "2024-04-02"],
    ["2023-12-28", "2023-12-29", "2024-01-02", "2024-01-03"],
])
def test_quarter_and_year_boundaries_use_market_sessions(monkeypatch, sessions):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts/research"))
    fn = importlib.import_module("build_overnight_target_calendar").target_calendar
    dates = pd.to_datetime(sessions)
    f = fn(dates, dates[:2])
    assert f.calendar_target_last_quarter_session.tolist() == [True, False]
    assert f.calendar_target_first_quarter_session.tolist() == [False, True]
