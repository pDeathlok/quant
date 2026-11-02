import pandas as pd
import pytest
from train_right_t012 import SIGNALS, choose, map_entries, summarize


def test_global_calendar_mapping_and_missing_stock_day() -> None:
    days = pd.to_datetime(["2025-09-05", "2025-09-08", "2025-09-09"])
    events = pd.DataFrame({"symbol": ["000001.SZ"], "date": [days[0]],
                           **{s: [s == "B2"] for s in SIGNALS}})
    mapped = map_entries(events, pd.DatetimeIndex(days))
    assert mapped.date.iloc[0] == days[1]
    assert mapped.planned_exit_date.iloc[0] == days[2]
    # A missing individual bar does not change the global entry date.
    bars = pd.DataFrame({"symbol": ["000001.SZ"], "date": [days[2]], "close": [10.]})
    merged = mapped.merge(bars, on=["symbol", "date"], how="left")
    assert merged.close.isna().all()
    with pytest.raises(ValueError, match="Invalid event keys"):
        map_entries(pd.concat([events, events]), pd.DatetimeIndex(days))


def test_selection_never_uses_future_outcomes() -> None:
    frame = pd.DataFrame({"symbol": ["a", "b"], "date": pd.to_datetime(["2025-09-08"] * 2),
                          "probability": [.8, .7], "expected_net": [.01, .02], "net": [-.1, .1]})
    original = choose(frame, .6, "probability")
    changed = choose(frame.assign(net=[.1, -.1]), .6, "probability")
    assert original.symbol.tolist() == changed.symbol.tolist() == ["a"]
    assert choose(frame, .9, "probability").empty


def test_empty_selection_is_valid_abstention() -> None:
    frame = pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "net": pd.Series(dtype=float),
                          "event_hit": pd.Series(dtype="boolean")})
    result = summarize(frame, pd.date_range("2025-09-01", periods=20))
    assert result["signals"] == 0
    assert result["unknown"] == 0
