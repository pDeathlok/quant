import pandas as pd
import pytest

from quant.research.overnight_context import add_watchlist_context


def test_context_uses_only_same_cutoff_inputs_not_labels_or_future_dates():
    dates = pd.to_datetime(["2025-01-02", "2025-01-02", "2025-01-03"])
    frame = pd.DataFrame({"symbol": ["A", "B", "A"], "date": dates,
        "cutoff": dates + pd.Timedelta(hours=14, minutes=55),
        "x_min_return_so_far": [.02, -.01, .5], "x_min_last30_return": [.01, 0., .2],
        "x_min_vwap_deviation": [.01, -.01, .1], "x_min_range_so_far": [.03, .02, .6],
        "x_min_amount_so_far": [100., 200., 300.], "target_net": [.1, -.2, .3],
        "entry_close_label": [100., 10., 200.], "decision_eligible": [True, False, True]})
    before = add_watchlist_context(frame)
    assert before.loc[0, "x_ctx_return_mean"] == pytest.approx(.005)
    assert before.loc[0, "x_ctx_positive_fraction"] == .5
    assert before.loc[0, "x_ctx_count"] == 2
    frame["target_net"] = 1e9
    frame["entry_close_label"] = -1e9
    frame["decision_eligible"] = False
    frame.loc[2, "x_min_return_so_far"] = -100.
    after = add_watchlist_context(frame)
    cols = [c for c in after if c.startswith("x_ctx_")]
    pd.testing.assert_frame_equal(before.loc[:1, cols], after.loc[:1, cols])
    with pytest.raises(ValueError, match="Duplicate"):
        add_watchlist_context(pd.concat([frame, frame]))
    frame.loc[0, "cutoff"] += pd.Timedelta(minutes=5)
    with pytest.raises(ValueError, match="14:55"):
        add_watchlist_context(frame)
