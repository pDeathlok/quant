"""Guards for daily extension parity and current-bar entry eligibility."""

import numpy as np
import pandas as pd
from build_overnight_event_extension_inputs import compare
from predict_overnight_event_extension import entry_gate


def test_boundary_comparison_preserves_nan_and_detects_differences() -> None:
    left = pd.DataFrame({"symbol": ["a", "b"], "date": pd.to_datetime(["2026-09-01"] * 2), "f": [np.nan, 1.]})
    assert compare(left, left.copy(), ["f"])[0]["different_cells"] == 0
    right = left.copy()
    right.loc[1, "f"] = 2.
    assert compare(left, right, ["f"])[0]["different_cells"] == 1


def test_entry_gate_rejects_missing_bar_upper_limit_and_invalid_ohlc() -> None:
    frame = pd.DataFrame({"symbol": list("abcd"), "date": pd.to_datetime(["2026-09-10"] * 4),
                          "up_limit": [11.] * 4, "down_limit": [9.] * 4})
    for column in ["open", "high", "low", "close", "pre_close"]:
        frame[f"today_raw_{column}"] = 10.
    frame["today_raw_vol"] = 100.
    frame.loc[1, "today_raw_close"] = np.nan
    frame.loc[2, ["today_raw_close", "today_raw_high"]] = 11.
    frame.loc[3, "today_raw_high"] = 9.
    result = entry_gate(frame)
    assert result.eligible.tolist() == [True, False, False, False]
    assert result.unknown_entry.tolist() == [False, True, False, True]
    assert result.at_upper.tolist() == [False, False, True, False]
    # Future outcomes are not inputs to the eligibility calculation.
    frame["next_close"] = [0., 100., 0., 100.]
    pd.testing.assert_frame_equal(result, entry_gate(frame))
