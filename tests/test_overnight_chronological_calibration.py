import numpy as np
import pandas as pd
from audit_overnight_probability_support import chronological_calibrators


def test_validation_and_boundary_outcomes_cannot_change_calibration() -> None:
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2025-07-01", "2025-07-02", "2025-07-31", "2025-08-01"]),
        "exit_date": pd.to_datetime(["2025-07-02", "2025-07-03", "2025-08-01", "2025-08-04"]),
        "net": [-.05, .02, -.5, .9], "raw_probability": [.2, .8, .4, .9],
        "raw_risk": [.9, .1, .3, .1],
    })
    baseline = chronological_calibrators(frame, "2025-08-01")
    frame.loc[2:, "net"] = [.99, -.99]
    replay = chronological_calibrators(frame, "2025-08-01")
    for old, new in zip(baseline, replay, strict=True):
        np.testing.assert_array_equal(old.X_thresholds_, new.X_thresholds_)
        np.testing.assert_array_equal(old.y_thresholds_, new.y_thresholds_)


def test_event_calibration_uses_event_not_profitable_fallback() -> None:
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2025-07-01", "2025-07-02", "2025-07-03", "2025-08-01"]),
        "exit_date": pd.to_datetime(["2025-07-02", "2025-07-03", "2025-07-04", "2025-08-04"]),
        "net": [-.05, .002, .008, .008], "event_hit": [False, False, True, True],
        "raw_probability": [.1, .5, .9, .9], "raw_risk": [.9, .3, .1, .1],
    })
    event, _ = chronological_calibrators(frame, "2025-08-01")
    profit, _ = chronological_calibrators(frame.drop(columns="event_hit"), "2025-08-01")
    assert event.predict([.5])[0] == 0
    assert profit.predict([.5])[0] == 1
    frame.loc[3, "event_hit"] = False
    replay, _ = chronological_calibrators(frame, "2025-08-01")
    np.testing.assert_array_equal(event.y_thresholds_, replay.y_thresholds_)
