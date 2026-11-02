import numpy as np
import pandas as pd
import pytest
from validate_overnight_sigmoid_calibration import fit_sigmoid, logit_input


def test_sigmoid_timing_and_finite_endpoints() -> None:
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2025-07-01", "2025-07-02", "2025-07-31", "2025-08-01"]),
        "exit_date": pd.to_datetime(["2025-07-02", "2025-07-03", "2025-08-01", "2025-08-04"]),
        "net": [-.05, .02, -.5, .9], "raw_probability": [.2, .8, .4, .9],
        "raw_risk": [.9, .1, .3, .1],
    })
    old = fit_sigmoid(frame, "2025-08-01")
    frame.loc[2:, "net"] = [.99, -.99]
    new = fit_sigmoid(frame, "2025-08-01")
    for key in old:
        np.testing.assert_array_equal(old[key].coef_, new[key].coef_)
        np.testing.assert_array_equal(old[key].intercept_, new[key].intercept_)
    assert np.isfinite(logit_input(pd.Series([0., 1.]))).all()
    with pytest.raises(ValueError):
        logit_input(pd.Series([-.1, np.nan]))
