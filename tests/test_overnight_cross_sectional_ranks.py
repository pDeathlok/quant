import numpy as np
import pandas as pd
import pytest
from train_overnight_unfiltered_comparison import RANK_INPUTS, cross_sectional_ranks


def test_rank_ties_missing_and_date_isolation() -> None:
    frame = pd.DataFrame({"symbol": ["a", "b", "c", "d", "a"],
                          "date": pd.to_datetime(["2025-07-01"] * 4 + ["2025-07-02"])})
    for name in RANK_INPUTS:
        frame[name] = [1., 1., 3., np.nan, -100.]
    result = cross_sectional_ranks(frame)
    assert len(result.columns) == 12
    np.testing.assert_allclose(result.iloc[:, 0], [.5, .5, 1., np.nan, 1.], equal_nan=True)
    frame.loc[4, RANK_INPUTS] = 9999.
    frame["next_return"] = [.9, -.9, .8, -.8, .1]
    pd.testing.assert_frame_equal(result, cross_sectional_ranks(frame))
    with pytest.raises(ValueError):
        cross_sectional_ranks(pd.concat([frame, frame.iloc[:1]], ignore_index=True))
