import pandas as pd
from audit_overnight_probability_support import support


def test_support_distinguishes_rows_from_dates() -> None:
    frame = pd.DataFrame({"date": pd.to_datetime(["2025-07-01"] * 9 + ["2025-07-02"]),
                          "net": [.01] * 9 + [-.2]})
    result = support(frame)
    assert result["rows"] == 10
    assert result["dates"] == 2
    assert result["largest_date_share"] == .9
    assert result["win_rate"] == .9
    assert result["mean_net"] < 0
    assert support(frame.iloc[:0])["dates"] == 0
