import pandas as pd
from train_overnight_unfiltered_comparison import partition_masks


def test_recent_fit_only_changes_training_start() -> None:
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2024-12-30", "2025-01-02", "2025-06-30", "2025-07-01", "2025-10-08"]),
        "exit_date": pd.to_datetime(["2024-12-31", "2025-01-03", "2025-07-01", "2025-07-02", "2025-10-09"]),
    })
    old = partition_masks(frame)
    recent = partition_masks(frame, recent_fit=True)
    assert old[0].tolist() == [True, True, False, False, False]
    assert recent[0].tolist() == [False, True, False, False, False]
    for i in [1, 2]:
        pd.testing.assert_series_equal(old[i], recent[i])
