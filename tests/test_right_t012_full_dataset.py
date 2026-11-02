import pandas as pd
import pytest
from build_right_t012_full_dataset import asof_history, episode_age, historical_group_flag
from train_right_t012_full_registry import sample_weights, split_masks


def test_historical_join_excludes_same_day_and_future() -> None:
    keys = pd.DataFrame({"symbol": ["a", "a"], "date": pd.to_datetime(["2025-01-03", "2025-02-03"])})
    source = pd.DataFrame({"symbol": ["a", "a", "a"],
                           "date": pd.to_datetime(["2025-01-02", "2025-01-03", "2025-02-04"]),
                           "value": [1., 99., 999.]})
    result = asof_history(keys, source)
    assert result.value.iloc[0] == 1.
    assert pd.isna(result.value.iloc[1])
    with pytest.raises(ValueError, match="Duplicate"):
        asof_history(keys, pd.concat([source, source]))


def test_episode_age_uses_only_past_consecutive_market_sessions() -> None:
    calendar = pd.bdate_range("2025-01-02", "2025-01-10")
    events = pd.DataFrame({"symbol": ["a"] * 4,
                           "screen_date": pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-07", "2025-01-08"])})
    assert episode_age(events, calendar).tolist() == [1, 2, 1, 2]
    pd.testing.assert_series_equal(episode_age(events, calendar).iloc[:2], episode_age(events.iloc[:2], calendar))


def test_model_stages_purge_boundary_exits() -> None:
    frame = pd.DataFrame({"date": pd.to_datetime(["2024-12-30", "2024-12-31", "2025-01-02", "2025-04-01", "2025-07-01"]),
                          "planned_exit_date": pd.to_datetime(["2024-12-31", "2025-01-02", "2025-01-03", "2025-04-02", "2025-07-02"])})
    masks = split_masks(frame, ("2025-01-01", "2025-04-01", "2025-07-01", "2026-01-01"))
    assert masks["fit"].tolist() == [True, False, False, False, False]
    assert masks["calibration"].tolist() == [False, False, True, False, False]
    assert masks["selection"].tolist() == [False, False, False, True, False]
    assert masks["diagnostic"].tolist() == [False, False, False, False, True]


def test_group_aggregate_and_unavailable_members() -> None:
    frame = pd.DataFrame({"LOW_PULLBACK": [True, None, False], "PINGHANG": [True, False, None]})
    low = historical_group_flag(frame, "LOW_PULLBACK")
    assert low.iloc[0] == 1 and pd.isna(low.iloc[1]) and low.iloc[2] == 0
    double = historical_group_flag(frame, "DOUBLE_YANG")
    assert double.iloc[0] == 1 and double.iloc[1:].isna().all()


def test_training_weights_balance_dates_and_discount_older_signals() -> None:
    frame = pd.DataFrame({"date": pd.to_datetime(["2025-01-02", "2025-01-02", "2025-01-03"]),
                          "episode_age": [1, 4, 1]})
    weights = pd.Series(sample_weights(frame))
    assert weights.iloc[0] == pytest.approx(2 * weights.iloc[1])
    totals = weights.groupby(frame.date).sum()
    assert totals.iloc[0] == pytest.approx(totals.iloc[1])
