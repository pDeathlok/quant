from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from quant.research.kronos_rolling import (  # noqa: E402
    mature_window,
    refit_segments,
    stacking_window,
)


def test_refit_segments_cover_month_or_week_once_with_trading_day_boundaries():
    dates = pd.to_datetime(
        ["2026-01-05", "2026-01-06", "2026-01-09", "2026-01-12", "2026-01-30", "2026-02-02"]
    )
    weekly = refit_segments(dates, "weekly")
    monthly = refit_segments(dates, "monthly")
    assert [t[0] for t in weekly] == list(dates[[0, 3, 4, 5]])
    assert [t[0] for t in monthly] == list(dates[[0, 5]])
    for segments in [weekly, monthly]:
        assert sorted(np.concatenate([x[1] for x in segments])) == sorted(dates)
        for date, subset in segments:
            assert (subset >= date).all()
    with pytest.raises(ValueError):
        refit_segments(dates, "daily")


def test_three_calendar_months_and_mature_labels_strictly_before_refit():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-09-30", "2025-10-01", "2025-12-20", "2025-12-30"]),
            "exit_date": pd.to_datetime(["2025-10-08", "2025-10-09", "2026-01-01", "2026-01-08"]),
        }
    )
    selected = mature_window(frame, pd.Timestamp("2026-01-01"))
    assert selected.index.tolist() == [1]
    changed = frame.copy()
    changed.loc[changed.exit_date >= "2026-01-01", "date"] = pd.Timestamp("2025-11-01")
    assert mature_window(changed, pd.Timestamp("2026-01-01")).index.tolist() == [1]


def test_stacking_uses_last_month_and_purges_auxiliary_head_boundary():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-10-10", "2025-11-28", "2025-12-01", "2025-12-20"]),
            "exit_date": pd.to_datetime(["2025-10-20", "2025-12-08", "2025-12-09", "2025-12-29"]),
        }
    )
    early, meta = stacking_window(frame, pd.Timestamp("2026-01-01"))
    assert early.index.tolist() == [0]
    assert meta.index.tolist() == [2, 3]
    assert early.exit_date.max() < meta.date.min()


def test_insufficient_mature_data_stays_cash_without_fitting(tmp_path):
    from quant.research.kronos_rolling import rolling_predictions

    dates = pd.to_datetime(["2025-11-01", "2025-11-03", "2026-01-05"])
    frame = pd.DataFrame(
        {
            "symbol": ["X"] * 3,
            "date": dates,
            "entry_date": dates + pd.Timedelta(days=1),
            "exit_date": dates + pd.Timedelta(days=8),
            "net_return": [0.01, -0.01, 0.02],
            "baseline_score": [0.2, 0.3, 0.4],
            "embedding_row": [0, 1, 2],
        }
    )
    prediction, audit = rolling_predictions(
        frame,
        np.zeros((3, 8), dtype=np.float32),
        pd.DatetimeIndex([dates[-1]]),
        "weekly",
        tmp_path / "models",
        epochs=1,
    )
    assert (
        prediction[["rolling_sequence", "fusion_only", "rolling_joint", "rolling_cash"]]
        .isna()
        .all()
        .all()
    )
    assert audit[0]["status"] == "insufficient_training_cash"
    assert not list((tmp_path / "models").iterdir())
