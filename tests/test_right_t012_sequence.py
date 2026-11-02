import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from train_right_t012_sequence import PriceSequence, sequence_windows  # noqa: E402


def test_sequence_ends_at_entry_and_rejects_missing_market_day() -> None:
    dates = pd.bdate_range("2025-01-01", periods=60)
    bars = pd.DataFrame({"symbol": "a", "date": dates, "open": 10., "high": 11.,
                         "low": 9., "close": 10., "pre_close": 10., "vol": 100., "valid_bar": True})
    events = pd.DataFrame({"symbol": ["a"], "date": [dates[55]]})
    original, ready = sequence_windows(bars, events, dates)
    changed = bars.copy()
    changed.loc[changed.date.gt(dates[55]), "close"] = 100.
    future_changed, _ = sequence_windows(changed, events, dates)
    assert ready.tolist() == [True]
    np.testing.assert_array_equal(original, future_changed)
    _, missing_ready = sequence_windows(bars.drop(index=40), events, dates)
    assert not missing_ready.any()


def test_price_heads_obey_ohlc_order() -> None:
    model = PriceSequence()
    values = model(torch.zeros((3, 32, 5)))
    assert values.shape == (3, 12)
    assert (values[:, 1] >= values[:, [0, 3]].max(dim=1).values).all()
    assert (values[:, 2] <= values[:, [0, 3]].min(dim=1).values).all()
