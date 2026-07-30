import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant.strategies.custom.vegas_tunnel import add_vegas_tunnel_signals


def _base_vegas_frame(name: str = "测试股份") -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=260, freq="D")
    close = 10 + np.arange(len(dates)) * 0.035
    open_ = close - 0.04
    high = close + 0.10
    low = close - 0.10
    volume = np.full(len(dates), 1000.0)

    probe = pd.DataFrame({"close": close})
    ema144 = probe["close"].ewm(span=144, adjust=False, min_periods=144).mean()
    ema169 = probe["close"].ewm(span=169, adjust=False, min_periods=169).mean()
    tunnel_upper = pd.concat([ema144, ema169], axis=1).max(axis=1)

    pullback_idx = 246
    signal_idx = 250
    low[pullback_idx] = tunnel_upper.iloc[pullback_idx] * 1.01
    close[pullback_idx] = max(tunnel_upper.iloc[pullback_idx] * 1.035, close[pullback_idx] * 0.98)
    open_[pullback_idx] = close[pullback_idx] + 0.03
    high[pullback_idx] = close[pullback_idx] + 0.08
    volume[pullback_idx] = 950

    close[signal_idx] = close[signal_idx - 1] * 1.012
    open_[signal_idx] = close[signal_idx] * 0.985
    high[signal_idx] = close[signal_idx] * 1.01
    low[signal_idx] = open_[signal_idx] * 0.995
    volume[signal_idx] = 1250

    return pd.DataFrame(
        {
            "date": dates,
            "symbol": "TEST.SZ",
            "name": name,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "pre_close": pd.Series(close).shift(1).fillna(close[0]),
            "volume": volume,
        }
    )


def test_vegas_tunnel_detects_pullback_rebound():
    df = _base_vegas_frame()

    out = add_vegas_tunnel_signals(df)

    signal_idx = 250
    assert out.loc[signal_idx, "vegas_recent_pullback"] == 1
    assert out.loc[signal_idx, "vegas_trend_stack"] == 1
    assert out.loc[signal_idx, "signal_vegas_tunnel"] == 1


def test_vegas_tunnel_excludes_st_names():
    df = _base_vegas_frame(name="*ST测试")

    out = add_vegas_tunnel_signals(df)

    assert out["signal_vegas_tunnel"].sum() == 0
