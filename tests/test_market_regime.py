import numpy as np
import pandas as pd
import pytest

from quant.features.market_regime import classify_market_regime


def _market_frame(periods: int = 80) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=periods, freq="B")
    rows = []
    for index, symbol in enumerate(["000001.SZ", "000002.SZ", "600000.SH", "600001.SH"]):
        for offset, date in enumerate(dates):
            rows.append(
                {
                    "trade_date": date.strftime("%Y%m%d"),
                    "ts_code": symbol,
                    "close": 10 + index + offset * 0.05,
                    "amount": 1_000_000 + offset * 10_000,
                }
            )
    return pd.DataFrame(rows)


def test_market_regime_uses_only_information_available_as_of_date() -> None:
    market = _market_frame()
    dates = sorted(market["trade_date"].unique())
    index = (
        market.loc[market["ts_code"] == "000001.SZ", ["trade_date", "close"]]
        .assign(ts_code="000300.SH")
    )
    cutoff = dates[-2]

    baseline = classify_market_regime(index, market, as_of=cutoff)
    future = pd.concat(
        [
            market,
            pd.DataFrame(
                {
                    "trade_date": [dates[-1]],
                    "ts_code": ["999999.SZ"],
                    "close": [1_000_000.0],
                    "amount": [1_000_000_000.0],
                }
            ),
        ],
        ignore_index=True,
    )

    assert classify_market_regime(index, future, as_of=cutoff) == baseline
    assert baseline["as_of"] == cutoff
    assert baseline["regime"] == "risk_on"
    assert baseline["signals"]["breadth_above_ma20"] == 1.0


def test_market_regime_rejects_insufficient_history() -> None:
    market = _market_frame(periods=10)
    index = market.loc[market["ts_code"] == "000001.SZ", ["trade_date", "close"]]

    with np.testing.assert_raises_regex(ValueError, "at least 60"):
        classify_market_regime(index, market)


def test_market_regime_rejects_mismatched_terminal_observation_dates() -> None:
    market = _market_frame()
    dates = sorted(market["trade_date"].unique())
    index = (
        market.loc[market["ts_code"] == "000001.SZ", ["trade_date", "close"]]
        .iloc[:-1]
        .assign(ts_code="000300.SH")
    )

    with pytest.raises(ValueError, match="terminal observation dates differ"):
        classify_market_regime(index, market, as_of=dates[-1])


def test_market_regime_rejects_inputs_stale_for_requested_decision_date() -> None:
    market = _market_frame()
    dates = sorted(market["trade_date"].unique())
    stale_market = market.loc[market["trade_date"] < dates[-1]].copy()
    index = (
        stale_market.loc[
            stale_market["ts_code"] == "000001.SZ",
            ["trade_date", "close"],
        ]
        .assign(ts_code="000300.SH")
    )

    with pytest.raises(ValueError, match="requested decision date"):
        classify_market_regime(index, stale_market, as_of=dates[-1])
