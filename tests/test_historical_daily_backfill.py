from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/backfill_historical_daily.py"
spec = importlib.util.spec_from_file_location("historical_daily_backfill", SCRIPT)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_merge_missing_bars_preserves_canonical_prices() -> None:
    existing = pd.DataFrame({
        "ts_code": ["000001.SZ"], "trade_date": ["20261015"], "close": [10.0],
    })
    incoming = pd.DataFrame({
        "ts_code": ["000001.SZ", "000005.SZ"],
        "trade_date": ["20261015", "20261015"], "close": [11.0, 3.0],
    })

    merged, added = module.merge_missing_bars(existing, incoming)

    assert added == 1
    assert merged.set_index("ts_code")["close"].to_dict() == {
        "000001.SZ": 10.0, "000005.SZ": 3.0,
    }


def test_merge_missing_bars_rejects_duplicate_source_keys() -> None:
    incoming = pd.DataFrame({
        "ts_code": ["000005.SZ", "000005.SZ"],
        "trade_date": ["20261015", "20261015"], "close": [3.0, 4.0],
    })

    with pytest.raises(ValueError, match="duplicate keys"):
        module.merge_missing_bars(pd.DataFrame(), incoming)


def test_sohu_bar_requires_matching_neighbors() -> None:
    rows = [
        ["2013-11-15", "13.74", "14.03", "0.28", "2.04%", "13.61", "14.28", "32605", "4584.01"],
        ["2013-11-14", "13.63", "13.75", "-0.03", "-0.22%", "13.30", "13.87", "26874", "3654.97"],
        ["2013-11-13", "14.10", "13.78", "0.01", "0.07%", "13.77", "14.29", "36073", "5062.88"],
    ]
    canonical = pd.DataFrame({
        "ts_code": ["000022.SZ", "000022.SZ"],
        "trade_date": ["20131113", "20131115"],
        "open": [14.10, 13.74], "high": [14.29, 14.28],
        "low": [13.77, 13.61], "close": [13.78, 14.03],
        "pre_close": [13.77, 13.75], "vol": [36072.95, 32605.24],
        "amount": [50628.8111, 45840.0898],
    })

    recovered = module.verified_sohu_bar("000022.SZ", "20131114", rows, canonical)
    assert recovered.iloc[0]["close"] == 13.75
    assert recovered.iloc[0]["amount"] == 36549.7

    canonical.loc[1, "pre_close"] = 13.9
    with pytest.raises(ValueError, match="pre_close"):
        module.verified_sohu_bar("000022.SZ", "20131114", rows, canonical)
