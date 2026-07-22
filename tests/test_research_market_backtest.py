from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_DIR = PROJECT_ROOT / "scripts" / "research"
if str(RESEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(RESEARCH_DIR))

from analyze_b1_entry_exit_grid import add_future_prices  # noqa: E402
from quant.data import MarketDataStore, MarketDataStoreConfig  # noqa: E402


def test_add_future_prices_reads_canonical_market_partitions(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    store = MarketDataStore(MarketDataStoreConfig(backend="parquet", root=raw_root))
    dates = ["20260715", "20260716", "20260717", "20260720"]
    store.write_market_batch(
        pd.DataFrame(
            {
                "ts_code": ["000001.SZ"] * 4,
                "trade_date": dates,
                "open": [10.0, 10.1, 10.2, 10.3],
                "high": [10.2, 10.3, 10.4, 10.5],
                "low": [9.9, 10.0, 10.1, 10.2],
                "close": [10.0, 10.1, 10.2, 10.3],
                "pre_close": [9.9, 10.0, 10.1, 10.2],
            }
        )
    )
    candidates = pd.DataFrame(
        {"symbol": ["000001.SZ"], "date": [pd.Timestamp("2026-07-15")], "close": [10.0]}
    )

    result = add_future_prices(candidates, raw_root / "daily", max_hold_days=2)

    assert len(result) == 1
    assert result.iloc[0]["entry_open"] == 10.1
    assert result.iloc[0]["date_t1"] == pd.Timestamp("2026-07-16")
    assert result.iloc[0]["close_t2"] == 10.2
