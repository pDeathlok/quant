from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quant.webapp import services


def _write_daily_history(path: Path) -> None:
    dates = pd.bdate_range("2024-01-01", periods=180)
    close = np.linspace(10.0, 15.0, len(dates))
    pd.DataFrame(
        {
            "ts_code": "000001.SZ",
            "trade_date": dates.strftime("%Y%m%d"),
            "open": close - 0.1,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "pct_chg": pd.Series(close).pct_change().fillna(0.0) * 100,
        }
    ).to_parquet(path, index=False)


def test_live_feature_mode_does_not_build_or_persist_daily_returns(monkeypatch, tmp_path: Path) -> None:
    module = services._long_research_module()
    daily_dir = tmp_path / "daily"
    cache_dir = tmp_path / "cache"
    daily_dir.mkdir()
    _write_daily_history(daily_dir / "000001.SZ.parquet")
    monkeypatch.setattr(module, "DAILY_DIR", daily_dir)
    monkeypatch.setattr(module, "RESEARCH_CACHE_DIR", cache_dir)
    stock_basic = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "name": ["测试股票"],
            "industry": ["测试行业"],
            "list_date": pd.to_datetime(["2010-01-01"]),
        }
    )

    features, daily_returns = module.load_daily_monthly_features(
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-12-31"),
        stock_basic,
        use_cache=False,
        include_daily_returns=False,
    )

    assert not features.empty
    assert daily_returns.empty
    assert not cache_dir.exists()
