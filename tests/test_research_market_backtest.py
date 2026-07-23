from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_DIR = PROJECT_ROOT / "scripts" / "research"
if str(RESEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(RESEARCH_DIR))

import analyze_b1_entry_exit_grid as entry_exit  # noqa: E402
from analyze_b1_entry_exit_grid import add_future_prices  # noqa: E402
from analyze_b1_formal_combos import COMBOS, combo_mask  # noqa: E402
from refresh_chan_model_live_scores import _build_recent_feature_dataset  # noqa: E402
from quant.data import MarketDataStore, MarketDataStoreConfig  # noqa: E402


class _UnifiedModel:
    feature_names_in_ = np.array(["factor_a", "factor_b"])
    selected_features_ = ["factor_a"]
    imputer = object()

    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        positive = np.full(len(frame), self.probability)
        return np.column_stack([1 - positive, positive])


def test_formal_prediction_adapter_supports_unified_models(monkeypatch) -> None:
    probabilities = {
        "up5_es": 0.6,
        "up8_es": 0.7,
        "up10": 0.3,
        "down2_es": 0.2,
        "down3_es": 0.1,
    }
    monkeypatch.setattr(
        entry_exit,
        "MODEL_PATHS",
        {name: Path(f"{name}.joblib") for name in probabilities},
    )
    monkeypatch.setattr(
        entry_exit.joblib,
        "load",
        lambda path: _UnifiedModel(probabilities[path.stem]),
    )

    result = entry_exit.predict_models(pd.DataFrame({"factor_a": [1.0]}))

    assert result.iloc[0]["pred_up10"] == 0.3
    assert result.iloc[0]["pred_up10_es"] == 0.3
    assert result.iloc[0]["pred_down2_es"] == 0.2


def test_calibrated_formal_combo_masks_use_unified_probability_names() -> None:
    candidates = pd.DataFrame(
        {
            "pred_up5_es": [0.1, 0.1],
            "pred_up8_es": [0.6, 0.8],
            "pred_up10_es": [0.3, 0.1],
            "pred_down2_es": [0.2, 0.2],
            "pred_down3_es": [0.3, 0.3],
        }
    )

    assert combo_mask(candidates, COMBOS[0]).tolist() == [True, False]
    assert combo_mask(candidates, COMBOS[1]).tolist() == [False, True]


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


def test_chan_live_features_read_canonical_market_partitions(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    daily_dir = raw_root / "daily"
    dates = pd.bdate_range("2026-01-01", periods=150)
    close = pd.Series(range(len(dates)), dtype=float) / 100 + 10.0
    store = MarketDataStore(MarketDataStoreConfig(backend="parquet", root=raw_root))
    store.write_market_batch(
        pd.DataFrame(
            {
                "ts_code": ["002440.SZ"] * len(dates),
                "trade_date": dates.strftime("%Y%m%d"),
                "date": dates,
                "open": close,
                "high": close + 0.2,
                "low": close - 0.2,
                "close": close,
                "pre_close": close.shift(1).fillna(close.iloc[0]),
                "vol": 1_000_000.0,
                "volume": 1_000_000.0,
                "amount": 10_000_000.0,
                "turnover_rate": 1.0,
            }
        )
    )
    signal_date = dates[-2]
    candidates = pd.DataFrame(
        {
            "symbol": ["002440.SZ"],
            "date": [signal_date],
            "signal_chan_daily_long": [1],
        }
    )

    result = _build_recent_feature_dataset(
        candidates,
        daily_dir,
        raw_root / "daily_basic",
        raw_root / "moneyflow",
        signal_date.strftime("%Y-%m-%d"),
        signal_date.strftime("%Y-%m-%d"),
    )

    assert len(result) == 1
    assert result.iloc[0]["symbol"] == "002440.SZ"
    assert pd.notna(result.iloc[0]["ret_20d"])
