from __future__ import annotations

import argparse
import runpy
from datetime import date, timedelta
from pathlib import Path

import polars as pl


SCRIPT = Path(__file__).parents[1] / "scripts/research/optimize_selector_buy_hold_scores.py"


def _module() -> dict:
    return runpy.run_path(str(SCRIPT))


def test_rolling_splits_exclude_unmatured_label_horizon() -> None:
    module = _module()
    dates = [date(2024, 1, 1) + timedelta(days=index) for index in range(20)]
    args = argparse.Namespace(
        train_start=None,
        train_end=None,
        valid_end=None,
        test_end=None,
        train_days=6,
        valid_days=4,
        test_days=5,
        label_horizon=3,
    )

    splits = module["resolve_rolling_splits"](args, dates)

    assert splits.market_end == "2024-01-20"
    assert splits.test_end == "2024-01-17"
    assert splits.train_start == "2024-01-03"
    assert splits.train_end == "2024-01-08"
    assert splits.valid_end == "2024-01-12"


def test_history_quality_rejects_duplicate_keys(tmp_path: Path) -> None:
    module = _module()
    path = tmp_path / "history.parquet"
    pl.DataFrame(
        {
            "symbol": ["000001.SZ", "000001.SZ"],
            "date": [date(2024, 1, 10), date(2024, 1, 10)],
            "future_return_t5_pct": [1.0, 1.0],
            "future_max_high_t5_pct": [2.0, 2.0],
            "selector_return_1d": [0.2, 0.2],
        }
    ).write_parquet(path)
    splits = module["RollingSplits"](
        train_start="2024-01-01",
        train_end="2024-01-05",
        valid_end="2024-01-08",
        test_end="2024-01-10",
        market_end="2024-01-10",
        label_horizon=5,
    )

    try:
        module["validate_history"](path, splits)
    except RuntimeError as exc:
        assert "duplicate_symbol_date_keys" in str(exc)
    else:
        raise AssertionError("duplicate history keys must fail validation")
