from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from quant.features import variable_library


def _write_daily_basic_history(root, periods: int = 90) -> list[pd.Timestamp]:
    dates = list(pd.bdate_range("2026-01-05", periods=periods))
    for index, trade_date in enumerate(dates):
        rows = [
            {
                "ts_code": "000001.SZ",
                "trade_date": trade_date.strftime("%Y%m%d"),
                "turnover_rate": 1.0 + index / 100,
                "turnover_rate_f": 1.2 + index / 100,
                "volume_ratio": 0.8 + index / 200,
                "pe_ttm": 10.0 + index / 10,
                "pb": 1.5 + index / 100,
                "ps_ttm": 2.0 + index / 100,
                "total_share": 1000.0,
                "float_share": 800.0,
                "free_share": 700.0,
                "total_mv": 10_000.0 + index,
                "circ_mv": 8_000.0 + index,
            }
        ]
        # A sparse history forces the bounded loader to expand beyond its
        # initial file window, exercising suspended-stock equivalence.
        if index % 3 == 0:
            rows.append(
                {
                    **rows[0],
                    "ts_code": "000002.SZ",
                    "turnover_rate": 2.0 + index / 100,
                    "total_mv": 20_000.0 + index,
                    "circ_mv": 16_000.0 + index,
                }
            )
        pd.DataFrame(rows).to_parquet(root / f"{trade_date:%Y%m%d}.parquet", index=False)
    return dates


def test_bounded_daily_basic_features_match_full_history_for_rolling_values(tmp_path) -> None:
    dates = _write_daily_basic_history(tmp_path)
    target_date = dates[-3].strftime("%Y%m%d")
    target_keys = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ"],
            "trade_date": [target_date, target_date],
        }
    )

    full = variable_library.load_daily_basic_features(tmp_path)
    bounded = variable_library.load_daily_basic_features(tmp_path, target_keys=target_keys)
    columns = sorted(set(full.columns) & set(bounded.columns))
    expected = (
        full.merge(target_keys, on=["ts_code", "trade_date"], how="inner")
        .sort_values(["ts_code", "trade_date"])[columns]
        .reset_index(drop=True)
    )
    actual = (
        bounded.merge(target_keys, on=["ts_code", "trade_date"], how="inner")
        .sort_values(["ts_code", "trade_date"])[columns]
        .reset_index(drop=True)
    )

    assert_frame_equal(actual, expected, check_exact=False, rtol=1e-13, atol=1e-13)


def test_incremental_merge_reads_only_bounded_files_and_matches_full_merge(monkeypatch, tmp_path) -> None:
    dates = _write_daily_basic_history(tmp_path)
    target_date = dates[-1].strftime("%Y%m%d")
    incremental = pd.DataFrame(
        {
            "symbol": ["000001.SZ"],
            "ts_code": ["000001.SZ"],
            "trade_date": [target_date],
            "date": [dates[-1]],
            "turnover_rate": [np.nan],
        }
    )
    full_daily_basic = variable_library.load_daily_basic_features(tmp_path)
    expected = incremental.drop(columns=["turnover_rate"]).merge(
        full_daily_basic,
        on=["ts_code", "trade_date"],
        how="left",
    )

    original_read = variable_library._read_daily_basic_file
    reads: list[str] = []

    def recording_read(path):
        reads.append(path.name)
        return original_read(path)

    monkeypatch.setattr(variable_library, "_read_daily_basic_file", recording_read)
    actual = variable_library.merge_daily_basic_features(incremental, tmp_path)

    expected = expected.sort_index(axis=1)
    actual = actual.sort_index(axis=1)
    assert_frame_equal(actual, expected, check_exact=False, rtol=1e-13, atol=1e-13)
    assert len(reads) < len(dates) / 2


def test_incremental_merge_does_not_scan_history_when_target_date_is_absent(monkeypatch, tmp_path) -> None:
    dates = _write_daily_basic_history(tmp_path, periods=30)
    target_date = (dates[-1] + pd.offsets.BDay()).strftime("%Y%m%d")
    incremental = pd.DataFrame(
        {
            "symbol": ["000001.SZ"],
            "ts_code": ["000001.SZ"],
            "trade_date": [target_date],
            "date": [pd.to_datetime(target_date)],
        }
    )
    original_read = variable_library._read_daily_basic_file
    reads: list[str] = []

    def recording_read(path):
        reads.append(path.name)
        return original_read(path)

    monkeypatch.setattr(variable_library, "_read_daily_basic_file", recording_read)
    actual = variable_library.merge_daily_basic_features(incremental, tmp_path)

    assert_frame_equal(actual, incremental)
    assert reads == []


def test_incremental_merge_can_gate_missing_daily_basic_coverage(tmp_path) -> None:
    dates = _write_daily_basic_history(tmp_path, periods=5)
    target_date = (dates[-1] + pd.offsets.BDay()).strftime("%Y%m%d")
    incremental = pd.DataFrame(
        {
            "symbol": ["000001.SZ"],
            "ts_code": ["000001.SZ"],
            "trade_date": [target_date],
            "date": [pd.to_datetime(target_date)],
        }
    )

    with pytest.raises(RuntimeError, match="coverage below required threshold"):
        variable_library.merge_daily_basic_features(
            incremental,
            tmp_path,
            min_match_rate=0.98,
        )
