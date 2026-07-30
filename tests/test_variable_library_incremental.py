from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from quant.features import variable_library


def _reference_continuous_ohlc(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    order = (
        out.assign(
            _order_date=pd.to_datetime(
                out["date"],
                errors="coerce",
            )
        )
        .sort_values("_order_date")
        .index
    )
    sorted_frame = out.loc[order].copy()
    factor = pd.Series(1.0, index=sorted_frame.index, dtype=float)
    close = pd.to_numeric(sorted_frame["close"], errors="coerce")
    pre_close = pd.to_numeric(
        sorted_frame["pre_close"],
        errors="coerce",
    )
    for position in range(len(sorted_frame) - 1, 0, -1):
        index = sorted_frame.index[position]
        previous_index = sorted_frame.index[position - 1]
        previous_close = close.loc[previous_index]
        current_pre_close = pre_close.loc[index]
        ratio = (
            current_pre_close / previous_close
            if pd.notna(current_pre_close)
            and pd.notna(previous_close)
            and previous_close
            else 1.0
        )
        if not np.isfinite(ratio) or ratio <= 0:
            ratio = 1.0
        factor.loc[previous_index] = factor.loc[index] * ratio
    for column in ("open", "high", "low", "close"):
        out[column] = (
            pd.to_numeric(out[column], errors="coerce")
            * factor.reindex(out.index).fillna(1.0)
        )
    return out


def test_vectorized_continuous_ohlc_matches_reference_with_repricing():
    rng = np.random.default_rng(20260728)
    size = 260
    dates = pd.bdate_range("2025-01-02", periods=size)
    close = rng.lognormal(2.5, 0.15, size)
    pre_close = np.roll(close, 1)
    pre_close[0] = close[0]
    pre_close[80] = close[79] * 0.83
    pre_close[170] = close[169] * 1.21
    pre_close[210] = 0.0
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": close * rng.uniform(0.98, 1.02, size),
            "high": close * rng.uniform(1.0, 1.05, size),
            "low": close * rng.uniform(0.95, 1.0, size),
            "close": close,
            "pre_close": pre_close,
        },
        index=rng.permutation(np.arange(1000, 1000 + size)),
    ).sample(frac=1.0, random_state=20260728)

    expected = _reference_continuous_ohlc(frame)
    actual = variable_library.build_continuous_ohlc(frame)

    assert_frame_equal(
        actual,
        expected,
        check_exact=False,
        rtol=1e-14,
        atol=1e-14,
    )


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
