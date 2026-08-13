from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd


RESEARCH_DIR = Path(__file__).resolve().parents[1] / "scripts" / "research"
if str(RESEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(RESEARCH_DIR))

import backtest_long_dividend_quality as module  # noqa: E402


def _write_snapshot(path: Path, trade_date: str, symbols: list[str]) -> None:
    rows = len(symbols)
    pd.DataFrame(
        {
            "ts_code": symbols,
            "trade_date": [trade_date] * rows,
            "turnover_rate": [1.0] * rows,
            "turnover_rate_f": [1.2] * rows,
            "pe_ttm": [10.0] * rows,
            "pb": [1.0] * rows,
            "ps_ttm": [2.0] * rows,
            "dv_ratio": [0.5] * rows,
            "dv_ttm": [float(index + 1) for index in range(rows)],
            "total_mv": [1_000_000.0] * rows,
            "circ_mv": [800_000.0] * rows,
            # Deliberately unused: the optimized path must project it out.
            "unused_large_payload": ["x" * 128] * rows,
        }
    ).to_parquet(path, index=False)


def _configure_paths(monkeypatch, tmp_path: Path) -> Path:
    source = tmp_path / "daily_basic"
    source.mkdir()
    monkeypatch.setattr(module, "DAILY_BASIC_DIR", source)
    monkeypatch.setattr(module, "DAILY_BASIC_CACHE_DIR", tmp_path / "fallback")
    monkeypatch.setattr(module, "RESEARCH_CACHE_DIR", tmp_path / "research_cache")
    return source


def _full_scan_reference(
    source: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(source.glob("*.parquet")):
        date = module._daily_basic_file_date(path)
        if date is None or date < start or date > end:
            continue
        frame = pd.read_parquet(path)
        if frame.empty or "ts_code" not in frame.columns:
            continue
        frames.append(frame[list(module.DAILY_BASIC_FEATURE_COLUMNS)].copy())
    basic = pd.concat(frames, ignore_index=True)
    basic["date"] = pd.to_datetime(
        basic["trade_date"].astype(str), format="%Y%m%d", errors="coerce"
    )
    sampled_dates = basic.groupby(basic["date"].dt.to_period("M"))["date"].max()
    sampled = basic[basic["date"].isin(set(sampled_dates))].copy()
    sampled = sampled.sort_values(["ts_code", "date"]).reset_index(drop=True)
    dv = pd.to_numeric(sampled["dv_ttm"], errors="coerce")
    sampled["dv_ttm_mean_36m"] = dv.groupby(sampled["ts_code"]).transform(
        lambda values: values.rolling(36, min_periods=12).mean()
    )
    sampled["dv_ttm_std_36m"] = dv.groupby(sampled["ts_code"]).transform(
        lambda values: values.rolling(36, min_periods=12).std()
    )
    sampled["dv_ttm_stability_36m"] = (
        sampled["dv_ttm_mean_36m"]
        / sampled["dv_ttm_std_36m"].replace(0, float("nan"))
    ).clip(0, 10)
    return sampled


def test_period_end_projection_matches_full_scan_golden_and_partial_latest_semantics(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = _configure_paths(monkeypatch, tmp_path)
    _write_snapshot(source / "20240126.parquet", "20240126", ["A", "B"])
    # Jan 30 is a non-natural month end and intentionally partial.  The old
    # groupby-last behavior keeps only this global latest-date cross-section;
    # it does not backfill B from Jan 26.
    _write_snapshot(source / "20240130.parquet", "20240130", ["A"])
    _write_snapshot(source / "20240328.parquet", "20240328", ["A", "B"])
    start = pd.Timestamp("2024-01-01")
    end = pd.Timestamp("2024-03-31")

    expected = _full_scan_reference(source, start, end)
    actual, coverage = module.load_daily_basic_monthly(start, end)

    pd.testing.assert_frame_equal(actual, expected)
    assert actual["date"].drop_duplicates().tolist() == [
        pd.Timestamp("2024-01-30"),
        pd.Timestamp("2024-03-28"),
    ]
    assert actual.loc[actual["date"].eq(pd.Timestamp("2024-01-30")), "ts_code"].tolist() == ["A"]
    assert coverage["loaded_trade_dates"] == 3
    assert coverage["read_trade_dates"] == 2
    assert coverage["refreshed_periods"] == 2


def test_period_end_cache_refreshes_when_latest_snapshot_is_backfilled(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = _configure_paths(monkeypatch, tmp_path)
    snapshot = source / "20240130.parquet"
    _write_snapshot(snapshot, "20240130", ["A"])
    start = pd.Timestamp("2024-01-01")
    end = pd.Timestamp("2024-01-31")

    first, first_coverage = module.load_daily_basic_monthly(start, end)
    cached, cached_coverage = module.load_daily_basic_monthly(start, end)
    _write_snapshot(snapshot, "20240130", ["A", "B"])
    repaired, repaired_coverage = module.load_daily_basic_monthly(start, end)

    assert first["ts_code"].tolist() == ["A"]
    pd.testing.assert_frame_equal(cached, first)
    assert first_coverage["refreshed_periods"] == 1
    assert cached_coverage["cache_hit_periods"] == 1
    assert repaired["ts_code"].tolist() == ["A", "B"]
    assert repaired_coverage["refreshed_periods"] == 1
    assert repaired_coverage["cache_hit_periods"] == 0


def test_period_end_cache_detects_same_size_content_change_with_preserved_mtime(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = _configure_paths(monkeypatch, tmp_path)
    snapshot = source / "20240130.parquet"
    _write_snapshot(snapshot, "20240130", ["A"])
    original_stat = snapshot.stat()
    start = pd.Timestamp("2024-01-01")
    end = pd.Timestamp("2024-01-31")
    original, _ = module.load_daily_basic_monthly(start, end)

    _write_snapshot(snapshot, "20240130", ["B"])
    assert snapshot.stat().st_size == original_stat.st_size
    os.utime(snapshot, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    changed, coverage = module.load_daily_basic_monthly(start, end)

    assert original["ts_code"].tolist() == ["A"]
    assert changed["ts_code"].tolist() == ["B"]
    assert coverage["cache_hit_periods"] == 0
    assert coverage["refreshed_periods"] == 1


def test_monthly_projection_microbenchmark_reads_one_file_per_available_month(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = _configure_paths(monkeypatch, tmp_path)
    dates = pd.bdate_range("2024-01-02", "2024-03-29")
    for date in dates:
        trade_date = date.strftime("%Y%m%d")
        _write_snapshot(source / f"{trade_date}.parquet", trade_date, ["A", "B"])
    start = pd.Timestamp("2024-01-01")
    end = pd.Timestamp("2024-03-31")
    real_read_parquet = pd.read_parquet
    source_reads: list[tuple[Path, tuple[str, ...] | None]] = []

    def recording_read(path, *args, **kwargs):
        resolved = Path(path)
        if resolved.parent == source:
            columns = kwargs.get("columns")
            source_reads.append((resolved, tuple(columns) if columns is not None else None))
        return real_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(module.pd, "read_parquet", recording_read)
    expected = _full_scan_reference(source, start, end)
    full_scan_reads = len(source_reads)
    source_reads.clear()

    actual, coverage = module.load_daily_basic_monthly(start, end)

    pd.testing.assert_frame_equal(actual, expected)
    assert full_scan_reads == len(dates)
    assert len(source_reads) == 3
    assert len(source_reads) * 10 < full_scan_reads
    assert all(columns == module.DAILY_BASIC_FEATURE_COLUMNS for _, columns in source_reads)
    assert coverage["read_source_files"] == 3
