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
    for position in range(1, len(sorted_frame)):
        index = sorted_frame.index[position]
        previous_index = sorted_frame.index[position - 1]
        previous_close = close.loc[previous_index]
        current_pre_close = pre_close.loc[index]
        ratio = (
            previous_close / current_pre_close
            if pd.notna(current_pre_close)
            and pd.notna(previous_close)
            and current_pre_close
            and previous_close
            else 1.0
        )
        if not np.isfinite(ratio) or ratio <= 0:
            ratio = 1.0
        factor.loc[index] = factor.loc[previous_index] * ratio
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

    with pytest.raises(RuntimeError, match="coverage below required threshold") as error:
        variable_library.merge_daily_basic_features(
            incremental,
            tmp_path,
            min_match_rate=0.98,
        )

    message = str(error.value)
    assert "total_rows=1 missing_rows=1 missing_symbols=1" in message
    assert type(error.value) is RuntimeError
    assert f"missing_date_range={target_date}..{target_date}" in message
    assert "missing_source_rows=1 turnover_rate_null_rows=0" in message
    assert f"samples=[('000001.SZ', '{target_date}')]" in message


def test_daily_basic_coverage_distinguishes_missing_rows_from_null_turnover(monkeypatch, tmp_path) -> None:
    frame = pd.DataFrame({
        "symbol": ["000001.SZ", "000002.SZ", "000003.SZ", "000003.SZ"],
        "trade_date": ["20260105", "20260105", "20260106", "20260106"],
        "turnover_rate": [99.0] * 4,
    })
    original = frame.copy(deep=True)
    source = pd.DataFrame({
        "ts_code": ["000001.SZ", "000002.SZ"],
        "trade_date": ["20260105", "20260105"],
        "turnover_rate": [1.0, np.nan],
    })
    monkeypatch.setattr(variable_library, "load_daily_basic_features", lambda *args, **kwargs: source)

    with pytest.raises(RuntimeError, match="matched=25.00% required=98.00%") as error:
        variable_library.merge_daily_basic_features(frame, tmp_path, min_match_rate=0.98)

    message = str(error.value)
    assert "total_rows=4 missing_rows=3 missing_symbols=2" in message
    assert type(error.value) is RuntimeError
    assert "missing_date_range=20260105..20260106" in message
    assert "missing_source_rows=2 turnover_rate_null_rows=1" in message
    assert "'20260105': {'total_rows': 2, 'missing_rows': 1, 'missing_source_rows': 0, 'turnover_rate_null_rows': 1}" in message
    assert "'20260106': {'total_rows': 2, 'missing_rows': 2, 'missing_source_rows': 2, 'turnover_rate_null_rows': 0}" in message
    assert "samples=[('000002.SZ', '20260105'), ('000003.SZ', '20260106')]" in message
    assert_frame_equal(frame, original)


def test_daily_basic_missing_turnover_column_cannot_reuse_cached_values(monkeypatch, tmp_path) -> None:
    frame = pd.DataFrame({"symbol": ["000001.SZ"], "trade_date": ["20260105"], "turnover_rate": [99.0]})
    source = pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": ["20260105"], "pb": [1.0]})
    monkeypatch.setattr(variable_library, "load_daily_basic_features", lambda *args, **kwargs: source)

    with pytest.raises(RuntimeError, match="matched=0.00% required=98.00%") as error:
        variable_library.merge_daily_basic_features(frame, tmp_path, min_match_rate=0.98)

    assert "missing_source_rows=0 turnover_rate_null_rows=1" in str(error.value)


def test_daily_basic_coverage_diagnostics_are_bounded(monkeypatch, tmp_path) -> None:
    dates = pd.bdate_range("2026-01-05", periods=30).strftime("%Y%m%d")
    frame = pd.DataFrame({"symbol": ["000001.SZ"] * 30, "trade_date": dates})
    monkeypatch.setattr(variable_library, "load_daily_basic_features", lambda *args, **kwargs: pd.DataFrame())

    with pytest.raises(RuntimeError) as error:
        variable_library.merge_daily_basic_features(frame, tmp_path, min_match_rate=0.98)

    message = str(error.value)
    assert "total_rows=30 missing_rows=30 missing_symbols=1" in message
    assert f"missing_date_range={dates[0]}..{dates[-1]}" in message
    assert message.count("'total_rows':") == 10
    assert message.count("('000001.SZ',") == 5
    assert "omitted_dates=20" in message
    assert len(message) < 2500


@pytest.mark.parametrize("available_rows", [97, 98])
def test_daily_basic_strict_threshold_and_unfilled_missing_values(monkeypatch, tmp_path, available_rows) -> None:
    symbols = [f"{number:06d}.SZ" for number in range(100)]
    frame = pd.DataFrame({"symbol": symbols, "trade_date": ["20260105"] * 100, "turnover_rate": [99.0] * 100})
    source = pd.DataFrame({
        "ts_code": symbols[:available_rows],
        "trade_date": ["20260105"] * available_rows,
        "turnover_rate": [0.0] * available_rows,
    })
    monkeypatch.setattr(variable_library, "load_daily_basic_features", lambda *args, **kwargs: source)

    if available_rows < 98:
        with pytest.raises(RuntimeError, match="matched=97.00% required=98.00%"):
            variable_library.merge_daily_basic_features(frame, tmp_path, min_match_rate=0.98)
    else:
        actual = variable_library.merge_daily_basic_features(frame, tmp_path, min_match_rate=0.98)
        assert len(actual) == len(frame)
        assert actual["turnover_rate"].iloc[:98].eq(0.0).all()
        assert actual["turnover_rate"].iloc[98:].isna().all()
        assert not any(column.startswith("_daily_basic_merge") for column in actual)


@pytest.mark.parametrize("date_column", ["trade_date", "date"])
def test_daily_basic_merge_normalizes_dates_before_deduplication(tmp_path, date_column) -> None:
    for name, date, turnover in [
        ("20260105", 20260105, 1.0),
        ("snapshot_20260105", "2026-01-05", 2.0),
    ]:
        pd.DataFrame({
            "ts_code": ["000001.SZ"], "trade_date": [date], "turnover_rate": [turnover],
        }).to_parquet(tmp_path / f"{name}.parquet", index=False)
    frame = pd.DataFrame({
        "symbol": ["000001.SZ"] * 3,
        date_column: [20260105, "2026-01-05", pd.Timestamp("2026-01-05")],
        "_daily_basic_merge": ["user value"] * 3,
    })

    actual = variable_library.merge_daily_basic_features(frame, tmp_path, min_match_rate=0.98)

    assert len(actual) == 3
    assert actual["trade_date"].eq("20260105").all()
    assert actual["turnover_rate"].eq(2.0).all()
    assert actual["_daily_basic_merge"].eq("user value").all()
    assert "_daily_basic_merge_" not in actual


def test_daily_basic_merge_rejects_nonunique_loaded_keys(monkeypatch, tmp_path) -> None:
    frame = pd.DataFrame({"symbol": ["000001.SZ"] * 2, "trade_date": ["20260105"] * 2})
    source = pd.DataFrame({
        "ts_code": ["000001.SZ"] * 2,
        "trade_date": [20260105, "2026-01-05"],
        "turnover_rate": [1.0, 2.0],
    })
    monkeypatch.setattr(variable_library, "load_daily_basic_features", lambda *args, **kwargs: source)

    with pytest.raises(pd.errors.MergeError, match="not a many-to-one merge"):
        variable_library.merge_daily_basic_features(frame, tmp_path, min_match_rate=0.98)


def test_daily_basic_null_date_keys_do_not_match(monkeypatch, tmp_path) -> None:
    frame = pd.DataFrame({"symbol": ["000001.SZ"] * 2, "trade_date": ["20260105", None]})
    source = pd.DataFrame({
        "ts_code": ["000001.SZ"] * 2,
        "trade_date": ["20260105", None],
        "turnover_rate": [1.0, 2.0],
    })
    monkeypatch.setattr(variable_library, "load_daily_basic_features", lambda *args, **kwargs: source)

    with pytest.raises(RuntimeError, match="matched=50.00% required=98.00%") as error:
        variable_library.merge_daily_basic_features(frame, tmp_path, min_match_rate=0.98)

    assert "missing_source_rows=1 turnover_rate_null_rows=0" in str(error.value)
    assert "samples=[('000001.SZ', '<unknown>')]" in str(error.value)


@pytest.fixture
def source_preflight_only(monkeypatch):
    from quant.data import market_snapshot

    monkeypatch.setattr(market_snapshot, "current_market_snapshot", lambda: None)

    def reject_feature_load(*args, **kwargs):
        pytest.fail("Source preflight must not load rolling features or history")

    monkeypatch.setattr(variable_library, "load_daily_basic_features", reject_feature_load)
    monkeypatch.setattr(variable_library, "_read_daily_basic_file", reject_feature_load)


def test_source_preflight_gates_each_date_not_global_average(tmp_path, source_preflight_only) -> None:
    symbols = [f"{number:06d}.SZ" for number in range(100)]
    source = pd.DataFrame({"ts_code": symbols, "trade_date": ["20260105"] * 100, "turnover_rate": [1.0] * 100})
    source.to_parquet(tmp_path / "20260105.parquet", index=False)
    targets = pd.concat([
        source[["ts_code", "trade_date"]],
        pd.DataFrame({"ts_code": [symbols[0]], "trade_date": ["20260106"]}),
    ], ignore_index=True)

    # Overall coverage is 100/101 > 98%, but the second date has no coverage.
    with pytest.raises(RuntimeError, match="date=20260106 matched=0.00% required=98.00%") as error:
        variable_library.validate_daily_basic_source_keys(targets, tmp_path)

    assert "total_rows=1 missing_rows=1 missing_symbols=1" in str(error.value)
    assert "missing_source_rows=1 turnover_rate_null_rows=0" in str(error.value)
    assert type(error.value) is variable_library.DailyBasicSourceCoverageError


def test_source_preflight_reads_only_target_keys_and_three_columns(monkeypatch, tmp_path, source_preflight_only) -> None:
    dates = _write_daily_basic_history(tmp_path, periods=35)
    target_date = dates[-1].strftime("%Y%m%d")
    targets = pd.DataFrame({"symbol": ["000001.SZ"] * 2, "date": [dates[-1], dates[-1]]})
    original = targets.copy(deep=True)
    reads = []
    read_parquet = pd.read_parquet

    def recording_read(path, **kwargs):
        reads.append((path.name, kwargs))
        return read_parquet(path, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", recording_read)

    result = variable_library.validate_daily_basic_source_keys(targets, tmp_path)

    assert result["total_rows"] == result["matched_rows"] == 1
    assert result["matched_rate"] == 1.0
    assert result["missing_rows"] == 0
    assert result["by_date"][target_date]["matched_rate"] == 1.0
    assert len(reads) == 1
    assert reads[0][0] == f"{target_date}.parquet"
    assert reads[0][1]["columns"] == ["ts_code", "trade_date", "turnover_rate"]
    assert reads[0][1]["filters"][0] == ("ts_code", "in", ["000001.SZ"])
    assert reads[0][1]["filters"][1][:2] == ("trade_date", "in")
    assert target_date in reads[0][1]["filters"][1][2]
    assert_frame_equal(targets, original)


def test_source_preflight_missing_target_does_not_read_or_borrow_history(monkeypatch, tmp_path, source_preflight_only) -> None:
    dates = _write_daily_basic_history(tmp_path, periods=5)
    targets = pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": ["20260112"]})
    assert dates[-1] < pd.Timestamp("20260112")

    def reject_read(*args, **kwargs):
        pytest.fail("Historical files must not be read for missing target dates")

    monkeypatch.setattr(pd, "read_parquet", reject_read)

    with pytest.raises(RuntimeError, match="missing_source_rows=1 turnover_rate_null_rows=0"):
        variable_library.validate_daily_basic_source_keys(targets, tmp_path)


def test_source_preflight_empty_candidates_succeed_without_io(monkeypatch, tmp_path, source_preflight_only) -> None:
    from quant.data import market_snapshot

    def reject_io(*args, **kwargs):
        pytest.fail("Empty candidates must return without source access")

    monkeypatch.setattr(market_snapshot, "current_market_snapshot", reject_io)
    monkeypatch.setattr(pd, "read_parquet", reject_io)

    result = variable_library.validate_daily_basic_source_keys(pd.DataFrame(), tmp_path)

    assert result == {
        "total_rows": 0, "matched_rows": 0, "missing_rows": 0,
        "missing_source_rows": 0, "turnover_rate_null_rows": 0,
        "matched_rate": 1.0, "min_match_rate": 0.98, "by_date": {},
    }


@pytest.mark.parametrize("date", [20260105, "2026-01-05", pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-05").date()])
def test_source_preflight_filters_legacy_date_types(tmp_path, source_preflight_only, date) -> None:
    pd.DataFrame({
        "ts_code": ["000001.SZ", "000002.SZ"], "trade_date": [date, date],
        "turnover_rate": [1.0, np.nan], "unneeded": [99.0, 99.0],
    }).to_parquet(tmp_path / "daily_basic.parquet", index=False)
    targets = pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": ["20260105"]})

    result = variable_library.validate_daily_basic_source_keys(targets, tmp_path)

    assert result["matched_rows"] == result["total_rows"] == 1


@pytest.mark.parametrize("include_turnover", [True, False])
def test_source_preflight_null_turnover_is_not_missing_source(tmp_path, source_preflight_only, include_turnover) -> None:
    source = pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": ["20260105"]})
    if include_turnover:
        source["turnover_rate"] = np.nan
    source.to_parquet(tmp_path / "20260105.parquet", index=False)

    with pytest.raises(RuntimeError, match="missing_source_rows=0 turnover_rate_null_rows=1") as error:
        variable_library.validate_daily_basic_source_keys(source, tmp_path)

    assert type(error.value) is variable_library.DailyBasicSourceCoverageError


def test_source_preflight_pinned_snapshot_uses_exact_dates_symbols_columns(monkeypatch, tmp_path, source_preflight_only) -> None:
    from quant.data import market_snapshot

    calls = []

    class Snapshot:
        def read(self, dataset, **kwargs):
            calls.append((dataset, kwargs))
            return pd.DataFrame({
                "ts_code": kwargs["symbols"],
                "trade_date": [kwargs["start_date"]],
                "turnover_rate": [1.0],
            })

    def reject_raw_read(*args, **kwargs):
        pytest.fail("Pinned preflight must not fall back to mutable parquet")

    monkeypatch.setattr(market_snapshot, "current_market_snapshot", lambda: Snapshot())
    monkeypatch.setattr(pd, "read_parquet", reject_raw_read)
    targets = pd.DataFrame({"ts_code": ["000001.SZ", "000002.SZ"], "trade_date": ["20260105", "20260107"]})

    result = variable_library.validate_daily_basic_source_keys(targets, tmp_path)

    assert result["matched_rows"] == 2
    assert result["matched_rate"] == 1.0
    assert calls == [
        ("daily_basic", {
            "start_date": date, "end_date": date, "symbols": [symbol],
            "columns": ["ts_code", "trade_date", "turnover_rate"],
        })
        for symbol, date in targets.itertuples(index=False, name=None)
    ]


def test_source_preflight_threshold_summary_counts_unique_keys(tmp_path, source_preflight_only) -> None:
    symbols = [f"{number:06d}.SZ" for number in range(100)]
    targets = pd.DataFrame({"ts_code": symbols, "trade_date": ["20260105"] * 100})
    source = targets.iloc[:99].assign(turnover_rate=[0.0] * 98 + [np.nan])
    source.to_parquet(tmp_path / "20260105.parquet", index=False)
    targets = pd.concat([targets, targets.iloc[:2]], ignore_index=True)

    result = variable_library.validate_daily_basic_source_keys(targets, tmp_path)

    counts = {
        "total_rows": 100, "matched_rows": 98, "missing_rows": 2,
        "missing_source_rows": 1, "turnover_rate_null_rows": 1, "matched_rate": 0.98,
    }
    assert result == {**counts, "min_match_rate": 0.98, "by_date": {"20260105": counts}}


@pytest.mark.parametrize("threshold", [np.nan, np.inf, -0.01, 1.01])
def test_source_preflight_rejects_invalid_threshold_even_for_empty_candidates(tmp_path, threshold) -> None:
    with pytest.raises(ValueError, match="min_match_rate must be between 0 and 1"):
        variable_library.validate_daily_basic_source_keys(pd.DataFrame(), tmp_path, min_match_rate=threshold)


def test_source_preflight_pinned_failure_does_not_fall_back(monkeypatch, tmp_path, source_preflight_only) -> None:
    from quant.data import market_snapshot

    targets = pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": ["20260105"]})
    targets.assign(turnover_rate=1.0).to_parquet(tmp_path / "20260105.parquet", index=False)

    class Snapshot:
        def read(self, *args, **kwargs):
            raise market_snapshot.MarketSnapshotError("Dataset was not sealed: daily_basic")

    monkeypatch.setattr(market_snapshot, "current_market_snapshot", lambda: Snapshot())

    with pytest.raises(market_snapshot.MarketSnapshotError, match="Dataset was not sealed") as error:
        variable_library.validate_daily_basic_source_keys(targets, tmp_path)

    assert not isinstance(error.value, variable_library.DailyBasicSourceCoverageError)


@pytest.mark.parametrize("stage", ["schema", "rows"])
@pytest.mark.parametrize("error_type", [OSError, RuntimeError])
def test_source_preflight_io_errors_are_not_coverage_failures(monkeypatch, tmp_path, source_preflight_only, stage, error_type) -> None:
    import pyarrow.parquet as pq

    targets = pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": ["20260105"]})
    targets.assign(turnover_rate=1.0).to_parquet(tmp_path / "20260105.parquet", index=False)
    failure = error_type("source read failed")

    def fail_read(*args, **kwargs):
        raise failure

    if stage == "schema":
        monkeypatch.setattr(pq, "read_schema", fail_read)
    else:
        monkeypatch.setattr(pd, "read_parquet", fail_read)

    with pytest.raises(error_type) as error:
        variable_library.validate_daily_basic_source_keys(targets, tmp_path)

    assert error.value is failure
    assert not isinstance(error.value, variable_library.DailyBasicSourceCoverageError)


def test_source_preflight_bad_schema_is_not_coverage_failure(tmp_path, source_preflight_only) -> None:
    pd.DataFrame({"turnover_rate": [1.0]}).to_parquet(tmp_path / "20260105.parquet", index=False)
    targets = pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": ["20260105"]})

    with pytest.raises(RuntimeError, match="source missing identity columns") as error:
        variable_library.validate_daily_basic_source_keys(targets, tmp_path)

    assert type(error.value) is RuntimeError
