from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quant.data.tushare_fetcher import TushareDataFetcher, validate_daily_basic_frame
from quant.routine import daily_basic_refresh


def _complete_daily_basic(trade_date: str = "20260722") -> pd.DataFrame:
    rows = 4
    return pd.DataFrame(
        {
            "ts_code": [f"00000{index}.SZ" for index in range(1, rows + 1)],
            "trade_date": [trade_date] * rows,
            "turnover_rate": [1.0] * rows,
            "turnover_rate_f": [1.2] * rows,
            "volume_ratio": [0.9] * rows,
            "pe": [10.0] * rows,
            "pe_ttm": [11.0] * rows,
            "pb": [1.1] * rows,
            "ps": [1.2] * rows,
            "ps_ttm": [1.3] * rows,
            "dv_ratio": [0.5, 0.4, None, None],
            "dv_ttm": [0.6, 0.5, None, None],
            "total_share": [100.0] * rows,
            "float_share": [80.0] * rows,
            "free_share": [70.0] * rows,
            "total_mv": [1000.0] * rows,
            "circ_mv": [800.0] * rows,
        }
    )


def test_validate_daily_basic_rejects_empty_and_wrong_date() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        validate_daily_basic_frame(pd.DataFrame(), "20260722")

    wrong_date = pd.DataFrame(
        {"ts_code": ["000001.SZ"], "trade_date": ["20260721"]}
    )
    with pytest.raises(ValueError, match="trade_date mismatch"):
        validate_daily_basic_frame(wrong_date, "20260722")


def test_get_daily_basic_does_not_keep_empty_cache(monkeypatch, tmp_path: Path) -> None:
    cache_path = tmp_path / "tushare_daily_basic_20260722.parquet"
    pd.DataFrame().to_parquet(cache_path, index=False)

    class EmptyPro:
        def daily_basic(self, **kwargs):
            return pd.DataFrame(columns=["ts_code", "trade_date"])

    fetcher = TushareDataFetcher.__new__(TushareDataFetcher)
    fetcher.cache_dir = tmp_path
    fetcher._memory_cache = {}
    fetcher.pro = EmptyPro()

    with pytest.raises(ValueError, match="returned 0 rows"):
        fetcher.get_daily_basic("20260722")

    assert not cache_path.exists()


def test_validate_daily_basic_rejects_model_feature_columns_with_no_current_values() -> None:
    partial = _complete_daily_basic()
    partial["volume_ratio"] = None

    with pytest.raises(ValueError, match="feature coverage"):
        validate_daily_basic_frame(
            partial,
            "20260722",
            required_feature_coverage=daily_basic_refresh.DAILY_BASIC_FEATURE_COVERAGE,
        )


def test_validate_daily_basic_reports_required_feature_coverage() -> None:
    frame = validate_daily_basic_frame(
        _complete_daily_basic(),
        "20260722",
        required_feature_coverage=daily_basic_refresh.DAILY_BASIC_FEATURE_COVERAGE,
    )

    assert frame.attrs["feature_coverage"]["volume_ratio"] == 1.0
    assert frame.attrs["feature_coverage"]["dv_ratio"] == 0.5


def test_fetch_one_trade_date_preserves_last_good_file_on_empty_response(
    monkeypatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "daily_basic"
    output_dir.mkdir()
    output_path = output_dir / "20260722.parquet"
    old = pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": ["20260722"]})
    old.to_parquet(output_path, index=False)

    class EmptyFetcher:
        def __init__(self, **kwargs):
            pass

        def get_daily_basic(self, trade_date: str) -> pd.DataFrame:
            return pd.DataFrame(columns=["ts_code", "trade_date"])

    monkeypatch.setattr(daily_basic_refresh, "TushareDataFetcher", EmptyFetcher)

    result = daily_basic_refresh.fetch_one_trade_date(
        "20260722",
        output_dir,
        tmp_path / "cache",
        daily_basic_refresh.RequestLimiter(0),
        retries=1,
        retry_base_delay=0,
        retry_max_delay=0,
    )

    assert result["status"] == "failed"
    pd.testing.assert_frame_equal(pd.read_parquet(output_path), old)
    assert result["requires_source_refresh"] is True
    assert daily_basic_refresh._daily_basic_provenance_path(output_path).exists()


def test_latest_daily_basic_waits_for_complete_model_fields(
    monkeypatch,
    tmp_path: Path,
) -> None:
    responses = [_complete_daily_basic(), _complete_daily_basic()]
    responses[0]["free_share"] = None

    class DelayedCompleteFetcher:
        def __init__(self, **kwargs):
            pass

        def get_daily_basic(self, trade_date: str) -> pd.DataFrame:
            return responses.pop(0)

    monkeypatch.setattr(
        daily_basic_refresh,
        "TushareDataFetcher",
        DelayedCompleteFetcher,
    )
    output_dir = tmp_path / "daily_basic"
    output_dir.mkdir()

    result = daily_basic_refresh.fetch_one_trade_date(
        "20260722",
        output_dir,
        tmp_path / "cache",
        daily_basic_refresh.RequestLimiter(0),
        retries=0,
        retry_base_delay=0,
        retry_max_delay=0,
        expected_rows=4,
        minimum_coverage_rate=1.0,
        availability_retry_failures=1,
        availability_retry_interval=0,
    )

    assert result["status"] == "success"
    assert result["attempts"] == 2
    assert result["feature_coverage"]["free_share"] == 1.0


def test_complete_local_daily_basic_is_validated_without_refetching(
    monkeypatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "daily_basic"
    output_dir.mkdir()
    _complete_daily_basic().to_parquet(output_dir / "20260722.parquet", index=False)

    class RefetchMustNotRun:
        def __init__(self, **kwargs):
            raise AssertionError("complete local daily_basic should not be refetched")

    monkeypatch.setattr(daily_basic_refresh, "TushareDataFetcher", RefetchMustNotRun)

    result = daily_basic_refresh.fetch_one_trade_date(
        "20260722",
        output_dir,
        tmp_path / "cache",
        daily_basic_refresh.RequestLimiter(0),
        retries=0,
        retry_base_delay=0,
        retry_max_delay=0,
        expected_rows=4,
        minimum_coverage_rate=1.0,
    )

    assert result["status"] == "success"
    assert result["source"] == "local_validated"
    assert result["attempts"] == 0


def test_forced_recent_recheck_bypasses_local_file_and_reports_source_change(
    monkeypatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "daily_basic"
    output_dir.mkdir()
    previous = _complete_daily_basic()
    previous.loc[0, "turnover_rate"] = 1.0
    previous.to_parquet(output_dir / "20260722.parquet", index=False)
    official = _complete_daily_basic()
    official.loc[0, "turnover_rate"] = 2.0
    calls = 0

    class Fetcher:
        def __init__(self, **kwargs):
            pass

        def get_daily_basic(self, trade_date: str) -> pd.DataFrame:
            nonlocal calls
            calls += 1
            return official.copy()

    monkeypatch.setattr(daily_basic_refresh, "TushareDataFetcher", Fetcher)

    result = daily_basic_refresh.fetch_one_trade_date(
        "20260722",
        output_dir,
        tmp_path / "cache",
        daily_basic_refresh.RequestLimiter(0),
        retries=0,
        retry_base_delay=0,
        retry_max_delay=0,
        expected_rows=4,
        minimum_coverage_rate=1.0,
        expected_symbols=set(official["ts_code"]),
        force_source_refresh=True,
    )

    assert result["status"] == "success"
    assert result["source"] == "tushare"
    assert result["source_rechecked"] is True
    assert result["source_changed_rows"] == 1
    assert "turnover_rate" in result["source_changed_columns"]
    assert calls == 1


def test_daily_basic_rejects_wrong_symbol_set_even_when_row_count_matches(
    monkeypatch,
    tmp_path: Path,
) -> None:
    wrong = _complete_daily_basic()
    wrong.loc[0, "ts_code"] = "999999.SZ"

    class Fetcher:
        def __init__(self, **kwargs):
            pass

        def get_daily_basic(self, trade_date: str) -> pd.DataFrame:
            return wrong.copy()

    monkeypatch.setattr(daily_basic_refresh, "TushareDataFetcher", Fetcher)
    result = daily_basic_refresh.fetch_one_trade_date(
        "20260722",
        tmp_path / "daily_basic",
        tmp_path / "cache",
        daily_basic_refresh.RequestLimiter(0),
        retries=0,
        retry_base_delay=0,
        retry_max_delay=0,
        expected_rows=4,
        minimum_coverage_rate=1.0,
        expected_symbols={f"00000{i}.SZ" for i in range(1, 5)},
    )

    assert result["status"] == "failed"
    assert "missing expected market symbols" in result["error"]


def test_latest_daily_basic_derives_delayed_volume_ratio_from_daily(
    monkeypatch,
    tmp_path: Path,
) -> None:
    daily_dir = tmp_path / "daily"
    partition = tmp_path / "daily_partitioned" / "year_month=202607"
    partition.mkdir(parents=True)
    rows = []
    for symbol_index in range(1, 5):
        symbol = f"00000{symbol_index}.SZ"
        for day, volume in zip(range(16, 23), (100, 110, 120, 130, 140, 150, 180)):
            rows.append(
                {"ts_code": symbol, "trade_date": f"202607{day:02d}", "vol": volume}
            )
    pd.DataFrame(rows).to_parquet(partition / "data.parquet", index=False)
    delayed = _complete_daily_basic()
    delayed["volume_ratio"] = None

    class DelayedVolumeRatioFetcher:
        def __init__(self, **kwargs):
            pass

        def get_daily_basic(self, trade_date: str) -> pd.DataFrame:
            return delayed

    monkeypatch.setattr(
        daily_basic_refresh,
        "TushareDataFetcher",
        DelayedVolumeRatioFetcher,
    )
    output_dir = tmp_path / "daily_basic"
    output_dir.mkdir()

    result = daily_basic_refresh.fetch_one_trade_date(
        "20260722",
        output_dir,
        tmp_path / "cache",
        daily_basic_refresh.RequestLimiter(0),
        retries=0,
        retry_base_delay=0,
        retry_max_delay=0,
        expected_rows=4,
        minimum_coverage_rate=1.0,
        daily_dir=daily_dir,
    )

    assert result["status"] == "success"
    assert result["derived_features"]["volume_ratio"]["filled_rows"] == 4
    saved = pd.read_parquet(output_dir / "20260722.parquet")
    assert saved["volume_ratio"].tolist() == [1.38] * 4


def test_latest_daily_basic_estimates_delayed_vendor_fields_on_final_attempt(
    monkeypatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "daily_basic"
    output_dir.mkdir()
    previous = _complete_daily_basic("20260721")
    previous.to_parquet(output_dir / "20260721.parquet", index=False)
    delayed = _complete_daily_basic()
    delayed["float_share"] = 88.0
    for column in daily_basic_refresh.DELAYED_DAILY_BASIC_COLUMNS:
        delayed[column] = None

    class DelayedVendorFetcher:
        def __init__(self, **kwargs):
            pass

        def get_daily_basic(self, trade_date: str) -> pd.DataFrame:
            return delayed.copy()

    monkeypatch.setattr(
        daily_basic_refresh,
        "TushareDataFetcher",
        DelayedVendorFetcher,
    )
    monkeypatch.setattr(daily_basic_refresh.time, "sleep", lambda _seconds: None)

    result = daily_basic_refresh.fetch_one_trade_date(
        "20260722",
        output_dir,
        tmp_path / "cache",
        daily_basic_refresh.RequestLimiter(0),
        retries=0,
        retry_base_delay=0,
        retry_max_delay=0,
        expected_rows=4,
        minimum_coverage_rate=1.0,
        availability_retry_failures=1,
        availability_retry_interval=0,
    )

    assert result["status"] == "success"
    assert result["source"] == "tushare_with_estimated_fallback"
    assert result["repair_status"] == "pending"
    assert result["requires_source_refresh"] is True
    assert set(result["derived_features"]) == {
        "free_share",
        "turnover_rate_f",
        "dv_ratio",
        "dv_ttm",
    }
    assert result["derived_features"]["free_share"]["filled_rows"] == 4
    saved = pd.read_parquet(output_dir / "20260722.parquet")
    assert saved["free_share"].tolist() == [77.0] * 4
    assert saved["turnover_rate_f"].tolist() == pytest.approx(
        [88.0 / 77.0] * 4
    )
    provenance = daily_basic_refresh._daily_basic_provenance_path(
        output_dir / "20260722.parquet"
    )
    assert provenance.exists()
    marker = daily_basic_refresh._read_source_refresh_marker(
        output_dir / "20260722.parquet"
    )
    assert set(marker["estimated_features"]) == {
        "free_share",
        "turnover_rate_f",
        "dv_ratio",
        "dv_ttm",
    }


def test_estimated_latest_daily_basic_is_replaced_by_official_source(
    monkeypatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "daily_basic"
    output_dir.mkdir()
    output_path = output_dir / "20260722.parquet"
    estimated = _complete_daily_basic()
    estimated["free_share"] = 69.0
    estimated.to_parquet(output_path, index=False)
    provenance = daily_basic_refresh._daily_basic_provenance_path(output_path)
    provenance.write_text(
        '{"requires_source_refresh": true}',
        encoding="utf-8",
    )

    class OfficialVendorFetcher:
        def __init__(self, **kwargs):
            pass

        def get_daily_basic(self, trade_date: str) -> pd.DataFrame:
            return _complete_daily_basic()

    monkeypatch.setattr(
        daily_basic_refresh,
        "TushareDataFetcher",
        OfficialVendorFetcher,
    )

    result = daily_basic_refresh.fetch_one_trade_date(
        "20260722",
        output_dir,
        tmp_path / "cache",
        daily_basic_refresh.RequestLimiter(0),
        retries=0,
        retry_base_delay=0,
        retry_max_delay=0,
        expected_rows=4,
        minimum_coverage_rate=1.0,
    )

    assert result["source"] == "tushare"
    assert result["repair_status"] == "repaired"
    assert result["repair_changed_rows"] == 4
    assert result["repair_changed_columns"] == ["free_share"]
    assert result["requires_source_refresh"] is False
    assert pd.read_parquet(output_path)["free_share"].tolist() == [70.0] * 4
    assert not provenance.exists()


def test_invalid_provenance_is_treated_as_requiring_source_refresh(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "20260722.parquet"
    provenance = daily_basic_refresh._daily_basic_provenance_path(output_path)
    provenance.write_text("not-json", encoding="utf-8")

    assert daily_basic_refresh._requires_source_refresh(output_path) is True


def test_refresh_manifest_exposes_durable_repair_queue(
    monkeypatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "daily_basic"
    output_dir.mkdir()
    pending_path = output_dir / "20260721.parquet"
    daily_basic_refresh._write_source_refresh_marker(
        pending_path,
        "20260721",
        reason="source_feature_coverage_incomplete",
    )

    monkeypatch.setattr(
        daily_basic_refresh,
        "load_trade_date_symbol_sets",
        lambda *_args, **_kwargs: {
            "20260721": {f"00000{i}.SZ" for i in range(1, 5)},
            "20260722": {f"00000{i}.SZ" for i in range(1, 5)},
        },
    )

    def fake_fetch(trade_date: str, *_args, **_kwargs) -> dict:
        if trade_date == "20260721":
            daily_basic_refresh._daily_basic_provenance_path(
                output_dir / f"{trade_date}.parquet"
            ).unlink()
            return {
                "trade_date": trade_date,
                "status": "success",
                "repair_requested": True,
                "repair_status": "repaired",
                "requires_source_refresh": False,
            }
        daily_basic_refresh._write_source_refresh_marker(
            output_dir / f"{trade_date}.parquet",
            trade_date,
            reason="estimated_fallback",
        )
        return {
            "trade_date": trade_date,
            "status": "success",
            "repair_requested": False,
            "repair_status": "pending",
            "requires_source_refresh": True,
        }

    monkeypatch.setattr(daily_basic_refresh, "fetch_one_trade_date", fake_fetch)
    monkeypatch.setattr(daily_basic_refresh, "AUDIT_ROOT", tmp_path / "audit")

    manifest = daily_basic_refresh.refresh_daily_basic(
        start_date="20260721",
        daily_dir=tmp_path / "daily",
        output_dir=output_dir,
        workers=1,
    )

    assert manifest["repair_requested_dates"] == ["20260721"]
    assert manifest["repaired_dates"] == ["20260721"]
    assert manifest["downstream_refresh_dates"] == ["20260721"]
    assert manifest["newly_flagged_dates"] == ["20260722"]
    assert manifest["pending_repair_dates"] == ["20260722"]
    assert manifest["repair_queue_size"] == 1
    assert manifest["data_quality_status"] == "provisional"
