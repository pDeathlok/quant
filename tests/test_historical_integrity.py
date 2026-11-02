from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import pytest

from quant.routine.daily_basic_refresh import (
    DAILY_BASIC_FEATURE_COVERAGE,
    _required_basic_symbols,
    acknowledge_daily_basic_change_dates,
    pending_daily_basic_change_dates,
    record_daily_basic_change_dates,
)
from quant.routine.historical_integrity import (
    acknowledge_backfill_change_dates,
    audit_historical_inputs,
    pending_backfill_change_dates,
)


def _write_inputs(
    root: Path, date: str, *, basic_codes: list[str], daily_codes: list[str] | None = None
) -> tuple[Path, Path]:
    daily_root = root / "daily_partitioned"
    basic_root = root / "daily_basic"
    partition = daily_root / f"year_month={date[:6]}"
    partition.mkdir(parents=True)
    basic_root.mkdir(parents=True)
    daily_codes = daily_codes or ["000001.SZ"]
    pd.DataFrame({
        "trade_date": [date] * len(daily_codes), "ts_code": daily_codes,
        "open": [1.0] * len(daily_codes), "high": [1.0] * len(daily_codes),
        "low": [1.0] * len(daily_codes), "close": [1.0] * len(daily_codes),
    }).to_parquet(partition / "data.parquet", index=False)
    basic = {"trade_date": [date] * len(basic_codes), "ts_code": basic_codes}
    basic.update({column: [1.0] * len(basic_codes) for column in DAILY_BASIC_FEATURE_COVERAGE})
    basic["turnover_rate"] = [1.0 if code == "000001.SZ" else 0.0 for code in basic_codes]
    pd.DataFrame(basic).to_parquet(basic_root / f"{date}.parquet", index=False)
    return daily_root, basic_root


def test_historical_audit_accepts_complete_date(tmp_path: Path) -> None:
    daily, basic = _write_inputs(tmp_path, "20261015", basic_codes=["000001.SZ", "000002.SZ"])

    result = audit_historical_inputs(["20261015"], "20261015", daily_root=daily, basic_root=basic)

    assert result["status"] == "success"
    assert result["calendar_dates"] == 1
    assert result["failure_count"] == 0


def test_historical_audit_fails_missing_calendar_date(tmp_path: Path) -> None:
    daily, basic = _write_inputs(tmp_path, "20261015", basic_codes=["000001.SZ"])

    result = audit_historical_inputs(
        ["20261014", "20261015"], "20261015", daily_root=daily, basic_root=basic
    )

    assert {item["reason"] for item in result["failures"]} == {"missing_trade_date", "missing_file"}


def test_historical_audit_fails_missing_required_feature(tmp_path: Path) -> None:
    daily, basic = _write_inputs(tmp_path, "20261015", basic_codes=["000001.SZ"])
    path = basic / "20261015.parquet"
    frame = pd.read_parquet(path)
    frame["free_share"] = float("nan")
    frame.to_parquet(path, index=False)

    result = audit_historical_inputs(["20261015"], "20261015", daily_root=daily, basic_root=basic)

    assert result["status"] == "failed"
    assert any(item.get("column") == "free_share" for item in result["failures"])


def test_historical_audit_accepts_same_day_confirmed_loss_only(tmp_path: Path) -> None:
    from quant.data.daily_basic_valuation import REFERENCE_SOURCE

    daily, basic = _write_inputs(tmp_path, "20261015", basic_codes=["000001.SZ"])
    path = basic / "20261015.parquet"
    frame = pd.read_parquet(path)
    frame["pe_ttm"] = float("nan")
    frame["pe_ttm_reference"] = -10.0
    frame["pe_ttm_reference_source"] = REFERENCE_SOURCE
    frame["pe_ttm_reference_date"] = "20261015"
    frame.to_parquet(path, index=False)
    result = audit_historical_inputs(["20261015"], "20261015", daily_root=daily, basic_root=basic)
    assert result["failure_count"] == 0
    assert result["valuation_verification"][0]["confirmed_loss_rows"] == 1
    frame["pe_ttm_reference_date"] = "20261014"
    frame.to_parquet(path, index=False)
    result = audit_historical_inputs(["20261015"], "20261015", daily_root=daily, basic_root=basic)
    assert any(item.get("column") == "pe_ttm" for item in result["failures"])


def test_historical_audit_recomputes_persisted_income_evidence(tmp_path):
    from quant.data.daily_basic_valuation import supplement_pe_ttm_from_income

    daily, basic = _write_inputs(tmp_path, "20261030", basic_codes=["000001.SZ"])
    path = basic / "20261030.parquet"
    frame = pd.read_parquet(path)
    frame["pe_ttm"] = float("nan")
    frame["total_mv"] = 100
    income_path = tmp_path / "income.parquet"
    pd.DataFrame({
        "ts_code": ["000001.SZ"] * 3,
        "end_date": ["20251231", "20260930", "20250930"],
        "ann_date": ["20260410", "20261025", "20251025"],
        "report_type": ["1"] * 3, "n_income_attr_p": [100000, -50000, 100000],
    }).to_parquet(income_path)
    frame, _ = supplement_pe_ttm_from_income(frame, "20261030", income_path)
    frame.to_parquet(path, index=False)
    result = audit_historical_inputs(["20261030"], "20261030", daily_root=daily, basic_root=basic)
    assert result["failure_count"] == 0
    assert result["valuation_verification"][0]["unresolved_rows"] == 0
    frame["total_mv"] = 101
    frame.to_parquet(path, index=False)
    result = audit_historical_inputs(["20261030"], "20261030", daily_root=daily, basic_root=basic)
    assert any(item.get("column") == "pe_ttm" for item in result["failures"])


def test_historical_audit_fails_missing_traded_price(tmp_path: Path) -> None:
    daily, basic = _write_inputs(tmp_path, "20261015", basic_codes=["000001.SZ", "000002.SZ"])
    path = basic / "20261015.parquet"
    frame = pd.read_parquet(path)
    frame.loc[frame["ts_code"] == "000002.SZ", "turnover_rate"] = 2.0
    frame.to_parquet(path, index=False)

    result = audit_historical_inputs(["20261015"], "20261015", daily_root=daily, basic_root=basic)

    assert result["status"] == "failed"
    assert result["missing_traded_price_symbols"] == ["000002.SZ"]


def test_legacy_bj_identifier_is_not_required_in_old_basic() -> None:
    codes = {"000001.SZ", "920964.BJ", "832317.BJ"}

    assert _required_basic_symbols(codes, "20181224") == {"000001.SZ"}
    assert _required_basic_symbols(codes, "20261015") == codes


def test_historical_audit_classifies_prelisting_bj_without_warning(tmp_path: Path) -> None:
    daily, basic = _write_inputs(
        tmp_path, "20130118", basic_codes=["000001.SZ"],
        daily_codes=["000001.SZ", "920139.BJ"],
    )
    history = tmp_path / "stock_basic_history.parquet"
    pd.DataFrame({"ts_code": ["920139.BJ"], "list_date": ["20221028"]}).to_parquet(history)

    result = audit_historical_inputs(
        ["20130118"], "20130118", daily_root=daily, basic_root=basic,
        stock_basic_history_path=history,
    )

    assert result["status"] == "success"
    assert result["warning_count"] == 0
    assert result["source_exclusions"]["legacy_bj_before_listing"]["symbol_count"] == 1


def test_historical_audit_excludes_bj_bar_before_exchange_open(tmp_path: Path) -> None:
    daily, basic = _write_inputs(
        tmp_path, "20211112", basic_codes=["000001.SZ"],
        daily_codes=["000001.SZ", "920198.BJ"],
    )
    history = tmp_path / "stock_basic_history.parquet"
    pd.DataFrame({"ts_code": ["920198.BJ"], "list_date": ["20200727"]}).to_parquet(history)

    result = audit_historical_inputs(
        ["20211112"], "20211112", daily_root=daily, basic_root=basic,
        stock_basic_history_path=history,
    )

    assert result["status"] == "success"
    assert result["source_exclusions"]["legacy_bj_before_listing"]["symbol_count"] == 1


def test_historical_audit_fails_closed_without_bj_identity_history(tmp_path: Path) -> None:
    daily, basic = _write_inputs(
        tmp_path, "20130118", basic_codes=["000001.SZ"],
        daily_codes=["000001.SZ", "920139.BJ"],
    )

    result = audit_historical_inputs(
        ["20130118"], "20130118", daily_root=daily, basic_root=basic,
        stock_basic_history_path=tmp_path / "missing.parquet",
    )

    assert result["status"] == "failed"
    assert {item["reason"] for item in result["failures"]} == {
        "unavailable_bj_listing_history", "missing_price_symbols",
    }


@pytest.mark.parametrize("date,list_date", [
    ("20211115", "20200727"),
    ("20221031", "20221031"),
])
def test_historical_audit_fails_missing_bj_after_listing(
    tmp_path: Path, date: str, list_date: str
) -> None:
    daily, basic = _write_inputs(
        tmp_path, date, basic_codes=["000001.SZ"],
        daily_codes=["000001.SZ", "920753.BJ"],
    )
    history = tmp_path / "stock_basic_history.parquet"
    pd.DataFrame({"ts_code": ["920753.BJ"], "list_date": [list_date]}).to_parquet(history)

    result = audit_historical_inputs(
        [date], date, daily_root=daily, basic_root=basic,
        stock_basic_history_path=history,
    )

    assert result["status"] == "failed"
    assert result["warning_count"] == 0
    assert any(item["reason"] == "missing_price_symbols" for item in result["failures"])


def test_historical_audit_rejects_calendar_without_target(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="incomplete trade calendar"):
        audit_historical_inputs([], "20261015", daily_root=tmp_path, basic_root=tmp_path)


def test_daily_basic_changes_remain_pending_until_acknowledged(tmp_path: Path) -> None:
    ledger = tmp_path / "dirty.json"

    record_daily_basic_change_dates(["20181224"], path=ledger)
    record_daily_basic_change_dates(["20260616", "20181224"], path=ledger)
    assert pending_daily_basic_change_dates(ledger) == ["20181224", "20260616"]
    assert acknowledge_daily_basic_change_dates(["20181224"], path=ledger) == ["20260616"]


def test_historical_backfill_dates_remain_pending_until_publication(tmp_path: Path) -> None:
    report = tmp_path / "backfill.json"
    report.write_text(json.dumps({
        "status": "success", "changed_dates": ["20181224", "20260616"],
    }))

    assert pending_backfill_change_dates(report) == ["20181224", "20260616"]
    acknowledge_backfill_change_dates(["20181224", "20260616"], report)
    assert pending_backfill_change_dates(report) == []


def test_incomplete_backfill_cannot_acknowledge_dirty_dates(tmp_path: Path) -> None:
    report = tmp_path / "backfill.json"
    report.write_text(json.dumps({"status": "running", "changed_dates": ["20181224"]}))

    assert pending_backfill_change_dates(report) == ["20181224"]
    with pytest.raises(RuntimeError, match="not complete"):
        acknowledge_backfill_change_dates(["20181224"], report)
