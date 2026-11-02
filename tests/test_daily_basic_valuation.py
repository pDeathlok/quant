from __future__ import annotations

import pandas as pd
import pytest
import os
import json

from quant.data.daily_basic_valuation import (
    REFERENCE_SOURCE,
    INCOME_SOURCE,
    DERIVATION_COLUMN,
    fetch_eastmoney_valuation,
    resolved_feature_mask,
    supplement_pe_ttm,
    supplement_pe_ttm_from_income,
)
from quant.data.tushare_fetcher import validate_daily_basic_frame


def inputs():
    frame = pd.DataFrame({
        "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
        "trade_date": ["20261028"] * 3,
        "pe_ttm": [10.0, None, None], "total_mv": [100.0] * 3,
    })
    reference = frame.copy()
    reference["pe_ttm"] = [99.0, 20.0, -30.0]
    return frame, reference


def test_supplement_preserves_official_values_and_tushare_loss_convention(tmp_path):
    frame, reference = inputs()
    out, report = supplement_pe_ttm(frame, "20261028", tmp_path, fetch=lambda _: reference)
    assert out.pe_ttm.iloc[:2].tolist() == [10, 20]
    assert pd.isna(out.pe_ttm.iloc[2])
    assert out.pe_ttm_reference.iloc[2] == -30
    assert report["filled_rows"] == report["confirmed_loss_rows"] == 1
    assert resolved_feature_mask(out, "pe_ttm").all()
    result = validate_daily_basic_frame(out, "20261028", required_feature_coverage={"pe_ttm": 0.7})
    assert result.attrs["feature_coverage"]["pe_ttm"] == 2 / 3
    assert result.attrs["resolved_feature_coverage"]["pe_ttm"] == 1
    loaded = pd.read_parquet(report["cache_path"])
    pd.testing.assert_frame_equal(loaded, reference)
    # An exact-date cache prevents repeat downloads during an immediate retry.
    supplement_pe_ttm(frame, "20261028", tmp_path, fetch=lambda _: pytest.fail("cache missed"))
    os.utime(report["cache_path"], (0, 0))
    with pytest.raises(RuntimeError, match="source unavailable"):
        def unavailable(_):
            raise RuntimeError("source unavailable")
        supplement_pe_ttm(frame, "20261028", tmp_path, fetch=unavailable)


@pytest.mark.parametrize("problem", ["wrong_date", "duplicate", "missing_column"])
def test_rejects_invalid_reference(tmp_path, problem):
    frame, reference = inputs()
    if problem == "wrong_date":
        reference["trade_date"] = "20261027"
    elif problem == "duplicate":
        reference = pd.concat([reference, reference.iloc[:1]])
    else:
        reference = reference.drop(columns="pe_ttm")
    with pytest.raises(ValueError):
        supplement_pe_ttm(frame, "20261028", tmp_path, fetch=lambda _: reference)
    assert not list(tmp_path.glob("*.parquet"))


@pytest.mark.parametrize("value,cap", [(0, 100), (float("inf"), 100), (-30, 200)])
def test_unverified_or_undefined_values_do_not_satisfy_gate(tmp_path, value, cap):
    frame, reference = inputs()
    reference.loc[1:, "pe_ttm"] = value
    reference.loc[1:, "total_mv"] = cap
    out, _ = supplement_pe_ttm(frame, "20261028", tmp_path, fetch=lambda _: reference)
    with pytest.raises(ValueError, match="coverage"):
        validate_daily_basic_frame(out, "20261028", required_feature_coverage={"pe_ttm": 0.7})


def test_wrong_day_evidence_does_not_satisfy_gate():
    frame, _ = inputs()
    frame["pe_ttm_reference"] = -30
    frame["pe_ttm_reference_source"] = REFERENCE_SOURCE
    frame["pe_ttm_reference_date"] = "20261027"
    assert resolved_feature_mask(frame, "pe_ttm").tolist() == [True, False, False]


def test_reference_retry_preserves_previously_verified_income_and_vendor_evidence(tmp_path):
    frame, reference = inputs()
    income_path = tmp_path / "income.parquet"
    income_records().to_parquet(income_path)
    frame, _ = supplement_pe_ttm_from_income(frame, "20261028", income_path)
    proof = frame.loc[2, DERIVATION_COLUMN]
    reference.loc[2, "total_mv"] = 200
    out, report = supplement_pe_ttm(frame, "20261028", tmp_path, fetch=lambda _: reference)
    assert out.loc[2, DERIVATION_COLUMN] == proof
    assert out.loc[2, "pe_ttm_reference_source"] == INCOME_SOURCE
    assert report["unresolved_rows"] == 0
    again, report = supplement_pe_ttm(out, "20261028", tmp_path)
    assert resolved_feature_mask(again, "pe_ttm").all()
    assert report["unresolved_rows"] == 0


def test_batch_fetch_paginates_and_converts_market_cap(monkeypatch):
    calls = []

    class Response:
        def __init__(self, page):
            self.page = page

        def raise_for_status(self):
            pass

        def json(self):
            return {"success": True, "result": {"count": 2, "pages": 2, "data": [{
                "SECUCODE": f"00000{self.page}.SZ", "TRADE_DATE": "2026-10-28",
                "PE_TTM": 10, "TOTAL_MARKET_CAP": 1000000,
            }]}}

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, url, *, params, timeout):
            calls.append(dict(params))
            assert timeout == 30
            return Response(params["pageNumber"])

    monkeypatch.setattr("quant.data.daily_basic_valuation.requests.Session", Session)
    out = fetch_eastmoney_valuation("20261028")
    assert len(out) == len(calls) == 2
    assert out.total_mv.tolist() == [100, 100]
    assert out.trade_date.tolist() == ["20261028"] * 2


def income_records(symbol="000003.SZ"):
    return pd.DataFrame({
        "ts_code": [symbol] * 3,
        "end_date": ["20251231", "20260930", "20250930"],
        "ann_date": ["20260410", "20261025", "20251025"],
        "report_type": ["1"] * 3,
        "n_income_attr_p": [100000.0, -50000.0, 100000.0],
    })


def test_income_fallback_resolves_cap_mismatch_without_relaxing_reference_gate(tmp_path):
    frame, reference = inputs()
    reference.loc[2, "total_mv"] = 98
    income = income_records()
    path = tmp_path / "income.parquet"
    income.to_parquet(path)
    out, _ = supplement_pe_ttm(frame, "20261028", tmp_path, fetch=lambda _: reference)
    assert not resolved_feature_mask(out, "pe_ttm").iloc[2]
    out, report = supplement_pe_ttm_from_income(out, "20261028", path)
    assert pd.isna(out.pe_ttm.iloc[2])
    assert out.pe_ttm_reference.iloc[2] == -20
    assert out.pe_ttm_reference_source.iloc[2] == INCOME_SOURCE
    assert resolved_feature_mask(out, "pe_ttm").all()
    assert report["confirmed_loss_rows"] == 1
    assert report["unresolved_rows"] == 0
    assert out.pe_ttm.iloc[0] == 10
    assert out.pe_ttm.iloc[1] == 20


@pytest.mark.parametrize("damage", ["future", "actual_future", "missing", "zero", "nan",
                                    "conflicting", "parent_only", "invalid_period"])
def test_income_fallback_rejects_unavailable_or_ambiguous_inputs(tmp_path, damage):
    frame, _ = inputs()
    income = income_records()
    if damage == "future":
        income.loc[1, "ann_date"] = "20261029"
    elif damage == "actual_future":
        income["f_ann_date"] = income.ann_date
        income.loc[1, "f_ann_date"] = "20261029"
    elif damage == "missing":
        income = income.iloc[1:]
    elif damage == "zero":
        income.loc[1, "n_income_attr_p"] = 0
    elif damage == "nan":
        income.loc[1, "n_income_attr_p"] = float("nan")
    elif damage == "conflicting":
        other = income.iloc[[1]].copy()
        other["n_income_attr_p"] = -100
        income = pd.concat([income, other])
    elif damage == "parent_only":
        income.report_type = "6"
    else:
        income.loc[1, "end_date"] = "20260929"
    path = tmp_path / "income.parquet"
    income.to_parquet(path)
    out, report = supplement_pe_ttm_from_income(frame, "20261028", path)
    assert report["confirmed_loss_rows"] == 0
    assert not resolved_feature_mask(out, "pe_ttm").iloc[2]


@pytest.mark.parametrize("damage", ["missing", "date", "symbol", "cap", "value", "shape"])
def test_income_loss_evidence_is_recomputed_by_quality_gate(tmp_path, damage):
    frame, _ = inputs()
    path = tmp_path / "income.parquet"
    income_records().to_parquet(path)
    out, _ = supplement_pe_ttm_from_income(frame, "20261028", path)
    assert resolved_feature_mask(out, "pe_ttm").iloc[2]
    proof = json.loads(out.loc[2, DERIVATION_COLUMN])
    if damage == "missing":
        out.loc[2, DERIVATION_COLUMN] = None
    elif damage == "date":
        proof["reports"][1]["available_date"] = "20261029"
    elif damage == "symbol":
        proof["ts_code"] = "000002.SZ"
    elif damage == "cap":
        out.loc[2, "total_mv"] = 101
    elif damage == "value":
        out.loc[2, "pe_ttm_reference"] = -21
    else:
        proof["reports"] = "invalid"
    if damage not in {"missing"}:
        out.loc[2, DERIVATION_COLUMN] = json.dumps(proof)
    assert not resolved_feature_mask(out, "pe_ttm").iloc[2]


def test_income_fallback_supports_annual_positive_earnings_and_adjusted_reports(tmp_path):
    frame, _ = inputs()
    frame["trade_date"] = "20260428"
    income = income_records().iloc[[0]].copy()
    income.ts_code = "000003.SZ"
    adjusted = income.copy()
    adjusted.report_type = "4"
    adjusted.ann_date = "20260420"
    adjusted.n_income_attr_p = 200000.0
    path = tmp_path / "income.parquet"
    pd.concat([income, adjusted]).to_parquet(path)
    out, report = supplement_pe_ttm_from_income(frame, "20260428", path)
    assert out.pe_ttm.iloc[2] == 5
    assert report["filled_rows"] == 1
