from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import requests

from quant.data.atomic_io import atomic_write_parquet


REFERENCE_SOURCE = "eastmoney_akshare_stock_value_em"
REFERENCE_COLUMNS = ["pe_ttm_reference", "pe_ttm_reference_source", "pe_ttm_reference_date"]
INCOME_SOURCE = "tushare_income_ttm"
DERIVATION_COLUMN = "pe_ttm_derivation"


def _income_pe(proof: str, trade_date: str, market_cap: float, ts_code: str) -> float:
    evidence = json.loads(proof)
    if not isinstance(evidence, dict) or evidence.get("ts_code") != ts_code:
        raise ValueError("income evidence symbol mismatch")
    rows = evidence["reports"]
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("invalid income reports")
    if len(rows) not in {1, 3} or not np.isfinite(market_cap) or market_cap <= 0:
        raise ValueError("invalid income evidence")
    target = pd.to_datetime(trade_date, format="%Y%m%d", errors="raise")
    periods = [pd.to_datetime(row["end_date"], format="%Y%m%d", errors="raise") for row in rows]
    latest = periods[0] if len(rows) == 1 else periods[1]
    expected = ([latest] if len(rows) == 1 else
                [pd.Timestamp(latest.year - 1, 12, 31), latest,
                 pd.Timestamp(latest.year - 1, latest.month, latest.day)])
    if (periods != expected or not latest.is_quarter_end
            or (latest.month == 12) != (len(rows) == 1)):
        raise ValueError("invalid TTM periods")
    if not 0 <= (target - latest).days <= 200:
        raise ValueError("stale income evidence")
    for row, period in zip(rows, periods):
        announced = pd.to_datetime(row["available_date"], format="%Y%m%d", errors="raise")
        if not period <= announced <= target or str(row["report_type"]) not in {"1", "4"}:
            raise ValueError("income evidence not available at target date")
    profits = np.array([float(row["n_income_attr_p"]) for row in rows])
    profit = profits[0] if len(rows) == 1 else profits[0] + profits[1] - profits[2]
    if not np.isfinite(profits).all() or not np.isfinite(profit) or profit == 0:
        raise ValueError("undefined TTM earnings")
    if not np.isclose(float(evidence["market_cap_yuan"]), market_cap, rtol=1e-10):
        raise ValueError("income evidence market cap mismatch")
    signed = market_cap / profit
    if not np.isfinite(signed) or signed == 0:
        raise ValueError("undefined derived PE")
    return signed


def resolved_feature_mask(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    resolved = values.notna() & np.isfinite(values)
    if column != "pe_ttm" or not set(REFERENCE_COLUMNS).issubset(frame.columns):
        return resolved
    reference = pd.to_numeric(frame["pe_ttm_reference"], errors="coerce")
    # A verified loss is an observed, inapplicable PE, not a missing response.
    verified_loss = (
        values.isna()
        & reference.lt(0) & np.isfinite(reference)
        & frame["pe_ttm_reference_source"].eq(REFERENCE_SOURCE)
        & frame["pe_ttm_reference_date"].eq(frame["trade_date"].astype(str))
    )
    if DERIVATION_COLUMN in frame and "total_mv" in frame:
        candidates = (values.isna() & reference.lt(0)
                      & frame["pe_ttm_reference_source"].eq(INCOME_SOURCE)
                      & frame["pe_ttm_reference_date"].eq(frame["trade_date"].astype(str)))
        for index in frame.index[candidates]:
            try:
                computed = _income_pe(frame.at[index, DERIVATION_COLUMN],
                                      str(frame.at[index, "trade_date"]),
                                      float(frame.at[index, "total_mv"]) * 10000,
                                      str(frame.at[index, "ts_code"]))
                verified_loss.at[index] = computed < 0 and np.isclose(
                    computed, reference.at[index], rtol=1e-10)
            except (ValueError, TypeError, KeyError, OverflowError):
                pass
    return resolved | verified_loss


def supplement_pe_ttm_from_income(
    frame: pd.DataFrame, trade_date: str, income_path: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Derive only unresolved observations from as-of-date consolidated earnings."""
    out = frame.copy()
    unresolved = ~resolved_feature_mask(out, "pe_ttm")
    report: dict[str, object] = {"source": INCOME_SOURCE, "trade_date": trade_date,
                                 "filled_rows": 0, "confirmed_loss_rows": 0,
                                 "unresolved_rows": int(unresolved.sum())}
    if not unresolved.any() or not income_path.is_file():
        return out, report
    income = pd.read_parquet(income_path, filters=[("ts_code", "in", out.loc[unresolved, "ts_code"].tolist())])
    required = {"ts_code", "ann_date", "end_date", "report_type", "n_income_attr_p"}
    if not required.issubset(income.columns):
        raise ValueError("invalid income schema for TTM derivation")
    for column in ("ann_date", "end_date"):
        income[column] = income[column].astype(str)
    income["available_date"] = income["ann_date"]
    if "f_ann_date" in income:
        actual = income["f_ann_date"].astype("string")
        present = actual.notna() & actual.ne("")
        income.loc[present, "available_date"] = pd.concat(
            [income.loc[present, "ann_date"], actual[present]], axis=1,
        ).max(axis=1)
    income["report_type"] = income["report_type"].astype(str)
    income = income[income.report_type.isin(["1", "4"])
                    & income.available_date.str.fullmatch(r"\d{8}")
                    & income.end_date.str.fullmatch(r"\d{8}")
                    & income.available_date.le(trade_date) & income.end_date.le(trade_date)]
    for column in [*REFERENCE_COLUMNS, DERIVATION_COLUMN]:
        if column not in out:
            out[column] = pd.Series(None, index=out.index, dtype=object)
    for index in out.index[unresolved]:
        records = income[income.ts_code.eq(out.at[index, "ts_code"])]
        if records.empty:
            continue
        latest = records.end_date.max()
        periods = ([latest] if latest.endswith("1231") else
                   [f"{int(latest[:4]) - 1}1231", latest,
                    f"{int(latest[:4]) - 1}{latest[4:]}"])
        reports = []
        for period in periods:
            candidates = records[records.end_date.eq(period)].sort_values(["available_date", "report_type"])
            if candidates.empty:
                break
            chosen = candidates.iloc[-1]
            peers = candidates[(candidates.available_date == chosen.available_date)
                               & (candidates.report_type == chosen.report_type)]
            if peers.n_income_attr_p.nunique(dropna=False) != 1:
                break
            try:
                reports.append({
                    "end_date": str(chosen["end_date"]),
                    "available_date": str(chosen["available_date"]),
                    "report_type": str(chosen["report_type"]),
                    "n_income_attr_p": float(chosen["n_income_attr_p"]),
                })
            except (TypeError, ValueError):
                break
        try:
            cap = float(out.at[index, "total_mv"]) * 10000
            symbol = str(out.at[index, "ts_code"])
            proof = json.dumps({"ts_code": symbol, "market_cap_yuan": cap, "reports": reports}, allow_nan=False)
            signed = _income_pe(proof, trade_date, cap, symbol)
            if not np.isfinite(signed):
                continue
        except (ValueError, TypeError, KeyError, OverflowError):
            continue
        out.at[index, "pe_ttm_reference"] = signed
        out.at[index, "pe_ttm_reference_source"] = INCOME_SOURCE
        out.at[index, "pe_ttm_reference_date"] = trade_date
        out.at[index, DERIVATION_COLUMN] = proof
        key = "confirmed_loss_rows" if signed < 0 else "filled_rows"
        report[key] = int(report[key]) + 1
        if signed > 0:
            out.at[index, "pe_ttm"] = signed
    report["unresolved_rows"] = int((~resolved_feature_mask(out, "pe_ttm")).sum())
    return out, report


def fetch_eastmoney_valuation(trade_date: str) -> pd.DataFrame:
    """Batch the historical endpoint used by AkShare's stock_value_em.

    Query an exact date instead of downloading every stock's entire history.
    """
    day = pd.to_datetime(trade_date, format="%Y%m%d").strftime("%Y-%m-%d")
    params = {
        "reportName": "RPT_VALUEANALYSIS_DET",
        "columns": "SECUCODE,TRADE_DATE,PE_TTM,TOTAL_MARKET_CAP",
        "filter": f"(TRADE_DATE='{day}')",
        "pageSize": "5000", "pageNumber": "1",
        "sortColumns": "SECUCODE", "sortTypes": "1",
        "source": "WEB", "client": "WEB",
    }
    records: list[dict] = []
    with requests.Session() as client:
        pages = 1
        page = 1
        while page <= pages:
            params["pageNumber"] = str(page)
            response = client.get(
                "https://datacenter-web.eastmoney.com/api/data/v1/get", params=params, timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            result = payload.get("result") or {}
            if payload.get("success") is not True or not result.get("data"):
                raise ValueError(f"Eastmoney valuation unavailable for {trade_date}")
            pages = int(result["pages"])
            if not 1 <= pages <= 20:
                raise ValueError("invalid Eastmoney valuation pagination")
            records.extend(result["data"])
            page += 1
    frame = pd.DataFrame(records).rename(columns={
        "SECUCODE": "ts_code", "TRADE_DATE": "trade_date",
        "PE_TTM": "pe_ttm", "TOTAL_MARKET_CAP": "total_mv",
    })
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.strftime("%Y%m%d")
    frame["total_mv"] = pd.to_numeric(frame["total_mv"], errors="coerce") / 10000
    if len(frame) != int(result["count"]):
        raise ValueError("incomplete Eastmoney valuation pagination")
    return frame[["ts_code", "trade_date", "pe_ttm", "total_mv"]]


def supplement_pe_ttm(
    frame: pd.DataFrame,
    trade_date: str,
    cache_dir: Path,
    *,
    fetch: Callable[[str], pd.DataFrame] = fetch_eastmoney_valuation,
) -> tuple[pd.DataFrame, dict[str, object]]:
    path = cache_dir / f"valuation_{trade_date}.parquet"
    cached = path.exists() and time.time() - path.stat().st_mtime < 600
    reference = pd.read_parquet(path) if cached else fetch(trade_date)
    required = {"ts_code", "trade_date", "pe_ttm", "total_mv"}
    if not required.issubset(reference) or reference.empty:
        raise ValueError("invalid historical valuation schema")
    if not reference["trade_date"].astype(str).eq(trade_date).all():
        raise ValueError("historical valuation trade date mismatch")
    if reference["ts_code"].isna().any() or reference["ts_code"].duplicated().any():
        raise ValueError("historical valuation duplicate or blank symbols")
    if not cached:
        atomic_write_parquet(reference, path, index=False)
    reference = reference.set_index("ts_code")
    out = frame.copy()
    unresolved = ~resolved_feature_mask(out, "pe_ttm")
    values = pd.to_numeric(out["pe_ttm"], errors="coerce")
    signed = pd.to_numeric(out["ts_code"].map(reference["pe_ttm"]), errors="coerce")
    market_cap = pd.to_numeric(out["total_mv"], errors="coerce")
    ref_cap = pd.to_numeric(out["ts_code"].map(reference["total_mv"]), errors="coerce")
    matched = (
        unresolved & values.isna() & np.isfinite(signed) & signed.ne(0)
        & market_cap.gt(0) & np.isfinite(market_cap) & np.isfinite(ref_cap)
        & ((ref_cap - market_cap).abs() / market_cap).le(0.01)
    )
    # Keep Tushare's null-for-loss convention, preserving signed evidence separately.
    out.loc[matched & signed.gt(0), "pe_ttm"] = signed[matched & signed.gt(0)]
    for column in REFERENCE_COLUMNS:
        if column not in out:
            out[column] = pd.Series(None, index=out.index, dtype=object)
    out.loc[matched, "pe_ttm_reference"] = signed[matched]
    out.loc[matched, "pe_ttm_reference_source"] = REFERENCE_SOURCE
    out.loc[matched, "pe_ttm_reference_date"] = trade_date
    return out, {
        "source": REFERENCE_SOURCE, "trade_date": trade_date,
        "filled_rows": int((matched & signed.gt(0)).sum()),
        "confirmed_loss_rows": int((matched & signed.lt(0)).sum()),
        "unresolved_rows": int((~resolved_feature_mask(out, "pe_ttm")).sum()),
        "cache_path": str(path), "quality": "same_date_cross_checked",
    }
