"""Incrementally refresh reference and slow-moving factor inputs."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from quant.data.tushare_fetcher import TushareDataFetcher
from quant.data.long_factor_backfill import (
    RequestPolicy,
    backfill_trade_date_partitions,
    refresh_holder_trade_recent,
)
from quant.data.tradability import build_daily_tradability
from quant.routine.paths import PROJECT_ROOT


RAW_DIR = PROJECT_ROOT / "data/raw"
AUDIT_ROOT = RAW_DIR / "source_audit"
REFERENCE_STATE_PATH = RAW_DIR / "reference_refresh_state.json"
INDEX_CODE = "000300.SH"

FINANCIAL_DATASETS: dict[str, dict[str, Any]] = {
    "fina_indicator": {
        "method": "fina_indicator_vip",
        "path": "fina_indicator.parquet",
        "fields": (
            "ts_code,end_date,ann_date,eps,roe,roe_waa,roe_dt,roa,"
            "netprofit_margin,grossprofit_margin,debt_to_assets,current_ratio,"
            "quick_ratio,ar_turn,inv_turn,assets_turn,profit_to_gr,basic_eps_yoy,or_yoy"
        ),
        "dedupe": ["ts_code", "ann_date", "end_date"],
    },
    "income": {
        "method": "income_vip",
        "path": "income.parquet",
        "fields": (
            "ts_code,ann_date,end_date,report_type,revenue,operate_profit,total_profit,"
            "n_income,n_income_attr_p,total_revenue,income_tax,minority_gain"
        ),
        "dedupe": ["ts_code", "ann_date", "end_date", "report_type"],
    },
    "cashflow": {
        "method": "cashflow_vip",
        "path": "cashflow.parquet",
        "fields": (
            "ts_code,ann_date,end_date,report_type,n_cashflow_act,"
            "c_pay_acq_const_fiolta,net_profit"
        ),
        "dedupe": ["ts_code", "ann_date", "end_date", "report_type"],
    },
}


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f".{os.getpid()}.tmp.parquet")
    frame.to_parquet(temp_path, index=False)
    os.replace(temp_path, path)


def _atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f".{os.getpid()}.tmp.json")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp_path, path)


def _financial_checkpoint_current(state_path: Path, end_date: str) -> bool:
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError):
        return False
    return (
        state.get("financial_status") == "success"
        and str(state.get("end_date") or "") == end_date
        and str(state.get("refreshed_on") or "") == datetime.now().date().isoformat()
    )


def _merge_frame(path: Path, new_frame: pd.DataFrame, dedupe: list[str], sort: list[str]) -> pd.DataFrame:
    existing = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    combined = pd.concat([existing, new_frame], ignore_index=True, sort=False)
    if combined.empty:
        return combined
    keys = [column for column in dedupe if column in combined.columns]
    if keys:
        combined = combined.drop_duplicates(keys, keep="last")
    order = [column for column in sort if column in combined.columns]
    if order:
        combined = combined.sort_values(order).reset_index(drop=True)
    _atomic_write_parquet(combined, path)
    return combined


def _report_periods(end_date: str, count: int) -> list[str]:
    cursor = pd.Timestamp(end_date) + pd.offsets.QuarterEnd(-1)
    return [(cursor - pd.offsets.QuarterEnd(index)).strftime("%Y%m%d") for index in range(max(1, count))]


def refresh_stock_basic(fetcher: TushareDataFetcher, raw_dir: Path) -> dict[str, Any]:
    # The daily market refresh already checks this 24-hour cache.  Reuse that
    # result here instead of making the same stock_basic request twice per run.
    frame = fetcher.get_stock_basic(force_refresh=False)
    if frame is None or frame.empty:
        raise RuntimeError("Tushare stock_basic returned no rows")
    frame = frame.drop_duplicates("ts_code", keep="last").sort_values("ts_code").reset_index(drop=True)
    path = raw_dir / "stock_basic.parquet"
    _atomic_write_parquet(frame, path)
    return {"status": "success", "rows": len(frame), "path": str(path)}


def refresh_index_daily(fetcher: TushareDataFetcher, raw_dir: Path, end_date: str) -> dict[str, Any]:
    path = raw_dir / f"index_{INDEX_CODE}.parquet"
    existing = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    if not existing.empty and "trade_date" in existing.columns:
        latest = pd.to_datetime(existing["trade_date"].astype(str), format="%Y%m%d", errors="coerce").max()
        start = (latest - pd.Timedelta(days=10)).strftime("%Y%m%d") if pd.notna(latest) else "20100101"
    else:
        start = "20100101"
    fresh = fetcher.pro.index_daily(ts_code=INDEX_CODE, start_date=start, end_date=end_date)
    if fresh is None:
        fresh = pd.DataFrame()
    combined = _merge_frame(
        path,
        fresh,
        dedupe=["ts_code", "trade_date"],
        sort=["trade_date", "ts_code"],
    )
    latest_text = str(combined["trade_date"].astype(str).max()) if not combined.empty else None
    return {
        "status": "success" if latest_text == end_date else "partial",
        "requested_start": start,
        "requested_end": end_date,
        "new_rows": len(fresh),
        "total_rows": len(combined),
        "latest_trade_date": latest_text,
        "path": str(path),
    }


def _request_frame_with_retries(
    operation,
    *,
    retries: int,
    sleep_seconds: float,
) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            frame = operation()
            return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()
        except Exception as exc:
            last_error = exc
            if attempt + 1 < max(1, retries) and sleep_seconds > 0:
                time.sleep(sleep_seconds * (attempt + 1))
    assert last_error is not None
    raise last_error


def refresh_daily_tradability(
    fetcher: TushareDataFetcher,
    raw_dir: Path,
    trade_date: str,
    *,
    stock_basic: pd.DataFrame | None = None,
    minimum_coverage_rate: float | None = None,
    retries: int | None = None,
) -> dict[str, Any]:
    """Refresh one idempotent Tushare A-share tradability partition."""

    basic = (
        stock_basic.copy()
        if stock_basic is not None
        else pd.read_parquet(raw_dir / "stock_basic.parquet")
    )
    retry_count = retries or int(os.getenv("ROUTINE_TRADABILITY_RETRIES", "3"))
    retry_sleep = float(os.getenv("ROUTINE_TRADABILITY_RETRY_SLEEP", "0.5"))
    limits = _request_frame_with_retries(
        lambda: fetcher.pro.stk_limit(trade_date=trade_date),
        retries=retry_count,
        sleep_seconds=retry_sleep,
    )
    suspensions = _request_frame_with_retries(
        lambda: fetcher.pro.suspend_d(suspend_type="S", trade_date=trade_date),
        retries=retry_count,
        sleep_seconds=retry_sleep,
    )
    st_stocks = _request_frame_with_retries(
        lambda: fetcher.pro.stock_st(trade_date=trade_date),
        retries=retry_count,
        sleep_seconds=retry_sleep,
    )
    frame, audit = build_daily_tradability(
        trade_date=trade_date,
        stock_basic=basic,
        limits=limits,
        suspensions=suspensions,
        st_stocks=st_stocks,
        minimum_coverage_rate=(
            minimum_coverage_rate
            if minimum_coverage_rate is not None
            else float(os.getenv("ROUTINE_TRADABILITY_MIN_COVERAGE_RATE", "0.98"))
        ),
    )
    path = raw_dir / "tradability" / f"{trade_date}.parquet"
    _atomic_write_parquet(frame, path)
    return {
        "status": "success",
        **audit,
        "path": str(path),
    }


def refresh_financial_periods(
    fetcher: TushareDataFetcher,
    raw_dir: Path,
    end_date: str,
    period_count: int = 4,
    sleep_seconds: float = 0.15,
) -> dict[str, Any]:
    periods = _report_periods(end_date, period_count)
    results: dict[str, Any] = {}
    for dataset, spec in FINANCIAL_DATASETS.items():
        frames: list[pd.DataFrame] = []
        errors: list[str] = []
        method = getattr(fetcher.pro, spec["method"])
        for period in periods:
            try:
                frame = method(period=period, fields=spec["fields"])
                if frame is not None and not frame.empty:
                    frames.append(frame)
            except Exception as exc:
                errors.append(f"{period}: {exc}")
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
        fresh = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
        path = raw_dir / spec["path"]
        combined = _merge_frame(
            path,
            fresh,
            dedupe=spec["dedupe"],
            sort=["ts_code", "ann_date", "end_date"],
        )
        results[dataset] = {
            "status": "success" if not errors else "partial",
            "periods": periods,
            "new_rows": len(fresh),
            "total_rows": len(combined),
            "latest_ann_date": str(combined["ann_date"].astype(str).max()) if "ann_date" in combined else None,
            "errors": errors,
            "path": str(path),
        }
    return {
        "status": "success" if all(item["status"] == "success" for item in results.values()) else "partial",
        "periods": periods,
        "datasets": results,
    }


def refresh_long_factor_daily_sources(
    fetcher: TushareDataFetcher,
    raw_dir: Path,
    audit_dir: Path,
    trade_date: str,
) -> dict[str, Any]:
    """Refresh cheap daily/event sources while leaving full pledge scans low-frequency."""

    required_methods = [
        "trade_cal",
        "margin_detail",
        "moneyflow",
        "top_list",
        "stk_holdertrade",
    ]
    missing_methods = [
        method for method in required_methods if not callable(getattr(fetcher.pro, method, None))
    ]
    if missing_methods:
        return {
            "status": "skipped",
            "reason": f"provider does not expose: {','.join(missing_methods)}",
        }
    policy = RequestPolicy(
        retries=max(1, int(os.getenv("ROUTINE_LONG_FACTOR_RETRIES", "3"))),
        sleep_seconds=max(0.0, float(os.getenv("ROUTINE_LONG_FACTOR_SLEEP", "0.15"))),
        max_retry_wait_seconds=max(
            0.0, float(os.getenv("ROUTINE_LONG_FACTOR_MAX_RETRY_WAIT", "60"))
        ),
    )
    datasets: dict[str, Any] = {}
    for dataset in ("margin_detail", "moneyflow", "top_list"):
        datasets[dataset] = backfill_trade_date_partitions(
            fetcher.pro,
            dataset,
            trade_date,
            trade_date,
            raw_dir,
            audit_dir,
            policy=policy,
        )
    datasets["holder_trade_recent"] = refresh_holder_trade_recent(
        fetcher.pro,
        trade_date,
        raw_dir,
        audit_dir,
        lookback_days=max(1, int(os.getenv("ROUTINE_HOLDER_TRADE_LOOKBACK_DAYS", "45"))),
        policy=policy,
    )
    statuses = {item.get("status") for item in datasets.values()}
    status = (
        "failed"
        if "failed" in statuses
        else ("partial" if statuses & {"partial", "deferred"} else "success")
    )
    return {"status": status, "datasets": datasets}


def refresh_reference_data(
    end_date: str,
    *,
    include_financials: bool = True,
    fetcher: TushareDataFetcher | None = None,
    raw_dir: Path = RAW_DIR,
    audit_root: Path = AUDIT_ROOT,
    financial_periods: int | None = None,
    force_financials: bool = False,
    include_long_factor_sources: bool = True,
    state_path: Path | None = None,
) -> dict[str, Any]:
    """Refresh all non-daily inputs with bounded, idempotent requests."""

    fetcher = fetcher or TushareDataFetcher(cache_dir=PROJECT_ROOT / "data/cache/source_merge/tushare")
    steps: dict[str, Any] = {}
    critical_errors: list[str] = []
    audit_dir = audit_root / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_reference_data"
    audit_dir.mkdir(parents=True, exist_ok=True)
    stock_basic_frame: pd.DataFrame | None = None
    for name, operation in [
        ("stock_basic", lambda: refresh_stock_basic(fetcher, raw_dir)),
        ("index_000300", lambda: refresh_index_daily(fetcher, raw_dir, end_date)),
    ]:
        try:
            steps[name] = operation()
            if name == "stock_basic":
                stock_basic_frame = pd.read_parquet(raw_dir / "stock_basic.parquet")
        except Exception as exc:
            steps[name] = {"status": "failed", "error": str(exc)}
            critical_errors.append(f"{name}: {exc}")
    if stock_basic_frame is None:
        steps["tradability"] = {
            "status": "failed",
            "error": "stock_basic is unavailable",
        }
        critical_errors.append("tradability: stock_basic is unavailable")
    else:
        try:
            steps["tradability"] = refresh_daily_tradability(
                fetcher,
                raw_dir,
                end_date,
                stock_basic=stock_basic_frame,
            )
        except Exception as exc:
            steps["tradability"] = {"status": "failed", "error": str(exc)}
            critical_errors.append(f"tradability: {exc}")
    if include_long_factor_sources:
        try:
            steps["long_factor_sources"] = refresh_long_factor_daily_sources(
                fetcher,
                raw_dir,
                audit_dir,
                end_date,
            )
        except Exception as exc:
            steps["long_factor_sources"] = {"status": "failed", "error": str(exc)}
    else:
        steps["long_factor_sources"] = {
            "status": "skipped",
            "reason": "not required for this refresh scope",
        }
    effective_state_path = state_path or raw_dir / REFERENCE_STATE_PATH.name
    financial_checkpoint_hit = (
        include_financials
        and not force_financials
        and _financial_checkpoint_current(effective_state_path, end_date)
        and all((raw_dir / spec["path"]).is_file() for spec in FINANCIAL_DATASETS.values())
    )
    if financial_checkpoint_hit:
        steps["financials"] = {
            "status": "skipped",
            "reason": "successful financial refresh for this trade date already completed today",
            "checkpoint": str(effective_state_path),
        }
    elif include_financials:
        steps["financials"] = refresh_financial_periods(
            fetcher,
            raw_dir,
            end_date,
            period_count=financial_periods or int(os.getenv("ROUTINE_FINANCIAL_PERIODS", "4")),
            sleep_seconds=float(os.getenv("ROUTINE_FINANCIAL_SLEEP", "0.15")),
        )
    else:
        steps["financials"] = {"status": "skipped", "reason": "not required for this refresh scope"}
    financial_status = steps["financials"].get("status")
    long_factor_status = steps["long_factor_sources"].get("status")
    status = (
        "failed"
        if critical_errors
        else (
            "partial"
            if financial_status == "partial"
            or long_factor_status in {"failed", "partial", "deferred"}
            else "success"
        )
    )
    manifest = {
        "status": status,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "end_date": end_date,
        "steps": steps,
        "critical_errors": critical_errors,
    }
    manifest_path = audit_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    if include_financials and financial_status == "success":
        _atomic_write_json(
            {
                "end_date": end_date,
                "refreshed_on": datetime.now().date().isoformat(),
                "refreshed_at": manifest["created_at"],
                "financial_status": "success",
                "manifest_path": str(manifest_path),
            },
            effective_state_path,
        )
    return manifest
