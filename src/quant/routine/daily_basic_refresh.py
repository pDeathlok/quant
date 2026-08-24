from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd

from quant.data.atomic_io import atomic_write_csv, atomic_write_json, atomic_write_parquet
from quant.data.tushare_fetcher import TushareDataFetcher, validate_daily_basic_frame
from quant.routine.data_refresh import RequestLimiter, _is_retryable_error
from quant.routine.paths import PROJECT_ROOT


DAILY_DIR = PROJECT_ROOT / "data/raw/daily"
DAILY_BASIC_DIR = PROJECT_ROOT / "data/raw/daily_basic"
AUDIT_ROOT = PROJECT_ROOT / "data/raw/source_audit"

# These are the raw daily_basic columns consumed by the production B1 and Chan
# feature builders. Row-count coverage alone is insufficient: Tushare can
# return a full stock cross-section while selected fields are entirely null.
# Dividend yield is naturally sparse, hence its lower (but still non-zero)
# threshold; the remaining market/share fields should cover virtually all
# listed stocks on the decision date.
DAILY_BASIC_FEATURE_COVERAGE: dict[str, float] = {
    "turnover_rate": 0.98,
    "turnover_rate_f": 0.98,
    "volume_ratio": 0.95,
    "pe": 0.70,
    "pe_ttm": 0.70,
    "pb": 0.95,
    "ps": 0.95,
    "ps_ttm": 0.95,
    "dv_ratio": 0.50,
    "dv_ttm": 0.50,
    "total_share": 0.98,
    "float_share": 0.98,
    "free_share": 0.98,
    "total_mv": 0.98,
    "circ_mv": 0.98,
}

DELAYED_DAILY_BASIC_COLUMNS: frozenset[str] = frozenset(
    {"turnover_rate_f", "dv_ratio", "dv_ttm", "free_share"}
)


def _daily_basic_provenance_path(output_path: Path) -> Path:
    return output_path.with_suffix(".provenance.json")


def _read_source_refresh_marker(output_path: Path) -> dict[str, object] | None:
    provenance_path = _daily_basic_provenance_path(output_path)
    try:
        payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "requires_source_refresh": True,
            "reason": "invalid_repair_marker",
            "marker_error": str(exc),
        }
    if not isinstance(payload, dict):
        return {
            "requires_source_refresh": True,
            "reason": "invalid_repair_marker",
            "marker_error": "marker payload is not an object",
        }
    return payload


def _requires_source_refresh(output_path: Path) -> bool:
    payload = _read_source_refresh_marker(output_path)
    return bool(payload and payload.get("requires_source_refresh") is True)


def list_pending_daily_basic_repairs(output_dir: Path) -> list[str]:
    """Return every durable source-repair marker, regardless of its age."""

    pending: list[str] = []
    for marker_path in output_dir.glob("????????.provenance.json"):
        trade_date = marker_path.name.removesuffix(".provenance.json")
        output_path = output_dir / f"{trade_date}.parquet"
        if trade_date.isdigit() and _requires_source_refresh(output_path):
            pending.append(trade_date)
    return sorted(set(pending))


def _write_source_refresh_marker(
    output_path: Path,
    trade_date: str,
    *,
    reason: str,
    estimated_features: dict[str, dict[str, object]] | None = None,
    error: str | None = None,
) -> dict[str, object]:
    now = datetime.now().isoformat(timespec="seconds")
    previous = _read_source_refresh_marker(output_path) or {}
    previous_attempts = int(previous.get("source_refresh_attempts") or 0)
    payload: dict[str, object] = {
        "marker_version": 1,
        "trade_date": trade_date,
        "status": "pending_source_repair",
        "requires_source_refresh": True,
        "reason": reason,
        "first_detected_at": previous.get("first_detected_at") or now,
        "last_attempt_at": now,
        "source_refresh_attempts": previous_attempts + 1,
    }
    if estimated_features:
        payload["estimated_features"] = estimated_features
    elif isinstance(previous.get("estimated_features"), dict):
        payload["estimated_features"] = previous["estimated_features"]
    if error:
        payload["last_error"] = error
    atomic_write_json(payload, _daily_basic_provenance_path(output_path))
    return payload


def _source_refresh_failure_reason(error: str) -> str:
    if "feature coverage" in error:
        return "source_feature_coverage_incomplete"
    if "returned 0 rows" in error or "empty" in error.lower():
        return "source_data_unavailable"
    return "source_refresh_failed"


def _repair_diff(
    previous_frame: pd.DataFrame | None,
    official_frame: pd.DataFrame,
    marker: dict[str, object],
) -> tuple[int, list[str]]:
    """Compare a provisional snapshot with its official replacement."""

    if previous_frame is None or "ts_code" not in previous_frame.columns:
        changed_columns = sorted(
            column
            for column in official_frame.columns
            if column not in {"ts_code", "trade_date"}
        )
        return len(official_frame), changed_columns

    estimated = marker.get("estimated_features")
    candidate_columns = (
        set(estimated)
        if isinstance(estimated, dict) and estimated
        else set(previous_frame.columns) & set(official_frame.columns)
    )
    candidate_columns -= {"ts_code", "trade_date"}
    previous = previous_frame.drop_duplicates("ts_code", keep="last").set_index(
        "ts_code"
    )
    official = official_frame.drop_duplicates("ts_code", keep="last").set_index(
        "ts_code"
    )
    symbols = previous.index.union(official.index)
    changed_symbols = set(previous.index.symmetric_difference(official.index))
    changed_columns: list[str] = []
    for column in sorted(candidate_columns):
        left = previous[column].reindex(symbols)
        right = official[column].reindex(symbols)
        same = left.eq(right) | (left.isna() & right.isna())
        changed = same.index[~same.fillna(False)]
        if len(changed):
            changed_columns.append(column)
            changed_symbols.update(changed.astype(str))
    return len(changed_symbols), changed_columns


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(float("nan"), index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _load_previous_official_daily_basic(
    output_dir: Path,
    trade_date: str,
) -> tuple[str, pd.DataFrame] | None:
    candidates = sorted(
        path
        for path in output_dir.glob("????????.parquet")
        if path.stem.isdigit() and path.stem < trade_date
    )
    for path in reversed(candidates):
        if _requires_source_refresh(path):
            continue
        try:
            frame = validate_daily_basic_frame(
                pd.read_parquet(path),
                path.stem,
                required_feature_coverage=DAILY_BASIC_FEATURE_COVERAGE,
            )
        except Exception:
            continue
        return path.stem, frame
    return None


def _derive_delayed_daily_basic_features(
    frame: pd.DataFrame,
    trade_date: str,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    """Estimate latest fields while Tushare's daily batch is incomplete.

    Only the four fields known to arrive late are eligible. The previous input
    must be an official, fully validated local cross-section so estimates never
    cascade from one delayed day into the next.
    """

    delayed = {
        column
        for column in DELAYED_DAILY_BASIC_COLUMNS
        if column in frame
        and pd.to_numeric(frame[column], errors="coerce").isna().any()
    }
    if not delayed:
        return frame, {}

    previous = _load_previous_official_daily_basic(output_dir, trade_date)
    if previous is None:
        return frame, {}
    previous_trade_date, previous_frame = previous
    previous_columns = [
        "ts_code",
        "free_share",
        "float_share",
        "total_share",
        "total_mv",
        "dv_ratio",
        "dv_ttm",
    ]
    available_previous = [
        column for column in previous_columns if column in previous_frame.columns
    ]
    previous_values = previous_frame[available_previous].rename(
        columns={
            column: f"previous_{column}"
            for column in available_previous
            if column != "ts_code"
        }
    )
    enriched = frame.merge(previous_values, on="ts_code", how="left")
    derived: dict[str, dict[str, object]] = {}

    if "free_share" in delayed:
        current = _numeric_column(enriched, "free_share")
        previous_free = _numeric_column(enriched, "previous_free_share")
        current_float = _numeric_column(enriched, "float_share")
        previous_float = _numeric_column(
            enriched, "previous_float_share"
        ).replace(0, float("nan"))
        estimate = previous_free * current_float / previous_float
        estimate = estimate.fillna(previous_free)
        filled = current.isna() & estimate.notna()
        enriched["free_share"] = current.fillna(estimate)
        if filled.any():
            derived["free_share"] = {
                "source": "previous_official_daily_basic",
                "previous_trade_date": previous_trade_date,
                "formula": "previous_free_share * float_share / previous_float_share",
                "quality": "estimated",
                "filled_rows": int(filled.sum()),
            }

    if "turnover_rate_f" in delayed:
        current = _numeric_column(enriched, "turnover_rate_f")
        free_share = _numeric_column(enriched, "free_share").replace(
            0, float("nan")
        )
        turnover_rate = _numeric_column(enriched, "turnover_rate")
        float_share = _numeric_column(enriched, "float_share")
        estimate = turnover_rate * float_share / free_share
        filled = current.isna() & estimate.notna()
        enriched["turnover_rate_f"] = current.fillna(estimate)
        if filled.any():
            derived["turnover_rate_f"] = {
                "source": "tushare_daily_basic_derived",
                "formula": "turnover_rate * float_share / free_share",
                "quality": "derived",
                "filled_rows": int(filled.sum()),
            }

    previous_price = (
        _numeric_column(enriched, "previous_total_mv")
        / _numeric_column(enriched, "previous_total_share").replace(0, float("nan"))
    )
    current_price = (
        _numeric_column(enriched, "total_mv")
        / _numeric_column(enriched, "total_share").replace(0, float("nan"))
    )
    price_ratio = previous_price / current_price.replace(0, float("nan"))
    for column in ("dv_ratio", "dv_ttm"):
        if column not in delayed:
            continue
        current = _numeric_column(enriched, column)
        previous_yield = _numeric_column(enriched, f"previous_{column}")
        estimate = previous_yield * price_ratio
        filled = current.isna() & estimate.notna()
        enriched[column] = current.fillna(estimate)
        if filled.any():
            derived[column] = {
                "source": "previous_official_daily_basic",
                "previous_trade_date": previous_trade_date,
                "formula": (
                    f"previous_{column} * previous_close / current_close "
                    "(close inferred from total_mv / total_share)"
                ),
                "quality": "estimated",
                "filled_rows": int(filled.sum()),
            }

    previous_value_columns = [
        column for column in enriched.columns if column.startswith("previous_")
    ]
    return enriched.drop(columns=previous_value_columns), derived


def _derive_volume_ratio_from_daily(
    frame: pd.DataFrame,
    trade_date: str,
    daily_dir: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Fill delayed Tushare volume_ratio values from canonical daily volume.

    Tushare occasionally publishes the latest daily_basic cross-section before
    ``volume_ratio`` is populated. Its value is reproducible from the same
    Tushare daily source as current volume divided by the prior five-session
    average volume, rounded to the vendor's two-decimal precision.
    """

    if "volume_ratio" not in frame.columns:
        return frame, {}
    missing = pd.to_numeric(frame["volume_ratio"], errors="coerce").isna()
    if not missing.any():
        return frame, {}

    target_month = trade_date[:6]
    partition_root = daily_dir.parent / f"{daily_dir.name}_partitioned"
    candidates = sorted(
        path
        for path in partition_root.glob("year_month=*/data.parquet")
        if path.parent.name.removeprefix("year_month=") <= target_month
    )[-2:]
    if not candidates:
        return frame, {}

    history = pd.concat(
        [pd.read_parquet(path, columns=["ts_code", "trade_date", "vol"]) for path in candidates],
        ignore_index=True,
    )
    history["trade_date"] = (
        history["trade_date"].astype(str).str.replace("-", "", regex=False)
    )
    history["vol"] = pd.to_numeric(history["vol"], errors="coerce")
    history = history[history["trade_date"] <= trade_date].dropna(
        subset=["ts_code", "vol"]
    )
    history = history.sort_values(["ts_code", "trade_date"]).drop_duplicates(
        ["ts_code", "trade_date"], keep="last"
    )
    prior_average = history.groupby("ts_code", sort=False)["vol"].transform(
        lambda values: values.shift(1).rolling(5, min_periods=5).mean()
    )
    history["derived_volume_ratio"] = (history["vol"] / prior_average).round(2)
    derived = history.loc[
        history["trade_date"] == trade_date,
        ["ts_code", "derived_volume_ratio"],
    ].dropna(subset=["derived_volume_ratio"])
    if derived.empty:
        return frame, {}

    enriched = frame.merge(derived, on="ts_code", how="left")
    before = pd.to_numeric(enriched["volume_ratio"], errors="coerce").notna()
    enriched["volume_ratio"] = pd.to_numeric(
        enriched["volume_ratio"], errors="coerce"
    ).fillna(enriched.pop("derived_volume_ratio"))
    filled_rows = int((~before & enriched["volume_ratio"].notna()).sum())
    if not filled_rows:
        return enriched, {}
    return enriched, {
        "volume_ratio": {
            "source": "tushare_daily_derived",
            "formula": "vol / prior_5_session_mean_vol",
            "filled_rows": filled_rows,
        }
    }


def load_trade_dates_from_daily(daily_dir: Path, start_date: str, end_date: str | None) -> list[str]:
    return sorted(load_trade_date_symbol_counts(daily_dir, start_date, end_date))


def load_trade_date_symbol_counts(
    daily_dir: Path,
    start_date: str,
    end_date: str | None,
) -> dict[str, int]:
    counts: dict[str, set[str]] = {}
    partition_root = daily_dir.parent / f"{daily_dir.name}_partitioned"
    for path in partition_root.glob("year_month=*/data.parquet"):
        try:
            frame = pd.read_parquet(path, columns=["trade_date", "ts_code"])
        except Exception as exc:
            raise RuntimeError(f"failed to read canonical daily partition {path}: {exc}") from exc
        frame["trade_date"] = frame["trade_date"].astype(str)
        frame = frame[frame["trade_date"] >= start_date]
        if end_date:
            frame = frame[frame["trade_date"] <= end_date]
        for trade_date, group in frame.groupby("trade_date", sort=False):
            counts.setdefault(str(trade_date), set()).update(
                group["ts_code"].dropna().astype(str)
            )
    return {trade_date: len(symbols) for trade_date, symbols in counts.items()}


def fetch_one_trade_date(
    trade_date: str,
    output_dir: Path,
    cache_dir: Path,
    limiter: RequestLimiter,
    retries: int,
    retry_base_delay: float,
    retry_max_delay: float,
    expected_rows: int | None = None,
    minimum_coverage_rate: float = 0.98,
    availability_retry_failures: int = 0,
    availability_retry_interval: float = 60.0,
    daily_dir: Path = DAILY_DIR,
) -> dict:
    attempts = max(1, retries + 1)
    maximum_attempts = attempts + max(0, availability_retry_failures)
    last_error = ""
    output_path = output_dir / f"{trade_date}.parquet"
    provenance_path = _daily_basic_provenance_path(output_path)
    cache_path = cache_dir / f"tushare_daily_basic_{trade_date}.parquet"
    repair_marker = _read_source_refresh_marker(output_path)
    repair_requested = bool(
        repair_marker
        and repair_marker.get("requires_source_refresh") is True
    )
    repair_frame: pd.DataFrame | None = None
    if repair_requested and output_path.exists():
        try:
            repair_frame = pd.read_parquet(output_path)
        except Exception:
            repair_frame = None
    minimum_rows = max(
        1,
        math.ceil((expected_rows or 1) * minimum_coverage_rate),
    )
    if output_path.exists() and not _requires_source_refresh(output_path):
        try:
            local = validate_daily_basic_frame(
                pd.read_parquet(output_path),
                trade_date,
                minimum_rows=minimum_rows,
                required_feature_coverage=DAILY_BASIC_FEATURE_COVERAGE,
            )
        except Exception:
            # Keep the last file recoverable until a complete replacement has
            # been fetched and atomically written below.
            pass
        else:
            return {
                "trade_date": trade_date,
                "source": "local_validated",
                "status": "success",
                "rows": len(local),
                "expected_rows": expected_rows,
                "minimum_rows": minimum_rows,
                "coverage_rate": (
                    round(len(local) / expected_rows, 6) if expected_rows else None
                ),
                "feature_coverage": local.attrs.get("feature_coverage", {}),
                "path": str(output_path),
                "attempts": 0,
                "repair_requested": False,
                "repair_status": "not_required",
                "requires_source_refresh": False,
                "error": None,
            }
    for attempt in range(1, maximum_attempts + 1):
        try:
            limiter.wait()
            fetcher = TushareDataFetcher(cache_dir=cache_dir)
            try:
                df = validate_daily_basic_frame(
                    fetcher.get_daily_basic(trade_date),
                    trade_date,
                    minimum_rows=minimum_rows,
                )
                df, derived_features = _derive_volume_ratio_from_daily(
                    df,
                    trade_date,
                    daily_dir,
                )
                if availability_retry_failures > 0 and attempt == maximum_attempts:
                    df, delayed_features = _derive_delayed_daily_basic_features(
                        df,
                        trade_date,
                        output_dir,
                    )
                    derived_features.update(delayed_features)
                df = validate_daily_basic_frame(
                    df,
                    trade_date,
                    minimum_rows=minimum_rows,
                    required_feature_coverage=DAILY_BASIC_FEATURE_COVERAGE,
                )
                df.attrs["derived_features"] = derived_features
            except ValueError:
                # The fetcher's basic schema check intentionally accepts small
                # frames. Remove a cross-section cache that fails this
                # market-relative coverage gate so the retry reaches Tushare.
                cache_path.unlink(missing_ok=True)
                raise
            estimated_features = {
                column: metadata
                for column, metadata in derived_features.items()
                if column in DELAYED_DAILY_BASIC_COLUMNS
            }
            current_marker: dict[str, object] = {}
            if estimated_features:
                current_marker = _write_source_refresh_marker(
                    output_path,
                    trade_date,
                    reason="estimated_fallback",
                    estimated_features=estimated_features,
                )
            atomic_write_parquet(df, output_path, index=False)
            if estimated_features:
                # Do not allow an incomplete request cache to prevent a later
                # same-day retry from replacing estimates with official data.
                cache_path.unlink(missing_ok=True)
            else:
                provenance_path.unlink(missing_ok=True)
            repair_changed_rows = 0
            repair_changed_columns: list[str] = []
            if repair_requested and not estimated_features:
                repair_changed_rows, repair_changed_columns = _repair_diff(
                    repair_frame,
                    df,
                    repair_marker or {},
                )
            return {
                "trade_date": trade_date,
                "source": (
                    "tushare_with_estimated_fallback"
                    if estimated_features
                    else "tushare"
                ),
                "status": "success",
                "rows": len(df),
                "expected_rows": expected_rows,
                "minimum_rows": minimum_rows,
                "coverage_rate": (
                    round(len(df) / expected_rows, 6) if expected_rows else None
                ),
                "feature_coverage": df.attrs.get("feature_coverage", {}),
                "derived_features": df.attrs.get("derived_features", {}),
                "path": str(output_path),
                "attempts": attempt,
                "repair_requested": repair_requested,
                "repair_status": (
                    "pending"
                    if estimated_features
                    else "repaired" if repair_requested else "not_required"
                ),
                "repair_reason": (
                    (repair_marker or {}).get("reason")
                    if repair_requested
                    else current_marker.get("reason")
                ),
                "repair_changed_rows": repair_changed_rows,
                "repair_changed_columns": repair_changed_columns,
                "requires_source_refresh": bool(estimated_features),
                "error": None,
            }
        except Exception as exc:
            last_error = str(exc)
            retryable = _is_retryable_error(last_error) or "daily_basic" in last_error
            availability_retry = (
                "feature coverage" in last_error
                and attempt <= availability_retry_failures
            )
            if attempt >= maximum_attempts or (not retryable and not availability_retry):
                repair_reason = _source_refresh_failure_reason(last_error)
                _write_source_refresh_marker(
                    output_path,
                    trade_date,
                    reason=repair_reason,
                    error=last_error,
                )
                return {
                    "trade_date": trade_date,
                    "source": "tushare",
                    "status": "failed",
                    "rows": 0,
                    "path": str(output_path),
                    "attempts": attempt,
                    "repair_requested": repair_requested,
                    "repair_status": "pending",
                    "repair_reason": repair_reason,
                    "repair_changed_rows": 0,
                    "repair_changed_columns": [],
                    "requires_source_refresh": True,
                    "error": last_error,
                }
            if availability_retry:
                time.sleep(max(0.0, availability_retry_interval))
            else:
                time.sleep(min(retry_max_delay, retry_base_delay * (2 ** (attempt - 1))))
    repair_reason = _source_refresh_failure_reason(last_error or "unknown error")
    _write_source_refresh_marker(
        output_path,
        trade_date,
        reason=repair_reason,
        error=last_error or "unknown error",
    )
    return {
        "trade_date": trade_date,
        "source": "tushare",
        "status": "failed",
        "rows": 0,
        "path": str(output_path),
        "attempts": maximum_attempts,
        "repair_requested": repair_requested,
        "repair_status": "pending",
        "repair_reason": repair_reason,
        "repair_changed_rows": 0,
        "repair_changed_columns": [],
        "requires_source_refresh": True,
        "error": last_error or "unknown error",
    }


def refresh_daily_basic(
    start_date: str,
    end_date: str | None = None,
    daily_dir: Path = DAILY_DIR,
    output_dir: Path = DAILY_BASIC_DIR,
    workers: int = 4,
    sleep_between: float = 0.25,
    retries: int = 3,
    retry_base_delay: float = 2.0,
    retry_max_delay: float = 60.0,
) -> dict:
    expected_rows_by_date = load_trade_date_symbol_counts(
        daily_dir,
        start_date,
        end_date,
    )
    trade_dates = sorted(expected_rows_by_date)
    if not trade_dates:
        raise RuntimeError(f"No trade dates found in {daily_dir}")
    minimum_coverage_rate = float(
        os.getenv("ROUTINE_DAILY_BASIC_MIN_COVERAGE_RATE", "0.98")
    )
    availability_retry_failures = int(
        os.getenv("ROUTINE_DAILY_BASIC_AVAILABILITY_RETRY_FAILURES", "2")
    )
    availability_retry_interval = float(
        os.getenv("ROUTINE_DAILY_BASIC_AVAILABILITY_RETRY_INTERVAL", "60")
    )
    if not 0 < minimum_coverage_rate <= 1:
        raise ValueError(
            "ROUTINE_DAILY_BASIC_MIN_COVERAGE_RATE must be in (0, 1]"
        )

    cache_dir = PROJECT_ROOT / "data/cache/source_merge/tushare"
    limiter = RequestLimiter(sleep_between)
    audits: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                fetch_one_trade_date,
                trade_date,
                output_dir,
                cache_dir,
                limiter,
                retries,
                retry_base_delay,
                retry_max_delay,
                expected_rows_by_date.get(trade_date),
                minimum_coverage_rate,
                availability_retry_failures if trade_date == max(trade_dates) else 0,
                availability_retry_interval,
                daily_dir,
            )
            for trade_date in trade_dates
        ]
        for n, future in enumerate(as_completed(futures), start=1):
            audits.append(future.result())
            if n % 100 == 0 or n == len(futures):
                ok = sum(1 for audit in audits if audit["status"] == "success")
                failed = sum(1 for audit in audits if audit["status"] == "failed")
                print(f"daily_basic progress: {n}/{len(futures)} done, success={ok}, failed={failed}", flush=True)

    audit_dir = AUDIT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_daily_basic")
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / "daily_basic_audit.csv"
    audit_df = pd.DataFrame(audits)
    atomic_write_csv(audit_df, audit_path, index=False)
    repair_requested_dates = sorted(
        str(audit["trade_date"])
        for audit in audits
        if audit.get("repair_requested")
    )
    repaired_dates = sorted(
        str(audit["trade_date"])
        for audit in audits
        if audit.get("repair_status") == "repaired"
    )
    newly_flagged_dates = sorted(
        str(audit["trade_date"])
        for audit in audits
        if audit.get("requires_source_refresh")
        and not audit.get("repair_requested")
    )
    pending_repair_dates = list_pending_daily_basic_repairs(output_dir)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_policy": "tushare_only_daily_basic",
        "start_date": start_date,
        "end_date": end_date,
        "daily_dir": str(daily_dir),
        "output_dir": str(output_dir),
        "audit_path": str(audit_path),
        "trade_dates": len(trade_dates),
        "latest_trade_date": max(trade_dates),
        "minimum_coverage_rate": minimum_coverage_rate,
        "required_feature_coverage": DAILY_BASIC_FEATURE_COVERAGE,
        "availability_retry_failures": availability_retry_failures,
        "availability_retry_interval_seconds": availability_retry_interval,
        "repair_requested_dates": repair_requested_dates,
        "repaired_dates": repaired_dates,
        "downstream_refresh_dates": repaired_dates,
        "newly_flagged_dates": newly_flagged_dates,
        "pending_repair_dates": pending_repair_dates,
        "repair_queue_size": len(pending_repair_dates),
        "data_quality_status": (
            "provisional" if pending_repair_dates else "official"
        ),
        "success": int((audit_df["status"] == "success").sum()),
        "failed": int((audit_df["status"] == "failed").sum()),
        "workers": workers,
        "min_request_interval_seconds": sleep_between,
        "retries": retries,
    }
    manifest_path = audit_dir / "manifest.json"
    atomic_write_json(manifest, manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh Tushare daily_basic data by trade date.")
    parser.add_argument("--start", default="20240101")
    parser.add_argument("--end", default=None)
    parser.add_argument("--daily-dir", type=Path, default=DAILY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DAILY_BASIC_DIR)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sleep", type=float, default=0.25)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    result = refresh_daily_basic(
        start_date=args.start,
        end_date=args.end,
        daily_dir=args.daily_dir,
        output_dir=args.output_dir,
        workers=args.workers,
        sleep_between=args.sleep,
        retries=args.retries,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
