from __future__ import annotations

import subprocess
import sys
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from quant.application.workspace_refresh import (
    WorkspaceRefreshOperations,
    refresh_daily_workspaces,
)
from quant.application.daily_dependencies import DEFAULT_DAILY_DEPENDENCY_REGISTRY
from quant.application.left_side_ranking import DEFAULT_LEFT_SIDE_RANKING_CONFIG
from quant.application.selector_ranking import (
    DEFAULT_SELECTOR_RANKING_CONFIG,
    SelectorRankingSource,
)
from quant.data import MarketDataStore, MarketDataStoreConfig
from quant.data.atomic_io import atomic_write_json as publish_json
from quant.features.market_regime import classify_market_regime
from quant.features.factor_registry import (
    FACTOR_REGISTRY_SCHEMA_VERSION,
    registry_frame,
    validate_registry,
)
from quant.features.project_factor_layer import (
    LEGACY_PRODUCTION_FACTOR_SCHEMA_VERSION,
    PROJECT_FACTOR_SCHEMA_VERSION,
    resolve_project_factor_schema,
)
from quant.routine.cache_retention import run_cache_cleanup
from quant.routine.dashboard import write_dashboard_json
from quant.routine.b1_daily_plan import write_daily_plan
from quant.routine.paths import CONFIG_PATH, PROJECT_ROOT, ROUTINE_DIR
from quant.routine.strategies import StrategyConfig, load_strategy_configs


def _incremental_daily_start(lookback_days: int | None = None) -> str:
    if lookback_days is None:
        lookback_days = int(os.getenv("ROUTINE_DAILY_LOOKBACK_DAYS", "0"))
    daily_dir = PROJECT_ROOT / "data/raw/daily"
    store = MarketDataStore(MarketDataStoreConfig.from_env(root=daily_dir.parent))
    latest = store.latest_dataset_trade_date(daily_dir.name)
    if latest is None:
        return "20100101"
    start = latest - pd.Timedelta(days=lookback_days)
    return start.strftime("%Y%m%d")


def _incremental_feature_start() -> str:
    # Daily inference publishes an exact-date active-candidate sidecar.  The
    # historical training table is maintained outside the latency-sensitive
    # routine path, so its watermark must not force an old multi-day rebuild.
    return _incremental_daily_start(lookback_days=0)


def _incremental_daily_basic_start() -> str:
    daily_basic_dir = PROJECT_ROOT / "data/raw/daily_basic"
    from quant.routine.daily_basic_refresh import list_pending_daily_basic_repairs

    pending_repairs = list_pending_daily_basic_repairs(daily_basic_dir)
    dates = [
        path.stem
        for path in daily_basic_dir.glob("*.parquet")
        if path.stem.isdigit() and len(path.stem) == 8
    ]
    if not dates:
        return min([_incremental_daily_start(), *pending_repairs])
    # Rolling model factors depend on the preceding 20 trading sessions.  A
    # previously cached cross-section may have complete rows but incomplete
    # feature columns, so revalidate a calendar window wide enough to cover
    # that dependency on every routine run. Complete local dates are skipped
    # without a network request by daily_basic_refresh.
    latest = pd.to_datetime(max(dates), format="%Y%m%d")
    rolling_start = (latest - pd.Timedelta(days=45)).strftime("%Y%m%d")
    return min([rolling_start, *pending_repairs])


def refresh_data(dry_run: bool = True, progress_callback=None) -> dict:
    started = time.monotonic()
    start_date = _incremental_daily_start()
    workers = os.getenv("ROUTINE_DAILY_WORKERS", "4")
    sleep_seconds = os.getenv("ROUTINE_DAILY_SLEEP", "0.08")
    command = [
        sys.executable,
        "-m",
        "quant.routine.data_refresh",
        "--start",
        start_date,
        "--adjust",
        "none",
        "--output-dir",
        "data/raw/daily",
        "--workers",
        workers,
        "--sleep",
        sleep_seconds,
        "--retries",
        "3",
        "--retry-base-delay",
        "2",
        "--retry-max-delay",
        "60",
        "--final-retry-rounds",
        os.getenv("ROUTINE_DAILY_FINAL_RETRY_ROUNDS", "2"),
        "--final-retry-workers",
        os.getenv("ROUTINE_DAILY_FINAL_RETRY_WORKERS", "4"),
        "--final-retry-sleep",
        os.getenv("ROUTINE_DAILY_FINAL_RETRY_SLEEP", "0.8"),
        "--availability-retry-failures",
        os.getenv("ROUTINE_DAILY_AVAILABILITY_RETRY_FAILURES", "12"),
        "--availability-retry-interval",
        os.getenv("ROUTINE_DAILY_AVAILABILITY_RETRY_INTERVAL", "300"),
    ]
    if dry_run:
        return {
            "status": "skipped",
            "reason": "dry_run=true；未访问外部数据源。正式刷新仅使用 Tushare 日线数据。",
            "command": " ".join(command),
            "start_date": start_date,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")}
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    stdout_lines: list[str] = []
    progress_pattern = re.compile(r"refresh progress: (\d+)/(\d+) done, success=(\d+), failed=(\d+)")
    availability_pattern = re.compile(
        r"market daily availability retry: "
        r"trade_date=(\d{8}) "
        r"failed_attempts=(\d+)(?:/(\d+))? "
        r"retry_in_seconds=([0-9.]+)(?: deadline=(\S+))?"
    )
    assert process.stdout is not None
    for line in process.stdout:
        stdout_lines.append(line)
        match = progress_pattern.search(line)
        if match and progress_callback is not None:
            done, total, ok, failed = map(int, match.groups())
            ratio = done / total if total else 0
            progress_callback(
                percent=10 + int(ratio * 25),
                message=f"正在拉取 Tushare 最新日线数据：{done}/{total}，成功 {ok}，失败 {failed}",
            )
            continue
        availability_match = availability_pattern.search(line)
        if availability_match and progress_callback is not None:
            trade_date, failed_attempts, tolerated_failures, retry_seconds, deadline = (
                availability_match.groups()
            )
            attempts_text = f"{failed_attempts}/{tolerated_failures}" if tolerated_failures else failed_attempts
            progress_callback(
                percent=10,
                message=(
                    f"{trade_date} 日线尚未完整发布；"
                    f"已失败 {attempts_text} 次，"
                    f"{retry_seconds} 秒后重试"
                    + ("，截止北京时间 17:20" if deadline else "")
                ),
            )
    returncode = process.wait()
    stdout = "".join(stdout_lines)
    manifest = _extract_last_json_object(stdout)
    expected_trade_date = str(manifest.get("expected_trade_date") or "")
    dataset_trade_date = str(manifest.get("dataset_trade_date") or "")
    complete = (
        returncode == 0
        and manifest.get("status") == "success"
        and bool(expected_trade_date)
        and dataset_trade_date == expected_trade_date
    )
    return {
        **manifest,
        "status": "success" if complete else "failed",
        "returncode": returncode,
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stdout[-4000:] if returncode else "",
        "command": " ".join(command),
        "start_date": start_date,
        "expected_trade_date": expected_trade_date or None,
        "dataset_trade_date": dataset_trade_date or None,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def refresh_daily_basic_data(dry_run: bool = True, progress_callback=None) -> dict:
    started = time.monotonic()
    start_date = _incremental_daily_basic_start()
    if dry_run:
        return {
            "status": "skipped",
            "reason": "dry_run=true；未访问 Tushare daily_basic。",
            "start_date": start_date,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
    if progress_callback is not None:
        progress_callback(percent=36, message=f"正在补齐 daily_basic：{start_date} 至最新交易日")
    from quant.routine.daily_basic_refresh import refresh_daily_basic

    manifest = refresh_daily_basic(
        start_date=start_date,
        workers=int(os.getenv("ROUTINE_DAILY_BASIC_WORKERS", "4")),
        sleep_between=float(os.getenv("ROUTINE_DAILY_BASIC_SLEEP", "0.25")),
        retries=int(os.getenv("ROUTINE_DAILY_BASIC_RETRIES", "3")),
        progress_callback=progress_callback,
    )
    failed = int(manifest.get("failed") or 0)
    return {
        **manifest,
        "status": "success" if failed == 0 else "failed",
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def refresh_reference_inputs(
    dry_run: bool = True,
    include_financials: bool = True,
    *,
    include_analyst: bool | None = None,
    include_stock_basic: bool = True,
    include_index: bool = True,
    include_market_regime: bool = True,
    include_tradability: bool = True,
    long_factor_datasets: tuple[str, ...] | None = None,
) -> dict:
    """Refresh only reference inputs selected by the active dependency closure.

    Defaults preserve the legacy all-input behavior for direct callers.  The
    Web coordinator passes switches compiled from the current production
    product and model contracts.
    """

    started = time.monotonic()
    end_date = _incremental_daily_start()
    effective_include_analyst = (
        include_financials if include_analyst is None else include_analyst
    )
    if dry_run:
        return {
            "status": "skipped",
            "reason": "dry_run=true；未刷新当前产品闭包所需的参考数据。",
            "end_date": end_date,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
    from quant.routine.reference_data_refresh import refresh_reference_data

    reference_kwargs = {
        "end_date": end_date,
        "include_financials": include_financials,
        "include_stock_basic": include_stock_basic,
        "include_index": include_index,
        "include_tradability": include_tradability,
        "include_long_factor_sources": long_factor_datasets is None
        or bool(long_factor_datasets),
        "long_factor_datasets": long_factor_datasets,
    }
    if effective_include_analyst:
        # Eastmoney/AkShare and Tushare use independent upstreams and write to
        # different datasets. Run them concurrently, then merge only their
        # small status payloads after both have completed.
        with ThreadPoolExecutor(max_workers=2) as executor:
            reference_future = executor.submit(
                refresh_reference_data,
                **reference_kwargs,
            )
            analyst_future = executor.submit(_refresh_analyst_forecast_snapshot)
            result = reference_future.result()
            analyst_result = analyst_future.result()
        result["execution_mode"] = "parallel_tushare_and_akshare"
        result.setdefault("steps", {})["analyst_forecast_snapshot"] = analyst_result
        if analyst_result.get("status") == "failed":
            result["status"] = "failed"
            result.setdefault("critical_errors", []).append(
                "analyst_forecast_snapshot: "
                + str(analyst_result.get("error") or "refresh failed")
            )
        elif analyst_result.get("status") == "degraded":
            result.setdefault("warnings", []).append(
                "analyst_forecast_snapshot: "
                + str(analyst_result.get("error") or "using last-known-good data")
            )
    else:
        result = refresh_reference_data(**reference_kwargs)
        result.setdefault("steps", {})["analyst_forecast_snapshot"] = {
            "status": "skipped",
            "reason": "not required for this refresh scope",
        }
    if (
        include_market_regime
        and result.get("end_date")
        and result.get("status") in {"success", "partial"}
    ):
        regime_result = refresh_market_regime_snapshot(str(result["end_date"]))
        result.setdefault("steps", {})["market_regime"] = regime_result
        if regime_result.get("status") == "failed":
            result["status"] = "failed"
            result.setdefault("critical_errors", []).append(
                "market_regime: " + str(regime_result.get("error") or "refresh failed")
            )
    elif not include_market_regime:
        result.setdefault("steps", {})["market_regime"] = {
            "status": "skipped",
            "reason": "not required for this refresh scope",
        }
    result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return result


def refresh_market_regime_snapshot(
    end_date: str,
    *,
    raw_dir: Path | None = None,
    output_dir: Path | None = None,
    store: MarketDataStore | None = None,
) -> dict:
    """Build the daily regime snapshot from already-refreshed project data."""

    effective_raw_dir = raw_dir or PROJECT_ROOT / "data/raw"
    effective_output_dir = output_dir or PROJECT_ROOT / "data/features/market_regime"
    index_path = effective_raw_dir / "index_000300.SH.parquet"
    try:
        if not index_path.is_file():
            raise FileNotFoundError(index_path)
        index_daily = pd.read_parquet(index_path)
        lookback_days = int(os.getenv("ROUTINE_MARKET_REGIME_LOOKBACK_DAYS", "252"))
        if lookback_days < 90:
            raise ValueError("ROUTINE_MARKET_REGIME_LOOKBACK_DAYS must be at least 90")
        start_date = (pd.Timestamp(end_date) - pd.Timedelta(days=lookback_days)).strftime("%Y%m%d")
        market_store = store or MarketDataStore(
            MarketDataStoreConfig.from_env(root=effective_raw_dir)
        )
        market_daily = market_store.read_market_range(
            "daily",
            start_date=start_date,
            end_date=end_date,
            columns=["trade_date", "ts_code", "close", "amount"],
        )
        snapshot = classify_market_regime(index_daily, market_daily, as_of=end_date)
        if snapshot["as_of"] != end_date:
            raise RuntimeError(
                f"market regime snapshot stale: expected {end_date}, got {snapshot['as_of']}"
            )
        dated_path = effective_output_dir / f"{end_date}.json"
        latest_path = effective_output_dir / "latest.json"
        publish_json(snapshot, dated_path)
        publish_json(snapshot, latest_path)
        return {
            "status": "success",
            "as_of": snapshot["as_of"],
            "regime": snapshot["regime"],
            "score": snapshot["score"],
            "path": str(dated_path),
            "latest_path": str(latest_path),
        }
    except Exception as exc:
        return {"status": "failed", "error": str(exc), "end_date": end_date}


def _latest_long_analyst_symbols(limit: int = 80) -> list[str]:
    snapshot_dir = PROJECT_ROOT / "data/long_stock_pool_snapshots"
    symbols: list[str] = []
    seen: set[str] = set()
    if not snapshot_dir.exists():
        return symbols
    for path in sorted(snapshot_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("variant") not in {"tea", "tea_safe", "v44"}:
            continue
        for item in payload.get("stocks") or []:
            if str(item.get("state") or "") == "EXIT":
                continue
            symbol = str(item.get("ts_code") or "")
            if symbol and symbol not in seen:
                seen.add(symbol)
                symbols.append(symbol)
                if len(symbols) >= limit:
                    return symbols
    return symbols


def _run_analyst_command(command: list[str], env: dict[str, str], timeout_seconds: int) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return subprocess.CompletedProcess(
            command,
            124,
            stdout=stdout,
            stderr=(stderr + f"\nanalyst refresh timed out after {timeout_seconds}s").strip(),
        )


def _extract_last_json_object(text: str) -> dict:
    decoder = json.JSONDecoder()
    parsed: dict | None = None
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and not text[index + end :].strip():
            parsed = value
    return parsed or {}


def _existing_analyst_symbols(output_path: Path, source: str) -> set[str]:
    if not output_path.exists():
        return set()
    try:
        frame = pd.read_parquet(output_path, columns=["source", "ts_code"])
    except Exception:
        return set()
    return set(frame.loc[frame["source"] == source, "ts_code"].dropna().astype(str))


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f".{os.getpid()}.tmp.json")
    try:
        temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _refresh_analyst_forecast_snapshot() -> dict:
    """Refresh daily consensus plus broker-level reports for current long candidates."""

    started = time.monotonic()
    output_path = PROJECT_ROOT / "data/raw/analyst_forecasts.parquet"
    research_marker = PROJECT_ROOT / "data/raw/analyst_research_refresh_status.json"
    today = pd.Timestamp.now().normalize()
    env = {**os.environ, "PYTHONPATH": f"{PROJECT_ROOT / 'src'}:{PROJECT_ROOT / 'scripts' / 'research'}"}
    steps: dict[str, dict] = {}

    latest: pd.Timestamp | None = None
    if output_path.exists():
        try:
            current = pd.read_parquet(output_path, columns=["source", "report_date"])
            dates = pd.to_datetime(
                current.loc[current["source"] == "akshare_em_snapshot", "report_date"],
                errors="coerce",
            )
            latest = dates.max() if not dates.empty else None
        except Exception:
            latest = None

    if latest == today:
        steps["consensus_snapshot"] = {
            "status": "skipped",
            "reason": "today's snapshot already exists",
            "polled_through": today.date().isoformat(),
        }
    else:
        command = [
            sys.executable,
            "scripts/research/refresh_analyst_forecasts.py",
            "--source",
            "akshare_em_snapshot",
        ]
        result = _run_analyst_command(
            command,
            env,
            timeout_seconds=int(os.getenv("ROUTINE_AKSHARE_SNAPSHOT_TIMEOUT_SECONDS", "180")),
        )
        if result.returncode == 0 and output_path.exists():
            current = pd.read_parquet(output_path, columns=["source", "report_date"])
            dates = pd.to_datetime(current.loc[current["source"] == "akshare_em_snapshot", "report_date"], errors="coerce")
            latest = dates.max() if not dates.empty else None
        consensus_success = result.returncode == 0 and latest == today
        steps["consensus_snapshot"] = {
            "status": "success" if consensus_success else ("degraded" if latest is not None else "failed"),
            "returncode": result.returncode,
            "command": " ".join(command),
            "stdout_tail": result.stdout[-2000:],
            "stderr_tail": result.stderr[-2000:],
            "fallback_latest_report_date": (
                latest.date().isoformat() if not consensus_success and latest is not None and pd.notna(latest) else None
            ),
            "polled_through": (
                today.date().isoformat() if consensus_success else None
            ),
        }

    symbols = _latest_long_analyst_symbols()
    research_enabled = os.getenv("ROUTINE_REFRESH_CANDIDATE_RESEARCH", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    marker_date = None
    if research_marker.exists():
        try:
            marker_date = json.loads(research_marker.read_text(encoding="utf-8")).get("date")
        except Exception:
            marker_date = None
    if not symbols:
        steps["candidate_research_reports"] = {"status": "skipped", "reason": "no long candidates"}
    elif not research_enabled:
        steps["candidate_research_reports"] = {
            "status": "skipped",
            "reason": "external candidate-report refresh is disabled; set ROUTINE_REFRESH_CANDIDATE_RESEARCH=1 after approval",
            "symbols_requested": len(symbols),
        }
    elif marker_date == today.date().isoformat():
        steps["candidate_research_reports"] = {
            "status": "skipped",
            "reason": "today's candidate reports already refreshed",
            "polled_through": today.date().isoformat(),
        }
    else:
        research_command = [
            sys.executable,
            "scripts/research/refresh_analyst_forecasts.py",
            "--source",
            "akshare_em_research",
            "--symbols",
            ",".join(symbols),
            "--refresh-existing",
            "--sleep",
            os.getenv("ROUTINE_AKSHARE_RESEARCH_SLEEP", "0.35"),
            "--retries",
            os.getenv("ROUTINE_AKSHARE_RESEARCH_RETRIES", "3"),
            "--circuit-breaker-failures",
            os.getenv("ROUTINE_AKSHARE_CIRCUIT_BREAKER_FAILURES", "6"),
            "--checkpoint-every",
            os.getenv("ROUTINE_AKSHARE_CHECKPOINT_EVERY", "10"),
        ]
        research_result = _run_analyst_command(
            research_command,
            env,
            timeout_seconds=int(os.getenv("ROUTINE_AKSHARE_RESEARCH_TIMEOUT_SECONDS", "900")),
        )
        research_success = research_result.returncode == 0
        research_payload = _extract_last_json_object(research_result.stdout)
        failed_symbols = sorted(str(item) for item in (research_payload.get("failed_symbols") or []) if item)
        deferred_symbols = sorted(str(item) for item in (research_payload.get("deferred_symbols") or []) if item)
        no_data_symbols = sorted(str(item) for item in (research_payload.get("no_data_symbols") or []) if item)
        degraded_symbols = sorted(set(failed_symbols) | set(deferred_symbols))
        failure_rate = (len(degraded_symbols) / len(symbols)) if symbols else 0.0
        soft_failure_threshold = float(os.getenv("ROUTINE_AKSHARE_RESEARCH_SOFT_FAILURE_RATE", "0.05"))
        if research_success:
            _atomic_write_json(
                research_marker,
                {"date": today.date().isoformat(), "symbols": symbols},
            )
        fallback_symbols = sorted(_existing_analyst_symbols(output_path, "akshare_em_research") & set(symbols))
        fallback_missing_symbols = sorted(set(symbols) - set(fallback_symbols))
        low_ratio_soft_failure = (
            not research_success
            and degraded_symbols
            and failure_rate <= soft_failure_threshold
        )
        if research_success or low_ratio_soft_failure:
            research_status = "success"
        elif fallback_symbols:
            research_status = "degraded"
        else:
            research_status = "failed"
        steps["candidate_research_reports"] = {
            "status": research_status,
            "returncode": research_result.returncode,
            "symbols_requested": len(symbols),
            "symbols_success": int(research_payload.get("success") or 0) if research_payload else None,
            "symbols_failed": len(failed_symbols),
            "symbols_deferred": len(deferred_symbols),
            "symbols_no_data": len(no_data_symbols),
            "failed_symbols": failed_symbols,
            "deferred_symbols": deferred_symbols,
            "no_data_symbols": no_data_symbols,
            "degraded_symbols": degraded_symbols,
            "failure_rate": round(failure_rate, 6),
            "soft_failure_threshold": soft_failure_threshold,
            "fallback_symbols": len(fallback_symbols),
            "fallback_symbol_list": fallback_symbols,
            "fallback_missing_symbols": fallback_missing_symbols,
            "coverage": {
                "requested": len(symbols),
                "fresh_success": int(research_payload.get("success") or 0) if research_payload else None,
                "no_data": len(no_data_symbols),
                "last_known_good": len(fallback_symbols),
                "degraded": len(degraded_symbols),
            },
            "warning": (
                f"isolated low-ratio research failures: {', '.join(degraded_symbols)}"
                if low_ratio_soft_failure
                else None
            ),
            "command": " ".join(research_command),
            "stdout_tail": research_result.stdout[-2000:],
            "stderr_tail": research_result.stderr[-2000:],
            "polled_through": (
                today.date().isoformat() if research_success else None
            ),
        }

    failed = [name for name, item in steps.items() if item.get("status") == "failed"]
    degraded = [name for name, item in steps.items() if item.get("status") == "degraded"]
    ran = any(item.get("status") == "success" for item in steps.values())
    consensus_polled_through = steps.get("consensus_snapshot", {}).get(
        "polled_through"
    )
    return {
        "status": "failed" if failed else ("degraded" if degraded else ("success" if ran else "skipped")),
        "latest_report_date": latest.date().isoformat() if latest is not None and pd.notna(latest) else None,
        "candidate_symbols": symbols,
        # This node's required daily poll is the full-market consensus
        # snapshot.  Candidate research has its own independent marker above;
        # its disabled/fallback state must not fabricate a watermark.
        "polled_through": consensus_polled_through,
        "steps": steps,
        "error": (
            f"failed analyst refresh steps: {', '.join(failed)}"
            if failed
            else (f"using last-known-good analyst data: {', '.join(degraded)}" if degraded else None)
        ),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def build_features(
    progress_callback=None,
    *,
    incremental_start_date: str | None = None,
) -> dict:
    started = time.monotonic()
    start_date = incremental_start_date or _incremental_feature_start()
    two_unified_rankers_active = (
        DEFAULT_SELECTOR_RANKING_CONFIG.source
        == SelectorRankingSource.RIGHT_SIDE_UNIFIED
        and DEFAULT_LEFT_SIDE_RANKING_CONFIG.enabled
    )
    default_factor_schema = (
        PROJECT_FACTOR_SCHEMA_VERSION
        if two_unified_rankers_active
        else LEGACY_PRODUCTION_FACTOR_SCHEMA_VERSION
    )
    production_factor_mode = os.getenv(
        "ROUTINE_PRODUCTION_FACTOR_SCHEMA",
        default_factor_schema,
    )
    production_factor_schema = resolve_project_factor_schema(production_factor_mode)
    command = [
        sys.executable,
        "scripts/research/refresh_b1_feature_cache.py",
        "--incremental-start-date",
        start_date,
        "--workers",
        os.getenv("ROUTINE_FEATURE_WORKERS", "8"),
        "--executor",
        os.getenv("ROUTINE_FEATURE_EXECUTOR", "processes"),
        "--no-adaptive-workers",
        "--gate-cache",
        "data/features/b1/b1_gate_candidates.parquet",
        "--gate-manifest",
        "data/features/b1/b1_gate_manifest.json",
        "--live-only",
    ]
    env = {
        **os.environ,
        "PYTHONPATH": f"{PROJECT_ROOT / 'src'}:{PROJECT_ROOT / 'scripts' / 'research'}",
        "PROJECT_FACTOR_COMPATIBILITY_MODE": production_factor_schema,
    }
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    stdout_lines: list[str] = []
    progress_pattern = re.compile(r"processed (\d+)/(\d+) daily files, frames=(\d+)")
    assert process.stdout is not None
    for line in process.stdout:
        stdout_lines.append(line)
        match = progress_pattern.search(line)
        if match and progress_callback is not None:
            done, total, frames = map(int, match.groups())
            ratio = done / total if total else 0
            progress_callback(
                percent=35 + int(ratio * 10),
                message=f"正在增量构建 B1 特征：{done}/{total}，命中 {frames}",
            )
    returncode = process.wait()
    stdout = "".join(stdout_lines)
    manifest = _extract_last_json_object(stdout)
    source_latest_trade_date = str(manifest.get("source_latest_trade_date") or "")
    expected_trade_date = pd.to_datetime(
        _incremental_daily_start(),
        format="%Y%m%d",
    ).strftime("%Y-%m-%d")
    complete = (
        returncode == 0
        and manifest.get("status") == "success"
        and source_latest_trade_date == expected_trade_date
    )
    return {
        **manifest,
        "factor_schema_version": production_factor_schema,
        "factor_calculator": "quant.features.project_factor_layer",
        "status": "success" if complete else "failed",
        "returncode": returncode,
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stdout[-4000:] if returncode else "",
        "command": " ".join(command),
        "start_date": start_date,
        "expected_trade_date": expected_trade_date,
        "source_latest_trade_date": source_latest_trade_date or None,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def refresh_factor_registry_snapshot() -> dict:
    """Publish the factor contract consumed by every daily/weekly application."""

    from quant.features.factor_execution import (
        factor_execution_plan_payload,
        validate_factor_execution_registry,
    )
    from quant.features.factor_registry import FACTOR_REGISTRY_CONFIG_SHA256

    started = time.monotonic()
    output_path = PROJECT_ROOT / "data/features/factor_registry/latest.json"
    try:
        validate_registry()
        validate_factor_execution_registry()
        registry = registry_frame()
        family_counts = {
            str(key): int(value)
            for key, value in registry["family"].value_counts().sort_index().items()
        }
        semantic_category_counts = {
            str(key): int(value)
            for key, value in registry["semantic_category"].value_counts().sort_index().items()
        }
        factor_level_counts = {
            str(key): int(value)
            for key, value in registry["factor_level"].value_counts().sort_index().items()
        }
        calculator_counts = {
            str(key): int(value)
            for key, value in registry["calculator_id"].value_counts().sort_index().items()
        }
        frequency_counts = {
            str(key): int(value)
            for key, value in registry["frequency"].value_counts().sort_index().items()
        }
        role_counts = {
            str(key): int(value)
            for key, value in registry["role"].value_counts().sort_index().items()
        }
        lifecycle_counts = {
            str(key): int(value)
            for key, value in registry["lifecycle"].value_counts().sort_index().items()
        }
        factor_layer_counts = {
            str(key): int(value)
            for key, value in registry["layer"].value_counts().sort_index().items()
        }
        refresh_cadence_counts = {
            str(key): int(value)
            for key, value in registry["refresh_cadence"].value_counts().sort_index().items()
        }
        dependency_summary = {
            "schema_version": DEFAULT_DAILY_DEPENDENCY_REGISTRY.schema_version,
            "node_count": len(DEFAULT_DAILY_DEPENDENCY_REGISTRY.nodes),
            "scope_roots": DEFAULT_DAILY_DEPENDENCY_REGISTRY.scope_roots,
            "layer_counts": {
                layer.value: sum(
                    1
                    for node in DEFAULT_DAILY_DEPENDENCY_REGISTRY.nodes.values()
                    if node.layer == layer
                )
                for layer in type(next(iter(DEFAULT_DAILY_DEPENDENCY_REGISTRY.nodes.values())).layer)
            },
            "runtime_snapshot": "data/contracts/daily_dependencies/latest.json",
        }
        contract_body = {
            "registry_schema_version": FACTOR_REGISTRY_SCHEMA_VERSION,
            "factor_schema_version": PROJECT_FACTOR_SCHEMA_VERSION,
            "factors": registry.to_dict(orient="records"),
            "daily_dependency_registry": dependency_summary,
        }
        contract_hash = hashlib.sha256(
            json.dumps(contract_body, ensure_ascii=True, sort_keys=True).encode("utf-8")
        ).hexdigest()
        try:
            existing = json.loads(output_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, ValueError):
            existing = {}
        payload = {
            "status": "success",
            "registry_schema_version": FACTOR_REGISTRY_SCHEMA_VERSION,
            "factor_schema_version": PROJECT_FACTOR_SCHEMA_VERSION,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "contract_hash": contract_hash,
            "factor_count": int(len(registry)),
            "canonical_factor_count": int(registry["role"].eq("feature").sum()),
            "compatibility_alias_count": int(
                registry["role"].eq("compatibility_alias").sum()
            ),
            "strategy_identity_count": int(
                registry["role"].eq("strategy_identity").sum()
            ),
            "family_counts": family_counts,
            "semantic_category_counts": semantic_category_counts,
            "factor_level_counts": factor_level_counts,
            "calculator_counts": calculator_counts,
            "governance_config_sha256": FACTOR_REGISTRY_CONFIG_SHA256,
            "daily_execution_plan": factor_execution_plan_payload(),
            "frequency_counts": frequency_counts,
            "role_counts": role_counts,
            "lifecycle_counts": lifecycle_counts,
            "factor_layer_counts": factor_layer_counts,
            "refresh_cadence_counts": refresh_cadence_counts,
            "point_in_time_factor_count": int(registry["point_in_time"].sum()),
            "calculation_entrypoint": "multiple; see factors[].calculation_entrypoint",
            "applications": {
                "short_daily_current": "gate first, then complete current causal project factors",
                "short_daily_released": (
                    "pinned pre-v4 models use the explicit legacy compatibility schema "
                    "until independently retrained and promoted"
                ),
                "long_weekly": "weekly last trading day with financial ann_date as-of",
            },
            "daily_dependency_registry": dependency_summary,
            "factors": contract_body["factors"],
        }
        checkpoint_reused = existing.get("contract_hash") == contract_hash
        if checkpoint_reused:
            payload["created_at"] = existing.get("created_at") or payload["created_at"]
        else:
            publish_json(payload, output_path)
        return {
            **{key: value for key, value in payload.items() if key != "factors"},
            "checkpoint_reused": checkpoint_reused,
            "path": str(output_path),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
    except Exception as exc:
        return {
            "status": "failed",
            "error": str(exc),
            "path": str(output_path),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }


def resolve_daily_dependency_source_options(scope: str) -> dict[str, Any]:
    """Compile active source switches from promoted product/model contracts."""

    from quant.routine.daily_dependency_runtime import resolve_active_source_options

    return resolve_active_source_options(PROJECT_ROOT, scope)


def publish_daily_dependency_contract(
    target_date: str,
    scope: str,
    results: dict[str, Any],
    *,
    phase: str,
    strict_freshness: bool,
) -> dict[str, Any]:
    """Publish the four-layer daily plan and optionally enforce its final gate."""

    from quant.routine.daily_dependency_runtime import publish_daily_dependency_snapshot

    return publish_daily_dependency_snapshot(
        PROJECT_ROOT,
        target_date,
        scope=scope,
        results=results,
        phase=phase,
        strict_models=True,
        strict_freshness=strict_freshness,
        raise_on_failure=False,
    )


def run_right_side_shadow_routine(
    target_date: str,
    *,
    upstream_results: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the isolated right-side shadow if its release config enables it.

    The production ``run_daily_pipeline`` intentionally does not invoke this
    function.  A scheduler may call it as a separate research job after the
    shared market/signal inputs are complete; failures are contained in the
    rightSideShadow status artifact and cannot fail ``all`` or ``short``.
    """

    from quant.routine.right_side_unified_shadow import (
        run_configured_right_side_shadow,
    )

    return run_configured_right_side_shadow(
        target_date,
        upstream_results=upstream_results,
    )


def run_promoted_right_side_ranking(target_date: str) -> dict[str, Any]:
    """Run the production ranker only after an explicit source promotion."""

    if (
        DEFAULT_SELECTOR_RANKING_CONFIG.source
        != SelectorRankingSource.RIGHT_SIDE_UNIFIED
    ):
        return {
            "status": "skipped",
            "reason": "selector ranking source remains legacy_z_skill",
        }
    from quant.routine.right_side_unified_production import (
        run_right_side_unified_production,
    )

    return run_right_side_unified_production(target_date)


def refresh_strategy_signal_cache(workers: int = 8, progress_callback=None) -> dict:
    started = time.monotonic()
    start_date = _incremental_daily_start()
    command = [
        sys.executable,
        "scripts/research/rebuild_strategy_signal_cache.py",
        "--workers",
        str(workers),
        "--incremental-start-date",
        start_date,
    ]
    production_factor_schema = resolve_project_factor_schema(
        os.getenv(
            "ROUTINE_PRODUCTION_FACTOR_SCHEMA",
            LEGACY_PRODUCTION_FACTOR_SCHEMA_VERSION,
        )
    )
    env = {
        **os.environ,
        "PYTHONPATH": f"{PROJECT_ROOT / 'src'}:{PROJECT_ROOT / 'scripts' / 'research'}",
        "PROJECT_FACTOR_COMPATIBILITY_MODE": production_factor_schema,
    }
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    stdout_lines: list[str] = []
    progress_pattern = re.compile(r"(combined|family|z-skill) signals: (\d+)/(\d+) symbols")
    assert process.stdout is not None
    for line in process.stdout:
        stdout_lines.append(line)
        match = progress_pattern.search(line)
        if match and progress_callback is not None:
            phase, done_text, total_text = match.groups()
            done = int(done_text)
            total = int(total_text)
            ratio = done / total if total else 0
            if phase == "family":
                percent = 46 + int(ratio * 10)
                label = "核心策略规则信号"
            elif phase == "z-skill":
                percent = 56 + int(ratio * 12)
                label = "扩展形态策略信号"
            else:
                percent = 50 + int(ratio * 15)
                label = "核心/扩展策略信号"
            progress_callback(
                percent=percent,
                message=f"正在重建{label}：{done}/{total}",
            )
    returncode = process.wait()
    stdout = "".join(stdout_lines)
    manifest = _extract_last_json_object(stdout)
    expected_trade_date = pd.to_datetime(
        _incremental_daily_start(),
        format="%Y%m%d",
    ).strftime("%Y-%m-%d")
    processed_through_date = str(manifest.get("processed_through_date") or "")
    complete = (
        returncode == 0
        and manifest.get("status") == "success"
        and processed_through_date == expected_trade_date
    )
    return {
        **manifest,
        "factor_schema_version": production_factor_schema,
        "status": "success" if complete else "failed",
        "returncode": returncode,
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stdout[-4000:] if returncode else "",
        "command": " ".join(command),
        "start_date": start_date,
        "expected_trade_date": expected_trade_date,
        "processed_through_date": processed_through_date or None,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def score_latest_models(workers: int = 8) -> dict:
    started = time.monotonic()
    expected_trade_date = pd.to_datetime(
        _incremental_daily_start(),
        format="%Y%m%d",
    ).strftime("%Y-%m-%d")
    executor_type = os.getenv(
        "ROUTINE_MODEL_SCORE_EXECUTOR",
        "processes",
    )
    batch_size = os.getenv("ROUTINE_MODEL_SCORE_BATCH_SIZE", "8")
    command = [
        sys.executable,
        "scripts/research/score_latest_strategy_models.py",
        "--target-date",
        expected_trade_date,
        "--workers",
        str(workers),
        "--executor",
        executor_type,
        "--batch-size",
        batch_size,
    ]
    if (
        DEFAULT_SELECTOR_RANKING_CONFIG.source
        == SelectorRankingSource.RIGHT_SIDE_UNIFIED
    ):
        command.extend(
            [
                "--signals",
                *DEFAULT_SELECTOR_RANKING_CONFIG.preserved_legacy_signals,
            ]
        )
    production_factor_schema = resolve_project_factor_schema(
        os.getenv(
            "ROUTINE_PRODUCTION_FACTOR_SCHEMA",
            LEGACY_PRODUCTION_FACTOR_SCHEMA_VERSION,
        )
    )
    env = {
        **os.environ,
        "PYTHONPATH": f"{PROJECT_ROOT / 'src'}:{PROJECT_ROOT / 'scripts' / 'research'}",
        "PROJECT_FACTOR_COMPATIBILITY_MODE": production_factor_schema,
    }
    result = subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=False, capture_output=True, text=True)
    manifest = _extract_last_json_object(result.stdout)
    scored_trade_date = str(manifest.get("target_date") or "")
    complete = (
        result.returncode == 0
        and manifest.get("status") == "success"
        and scored_trade_date == expected_trade_date
    )
    return {
        **manifest,
        "status": "success" if complete else "failed",
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
        "command": " ".join(command),
        "factor_schema_version": production_factor_schema,
        "expected_trade_date": expected_trade_date,
        "scored_trade_date": scored_trade_date or None,
        "script_elapsed_seconds": manifest.get("elapsed_seconds"),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def refresh_chan_model_scores(progress_callback=None, workers: int | None = None) -> dict:
    started = time.monotonic()
    scored_path = PROJECT_ROOT / "reports/chan_daily/model_filter/chan_model_scored_candidates.parquet"
    manifest_path = scored_path.parent / "live_refresh_manifest.json"
    candidate_path = PROJECT_ROOT / "reports/chan_daily/chan_daily_candidates.parquet"
    end_date = _incremental_daily_start()
    start_date = end_date
    if scored_path.exists():
        try:
            scored = pd.read_parquet(scored_path, columns=["date"])
            latest = pd.to_datetime(scored["date"], errors="coerce").max()
            if pd.notna(latest):
                start_date = min(latest.strftime("%Y%m%d"), end_date)
        except Exception:
            pass
    if progress_callback is not None:
        progress_callback(percent=72, message=f"正在刷新缠论实时评分：{start_date} 至 {end_date}")
    command = [
        sys.executable,
        "scripts/research/refresh_chan_model_live_scores.py",
        "--start",
        start_date,
        "--end",
        end_date,
        "--daily-dir",
        "data/raw/daily",
        "--daily-basic-dir",
        "data/raw/daily_basic",
        "--max-workers",
        str(workers or int(os.getenv("ROUTINE_CHAN_WORKERS", "8"))),
        "--executor",
        os.getenv("ROUTINE_CHAN_EXECUTOR", "processes"),
        "--batch-size",
        os.getenv("ROUTINE_CHAN_BATCH_SIZE", "16"),
        "--skip-backfill-snapshots",
    ]
    candidate_latest = pd.NaT
    if candidate_path.exists():
        try:
            candidate_dates = pd.read_parquet(candidate_path, columns=["date"])["date"]
            candidate_latest = pd.to_datetime(candidate_dates, errors="coerce").max()
        except Exception:
            candidate_latest = pd.NaT
    if pd.isna(candidate_latest) or candidate_latest.normalize() < pd.to_datetime(end_date).normalize():
        command.append("--rebuild-candidates")
    env = {**os.environ, "PYTHONPATH": f"{PROJECT_ROOT / 'src'}:{PROJECT_ROOT / 'scripts' / 'research'}"}
    previous_manifest_mtime = manifest_path.stat().st_mtime_ns if manifest_path.exists() else None
    result = subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=False, capture_output=True, text=True)
    manifest: dict = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
    processed_through = str(manifest.get("processed_through") or "").replace("-", "")
    current_manifest_mtime = manifest_path.stat().st_mtime_ns if manifest_path.exists() else None
    manifest_updated = current_manifest_mtime is not None and (
        previous_manifest_mtime is None or current_manifest_mtime > previous_manifest_mtime
    )
    success = result.returncode == 0 and manifest_updated and processed_through == end_date
    return {
        "status": "success" if success else "failed",
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
        "command": " ".join(command),
        "start_date": start_date,
        "end_date": end_date,
        "processed_through": manifest.get("processed_through"),
        "manifest_path": str(manifest_path),
        "candidate_checkpoint_reused": "--rebuild-candidates" not in command,
        "candidate_refresh": manifest.get("candidate_refresh"),
        "script_elapsed_seconds": manifest.get("elapsed_seconds"),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def run_selected_strategies() -> dict:
    command = [
        sys.executable,
        "-m",
        "quant.research.b1_formal_combos",
    ]
    result = subprocess.run(command, cwd=PROJECT_ROOT, check=False, capture_output=True, text=True)
    return {
        "status": "success" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
        "command": " ".join(command),
    }


def generate_dashboard(*, allow_incompatible: bool = False) -> dict:
    """Publish the formal B1 dashboard when its model audit is valid.

    The daily refresh does not retrain or recalibrate the formal B1 models.  An
    incompatible audit must therefore keep blocking publication, but it should
    not prevent independent daily strategies from being refreshed.
    """

    try:
        output = write_dashboard_json()
    except RuntimeError as exc:
        if allow_incompatible and str(exc).startswith(
            "B1 dashboard publication blocked by model compatibility audit:"
        ):
            return {
                "status": "skipped",
                "reason": str(exc),
                "output_preserved": str(PROJECT_ROOT / "web/data/dashboard.json"),
            }
        raise
    return {"status": "success", "output": str(output)}


def generate_daily_plan() -> dict:
    output = write_daily_plan()
    return {"status": "success", "output": str(output)}


def refresh_daily_web_workspaces(
    operations: WorkspaceRefreshOperations,
    max_workers: int | None = None,
) -> dict[str, dict]:
    """Compatibility wrapper around the application-layer workspace coordinator."""

    return refresh_daily_workspaces(operations, max_workers=max_workers)


def write_run_manifest(results: dict, strategies: list[StrategyConfig]) -> Path:
    run_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ROUTINE_DIR / run_date
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_date,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "strategies": [asdict(strategy) for strategy in strategies],
        "steps": results,
    }
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def run_daily_pipeline(
    skip_data: bool = True,
    skip_backtest: bool = False,
    *,
    workspace_operations: WorkspaceRefreshOperations | None = None,
) -> dict:
    """Run the routine B1 workflow.

    The routine keeps only selected production strategies from
    configs/strategies/b1_selected.yaml. Their full rationale is documented in
    docs/strategies/b1_selected_strategy_record.md so old experimental scripts
    and invalid model artifacts can be removed without losing review context.
    """

    pipeline_started = time.monotonic()
    results: dict[str, dict] = {}
    results["cache_cleanup"] = run_cache_cleanup(PROJECT_ROOT)
    strategies = load_strategy_configs(CONFIG_PATH)
    pipeline_status = "success"
    pipeline_error: str | None = None

    def require_complete(step_name: str, *, allow_skipped: bool = False) -> None:
        status = str(results[step_name].get("status") or "")
        allowed = {"success"} | ({"skipped"} if allow_skipped else set())
        if status not in allowed:
            detail = results[step_name].get("stderr_tail") or results[step_name].get("critical_errors") or status
            raise RuntimeError(f"{step_name} incomplete: {detail}")

    try:
        source_options = resolve_daily_dependency_source_options("short")
        results["dependency_source_options"] = {
            "status": "success",
            **source_options,
        }
        results["refresh_data"] = refresh_data(dry_run=skip_data)
        require_complete("refresh_data", allow_skipped=skip_data)
        results["refresh_daily_basic"] = refresh_daily_basic_data(dry_run=skip_data)
        require_complete("refresh_daily_basic", allow_skipped=skip_data)
        results["refresh_reference_inputs"] = refresh_reference_inputs(
            dry_run=skip_data,
            include_financials=source_options["include_financials"],
            include_analyst=source_options["include_analyst"],
            include_stock_basic=source_options["include_stock_basic"],
            include_index=source_options["include_index"],
            include_market_regime=source_options["include_market_regime"],
            include_tradability=source_options["include_tradability"],
            long_factor_datasets=source_options["long_factor_datasets"],
        )
        require_complete("refresh_reference_inputs", allow_skipped=skip_data)
        results["refresh_factor_registry"] = refresh_factor_registry_snapshot()
        require_complete("refresh_factor_registry")
        # The signal pass owns the target-date B1 gate.  Run it first so the
        # expensive six-year feature pass is limited to the B1/Z candidate
        # union instead of falling back to an all-market scan.
        results["refresh_strategy_signal_cache"] = refresh_strategy_signal_cache()
        require_complete("refresh_strategy_signal_cache")
        if (
            DEFAULT_SELECTOR_RANKING_CONFIG.source
            == SelectorRankingSource.RIGHT_SIDE_UNIFIED
        ):
            results["right_side_unified_ranking"] = run_promoted_right_side_ranking(
                str(results["refresh_strategy_signal_cache"]["processed_through_date"])
            )
            require_complete("right_side_unified_ranking")
        results["build_features"] = build_features()
        require_complete("build_features")
        results["score_latest_models"] = score_latest_models()
        require_complete("score_latest_models")
        results["refresh_chan_model_scores"] = refresh_chan_model_scores()
        require_complete("refresh_chan_model_scores")
        if skip_backtest:
            results["run_selected_strategies"] = {"status": "skipped", "reason": "skip_backtest=true"}
        else:
            results["run_selected_strategies"] = run_selected_strategies()
            require_complete("run_selected_strategies")
        with ThreadPoolExecutor(max_workers=2) as executor:
            output_futures = {
                executor.submit(generate_daily_plan): "generate_daily_plan",
                executor.submit(
                    generate_dashboard,
                    allow_incompatible=skip_backtest,
                ): "generate_dashboard",
            }
            for future in as_completed(output_futures):
                step_name = output_futures[future]
                results[step_name] = future.result()
                require_complete(
                    step_name,
                    allow_skipped=step_name == "generate_dashboard" and skip_backtest,
                )
        if workspace_operations is None:
            results["refresh_daily_web_workspaces"] = {
                "status": "skipped",
                "reason": (
                    "workspace operations were not provided; canonical production "
                    "refresh runs through web-refresh"
                ),
            }
        else:
            workspace_results = refresh_daily_web_workspaces(workspace_operations)
            results["refresh_daily_web_workspaces"] = workspace_results
            failed_workspaces = [
                name
                for name, payload in workspace_results.items()
                if payload.get("status") != "success"
            ]
            if failed_workspaces:
                raise RuntimeError(
                    f"downstream workspaces incomplete: {', '.join(failed_workspaces)}"
                )
    except Exception as exc:
        pipeline_status = "failed"
        pipeline_error = str(exc)
    finally:
        # Long-line caches are produced near the end of a run. Cleaning again
        # here enforces the configured retention immediately instead of
        # carrying obsolete multi-GB versions until tomorrow.
        results["cache_cleanup_after"] = run_cache_cleanup(PROJECT_ROOT)
        results["pipeline"] = {
            "status": pipeline_status,
            "error": pipeline_error,
            "elapsed_seconds": round(time.monotonic() - pipeline_started, 3),
        }
    manifest = write_run_manifest(results, strategies)
    return {"status": pipeline_status, "manifest": str(manifest), "steps": results}
