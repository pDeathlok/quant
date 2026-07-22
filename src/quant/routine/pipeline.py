from __future__ import annotations

import subprocess
import sys
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from quant.data import MarketDataStore, MarketDataStoreConfig
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
    dataset = PROJECT_ROOT / "data/features/b1/training_xgb_project_vars.parquet"
    if not dataset.exists():
        return "20200101"
    try:
        frame = pd.read_parquet(dataset, columns=["date"])
    except Exception:
        return "20200101"
    if frame.empty:
        return "20200101"
    dates = pd.to_datetime(frame["date"], errors="coerce")
    latest = dates.max()
    if pd.isna(latest):
        return "20200101"
    return latest.strftime("%Y%m%d")


def _incremental_daily_basic_start() -> str:
    daily_basic_dir = PROJECT_ROOT / "data/raw/daily_basic"
    dates = [
        path.stem
        for path in daily_basic_dir.glob("*.parquet")
        if path.stem.isdigit() and len(path.stem) == 8
    ]
    return max(dates) if dates else _incremental_daily_start()


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
        start_new_session=True,
    )
    stdout_lines: list[str] = []
    progress_pattern = re.compile(r"refresh progress: (\d+)/(\d+) done, success=(\d+), failed=(\d+)")
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
    returncode = process.wait()
    stdout = "".join(stdout_lines)
    return {
        "status": "success" if returncode == 0 else "failed",
        "returncode": returncode,
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stdout[-4000:] if returncode else "",
        "command": " ".join(command),
        "start_date": start_date,
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
    )
    failed = int(manifest.get("failed") or 0)
    return {
        **manifest,
        "status": "success" if failed == 0 else "failed",
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def refresh_reference_inputs(dry_run: bool = True, include_financials: bool = True) -> dict:
    started = time.monotonic()
    end_date = _incremental_daily_start()
    if dry_run:
        return {
            "status": "skipped",
            "reason": "dry_run=true；未刷新 stock_basic、沪深300和财务报表。",
            "end_date": end_date,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
    from quant.routine.reference_data_refresh import refresh_reference_data

    if include_financials:
        # Eastmoney/AkShare and Tushare use independent upstreams and write to
        # different datasets. Run them concurrently, then merge only their
        # small status payloads after both have completed.
        with ThreadPoolExecutor(max_workers=2) as executor:
            reference_future = executor.submit(
                refresh_reference_data,
                end_date=end_date,
                include_financials=True,
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
        result = refresh_reference_data(end_date=end_date, include_financials=False)
    result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return result


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
        steps["consensus_snapshot"] = {"status": "skipped", "reason": "today's snapshot already exists"}
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
        steps["candidate_research_reports"] = {"status": "skipped", "reason": "today's candidate reports already refreshed"}
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
        if research_success:
            _atomic_write_json(
                research_marker,
                {"date": today.date().isoformat(), "symbols": symbols},
            )
        fallback_symbols = sorted(_existing_analyst_symbols(output_path, "akshare_em_research") & set(symbols))
        steps["candidate_research_reports"] = {
            "status": "success" if research_success else ("degraded" if fallback_symbols else "failed"),
            "returncode": research_result.returncode,
            "symbols_requested": len(symbols),
            "fallback_symbols": len(fallback_symbols),
            "command": " ".join(research_command),
            "stdout_tail": research_result.stdout[-2000:],
            "stderr_tail": research_result.stderr[-2000:],
        }

    failed = [name for name, item in steps.items() if item.get("status") == "failed"]
    degraded = [name for name, item in steps.items() if item.get("status") == "degraded"]
    ran = any(item.get("status") == "success" for item in steps.values())
    return {
        "status": "failed" if failed else ("degraded" if degraded else ("success" if ran else "skipped")),
        "latest_report_date": latest.date().isoformat() if latest is not None and pd.notna(latest) else None,
        "candidate_symbols": symbols,
        "steps": steps,
        "error": (
            f"failed analyst refresh steps: {', '.join(failed)}"
            if failed
            else (f"using last-known-good analyst data: {', '.join(degraded)}" if degraded else None)
        ),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def build_features(progress_callback=None) -> dict:
    started = time.monotonic()
    start_date = _incremental_feature_start()
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
    ]
    env = {**os.environ, "PYTHONPATH": f"{PROJECT_ROOT / 'src'}:{PROJECT_ROOT / 'scripts' / 'research'}"}
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
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
    return {
        "status": "success" if returncode == 0 else "failed",
        "returncode": returncode,
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stdout[-4000:] if returncode else "",
        "command": " ".join(command),
        "start_date": start_date,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


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
    env = {**os.environ, "PYTHONPATH": f"{PROJECT_ROOT / 'src'}:{PROJECT_ROOT / 'scripts' / 'research'}"}
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    stdout_lines: list[str] = []
    progress_pattern = re.compile(r"(family|z-skill) signals: (\d+)/(\d+) (?:files|symbols)")
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
                percent = 50 + int(ratio * 7)
                label = "核心策略规则信号"
            else:
                percent = 57 + int(ratio * 8)
                label = "扩展策略规则信号"
            progress_callback(
                percent=percent,
                message=f"正在增量重建{label}：{done}/{total}",
            )
    returncode = process.wait()
    stdout = "".join(stdout_lines)
    return {
        "status": "success" if returncode == 0 else "failed",
        "returncode": returncode,
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stdout[-4000:] if returncode else "",
        "command": " ".join(command),
        "start_date": start_date,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def score_latest_models(workers: int = 8) -> dict:
    started = time.monotonic()
    command = [
        sys.executable,
        "scripts/research/score_latest_strategy_models.py",
        "--workers",
        str(workers),
    ]
    env = {**os.environ, "PYTHONPATH": f"{PROJECT_ROOT / 'src'}:{PROJECT_ROOT / 'scripts' / 'research'}"}
    result = subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=False, capture_output=True, text=True)
    return {
        "status": "success" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
        "command": " ".join(command),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def refresh_chan_model_scores(progress_callback=None) -> dict:
    started = time.monotonic()
    scored_path = PROJECT_ROOT / "reports/chan_daily/model_filter/chan_model_scored_candidates.parquet"
    manifest_path = scored_path.parent / "live_refresh_manifest.json"
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
        os.getenv("ROUTINE_CHAN_WORKERS", "8"),
        "--rebuild-candidates",
        "--skip-backfill-snapshots",
    ]
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
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def run_selected_strategies() -> dict:
    command = [sys.executable, "scripts/research/analyze_b1_formal_combos.py"]
    result = subprocess.run(command, cwd=PROJECT_ROOT, check=False, capture_output=True, text=True)
    return {
        "status": "success" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
        "command": " ".join(command),
    }


def generate_dashboard() -> dict:
    output = write_dashboard_json()
    return {"status": "success", "output": str(output)}


def generate_daily_plan() -> dict:
    output = write_daily_plan()
    return {"status": "success", "output": str(output)}


def refresh_daily_web_workspaces(max_workers: int | None = None) -> dict[str, dict]:
    """Refresh every non-short-line web workspace with bounded concurrency.

    The short-line workspace is produced by the core daily pipeline. These six
    downstream workspaces only read the refreshed shared datasets and write
    separate snapshots, so they can safely run in parallel.
    """

    from quant.webapp.services import (
        _latest_candidate_signal_date,
        get_byd_daily_strategy,
        get_convertible_bond_allotments,
        get_convertible_bond_grid_plan,
        get_chan_model_strategy_plan,
        get_long_stock_pool,
        refresh_similar_pattern_analysis,
    )

    signal_date = _latest_candidate_signal_date()
    trade_date = str(signal_date).replace("-", "") if signal_date else None

    def refresh_chan() -> dict:
        payload = get_chan_model_strategy_plan(top_n=20, refresh=True, signal_date=signal_date)
        return {
            "status": "success",
            "signal_date": payload.get("signal_date"),
            "candidates": len(payload.get("candidates") or []),
        }

    def refresh_long() -> dict:
        variants = []
        for variant in ("tea", "tea_safe", "v44"):
            payload = get_long_stock_pool(variant=variant, signal_date=signal_date, refresh=True)
            variants.append(
                {
                    "variant": variant,
                    "signal_date": payload.get("signal_date"),
                    "stocks": len(payload.get("stocks") or []),
                }
            )
        return {"status": "success", "variants": variants}

    def refresh_convertible_bonds() -> dict:
        payload = get_convertible_bond_grid_plan(trade_date=trade_date, limit=18, refresh=bool(trade_date))
        return {
            "status": "success",
            "trade_date": payload.get("trade_date") or signal_date,
            "candidates": len(payload.get("candidates") or payload.get("items") or []),
        }

    def refresh_allotments() -> dict:
        allotments = get_convertible_bond_allotments(refresh=True)
        return {
            "status": "success",
            "generated_at": allotments.get("generated_at"),
            "records": len(allotments.get("records") or []),
        }

    def refresh_byd() -> dict:
        payload = get_byd_daily_strategy(refresh=True)
        planned_t = payload.get("planned_t") or {}
        return {
            "status": "success",
            "signal_date": planned_t.get("signal_date"),
            "alerts": len(payload.get("alerts") or []),
        }

    def refresh_similar() -> dict:
        similar = refresh_similar_pattern_analysis()
        return {
            "status": "success",
            "generated_at": similar.get("generated_at"),
            "targets": len(similar.get("results") or []),
        }

    jobs = {
        "chan_model_strategy": refresh_chan,
        "long_stock_pool": refresh_long,
        "convertible_bond_plan": refresh_convertible_bonds,
        "convertible_bond_allotments": refresh_allotments,
        "byd_daily_plan": refresh_byd,
        "similar_patterns": refresh_similar,
    }
    worker_count = max_workers or int(os.getenv("ROUTINE_WEB_WORKSPACE_WORKERS", "6"))
    worker_count = max(1, min(worker_count, len(jobs)))
    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(job): name for name, job in jobs.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as exc:
                results[name] = {"status": "failed", "error": str(exc)}
    return results


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


def run_daily_pipeline(skip_data: bool = True, skip_backtest: bool = False) -> dict:
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
        results["refresh_data"] = refresh_data(dry_run=skip_data)
        require_complete("refresh_data", allow_skipped=skip_data)
        results["refresh_daily_basic"] = refresh_daily_basic_data(dry_run=skip_data)
        require_complete("refresh_daily_basic", allow_skipped=skip_data)
        results["refresh_reference_inputs"] = refresh_reference_inputs(
            dry_run=skip_data,
            include_financials=True,
        )
        require_complete("refresh_reference_inputs", allow_skipped=skip_data)
        with ThreadPoolExecutor(max_workers=2) as executor:
            upstream_futures = {
                executor.submit(build_features): "build_features",
                executor.submit(refresh_strategy_signal_cache): "refresh_strategy_signal_cache",
            }
            for future in as_completed(upstream_futures):
                step_name = upstream_futures[future]
                results[step_name] = future.result()
                require_complete(step_name)
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
                executor.submit(generate_dashboard): "generate_dashboard",
            }
            for future in as_completed(output_futures):
                step_name = output_futures[future]
                results[step_name] = future.result()
                require_complete(step_name)
        results["refresh_daily_web_workspaces"] = refresh_daily_web_workspaces()
        failed_workspaces = [
            name
            for name, payload in results["refresh_daily_web_workspaces"].items()
            if isinstance(payload, dict) and payload.get("status") != "success"
        ]
        if failed_workspaces:
            raise RuntimeError(f"downstream workspaces incomplete: {', '.join(failed_workspaces)}")
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
