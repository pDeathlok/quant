from __future__ import annotations

import subprocess
import sys
import json
import os
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from quant.routine.dashboard import write_dashboard_json
from quant.routine.b1_daily_plan import write_daily_plan
from quant.routine.paths import CONFIG_PATH, PROJECT_ROOT, ROUTINE_DIR
from quant.routine.strategies import StrategyConfig, load_strategy_configs


def _incremental_daily_start(lookback_days: int | None = None) -> str:
    if lookback_days is None:
        lookback_days = int(os.getenv("ROUTINE_DAILY_LOOKBACK_DAYS", "0"))
    latest_dates: list[pd.Timestamp] = []
    daily_dir = PROJECT_ROOT / "data/raw/daily"
    if daily_dir.exists():
        for path in daily_dir.glob("*.parquet"):
            try:
                frame = pd.read_parquet(path, columns=["trade_date"])
            except Exception:
                try:
                    frame = pd.read_parquet(path, columns=["date"])
                except Exception:
                    continue
            if frame.empty:
                continue
            if "trade_date" in frame.columns:
                dates = pd.to_datetime(frame["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
            else:
                dates = pd.to_datetime(frame["date"], errors="coerce")
            latest = dates.max()
            if pd.notna(latest):
                latest_dates.append(latest)
    if not latest_dates:
        return "20100101"
    start = max(latest_dates) - pd.Timedelta(days=lookback_days)
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


def refresh_data(dry_run: bool = True, progress_callback=None) -> dict:
    start_date = _incremental_daily_start()
    workers = os.getenv("ROUTINE_DAILY_WORKERS", "16")
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
        }
    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")}
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
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
    stderr = process.stderr.read() if process.stderr is not None else ""
    returncode = process.wait()
    stdout = "".join(stdout_lines)
    return {
        "status": "success" if returncode == 0 else "failed",
        "returncode": returncode,
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr[-4000:],
        "command": " ".join(command),
        "start_date": start_date,
    }


def build_features(progress_callback=None) -> dict:
    start_date = _incremental_feature_start()
    command = [
        sys.executable,
        "scripts/research/refresh_b1_feature_cache.py",
        "--incremental-start-date",
        start_date,
        "--workers",
        os.getenv("ROUTINE_FEATURE_WORKERS", "96"),
    ]
    env = {**os.environ, "PYTHONPATH": f"{PROJECT_ROOT / 'src'}:{PROJECT_ROOT / 'scripts' / 'research'}"}
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
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
    stderr = process.stderr.read() if process.stderr is not None else ""
    returncode = process.wait()
    stdout = "".join(stdout_lines)
    return {
        "status": "success" if returncode == 0 else "failed",
        "returncode": returncode,
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr[-4000:],
        "command": " ".join(command),
        "start_date": start_date,
    }


def refresh_strategy_signal_cache(workers: int = 96, progress_callback=None) -> dict:
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
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    stdout_lines: list[str] = []
    progress_pattern = re.compile(r"(family|z-skill) signals: (\d+)/(\d+) files")
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
    stderr = process.stderr.read() if process.stderr is not None else ""
    returncode = process.wait()
    stdout = "".join(stdout_lines)
    return {
        "status": "success" if returncode == 0 else "failed",
        "returncode": returncode,
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr[-4000:],
        "command": " ".join(command),
        "start_date": start_date,
    }


def score_latest_models(workers: int = 96) -> dict:
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

    strategies = load_strategy_configs(CONFIG_PATH)
    results: dict[str, dict] = {}
    results["refresh_data"] = refresh_data(dry_run=skip_data)
    results["build_features"] = build_features()
    results["refresh_strategy_signal_cache"] = refresh_strategy_signal_cache()
    results["score_latest_models"] = score_latest_models()
    if skip_backtest:
        results["run_selected_strategies"] = {"status": "skipped", "reason": "skip_backtest=true"}
    else:
        results["run_selected_strategies"] = run_selected_strategies()
    results["generate_daily_plan"] = generate_daily_plan()
    results["generate_dashboard"] = generate_dashboard()
    manifest = write_run_manifest(results, strategies)
    return {"manifest": str(manifest), "steps": results}
