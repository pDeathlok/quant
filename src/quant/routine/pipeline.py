from __future__ import annotations

import subprocess
import sys
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from quant.routine.dashboard import write_dashboard_json
from quant.routine.b1_daily_plan import write_daily_plan
from quant.routine.paths import CONFIG_PATH, PROJECT_ROOT, ROUTINE_DIR
from quant.routine.strategies import StrategyConfig, load_strategy_configs


def refresh_data(dry_run: bool = True) -> dict:
    command = [
        sys.executable,
        "-m",
        "quant.routine.data_refresh",
        "--start",
        "20100101",
        "--adjust",
        "none",
        "--output-dir",
        "data/raw/daily",
        "--workers",
        "2",
        "--sleep",
        "0.25",
        "--retries",
        "3",
        "--retry-base-delay",
        "2",
        "--retry-max-delay",
        "60",
    ]
    if dry_run:
        return {
            "status": "skipped",
            "reason": "dry_run=true；未访问外部数据源。正式刷新仅使用 Tushare 日线数据。",
            "command": " ".join(command),
        }
    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")}
    result = subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=False, capture_output=True, text=True)
    return {
        "status": "success" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
        "command": " ".join(command),
    }


def build_features() -> dict:
    command = [
        sys.executable,
        "scripts/research/analyze_b1_entry_exit_grid.py",
        "--candidate-mode",
        "strict_no_volume",
        "--entry-mode",
        "threshold",
    ]
    return {
        "status": "planned",
        "reason": "当前候选池已经存在；正式调度时执行该命令刷新 B1 候选与模型预测",
        "command": " ".join(command),
    }


def refresh_strategy_signal_cache(workers: int = 96) -> dict:
    command = [
        sys.executable,
        "scripts/research/rebuild_strategy_signal_cache.py",
        "--workers",
        str(workers),
        "--force-refresh",
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
