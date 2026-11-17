"""Refresh the isolated index-futures preview (no trading API)."""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

from quant.routine.tushare_availability import market_now

ROOT = Path(__file__).resolve().parents[3]
_lock = threading.Lock()
_state: dict = {'status': 'idle'}


def refresh_status() -> dict:
    with _lock:
        return dict(_state)


def refresh_preview(root: Path, python: str | Path, trade_date: date | None = None) -> None:
    folder = root / 'data/research/index_futures/workspace_live'
    folder.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env['PYTHONPATH'] = str(root / 'src') + os.pathsep + env.get('PYTHONPATH', '')
    cmd = [str(python), str(root / 'scripts/research/build_index_futures_signal.py'), '--refresh', '--publish']
    if trade_date is not None:
        cmd += ['--trade-date', trade_date.isoformat()]
    with (folder / 'refresh.log').open('a') as log:
        subprocess.run(cmd, cwd=root, env=env, stdout=log, stderr=log, check=True, timeout=300)


def refresh_daily_preview(root: Path, day: date, report: Callable[[str], None]) -> dict:
    now = market_now()
    if day != now.date():
        return {"status": "skipped", "reason": "历史补跑跳过股指收盘预案"}
    ready_at = now.replace(hour=18, minute=0, second=0, microsecond=0)
    deadline = ready_at + timedelta(hours=1)
    if now < ready_at:
        report("股指战法等待北京时间 18:00 数据检查窗口")
        time.sleep((ready_at - now).total_seconds())

    for attempt in range(1, 8):
        report(f"正在生成 {day.isoformat()} 股指收盘预案（第 {attempt} 次）")
        try:
            refresh_preview(root, sys.executable, day)
        except (OSError, subprocess.SubprocessError):
            remaining = (deadline - market_now()).total_seconds()
            if remaining <= 0 or attempt == 7:
                return {
                    "status": "failed", "attempts": attempt,
                    "error": "股指预案未发布，已保留上次结果；请检查 workspace_live/refresh.log",
                }
            delay = min(600, remaining)
            report(f"股指预案本次未发布，{delay:g} 秒后重试")
            time.sleep(delay)
        else:
            return {"status": "success", "signal_date": day.isoformat(), "attempts": attempt}


def _refresh() -> None:
    try:
        refresh_preview(ROOT, sys.executable)
        result = {'status':'complete','message':'行情与评分已更新，历史回测保持冻结。'}
    except Exception:
        result = {'status':'failed','message':'行情或模型检查未通过，已保留上次成功结果。请检查数据权限、覆盖和本地 refresh.log。'}
    with _lock:
        _state.update(result)


def start_refresh() -> dict:
    with _lock:
        if _state['status']=='running':
            return dict(_state)
        _state.update(status='running',message='正在更新行情并核验模型与开仓条件。')
        try:
            threading.Thread(target=_refresh,name='index-futures-preview',daemon=True).start()
        except Exception:
            _state.update(status='failed',message='无法启动更新任务。')
        return dict(_state)
