"""On-demand refresh for the isolated index-futures preview (no trading API)."""
from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
_lock = threading.Lock()
_state: dict = {'status': 'idle'}


def refresh_status() -> dict:
    with _lock:
        return dict(_state)


def _refresh() -> None:
    try:
        folder = ROOT/'data/research/index_futures/workspace_live'
        folder.mkdir(parents=True,exist_ok=True)
        env = dict(os.environ)
        env['PYTHONPATH'] = str(ROOT/'src')+os.pathsep+env.get('PYTHONPATH','')
        with (folder/'refresh.log').open('w') as log:
            subprocess.run([sys.executable,str(ROOT/'scripts/research/build_index_futures_signal.py'),
                            '--refresh','--publish'],cwd=ROOT,env=env,stdout=log,stderr=log,check=True,timeout=300)
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
