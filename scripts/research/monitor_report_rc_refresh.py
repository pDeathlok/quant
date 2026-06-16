"""Print status for the long-running report_rc refresh jobs."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = PROJECT_ROOT / ".run"
REPORT_RC_PATH = PROJECT_ROOT / "data/raw/report_rc.parquet"
ANALYST_FORECAST_PATH = PROJECT_ROOT / "data/raw/analyst_forecasts.parquet"
PID_PATH = RUN_DIR / "tushare_report_rc_refresh.pid"
LOG_PATH = RUN_DIR / "tushare_report_rc_refresh.log"
PAUSED_PATH = RUN_DIR / "tushare_report_rc_refresh.paused"
DATAYES_PID_PATH = RUN_DIR / "datayes_consensus_refresh.pid"
DATAYES_LOG_PATH = RUN_DIR / "datayes_consensus_refresh.log"


def read_pid(path: Path) -> str | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return text or None


def process_status(pid: str | None) -> str:
    if not pid:
        return "not started"
    try:
        result = subprocess.run(
            ["ps", "-p", pid, "-o", "pid,ppid,stat,etime,command"],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        output = result.stdout.strip()
        return output if output else f"pid {pid} not running"
    except PermissionError as exc:
        return f"pid {pid}; ps unavailable in sandbox: {exc}"


def parquet_summary(path: Path, source_column: bool = False) -> str:
    if not path.exists():
        return f"{path}: missing"
    columns = ["ts_code", "report_date"]
    if source_column:
        columns.insert(0, "source")
    frame = pd.read_parquet(path, columns=columns)
    lines = [
        f"{path}",
        f"  rows={len(frame)} symbols={frame['ts_code'].nunique()} first={frame['report_date'].min()} last={frame['report_date'].max()}",
    ]
    if source_column and not frame.empty:
        summary = frame.groupby("source")["ts_code"].agg(["count", "nunique"]).reset_index()
        lines.append(summary.to_string(index=False))
    return "\n".join(lines)


def tail_log(path: Path, lines: int = 20) -> str:
    if not path.exists():
        return f"{path}: missing"
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(text[-lines:])


def main() -> None:
    tushare_pid = read_pid(PID_PATH)
    datayes_pid = read_pid(DATAYES_PID_PATH)
    print("## tushare process")
    if PAUSED_PATH.exists():
        print(PAUSED_PATH.read_text(encoding="utf-8").strip() or "paused")
    else:
        print(process_status(tushare_pid))
    print("\n## datayes process")
    print(process_status(datayes_pid))
    print("\n## tushare report_rc")
    print(parquet_summary(REPORT_RC_PATH))
    print("\n## analyst forecasts")
    print(parquet_summary(ANALYST_FORECAST_PATH, source_column=True))
    print("\n## tushare recent log")
    print(tail_log(LOG_PATH))
    print("\n## datayes recent log")
    print(tail_log(DATAYES_LOG_PATH))


if __name__ == "__main__":
    main()
