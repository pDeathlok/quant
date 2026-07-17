from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable, Mapping

import requests

from quant.data.tushare_fetcher import TushareDataFetcher
from quant.routine.cache_retention import cleanup_daily_caches


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_LOG_PATH = PROJECT_ROOT / ".run" / "daily_web_refresh.log"
DEFAULT_PID_PATH = PROJECT_ROOT / ".run" / "daily_web_refresh.pid"


@dataclass
class RefreshRunnerConfig:
    project_root: Path = PROJECT_ROOT
    env_path: Path = DEFAULT_ENV_PATH
    base_url: str = "http://127.0.0.1:8088/api"
    frontend_url: str | None = None
    scope: str = "all"
    poll_seconds: float = 20.0
    health_timeout_seconds: float = 60.0
    no_progress_timeout_seconds: float = 35 * 60.0
    retry_delay_seconds: float = 5.0
    max_attempts: int = 3
    service_log_path: Path = DEFAULT_LOG_PATH
    service_pid_path: Path = DEFAULT_PID_PATH
    restart_service: bool = True


@dataclass
class TradeDayDecision:
    should_run: bool
    reason: str
    trade_date: str
    raw: dict[str, Any] | None = None


class RefreshApiClient:
    def __init__(self, base_url: str, session: requests.Session | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()

    def get_health(self) -> dict[str, Any]:
        response = self.session.get(f"{self.base_url}/health", timeout=10)
        response.raise_for_status()
        return response.json()

    def get_frontend(self, frontend_url: str) -> str:
        response = self.session.get(frontend_url, timeout=10)
        response.raise_for_status()
        return response.text

    def get_status(self) -> dict[str, Any]:
        response = self.session.get(f"{self.base_url}/selector/refresh-latest/status", timeout=30)
        response.raise_for_status()
        return response.json()

    def start_refresh(self, scope: str) -> dict[str, Any]:
        response = self.session.post(
            f"{self.base_url}/selector/refresh-latest",
            json={"scope": scope},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()


def default_frontend_url(base_url: str) -> str:
    if base_url.rstrip("/").endswith("/api"):
        return base_url.rstrip("/")[:-4] + "/"
    return base_url.rstrip("/") + "/"


def check_local_web_stack(client: RefreshApiClient, frontend_url: str) -> tuple[dict[str, Any], bool]:
    health = client.get_health()
    frontend_html = client.get_frontend(frontend_url)
    if not frontend_html.strip():
        raise RuntimeError(f"前端首页响应为空: {frontend_url}")
    return health, True


def read_service_pid(pid_path: Path) -> int | None:
    try:
        raw_pid = pid_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except OSError:
        return None
    if not raw_pid:
        return None
    try:
        return int(raw_pid)
    except ValueError:
        return None


def process_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def stop_service_from_pid_file(
    pid_path: Path,
    timeout_seconds: float = 15.0,
    sleep_fn: Callable[[float], None] = time.sleep,
    monotonic_fn: Callable[[], float] = time.monotonic,
    print_fn: Callable[[str], None] = print,
) -> bool:
    pid = read_service_pid(pid_path)
    if pid is None:
        return False
    if not process_is_running(pid):
        try:
            pid_path.unlink()
        except FileNotFoundError:
            pass
        print_fn(f"[service] 清理旧 pid 文件，进程已不存在: pid={pid}")
        return False

    print_fn(f"[service] 准备重启常驻 web 服务，停止旧进程组 pid={pid}")
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except PermissionError:
        os.kill(pid, signal.SIGTERM)

    deadline = monotonic_fn() + timeout_seconds
    while monotonic_fn() < deadline:
        if not process_is_running(pid):
            try:
                pid_path.unlink()
            except FileNotFoundError:
                pass
            print_fn(f"[service] 旧 web 服务已停止: pid={pid}")
            return True
        sleep_fn(0.5)

    print_fn(f"[service] 旧 web 服务未在 {timeout_seconds:.0f}s 内退出，发送 SIGKILL: pid={pid}")
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError:
        os.kill(pid, signal.SIGKILL)
    try:
        pid_path.unlink()
    except FileNotFoundError:
        pass
    return True


def load_env_file(env_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not env_path.exists():
        return values
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key] = value
    return values


def decide_trade_day(
    target_date: date | None = None,
    fetcher_factory: Callable[..., TushareDataFetcher] = TushareDataFetcher,
) -> TradeDayDecision:
    day = target_date or date.today()
    trade_date = day.strftime("%Y%m%d")
    try:
        fetcher = fetcher_factory()
        cal = fetcher.get_trade_calendar(
            start_date=trade_date,
            end_date=trade_date,
            exchange="SSE",
            is_open="1",
        )
    except Exception as exc:
        return TradeDayDecision(
            should_run=False,
            reason=f"无法可靠确认 {trade_date} 是否为 A 股交易日: {exc}",
            trade_date=trade_date,
        )

    if cal.empty:
        return TradeDayDecision(
            should_run=False,
            reason=f"{trade_date} 非 A 股交易日，跳过刷新",
            trade_date=trade_date,
            raw={},
        )

    row = cal.iloc[0].to_dict()
    is_open = str(row.get("is_open", "0")) == "1"
    if not is_open:
        return TradeDayDecision(
            should_run=False,
            reason=f"{trade_date} 非 A 股交易日，跳过刷新",
            trade_date=trade_date,
            raw=row,
        )
    return TradeDayDecision(
        should_run=True,
        reason=f"{trade_date} 为 A 股交易日，执行刷新",
        trade_date=trade_date,
        raw=row,
    )


def ensure_local_service(
    config: RefreshRunnerConfig,
    client: RefreshApiClient,
    env: Mapping[str, str] | None = None,
    force_restart: bool = False,
    sleep_fn: Callable[[float], None] = time.sleep,
    monotonic_fn: Callable[[], float] = time.monotonic,
    print_fn: Callable[[str], None] = print,
) -> subprocess.Popen[str] | None:
    frontend_url = config.frontend_url or default_frontend_url(config.base_url)
    if force_restart:
        stop_service_from_pid_file(
            config.service_pid_path,
            sleep_fn=sleep_fn,
            monotonic_fn=monotonic_fn,
            print_fn=print_fn,
        )
    else:
        try:
            health, _ = check_local_web_stack(client, frontend_url)
            print_fn(f"[service] 前后端已就绪: api={json.dumps(health, ensure_ascii=False)} frontend={frontend_url}")
            return None
        except Exception as exc:
            print_fn(f"[service] 前后端未就绪，准备启动本地 web 服务: {exc}")

    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    merged_env["PYTHONPATH"] = str(config.project_root / "src")
    config.service_log_path.parent.mkdir(parents=True, exist_ok=True)
    config.service_pid_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = config.service_log_path.open("a", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, "scripts/run_webapp.py"],
        cwd=config.project_root,
        env=merged_env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
        close_fds=True,
    )
    config.service_pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
    deadline = monotonic_fn() + config.health_timeout_seconds
    while monotonic_fn() < deadline:
        if process.poll() is not None:
            log_handle.flush()
            raise RuntimeError(
                f"本地 Quant web 服务启动失败，进程已退出，日志见 {config.service_log_path}"
            )
        try:
            health, _ = check_local_web_stack(client, frontend_url)
            print_fn(
                f"[service] 已常驻后台启动前后端 pid={process.pid}: api={json.dumps(health, ensure_ascii=False)} frontend={frontend_url}"
            )
            return process
        except Exception:
            sleep_fn(1.0)
    raise TimeoutError(f"本地 Quant web 前后端在 {config.health_timeout_seconds:.0f}s 内未就绪")


def summarize_error(status: Mapping[str, Any]) -> str | None:
    error = status.get("error")
    if isinstance(error, str) and error.strip():
        return error.strip().splitlines()[0]
    results = status.get("result")
    if not isinstance(results, Mapping):
        return None
    for key, payload in results.items():
        if not isinstance(payload, Mapping):
            continue
        if payload.get("status") in {"failed", "error"}:
            detail = payload.get("error") or payload.get("stderr_tail") or payload.get("stdout_tail")
            if detail:
                return f"{key}: {str(detail).strip().splitlines()[0]}"
    return None


def extract_failed_count(status: Mapping[str, Any]) -> int | None:
    results = status.get("result")
    if not isinstance(results, Mapping):
        return None
    refresh_data = results.get("refresh_data")
    if isinstance(refresh_data, Mapping):
        failed = refresh_data.get("failed")
        if isinstance(failed, int):
            return failed
        stdout_tail = refresh_data.get("stdout_tail")
        if isinstance(stdout_tail, str):
            marker = '"failed":'
            if marker in stdout_tail:
                try:
                    suffix = stdout_tail[stdout_tail.rfind("{") :]
                    parsed = json.loads(suffix)
                    failed_value = parsed.get("failed")
                    if isinstance(failed_value, int):
                        return failed_value
                except Exception:
                    return None
    return None


def _status_signature(status: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        status.get("status"),
        status.get("percent"),
        status.get("current_step"),
        status.get("updated_at"),
        status.get("message"),
    )


def run_cache_cleanup(
    project_root: Path,
    reference_date: date,
    cleanup_fn: Callable[[Path, date | None], dict[str, Any]] = cleanup_daily_caches,
) -> dict[str, Any]:
    try:
        return cleanup_fn(project_root, reference_date)
    except Exception as exc:
        return {
            "status": "failed",
            "reference_date": reference_date.isoformat(),
            "reclaimed_bytes": 0,
            "errors": [str(exc)],
        }


def wait_for_terminal_status(
    client: RefreshApiClient,
    config: RefreshRunnerConfig,
    sleep_fn: Callable[[float], None] = time.sleep,
    monotonic_fn: Callable[[], float] = time.monotonic,
    print_fn: Callable[[str], None] = print,
) -> dict[str, Any]:
    last_signature: tuple[Any, ...] | None = None
    last_change_at = monotonic_fn()
    while True:
        status = client.get_status()
        signature = _status_signature(status)
        if signature != last_signature:
            last_signature = signature
            last_change_at = monotonic_fn()
            print_fn(
                "[progress] "
                f"status={status.get('status')} "
                f"percent={status.get('percent')} "
                f"step={status.get('current_step')} "
                f"updated_at={status.get('updated_at')} "
                f"message={status.get('message')}"
            )
        if status.get("status") in {"success", "failed", "error"}:
            return status
        if monotonic_fn() - last_change_at > config.no_progress_timeout_seconds:
            raise TimeoutError(
                f"刷新进展超过 {config.no_progress_timeout_seconds:.0f}s 未变化，停止等待"
            )
        sleep_fn(config.poll_seconds)


def run_refresh_workflow(
    config: RefreshRunnerConfig,
    target_date: date | None = None,
    session: requests.Session | None = None,
    fetcher_factory: Callable[..., TushareDataFetcher] = TushareDataFetcher,
    sleep_fn: Callable[[float], None] = time.sleep,
    monotonic_fn: Callable[[], float] = time.monotonic,
    print_fn: Callable[[str], None] = print,
) -> dict[str, Any]:
    env_values = load_env_file(config.env_path)
    os.environ.update(env_values)

    decision = decide_trade_day(target_date=target_date, fetcher_factory=fetcher_factory)
    print_fn(f"[trade-day] {decision.reason}")
    if not decision.should_run:
        return {
            "status": "skipped",
            "trade_date": decision.trade_date,
            "reason": decision.reason,
            "attempts": 0,
        }

    client = RefreshApiClient(config.base_url, session=session)
    service_process = ensure_local_service(
        config=config,
        client=client,
        env=env_values,
        force_restart=config.restart_service,
        sleep_fn=sleep_fn,
        monotonic_fn=monotonic_fn,
        print_fn=print_fn,
    )
    if service_process is not None:
        print_fn(f"[service] 运行日志: {config.service_log_path}")

    attempts = 0
    while attempts < config.max_attempts:
        status = client.get_status()
        if status.get("status") in {"running", "queued"}:
            print_fn("[refresh] 检测到已有运行中的刷新任务，转为接管监控")
        else:
            attempts += 1
            print_fn(f"[refresh] 提交刷新，第 {attempts}/{config.max_attempts} 次")
            status = client.start_refresh(config.scope)
            print_fn(
                "[refresh] "
                f"status={status.get('status')} percent={status.get('percent')} message={status.get('message')}"
            )

        try:
            terminal = wait_for_terminal_status(
                client=client,
                config=config,
                sleep_fn=sleep_fn,
                monotonic_fn=monotonic_fn,
                print_fn=print_fn,
            )
        except Exception as exc:
            if attempts >= config.max_attempts:
                raise
            print_fn(f"[retry] 监控失败: {exc}")
            ensure_local_service(
                config=config,
                client=client,
                env=env_values,
                force_restart=False,
                sleep_fn=sleep_fn,
                monotonic_fn=monotonic_fn,
                print_fn=print_fn,
            )
            sleep_fn(config.retry_delay_seconds)
            continue

        if terminal.get("status") == "success":
            failed_count = extract_failed_count(terminal)
            cache_cleanup = run_cache_cleanup(
                config.project_root,
                target_date or date.today(),
            )
            print_fn(
                "[cache-cleanup] "
                f"status={cache_cleanup['status']} "
                f"reclaimed_bytes={cache_cleanup['reclaimed_bytes']} "
                f"errors={len(cache_cleanup['errors'])}"
            )
            return {
                "status": "success",
                "trade_date": decision.trade_date,
                "attempts": max(attempts, 1),
                "started_at": terminal.get("started_at"),
                "finished_at": terminal.get("finished_at"),
                "failed_count": failed_count,
                "error_summary": summarize_error(terminal),
                "result": terminal.get("result"),
                "cache_cleanup": cache_cleanup,
            }

        if attempts >= config.max_attempts:
            return {
                "status": str(terminal.get("status") or "failed"),
                "trade_date": decision.trade_date,
                "attempts": attempts,
                "started_at": terminal.get("started_at"),
                "finished_at": terminal.get("finished_at"),
                "failed_count": extract_failed_count(terminal),
                "error_summary": summarize_error(terminal),
                "result": terminal.get("result"),
            }

        print_fn(
            "[retry] "
            f"检测到终态 {terminal.get('status')}，准备在 {config.retry_delay_seconds:.0f}s 后自动重试"
        )
        ensure_local_service(
            config=config,
            client=client,
            env=env_values,
            force_restart=False,
            sleep_fn=sleep_fn,
            monotonic_fn=monotonic_fn,
            print_fn=print_fn,
        )
        sleep_fn(config.retry_delay_seconds)

    raise RuntimeError("刷新重试流程意外结束")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="一键执行 A 股交易日页面全量刷新，自动检查服务并在失败后重试。",
    )
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_PATH), help="env 文件路径")
    parser.add_argument("--base-url", default="http://127.0.0.1:8088/api", help="本地 web API 根地址")
    parser.add_argument("--frontend-url", default=None, help="本地前端首页地址，默认由 --base-url 推导")
    parser.add_argument("--scope", default="all", help="刷新范围，默认 all")
    parser.add_argument("--poll-seconds", type=float, default=20.0, help="状态轮询间隔秒数")
    parser.add_argument("--health-timeout", type=float, default=60.0, help="服务启动等待秒数")
    parser.add_argument(
        "--no-progress-timeout",
        type=float,
        default=35 * 60.0,
        help="状态长时间无变化时的超时秒数",
    )
    parser.add_argument("--retry-delay", type=float, default=5.0, help="失败后重试前等待秒数")
    parser.add_argument("--max-attempts", type=int, default=3, help="最多触发刷新次数")
    parser.add_argument("--log-file", default=str(DEFAULT_LOG_PATH), help="服务启动日志路径")
    parser.add_argument("--pid-file", default=str(DEFAULT_PID_PATH), help="常驻后台服务 pid 文件路径")
    parser.add_argument(
        "--no-restart-service",
        action="store_true",
        help="不在刷新前重启本地 web 服务，调试时使用",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = RefreshRunnerConfig(
        env_path=Path(args.env_file).expanduser().resolve(),
        base_url=args.base_url,
        frontend_url=args.frontend_url,
        scope=args.scope,
        poll_seconds=args.poll_seconds,
        health_timeout_seconds=args.health_timeout,
        no_progress_timeout_seconds=args.no_progress_timeout,
        retry_delay_seconds=args.retry_delay,
        max_attempts=args.max_attempts,
        service_log_path=Path(args.log_file).expanduser().resolve(),
        service_pid_path=Path(args.pid_file).expanduser().resolve(),
        restart_service=not args.no_restart_service,
    )
    result = run_refresh_workflow(config=config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") in {"success", "skipped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
