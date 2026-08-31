from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from quant.routine import daily_basic_refresh, data_refresh, web_refresh_runner as runner
from quant.routine import tushare_availability as availability


class Clock:
    def __init__(self, hour=16, minute=0):
        self.value = datetime(2026, 8, 28, hour, minute, tzinfo=availability.MARKET_TIMEZONE)
        self.sleeps = []

    def now(self):
        return self.value

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.value += timedelta(seconds=seconds)


def basic_frame(*, missing=False):
    frame = pd.DataFrame({
        "ts_code": ["000001.SZ"],
        "trade_date": ["20260828"],
        **{column: [1.0] for column in daily_basic_refresh.DAILY_BASIC_FEATURE_COVERAGE},
    })
    if missing:
        frame[["pe", "ps"]] = float("nan")
    return frame


def test_deadline_uses_shanghai_time_and_shortens_last_wait():
    now = datetime(2026, 8, 28, 9, 15, tzinfo=timezone.utc)
    deadline = availability.tushare_retry_deadline("20260828", now)

    assert deadline.isoformat() == "2026-08-28T17:20:00+08:00"
    assert availability.tushare_retry_delay(deadline, now) == 300
    assert availability.tushare_retry_delay(deadline, deadline) is None
    assert availability.tushare_retry_delay(deadline, deadline + timedelta(days=1)) is None
    assert availability.tushare_retry_deadline("20260827", now) is None


@pytest.mark.parametrize("error, expected", [
    ("Tushare daily_basic model feature coverage below threshold for 20260828", True),
    ("Tushare daily_basic returned 0 rows for 20260828; minimum is 1", True),
    ("Tushare daily_basic returned 100 rows for 20260828; minimum is 5400", True),
    ("Tushare daily returned no market rows for 20260828", True),
    ("Tushare daily market coverage below 99.50%", True),
    ("Tushare daily did not reach expected trade date for 000001.SZ", True),
    ("Tushare daily_basic missing model feature columns for 20260828", True),
    ("Tushare stock_basic returned no rows", True),
    ("Tushare daily_basic duplicate ts_code", False),
    ("Tushare token permission denied", False),
    ("Tushare daily_basic Connection timed out", False),
    ("model feature coverage below threshold", False),
])
def test_only_tushare_missing_data_uses_deadline(error, expected):
    assert availability.is_tushare_data_missing(error) is expected


def run_basic(monkeypatch, tmp_path, clock, frames):
    class Fetcher:
        def __init__(self, **kwargs):
            pass

        def get_daily_basic(self, trade_date):
            response = frames.pop(0)
            if isinstance(response, Exception):
                raise response
            return response

    monkeypatch.setattr(daily_basic_refresh, "TushareDataFetcher", Fetcher)
    progress = []
    result = daily_basic_refresh.fetch_one_trade_date(
        "20260828", tmp_path / "basic", tmp_path / "cache",
        daily_basic_refresh.RequestLimiter(0),
        retries=1, retry_base_delay=2, retry_max_delay=60,
        expected_rows=1, availability_retry_failures=1,
        daily_dir=tmp_path / "daily", now_fn=clock.now,
        sleep_fn=clock.sleep, progress_callback=lambda **kwargs: progress.append(kwargs),
    )
    return result, progress


def test_daily_basic_waits_beyond_attempt_limit_and_continues_on_complete_data(monkeypatch, tmp_path):
    clock = Clock(16, 20)
    frames = [basic_frame(missing=True) for _ in range(4)] + [basic_frame()]

    result, progress = run_basic(monkeypatch, tmp_path, clock, frames)

    assert result["status"] == "success"
    assert result["attempts"] == 5
    assert clock.sleeps == [600] * 4
    assert len(progress) == 4
    assert all("17:20" in item["message"] for item in progress)
    assert pd.read_parquet(tmp_path / "basic/20260828.parquet")["pe"].notna().all()


@pytest.mark.parametrize("empty", [False, True])
def test_daily_basic_checks_at_deadline_then_fails_without_publishing(monkeypatch, tmp_path, empty):
    clock = Clock(17, 5)
    frames = [pd.DataFrame() if empty else basic_frame(missing=True) for _ in range(3)]

    result, _ = run_basic(monkeypatch, tmp_path, clock, frames)

    assert result["status"] == "failed"
    assert result["attempts"] == 3
    assert clock.sleeps == [600, 300]
    assert result["data_missing"] is True
    assert not (tmp_path / "basic/20260828.parquet").exists()
    assert (tmp_path / "basic/20260828.provenance.json").exists()


def test_daily_basic_other_errors_keep_normal_retry_budget(monkeypatch, tmp_path):
    clock = Clock()
    frames = [RuntimeError("permission denied") for _ in range(3)]

    result, _ = run_basic(monkeypatch, tmp_path, clock, frames)

    assert result["status"] == "failed"
    assert result["data_missing"] is False
    assert clock.sleeps == []


def run_market(clock, frames):
    class Fetcher:
        @property
        def pro(self):
            return self

        def daily(self, **kwargs):
            return frames.pop(0)

    return data_refresh._wait_for_market_daily_availability(
        Fetcher(), "20260828", [("000001.SZ", "test")],
        sleep_between=0, retry_failures=0, retry_interval_seconds=300,
        sleep_fn=clock.sleep, now_fn=clock.now,
    )


def test_market_daily_waits_until_complete_despite_zero_legacy_retries():
    clock = Clock(16, 50)
    complete = pd.DataFrame([{
        "ts_code": "000001.SZ", "trade_date": "20260828",
        "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0, "vol": 100.0,
    }])

    _, probe = run_market(clock, [pd.DataFrame(), pd.DataFrame(), complete])

    assert clock.sleeps == [600, 600]
    assert probe["attempts"] == 3
    assert probe["retry_interval_seconds"] == 600


def test_market_daily_failure_stops_at_deadline():
    clock = Clock(17, 15)

    with pytest.raises(data_refresh.MarketDataNotReadyError, match="17:20"):
        run_market(clock, [pd.DataFrame(), pd.DataFrame()])

    assert clock.sleeps == [300]


def missing_status():
    return {
        "status": "failed", "error": "Tushare daily_basic refresh failed",
        "result": {
            "refresh_data": {"status": "success", "failed": 0},
            "refresh_daily_basic": {
                "status": "failed", "failed": 1, "data_missing": True,
                "error_summary": "Tushare daily_basic model feature coverage below threshold: pe, ps",
            },
        },
    }


def run_workflow(monkeypatch, tmp_path, clock, terminals, max_attempts=1):
    class Client:
        terminal = {"status": "idle"}
        starts = 0

        def get_status(self):
            return self.terminal

        def start_refresh(self, scope):
            self.starts += 1
            self.terminal = terminals.pop(0)
            return {"status": "queued"}

    client = Client()
    monkeypatch.setattr(runner, "RefreshApiClient", lambda *args, **kwargs: client)
    monkeypatch.setattr(runner, "ensure_local_service", lambda **kwargs: None)
    monkeypatch.setattr(runner, "decide_trade_day", lambda **kwargs: runner.TradeDayDecision(True, "open", "20260828"))
    config = runner.RefreshRunnerConfig(
        project_root=tmp_path, env_path=tmp_path / "missing.env", max_attempts=max_attempts,
    )
    result = runner.run_refresh_workflow(
        config, target_date=date(2026, 8, 28), now_fn=clock.now,
        sleep_fn=clock.sleep, print_fn=lambda _: None,
    )
    return result, client.starts


def test_workflow_missing_data_does_not_exhaust_normal_attempts(monkeypatch, tmp_path):
    clock = Clock(16, 30)
    terminals = [missing_status() for _ in range(3)] + [{"status": "success"}]

    result, starts = run_workflow(monkeypatch, tmp_path, clock, terminals)

    assert result["status"] == "success"
    assert starts == result["attempts"] == 4
    assert clock.sleeps == [600] * 3


@pytest.mark.parametrize("hour, minute, sleeps, starts", [(17, 15, [300], 2), (17, 20, [], 1), (18, 0, [], 1)])
def test_workflow_does_not_retry_missing_data_past_deadline(monkeypatch, tmp_path, hour, minute, sleeps, starts):
    clock = Clock(hour, minute)

    result, actual_starts = run_workflow(monkeypatch, tmp_path, clock, [missing_status()] * starts, max_attempts=3)

    assert result["status"] == "failed"
    assert actual_starts == starts
    assert clock.sleeps == sleeps
    assert result["failed_count"] == 1
    assert "pe, ps" in result["error_summary"]


def test_workflow_other_failures_keep_normal_retry_limit(monkeypatch, tmp_path):
    failure = {"status": "failed", "error": "Tushare permission denied"}
    clock = Clock()

    result, starts = run_workflow(monkeypatch, tmp_path, clock, [failure, failure], max_attempts=2)

    assert result["status"] == "failed"
    assert starts == 2
    assert clock.sleeps == [5]


def test_workflow_success_after_deadline_is_not_interrupted(monkeypatch, tmp_path):
    clock = Clock(18, 0)

    result, starts = run_workflow(monkeypatch, tmp_path, clock, [{"status": "success"}])

    assert result["status"] == "success"
    assert starts == 1


def test_workflow_does_not_misclassify_old_missing_logs_as_current_network_failure():
    status = {
        "status": "failed",
        "result": {"refresh_data": {
            "status": "failed",
            "stderr_tail": (
                "Tushare daily returned no market rows for 20260828\n"
                "requests.exceptions.ConnectionError: connection refused"
            ),
        }},
    }

    assert runner._tushare_missing_error(status) is None


def test_workflow_explicit_mixed_failure_does_not_enter_missing_data_wait():
    status = missing_status()
    status["result"]["refresh_daily_basic"]["data_missing"] = False

    assert runner._tushare_missing_error(status) is None


def test_workflow_recognizes_reference_source_missing_rows():
    status = {
        "status": "failed",
        "result": {"refresh_reference_inputs": {
            "status": "failed",
            "critical_errors": ["stock_basic: Tushare stock_basic returned no rows"],
        }},
    }

    assert runner._tushare_missing_error(status) is not None
