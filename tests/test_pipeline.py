import io
import threading
import time
import json
from dataclasses import replace

import pandas as pd
import pytest

from quant.routine import pipeline


def test_daily_basic_incremental_start_revalidates_rolling_window(
    monkeypatch,
    tmp_path,
) -> None:
    daily_basic_dir = tmp_path / "data/raw/daily_basic"
    daily_basic_dir.mkdir(parents=True)
    (daily_basic_dir / "20260812.parquet").touch()
    monkeypatch.setattr(pipeline, "PROJECT_ROOT", tmp_path)

    assert pipeline._incremental_daily_basic_start() == "20260628"


def test_daily_basic_incremental_start_includes_oldest_pending_repair(
    monkeypatch,
    tmp_path,
) -> None:
    daily_basic_dir = tmp_path / "data/raw/daily_basic"
    daily_basic_dir.mkdir(parents=True)
    (daily_basic_dir / "20260812.parquet").touch()
    (daily_basic_dir / "20260105.provenance.json").write_text(
        '{"requires_source_refresh": true}',
        encoding="utf-8",
    )
    monkeypatch.setattr(pipeline, "PROJECT_ROOT", tmp_path)

    assert pipeline._incremental_daily_basic_start() == "20260105"


def test_refresh_daily_basic_reports_partial_failure(monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "_incremental_daily_basic_start", lambda: "20260720")
    monkeypatch.setattr(
        "quant.routine.daily_basic_refresh.refresh_daily_basic",
        lambda **kwargs: {"trade_dates": 2, "success": 1, "failed": 1, "start_date": kwargs["start_date"]},
    )

    result = pipeline.refresh_daily_basic_data(dry_run=False)

    assert result["status"] == "failed"
    assert result["start_date"] == "20260720"


def test_reference_and_analyst_interfaces_run_in_parallel(monkeypatch) -> None:
    barrier = threading.Barrier(2, timeout=2)
    thread_ids: set[int] = set()

    def fake_reference_data(**kwargs):
        thread_ids.add(threading.get_ident())
        barrier.wait()
        return {"status": "success", "steps": {}, "critical_errors": []}

    def fake_analyst_refresh():
        thread_ids.add(threading.get_ident())
        barrier.wait()
        return {"status": "success", "steps": {}}

    monkeypatch.setattr(
        "quant.routine.reference_data_refresh.refresh_reference_data",
        fake_reference_data,
    )
    monkeypatch.setattr(pipeline, "_refresh_analyst_forecast_snapshot", fake_analyst_refresh)

    result = pipeline.refresh_reference_inputs(dry_run=False, include_financials=True)

    assert result["status"] == "success"
    assert result["execution_mode"] == "parallel_tushare_and_akshare"
    assert result["steps"]["analyst_forecast_snapshot"]["status"] == "success"
    assert len(thread_ids) == 2


def test_market_regime_snapshot_is_published_from_canonical_data(tmp_path) -> None:
    dates = pd.date_range("2026-01-01", periods=80, freq="B")
    end_date = dates[-1].strftime("%Y%m%d")
    index = pd.DataFrame(
        {
            "trade_date": [date.strftime("%Y%m%d") for date in dates],
            "ts_code": "000300.SH",
            "close": range(100, 180),
        }
    )
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    index.to_parquet(raw_dir / "index_000300.SH.parquet", index=False)

    class Store:
        def read_market_range(self, *args, **kwargs):
            return pd.DataFrame(
                [
                    {
                        "trade_date": date.strftime("%Y%m%d"),
                        "ts_code": symbol,
                        "close": 10 + offset,
                        "amount": 1_000_000 + offset,
                    }
                    for offset, date in enumerate(dates)
                    for symbol in ("000001.SZ", "600000.SH")
                ]
            )

    output_dir = tmp_path / "features/market_regime"
    result = pipeline.refresh_market_regime_snapshot(
        end_date,
        raw_dir=raw_dir,
        output_dir=output_dir,
        store=Store(),
    )

    assert result["status"] == "success"
    assert (output_dir / f"{end_date}.json").is_file()
    assert json.loads((output_dir / "latest.json").read_text())["as_of"] == end_date


def test_analyst_snapshot_reuses_today_checkpoint(monkeypatch, tmp_path) -> None:
    output = tmp_path / "data/raw/analyst_forecasts.parquet"
    output.parent.mkdir(parents=True)
    today = pd.Timestamp.now().normalize()
    pd.DataFrame(
        {
            "source": ["akshare_em_snapshot"],
            "report_date": [today],
        }
    ).to_parquet(output, index=False)
    monkeypatch.setattr(pipeline, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        pipeline.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("subprocess should not run")),
    )

    result = pipeline._refresh_analyst_forecast_snapshot()

    assert result["status"] == "skipped"
    assert result["latest_report_date"] == today.date().isoformat()
    assert result["polled_through"] == today.date().isoformat()


def test_daily_analyst_refresh_fetches_broker_reports_for_long_candidates(monkeypatch, tmp_path) -> None:
    output = tmp_path / "data/raw/analyst_forecasts.parquet"
    output.parent.mkdir(parents=True)
    today = pd.Timestamp.now().normalize()
    pd.DataFrame({"source": ["akshare_em_snapshot"], "report_date": [today]}).to_parquet(output, index=False)
    snapshot_dir = tmp_path / "data/long_stock_pool_snapshots"
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "tea.json").write_text(
        json.dumps(
            {
                "variant": "tea",
                "stocks": [
                    {"ts_code": "000001.SZ", "state": "CORE"},
                    {"ts_code": "000002.SZ", "state": "EXIT"},
                ],
            }
        ),
        encoding="utf-8",
    )
    calls = []

    class Result:
        returncode = 0
        stdout = "ok"
        stderr = ""

    monkeypatch.setattr(pipeline, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("ROUTINE_REFRESH_CANDIDATE_RESEARCH", "1")
    monkeypatch.setattr(pipeline.subprocess, "run", lambda command, **kwargs: calls.append(command) or Result())

    result = pipeline._refresh_analyst_forecast_snapshot()

    assert result["status"] == "success"
    assert result["candidate_symbols"] == ["000001.SZ"]
    assert len(calls) == 1
    assert "akshare_em_research" in calls[0]
    assert "000001.SZ" in calls[0]
    assert "--refresh-existing" in calls[0]


def test_daily_analyst_timeout_uses_last_known_good_research(monkeypatch, tmp_path) -> None:
    output = tmp_path / "data/raw/analyst_forecasts.parquet"
    output.parent.mkdir(parents=True)
    today = pd.Timestamp.now().normalize()
    pd.DataFrame(
        {
            "source": ["akshare_em_snapshot", "akshare_em_research"],
            "ts_code": ["000001.SZ", "000001.SZ"],
            "report_date": [today, today - pd.Timedelta(days=7)],
        }
    ).to_parquet(output, index=False)
    snapshot_dir = tmp_path / "data/long_stock_pool_snapshots"
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "tea.json").write_text(
        json.dumps({"variant": "tea", "stocks": [{"ts_code": "000001.SZ", "state": "CORE"}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(pipeline, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("ROUTINE_REFRESH_CANDIDATE_RESEARCH", "1")
    monkeypatch.setattr(
        pipeline.subprocess,
        "run",
        lambda command, **kwargs: (_ for _ in ()).throw(
            pipeline.subprocess.TimeoutExpired(command, kwargs["timeout"])
        ),
    )

    result = pipeline._refresh_analyst_forecast_snapshot()

    assert result["status"] == "degraded"
    step = result["steps"]["candidate_research_reports"]
    assert step["status"] == "degraded"
    assert step["returncode"] == 124
    assert step["fallback_symbols"] == 1
    assert step["polled_through"] is None
    assert result["polled_through"] == today.date().isoformat()
    assert not (tmp_path / "data/raw/analyst_research_refresh_status.json").exists()


def test_daily_analyst_low_ratio_research_failures_are_isolated(monkeypatch, tmp_path) -> None:
    output = tmp_path / "data/raw/analyst_forecasts.parquet"
    output.parent.mkdir(parents=True)
    today = pd.Timestamp.now().normalize()
    symbols = [f"{index:06d}.SZ" for index in range(1, 54)]
    pd.DataFrame(
        {
            "source": ["akshare_em_snapshot", *["akshare_em_research"] * 52],
            "ts_code": [symbols[0], *symbols[:52]],
            "report_date": [today, *[today - pd.Timedelta(days=7)] * 52],
        }
    ).to_parquet(output, index=False)
    snapshot_dir = tmp_path / "data/long_stock_pool_snapshots"
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "tea.json").write_text(
        json.dumps(
            {
                "variant": "tea",
                "stocks": [{"ts_code": symbol, "state": "CORE"} for symbol in symbols],
            }
        ),
        encoding="utf-8",
    )

    class Result:
        returncode = 1
        stderr = ""
        stdout = (
            "akshare_em_research progress: 53/53 success=52 failed=1 deferred=0\n"
            + json.dumps(
                {
                    "source": "akshare_em_research",
                    "symbols_requested": 53,
                    "success": 52,
                    "failed": 1,
                    "deferred": 0,
                    "failed_symbols": [symbols[-1]],
                    "deferred_symbols": [],
                },
                indent=2,
            )
        )

    monkeypatch.setattr(pipeline, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("ROUTINE_REFRESH_CANDIDATE_RESEARCH", "1")
    monkeypatch.setattr(pipeline.subprocess, "run", lambda command, **kwargs: Result())

    result = pipeline._refresh_analyst_forecast_snapshot()

    assert result["status"] == "success"
    step = result["steps"]["candidate_research_reports"]
    assert step["status"] == "success"
    assert step["failed_symbols"] == [symbols[-1]]
    assert step["failure_rate"] < step["soft_failure_threshold"]
    assert "isolated low-ratio" in step["warning"]
    assert step["polled_through"] is None


def test_daily_analyst_no_data_research_symbols_are_not_degraded(monkeypatch, tmp_path) -> None:
    output = tmp_path / "data/raw/analyst_forecasts.parquet"
    output.parent.mkdir(parents=True)
    today = pd.Timestamp.now().normalize()
    symbols = ["600177.SH", "600178.SH"]
    pd.DataFrame(
        {
            "source": ["akshare_em_snapshot"],
            "ts_code": [symbols[0]],
            "report_date": [today],
        }
    ).to_parquet(output, index=False)
    snapshot_dir = tmp_path / "data/long_stock_pool_snapshots"
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "tea.json").write_text(
        json.dumps(
            {
                "variant": "tea",
                "stocks": [{"ts_code": symbol, "state": "CORE"} for symbol in symbols],
            }
        ),
        encoding="utf-8",
    )

    class Result:
        returncode = 0
        stderr = ""
        stdout = json.dumps(
            {
                "source": "akshare_em_research",
                "symbols_requested": 2,
                "success": 1,
                "failed": 0,
                "deferred": 0,
                "no_data": 1,
                "failed_symbols": [],
                "deferred_symbols": [],
                "no_data_symbols": [symbols[0]],
            },
            indent=2,
        )

    monkeypatch.setattr(pipeline, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("ROUTINE_REFRESH_CANDIDATE_RESEARCH", "1")
    monkeypatch.setattr(pipeline.subprocess, "run", lambda command, **kwargs: Result())

    result = pipeline._refresh_analyst_forecast_snapshot()

    assert result["status"] == "success"
    step = result["steps"]["candidate_research_reports"]
    assert step["status"] == "success"
    assert step["symbols_no_data"] == 1
    assert step["no_data_symbols"] == [symbols[0]]
    assert step["degraded_symbols"] == []
    assert step["failure_rate"] == 0.0
    assert step["warning"] is None
    assert step["polled_through"] == today.date().isoformat()


def test_analyst_consensus_fallback_does_not_advance_poll_watermark(
    monkeypatch,
    tmp_path,
) -> None:
    output = tmp_path / "data/raw/analyst_forecasts.parquet"
    output.parent.mkdir(parents=True)
    prior = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
    pd.DataFrame(
        {
            "source": ["akshare_em_snapshot"],
            "ts_code": ["000001.SZ"],
            "report_date": [prior],
        }
    ).to_parquet(output, index=False)

    class Result:
        returncode = 1
        stdout = ""
        stderr = "provider unavailable"

    monkeypatch.setattr(pipeline, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(pipeline.subprocess, "run", lambda command, **kwargs: Result())

    result = pipeline._refresh_analyst_forecast_snapshot()

    assert result["status"] == "degraded"
    assert result["polled_through"] is None
    assert result["steps"]["consensus_snapshot"]["polled_through"] is None


def test_daily_candidate_report_refresh_requires_explicit_external_opt_in(monkeypatch, tmp_path) -> None:
    output = tmp_path / "data/raw/analyst_forecasts.parquet"
    output.parent.mkdir(parents=True)
    today = pd.Timestamp.now().normalize()
    pd.DataFrame({"source": ["akshare_em_snapshot"], "report_date": [today]}).to_parquet(output, index=False)
    snapshot_dir = tmp_path / "data/long_stock_pool_snapshots"
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "tea.json").write_text(
        json.dumps({"variant": "tea", "stocks": [{"ts_code": "000001.SZ", "state": "CORE"}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(pipeline, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("ROUTINE_REFRESH_CANDIDATE_RESEARCH", raising=False)
    monkeypatch.setattr(
        pipeline.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("external refresh should remain disabled")),
    )

    result = pipeline._refresh_analyst_forecast_snapshot()

    assert result["status"] == "skipped"
    assert "disabled" in result["steps"]["candidate_research_reports"]["reason"]


def test_chan_refresh_requires_current_completion_manifest(monkeypatch, tmp_path) -> None:
    report_dir = tmp_path / "reports/chan_daily/model_filter"
    report_dir.mkdir(parents=True)
    scored_path = report_dir / "chan_model_scored_candidates.parquet"
    monkeypatch.setattr(pipeline, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "_incremental_daily_start", lambda: "20260721")

    class Result:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def run_without_manifest(*args, **kwargs):
        return Result()

    monkeypatch.setattr(pipeline.subprocess, "run", run_without_manifest)

    stale = pipeline.refresh_chan_model_scores()
    assert stale["status"] == "failed"

    (report_dir / "live_refresh_manifest.json").write_text(
        json.dumps({"processed_through": "2026-07-20"}),
        encoding="utf-8",
    )

    def run_with_current_manifest(*args, **kwargs):
        (report_dir / "live_refresh_manifest.json").write_text(
            json.dumps({"processed_through": "2026-07-21"}),
            encoding="utf-8",
        )
        return Result()

    monkeypatch.setattr(pipeline.subprocess, "run", run_with_current_manifest)
    current = pipeline.refresh_chan_model_scores()
    assert current["status"] == "success"
    assert current["processed_through"] == "2026-07-21"


def test_chan_refresh_uses_explicit_worker_budget(monkeypatch, tmp_path) -> None:
    report_dir = tmp_path / "reports/chan_daily/model_filter"
    manifest_path = report_dir / "live_refresh_manifest.json"
    captured: dict[str, object] = {}
    monkeypatch.setattr(pipeline, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "_incremental_daily_start", lambda: "20260723")

    class Result:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def run(command, **kwargs):
        captured["command"] = command
        report_dir.mkdir(parents=True)
        manifest_path.write_text(
            json.dumps({"processed_through": "2026-07-23"}),
            encoding="utf-8",
        )
        return Result()

    monkeypatch.setattr(pipeline.subprocess, "run", run)

    result = pipeline.refresh_chan_model_scores(workers=3)

    command = captured["command"]
    assert command[command.index("--max-workers") + 1] == "3"
    assert command[command.index("--executor") + 1] == "processes"
    assert command[command.index("--batch-size") + 1] == "16"
    assert result["status"] == "success"


def test_build_features_uses_process_executor_by_default(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeProcess:
        stdout = io.StringIO("")

        def __init__(self, command, **kwargs) -> None:
            self.stdout = io.StringIO(
                json.dumps(
                    {
                        "status": "success",
                        "source_latest_trade_date": "2026-07-21",
                    }
                )
            )
            captured["command"] = command
            captured["kwargs"] = kwargs

        def wait(self) -> int:
            return 0

    monkeypatch.delenv("ROUTINE_FEATURE_EXECUTOR", raising=False)
    monkeypatch.setattr(pipeline.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(pipeline, "_incremental_feature_start", lambda: "20260721")
    monkeypatch.setattr(pipeline, "_incremental_daily_start", lambda: "20260721")

    result = pipeline.build_features()

    command = captured["command"]
    kwargs = captured["kwargs"]
    assert command[command.index("--executor") + 1] == "processes"
    assert command[command.index("--gate-cache") + 1] == (
        "data/features/b1/b1_gate_candidates.parquet"
    )
    assert command[command.index("--gate-manifest") + 1] == (
        "data/features/b1/b1_gate_manifest.json"
    )
    assert "--live-only" in command
    assert "start_new_session" not in kwargs
    assert result["status"] == "success"


def test_build_features_accepts_daily_basic_repair_start(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeProcess:
        def __init__(self, command, **kwargs) -> None:
            captured["command"] = command
            self.stdout = io.StringIO(
                json.dumps(
                    {
                        "status": "success",
                        "source_latest_trade_date": "2026-07-22",
                    }
                )
            )

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(pipeline.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(pipeline, "_incremental_daily_start", lambda: "20260722")

    result = pipeline.build_features(incremental_start_date="20260718")

    command = captured["command"]
    assert command[command.index("--incremental-start-date") + 1] == "20260718"
    assert result["status"] == "success"


@pytest.mark.parametrize(
    "operation",
    [
        lambda: pipeline.refresh_data(dry_run=False),
        lambda: pipeline.refresh_strategy_signal_cache(workers=1),
    ],
)
def test_pipeline_subprocesses_stay_in_web_service_process_group(
    monkeypatch,
    operation,
) -> None:
    captured: list[dict[str, object]] = []

    class FakeProcess:
        stdout = io.StringIO("")

        def __init__(self, command, **kwargs) -> None:
            if "quant.routine.data_refresh" in command:
                payload = {
                    "status": "success",
                    "expected_trade_date": "20260721",
                    "dataset_trade_date": "20260721",
                }
            else:
                payload = {
                    "status": "success",
                    "processed_through_date": "2026-07-21",
                }
            self.stdout = io.StringIO(json.dumps(payload))
            captured.append(kwargs)

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(pipeline.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(pipeline, "_incremental_daily_start", lambda: "20260721")

    result = operation()

    assert result["status"] == "success"
    assert "start_new_session" not in captured[0]


def test_refresh_data_reports_market_availability_retry_progress(monkeypatch) -> None:
    progress: list[tuple[int, str]] = []

    class FakeProcess:
        def __init__(self, command, **kwargs) -> None:
            assert command[command.index("--availability-retry-failures") + 1] == "12"
            assert command[command.index("--availability-retry-interval") + 1] == "300"
            self.stdout = io.StringIO(
                "\n".join(
                    [
                        "market daily availability retry: trade_date=20260803 "
                        "failed_attempts=1/12 retry_in_seconds=300 error=not ready",
                        json.dumps(
                            {
                                "status": "success",
                                "expected_trade_date": "20260803",
                                "dataset_trade_date": "20260803",
                            }
                        ),
                    ]
                )
            )

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(pipeline.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(pipeline, "_incremental_daily_start", lambda: "20260731")

    result = pipeline.refresh_data(
        dry_run=False,
        progress_callback=lambda percent, message: progress.append((percent, message)),
    )

    assert result["status"] == "success"
    assert progress == [
        (10, "20260803 日线尚未完整发布；已失败 1/12 次，300 秒后重试")
    ]


def test_strategy_signal_cache_reports_family_and_z_skill_progress(monkeypatch) -> None:
    progress: list[tuple[int, str]] = []

    class FakeProcess:
        def __init__(self, command, **kwargs) -> None:
            self.stdout = io.StringIO(
                "\n".join(
                    [
                        "  family signals: 500/1000 symbols",
                        "  z-skill signals: 250/500 symbols",
                        json.dumps(
                            {
                                "status": "success",
                                "processed_through_date": "2026-07-21",
                            }
                        ),
                    ]
                )
            )

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(pipeline.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(pipeline, "_incremental_daily_start", lambda: "20260721")

    result = pipeline.refresh_strategy_signal_cache(
        workers=1,
        progress_callback=lambda percent, message: progress.append((percent, message)),
    )

    assert result["status"] == "success"
    assert progress == [
        (51, "正在重建核心策略规则信号：500/1000"),
        (62, "正在重建扩展形态策略信号：250/500"),
    ]


def test_model_scoring_uses_batched_processes_and_exposes_manifest(
    monkeypatch,
) -> None:
    captured: list[str] = []

    class Result:
        returncode = 0
        stdout = json.dumps(
            {
                "status": "success",
                "target_date": "2026-08-12",
                "executor_type": "processes",
                "batch_size": 8,
                "feature_elapsed_seconds": 73.4,
                "elapsed_seconds": 76.8,
            }
        )
        stderr = ""

    def fake_run(command, **kwargs):
        captured.extend(command)
        return Result()

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    monkeypatch.setattr(pipeline, "_incremental_daily_start", lambda: "20260812")
    monkeypatch.setenv("ROUTINE_MODEL_SCORE_EXECUTOR", "processes")
    monkeypatch.setenv("ROUTINE_MODEL_SCORE_BATCH_SIZE", "8")

    result = pipeline.score_latest_models(workers=4)

    assert captured[2:10] == [
        "--target-date",
        "2026-08-12",
        "--workers",
        "4",
        "--executor",
        "processes",
        "--batch-size",
        "8",
    ]
    assert captured[10:] == [
        "--signals",
        "DUICHEN_VA",
        "NANA",
        "YIDONG_DILIAN",
    ]
    assert result["status"] == "success"
    assert result["executor_type"] == "processes"
    assert result["feature_elapsed_seconds"] == 73.4
    assert result["script_elapsed_seconds"] == 76.8
    assert result["expected_trade_date"] == "2026-08-12"
    assert result["scored_trade_date"] == "2026-08-12"


def test_model_scoring_rejects_a_stale_success_manifest(monkeypatch) -> None:
    class Result:
        returncode = 0
        stdout = json.dumps(
            {
                "status": "success",
                "target_date": "2026-08-11",
                "feature_coverage": {"status": "valid"},
            }
        )
        stderr = ""

    monkeypatch.setattr(pipeline.subprocess, "run", lambda *args, **kwargs: Result())
    monkeypatch.setattr(pipeline, "_incremental_daily_start", lambda: "20260812")

    result = pipeline.score_latest_models(workers=1)

    assert result["status"] == "failed"
    assert result["expected_trade_date"] == "2026-08-12"
    assert result["scored_trade_date"] == "2026-08-11"


def test_promoted_model_scoring_runs_only_preserved_legacy_signals(
    monkeypatch,
) -> None:
    captured: list[str] = []

    class Result:
        returncode = 0
        stdout = json.dumps(
            {
                "status": "success",
                "target_date": "2026-08-12",
                "scored_signals": ["DUICHEN_VA", "NANA", "YIDONG_DILIAN"],
            }
        )
        stderr = ""

    def fake_run(command, **kwargs):
        captured.extend(command)
        return Result()

    promoted = replace(
        pipeline.DEFAULT_SELECTOR_RANKING_CONFIG,
        source=pipeline.SelectorRankingSource.RIGHT_SIDE_UNIFIED,
        promotion_enabled=True,
    )
    monkeypatch.setattr(pipeline, "DEFAULT_SELECTOR_RANKING_CONFIG", promoted)
    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    monkeypatch.setattr(pipeline, "_incremental_daily_start", lambda: "20260812")

    result = pipeline.score_latest_models(workers=1)

    signals_index = captured.index("--signals")
    assert captured[signals_index + 1 :] == [
        "DUICHEN_VA",
        "NANA",
        "YIDONG_DILIAN",
    ]
    assert result["status"] == "success"


def test_run_selected_strategies_uses_packaged_module(monkeypatch) -> None:
    captured: list[str] = []

    class Result:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def fake_run(command, **kwargs):
        captured.extend(command)
        return Result()

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)

    result = pipeline.run_selected_strategies()

    assert captured == [
        pipeline.sys.executable,
        "-m",
        "quant.research.b1_formal_combos",
    ]
    assert result["status"] == "success"


def test_daily_pipeline_bounds_cpu_stages_and_parallelizes_outputs(monkeypatch, tmp_path) -> None:
    active = 0
    max_active = 0
    shadow_called = False
    call_order: list[str] = []
    lock = threading.Lock()

    def parallel_step() -> dict:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return {"status": "success"}

    monkeypatch.setattr(pipeline, "load_strategy_configs", lambda path: [])
    monkeypatch.setattr(
        pipeline,
        "run_cache_cleanup",
        lambda project_root: call_order.append("cleanup") or {"status": "success"},
    )
    monkeypatch.setattr(
        pipeline,
        "refresh_data",
        lambda dry_run: call_order.append("refresh") or {"status": "success"},
    )
    monkeypatch.setattr(
        pipeline,
        "refresh_daily_basic_data",
        lambda dry_run: call_order.append("daily_basic") or {"status": "success"},
    )
    monkeypatch.setattr(
        pipeline,
        "refresh_reference_inputs",
        lambda *args, **kwargs: call_order.append("reference") or {"status": "success"},
    )
    monkeypatch.setattr(
        pipeline,
        "resolve_daily_dependency_source_options",
        lambda scope: {
            "include_financials": False,
            "include_analyst": False,
            "include_stock_basic": False,
            "include_index": False,
            "include_market_regime": False,
            "include_tradability": False,
            "long_factor_datasets": (),
        },
    )
    monkeypatch.setattr(
        pipeline,
        "refresh_factor_registry_snapshot",
        lambda: {"status": "success"},
    )
    monkeypatch.setattr(pipeline, "build_features", parallel_step)

    def refresh_strategy_signals() -> dict:
        result = parallel_step()
        result["processed_through_date"] = "2026-08-13"
        return result

    monkeypatch.setattr(
        pipeline,
        "refresh_strategy_signal_cache",
        refresh_strategy_signals,
    )
    monkeypatch.setattr(
        pipeline,
        "run_promoted_right_side_ranking",
        lambda target_date: {
            "status": "success",
            "target_date": target_date,
            "checkpoint_reused": True,
        },
    )
    monkeypatch.setattr(pipeline, "score_latest_models", lambda: {"status": "success"})
    monkeypatch.setattr(pipeline, "refresh_chan_model_scores", lambda: {"status": "success"})

    def shadow_must_remain_separate(*args, **kwargs):
        nonlocal shadow_called
        shadow_called = True
        raise AssertionError("production daily pipeline invoked research shadow")

    monkeypatch.setattr(
        pipeline,
        "run_right_side_shadow_routine",
        shadow_must_remain_separate,
    )
    monkeypatch.setattr(pipeline, "generate_daily_plan", parallel_step)
    monkeypatch.setattr(pipeline, "generate_dashboard", lambda **kwargs: parallel_step())
    monkeypatch.setattr(pipeline, "write_run_manifest", lambda results, strategies: tmp_path / "manifest.json")

    result = pipeline.run_daily_pipeline(skip_data=False, skip_backtest=True)

    assert call_order[:4] == ["cleanup", "refresh", "daily_basic", "reference"]
    assert max_active == 2
    assert result["steps"]["cache_cleanup"]["status"] == "success"
    assert result["steps"]["build_features"]["status"] == "success"
    assert result["steps"]["generate_dashboard"]["status"] == "success"
    assert result["steps"]["refresh_daily_web_workspaces"]["status"] == "skipped"
    assert shadow_called is False


def test_daily_pipeline_stops_before_features_when_source_refresh_is_incomplete(monkeypatch, tmp_path) -> None:
    feature_called = False

    def build_should_not_run():
        nonlocal feature_called
        feature_called = True
        return {"status": "success"}

    monkeypatch.setattr(pipeline, "load_strategy_configs", lambda path: [])
    monkeypatch.setattr(pipeline, "run_cache_cleanup", lambda project_root: {"status": "success"})
    monkeypatch.setattr(pipeline, "refresh_data", lambda dry_run: {"status": "failed", "failed": 1})
    monkeypatch.setattr(pipeline, "build_features", build_should_not_run)
    monkeypatch.setattr(pipeline, "write_run_manifest", lambda results, strategies: tmp_path / "manifest.json")

    result = pipeline.run_daily_pipeline(skip_data=False, skip_backtest=True)

    assert result["status"] == "failed"
    assert result["steps"]["pipeline"]["status"] == "failed"
    assert feature_called is False
