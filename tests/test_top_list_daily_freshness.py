from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from quant.application.daily_dependencies import (
    DEFAULT_DAILY_DEPENDENCY_REGISTRY,
    ModelContract,
    build_dependency_plan,
)
from quant.data import long_factor_backfill as backfill
from quant.routine import daily_dependency_runtime as runtime
from quant.routine.reference_data_refresh import refresh_reference_data
from quant.routine.web_refresh_runner import _tushare_missing_error


TARGET = "20260908"


def _frame(trade_date: str = TARGET, amount: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame({
        "trade_date": [trade_date],
        "ts_code": ["000001.SZ"],
        "reason": ["test"],
        "net_amount": [amount],
        "amount": [10.0],
        "net_rate": [0.1],
        "pct_change": [5.0],
    })


class _Pro:
    def __init__(self, *responses: object) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def trade_cal(self, **kwargs: str) -> pd.DataFrame:
        dates = pd.bdate_range(kwargs["start_date"], kwargs["end_date"]).strftime("%Y%m%d")
        return pd.DataFrame({"cal_date": dates, "is_open": "1"})

    def top_list(self, *, trade_date: str) -> pd.DataFrame:
        self.calls.append(trade_date)
        response = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(response, Exception):
            raise response
        return response


@pytest.fixture(autouse=True)
def _no_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROUTINE_LONG_FACTOR_RETRIES", "2")
    monkeypatch.setenv("ROUTINE_LONG_FACTOR_SLEEP", "0")
    monkeypatch.setenv("ROUTINE_LONG_FACTOR_MAX_RETRY_WAIT", "60")
    monkeypatch.setattr(backfill.time, "sleep", lambda _: None)


def _refresh(root: Path, pro: object, trade_date: str = TARGET) -> dict:
    def cached_read_must_not_run(*args, **kwargs):
        pytest.fail("daily top_list must bypass fetcher caches")

    return refresh_reference_data(
        trade_date,
        fetcher=SimpleNamespace(pro=pro, get_top_list=cached_read_must_not_run),
        raw_dir=root / "data/raw",
        audit_root=root / "audit",
        include_financials=False,
        include_stock_basic=False,
        include_index=False,
        include_tradability=False,
        long_factor_datasets=("top_list",),
    )


def _dataset(result: dict) -> dict:
    return result["steps"]["long_factor_sources"]["datasets"]["top_list"]


def _path(root: Path, trade_date: str = TARGET) -> Path:
    return root / "data/raw/top_list" / f"tushare_top_list_{trade_date}.parquet"


def _states(root: Path, result: dict) -> dict:
    return runtime.collect_node_states(
        DEFAULT_DAILY_DEPENDENCY_REGISTRY, root, {"refresh_reference_inputs": result}
    )


@pytest.mark.parametrize("scope", ["all", "chan"])
def test_active_source_options_keep_top_list_with_zero_effective_importance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scope: str,
) -> None:
    contract = ModelContract(
        node_id="score.chan",
        artifact_hashes=(),
        features_by_artifact={"chan": ("close", "top_list_count")},
        effective_features_by_artifact={"chan": ("close",)},
        required_feature_union=("close", "top_list_count"),
        effective_feature_union=("close",),
        combined_hash="test",
    )
    monkeypatch.setattr(
        runtime, "resolve_model_contracts", lambda *args, **kwargs: ({"score.chan": contract}, {})
    )
    options = runtime.resolve_active_source_options(tmp_path, scope)
    assert "data.top_list" in options["active_source_nodes"]
    assert "top_list" in options["long_factor_datasets"]
    plan = build_dependency_plan(
        DEFAULT_DAILY_DEPENDENCY_REGISTRY, scope, date(2026, 9, 8),
        effective_feature_requirements=options["effective_feature_requirements"],
    )
    entry = next(item for item in plan if item.node_id == "data.top_list")
    assert entry.active and entry.action == "poll"


@pytest.mark.parametrize("empty", [False, True])
def test_daily_poll_replaces_existing_partition_and_repolls_same_date(
    tmp_path: Path, empty: bool,
) -> None:
    path = _path(tmp_path)
    path.parent.mkdir(parents=True)
    _frame(amount=99.0).to_parquet(path, index=False)
    response = pd.DataFrame() if empty else _frame(amount=2.0)
    pro = _Pro(response)

    for _ in range(2):
        result = _refresh(tmp_path, pro)
        poll = _dataset(result)
        assert result["status"] == "success"
        assert poll["requested"] == poll["success"] == 1
        assert poll["already_complete"] == 0
        assert poll["polled_through"] == TARGET
        assert poll["polled_partitions"][TARGET] == runtime._sha256(path)
        assert _states(tmp_path, result)["data.top_list"].polled_through == date(2026, 9, 8)
    assert pro.calls == [TARGET, TARGET]
    stored = pd.read_parquet(path)
    assert {"trade_date", "ts_code", "reason"} <= set(stored.columns)
    if empty:
        assert stored.empty
    else:
        assert stored["net_amount"].tolist() == [2.0]


def test_empty_days_advance_poll_watermark_without_fabricating_rows(tmp_path: Path) -> None:
    pro = _Pro(pd.DataFrame())
    for target in ("20260907", TARGET):
        result = _refresh(tmp_path, pro, target)
        assert _dataset(result)["polled_through"] == target
        assert pd.read_parquet(_path(tmp_path, target)).empty
    assert pro.calls == ["20260907", "20260907", TARGET]
    assert len(list(_path(tmp_path).parent.glob("*.parquet"))) == 2


@pytest.mark.parametrize("first", [
    RuntimeError("provider unavailable"),
    _frame("20260803"),
    pd.DataFrame({"trade_date": [TARGET]}),
    _frame().drop(columns=["net_rate"]),
    None,
])
def test_retry_calls_provider_again_after_error_or_invalid_response(
    tmp_path: Path, first: object,
) -> None:
    pro = _Pro(first, _frame(amount=7.0))
    result = _refresh(tmp_path, pro)
    assert pro.calls == [TARGET, TARGET]
    assert result["status"] == "success"
    assert pd.read_parquet(_path(tmp_path))["net_amount"].tolist() == [7.0]
    assert _dataset(result)["polled_through"] == TARGET


@pytest.mark.parametrize(("response", "missing"), [
    (RuntimeError("provider unavailable"), False),
    (_frame("20260803"), True),
    (pd.DataFrame({"trade_date": [TARGET]}), True),
    (_frame().drop(columns=["net_rate"]), True),
    (None, True),
])
def test_failed_poll_preserves_old_file_but_never_uses_it_as_poll_evidence(
    tmp_path: Path, response: object, missing: bool,
) -> None:
    path = _path(tmp_path)
    path.parent.mkdir(parents=True)
    _frame(amount=99.0).to_parquet(path, index=False)
    old_bytes = path.read_bytes()
    pro = _Pro(response)
    result = _refresh(tmp_path, pro)
    assert pro.calls == [TARGET, TARGET]
    assert result["status"] == "failed"
    assert result["data_missing"] is missing
    assert result["error_summary"]
    retry_error = _tushare_missing_error({"result": {"refresh_reference_inputs": result}})
    assert bool(retry_error) is missing
    poll = _dataset(result)
    assert poll["status"] == "failed"
    assert not poll.get("polled_through")
    assert poll["polled_partitions"] == {}
    assert path.read_bytes() == old_bytes
    assert "data.top_list" not in _states(tmp_path, result)


def test_deferred_poll_is_not_missing_data_or_success(tmp_path: Path) -> None:
    pro = _Pro(RuntimeError("1次/小时"))
    result = _refresh(tmp_path, pro)
    assert pro.calls == [TARGET]
    assert result["status"] == "failed"
    assert result["data_missing"] is False
    assert _dataset(result)["status"] == "deferred"
    assert not _path(tmp_path).exists()
    assert "data.top_list" not in _states(tmp_path, result)


def test_missing_provider_method_fails_instead_of_skipping(tmp_path: Path) -> None:
    result = _refresh(tmp_path, SimpleNamespace())
    assert result["status"] == "failed"
    assert result["data_missing"] is False
    assert "provider does not expose" in result["error_summary"]


def test_historical_resume_never_mints_a_poll_receipt(tmp_path: Path) -> None:
    pro = _Pro(_frame())
    _refresh(tmp_path, pro)
    result = backfill.backfill_trade_date_partitions(
        pro, "top_list", TARGET, TARGET, tmp_path / "data/raw", tmp_path / "audit",
        policy=backfill.RequestPolicy(sleep_seconds=0),
    )
    assert result["status"] == "success"
    assert result["already_complete"] == 1
    assert result["requested"] == 0
    assert result["polled_partitions"] == {}
    assert pro.calls == [TARGET]
    result["polled_through"] = TARGET
    assert runtime._evidence_value(
        tmp_path, {"poll": result}, "top_list_poll", "poll", "polled_through"
    ) is None


@pytest.mark.parametrize("change", ["missing", "replaced", "no_receipt", "failed", "skipped"])
def test_freshness_rejects_missing_or_changed_output_and_unverified_results(
    tmp_path: Path, change: str,
) -> None:
    result = _refresh(tmp_path, _Pro(_frame()))
    assert "data.top_list" in _states(tmp_path, result)
    if change == "missing":
        _path(tmp_path).unlink()
    elif change == "replaced":
        _frame(amount=99.0).to_parquet(_path(tmp_path), index=False)
    elif change == "no_receipt":
        _dataset(result).pop("polled_partitions")
    else:
        _dataset(result)["status"] = change
    assert "data.top_list" not in _states(tmp_path, result)


def test_wrong_calendar_date_cannot_trigger_out_of_range_write(tmp_path: Path) -> None:
    pro = _Pro(_frame("20260803"))
    pro.trade_cal = lambda **kwargs: pd.DataFrame({"cal_date": ["20260803"], "is_open": ["1"]})
    result = _refresh(tmp_path, pro)
    assert result["status"] == "failed"
    assert pro.calls == []
    assert not list((tmp_path / "data/raw/top_list").glob("*.parquet"))


def test_partition_normalizes_dates_and_preserves_distinct_reasons(tmp_path: Path) -> None:
    frame = pd.concat([_frame("2026-09-08")] * 3, ignore_index=True)
    frame.loc[2, "reason"] = "second reason"
    result = _refresh(tmp_path, _Pro(frame))
    assert result["status"] == "success"
    stored = pd.read_parquet(_path(tmp_path))
    assert stored["trade_date"].tolist() == [TARGET, TARGET]
    assert set(stored["reason"]) == {"test", "second reason"}


def test_provider_limit_is_not_written_as_a_complete_poll(tmp_path: Path) -> None:
    frame = pd.concat([_frame()] * 10000, ignore_index=True)
    pro = _Pro(frame)
    result = _refresh(tmp_path, pro)
    assert result["status"] == "failed"
    assert result["data_missing"] is True
    assert "provider limit" in result["error_summary"]
    assert pro.calls == [TARGET, TARGET]
    assert not _path(tmp_path).exists()


class _HistoryPro(_Pro):
    def __init__(self, responses: dict[str, list[object]] | None = None) -> None:
        super().__init__()
        self.by_date = responses or {}
        self.calendar_calls: list[tuple[str, str]] = []

    def trade_cal(self, **kwargs: str) -> pd.DataFrame:
        self.calendar_calls.append((kwargs["start_date"], kwargs["end_date"]))
        return super().trade_cal(**kwargs)

    def top_list(self, *, trade_date: str) -> pd.DataFrame:
        self.calls.append(trade_date)
        remaining = self.by_date.get(trade_date, [])
        response = remaining.pop(0) if remaining else _frame(trade_date)
        if isinstance(response, Exception):
            raise response
        return response


def _seed(root: Path, trade_date: str, frame: pd.DataFrame | None = None) -> Path:
    path = _path(root, trade_date)
    path.parent.mkdir(parents=True, exist_ok=True)
    (frame if frame is not None else _frame(trade_date)).to_parquet(path, index=False)
    return path


def _sessions(start: str, end: str = TARGET) -> list[str]:
    # This fake provider's calendar is authoritative for these offline tests.
    return pd.bdate_range(start, end).strftime("%Y%m%d").tolist()


def test_catches_up_august_3_through_september_8_using_provider_calendar(tmp_path: Path) -> None:
    _seed(tmp_path, "20260803")
    pro = _HistoryPro({"20260807": [pd.DataFrame()]})
    result = _refresh(tmp_path, pro)
    poll = _dataset(result)
    assert result["status"] == "success"
    assert pro.calendar_calls == [("20260803", TARGET)]
    assert pro.calls == _sessions("20260803")
    assert poll["last_available_trade_date"] == "20260803"
    assert poll["coverage_start"] == "20260803"
    assert poll["expected_trade_dates"] == pro.calls
    assert poll["unresolved_dates"] == []
    assert poll["polled_through"] == TARGET
    assert pd.read_parquet(_path(tmp_path, "20260807")).empty
    assert "data.top_list" in _states(tmp_path, result)


def test_repoll_overlap_is_three_calendar_days_before_last_valid_partition(tmp_path: Path) -> None:
    for value in ("20260730", "20260731", "20260803"):
        _seed(tmp_path, value)
    pro = _HistoryPro()
    result = _refresh(tmp_path, pro)
    poll = _dataset(result)
    assert result["status"] == "success"
    assert poll["repoll_from"] == "20260731"
    assert poll["poll_overlap_calendar_days"] == 3
    assert pro.calls == _sessions("20260731")
    assert poll["already_complete"] == 1
    assert "20260730" in poll["validated_partitions"]
    assert "20260730" not in poll["polled_partitions"]


def test_latest_cached_file_does_not_hide_earlier_missing_or_invalid_sessions(tmp_path: Path) -> None:
    for value in ("20260803", "20260805", "20260904", "20260907", TARGET):
        _seed(tmp_path, value)
    _seed(tmp_path, "20260806", _frame("20260805"))
    _path(tmp_path, "20260807").write_bytes(b"corrupt parquet")
    pro = _HistoryPro()
    result = _refresh(tmp_path, pro)
    poll = _dataset(result)
    assert result["status"] == "success"
    assert poll["last_available_trade_date"] == TARGET
    assert poll["repoll_from"] == "20260905"
    assert set(pro.calls) == set(_sessions("20260803")) - {"20260803", "20260805", "20260904"}
    assert {"20260804", "20260806", "20260807", "20260907", TARGET} <= set(pro.calls)
    assert poll["unresolved_dates"] == []


def test_invalid_latest_file_cannot_advance_catchup_anchor(tmp_path: Path) -> None:
    _seed(tmp_path, "20260803")
    _seed(tmp_path, TARGET, _frame("20260803"))
    result = _refresh(tmp_path, _HistoryPro())
    assert result["status"] == "success"
    assert _dataset(result)["last_available_trade_date"] == "20260803"


def test_historical_hole_survives_partial_run_and_is_retried_despite_new_target_file(tmp_path: Path) -> None:
    _seed(tmp_path, "20260803")
    pro = _HistoryPro({"20260804": [_frame("20260803"), _frame("20260803")]})
    first = _refresh(tmp_path, pro)
    assert first["status"] == "failed"
    assert first["data_missing"] is True
    assert _dataset(first)["unresolved_dates"] == ["20260804"]
    assert not _dataset(first).get("polled_through")
    assert _path(tmp_path).exists()
    assert not _path(tmp_path, "20260804").exists()
    assert "data.top_list" not in _states(tmp_path, first)

    pro.calls.clear()
    second = _refresh(tmp_path, pro)
    assert second["status"] == "success"
    assert pro.calls == ["20260804", "20260907", TARGET]
    assert _dataset(second)["polled_through"] == TARGET
    assert "data.top_list" in _states(tmp_path, second)


def test_catchup_retries_upstream_for_historical_response(tmp_path: Path) -> None:
    _seed(tmp_path, "20260803")
    pro = _HistoryPro({"20260804": [RuntimeError("connection failed"), _frame("20260804")]})
    result = _refresh(tmp_path, pro)
    assert result["status"] == "success"
    assert pro.calls.count("20260804") == 2
    assert pd.read_parquet(_path(tmp_path, "20260804"))["trade_date"].tolist() == ["20260804"]


def test_bounded_recovery_does_not_backfill_all_existing_history(tmp_path: Path) -> None:
    old = _seed(tmp_path, "20130104")
    old_bytes = old.read_bytes()
    _seed(tmp_path, "20260803")
    pro = _HistoryPro()
    result = _refresh(tmp_path, pro)
    poll = _dataset(result)
    assert result["status"] == "success"
    assert poll["coverage_start"] == "20260511"
    assert pro.calendar_calls == [("20260511", TARGET)]
    assert poll["history_before_coverage_not_checked"] is True
    assert old.read_bytes() == old_bytes
    assert min(pro.calls) == "20260511"
    assert not _path(tmp_path, "20130107").exists()


def test_no_history_bootstraps_target_only_and_reports_boundary(tmp_path: Path) -> None:
    pro = _HistoryPro()
    result = _refresh(tmp_path, pro)
    poll = _dataset(result)
    assert result["status"] == "success"
    assert pro.calls == [TARGET]
    assert poll["bootstrap_target_only"] is True
    assert poll["history_start"] is None
    assert poll["coverage_start"] == TARGET


def test_historical_partition_mutation_invalidates_catchup_receipt(tmp_path: Path) -> None:
    _seed(tmp_path, "20260803")
    result = _refresh(tmp_path, _HistoryPro())
    assert "data.top_list" in _states(tmp_path, result)
    _seed(tmp_path, "20260804", _frame("20260804", amount=99.0))
    assert "data.top_list" not in _states(tmp_path, result)


def test_failed_old_overlap_poll_stays_pending_after_newer_files_succeed(tmp_path: Path) -> None:
    old = _seed(tmp_path, "20260803")
    old_bytes = old.read_bytes()
    pro = _HistoryPro({"20260803": [RuntimeError("offline"), RuntimeError("offline")]})
    first = _refresh(tmp_path, pro)
    assert first["status"] == "failed"
    assert old.read_bytes() == old_bytes
    assert _path(tmp_path).exists()
    state_path = old.parent / "daily_poll_state.json"
    assert json.loads(state_path.read_text())["pending_trade_dates"] == ["20260803"]
    pro.calls.clear()

    second = _refresh(tmp_path, pro)
    assert second["status"] == "success"
    assert pro.calls == ["20260803", "20260907", TARGET]
    assert json.loads(state_path.read_text())["pending_trade_dates"] == []


def test_pending_date_restores_boundary_when_failed_partition_never_existed(tmp_path: Path) -> None:
    _seed(tmp_path, TARGET)
    state_path = _path(tmp_path).parent / "daily_poll_state.json"
    state_path.write_text(json.dumps({
        "schema_version": "top_list_daily_poll_v1", "pending_trade_dates": ["20260803"],
    }))
    pro = _HistoryPro()
    result = _refresh(tmp_path, pro)
    assert result["status"] == "success"
    assert _dataset(result)["coverage_start"] == "20260803"
    assert pro.calls == _sessions("20260803")


def test_calendar_holiday_is_not_invented_as_a_missing_session(tmp_path: Path) -> None:
    _seed(tmp_path, "20260803")
    pro = _HistoryPro()
    calendar = pro.trade_cal

    def holiday_calendar(**kwargs):
        frame = calendar(**kwargs)
        return frame.loc[frame["cal_date"] != "20260805"]

    pro.trade_cal = holiday_calendar
    result = _refresh(tmp_path, pro)
    assert result["status"] == "success"
    assert "20260805" not in pro.calls
    assert not _path(tmp_path, "20260805").exists()


def test_reference_refresh_uses_declared_history_and_overlap_policy(tmp_path: Path, monkeypatch) -> None:
    from quant.application import daily_dependencies

    node = DEFAULT_DAILY_DEPENDENCY_REGISTRY.nodes["data.top_list"]
    node = replace(node, incremental=replace(
        node.incremental, context_lookback_calendar_days=7, poll_overlap_calendar_days=1,
    ))
    monkeypatch.setattr(daily_dependencies, "DEFAULT_DAILY_DEPENDENCY_REGISTRY", SimpleNamespace(
        nodes={"data.top_list": node},
    ))
    for value in ("20260803", "20260904", "20260907", TARGET):
        _seed(tmp_path, value)
    pro = _HistoryPro()
    result = _refresh(tmp_path, pro)
    assert result["status"] == "success"
    assert _dataset(result)["coverage_start"] == "20260901"
    assert _dataset(result)["repoll_from"] == "20260907"
    assert pro.calls == ["20260901", "20260902", "20260903", "20260907", TARGET]


def test_corrupt_pending_state_fails_closed_without_provider_poll(tmp_path: Path) -> None:
    _seed(tmp_path, TARGET)
    (_path(tmp_path).parent / "daily_poll_state.json").write_text("not json")
    pro = _HistoryPro()
    result = _refresh(tmp_path, pro)
    assert result["status"] == "failed"
    assert pro.calls == []


def test_calendar_cannot_silently_drop_a_known_pending_session(tmp_path: Path) -> None:
    _seed(tmp_path, TARGET)
    state_path = _path(tmp_path).parent / "daily_poll_state.json"
    state_path.write_text(json.dumps({
        "schema_version": "top_list_daily_poll_v1", "pending_trade_dates": ["20260907"],
    }))
    pro = _HistoryPro()
    pro.trade_cal = lambda **kwargs: pd.DataFrame({"cal_date": [TARGET], "is_open": ["1"]})
    result = _refresh(tmp_path, pro)
    assert result["status"] == "failed"
    assert result["data_missing"] is True
    assert _dataset(result)["unresolved_dates"] == ["20260907"]
    assert not _dataset(result).get("polled_through")
    assert json.loads(state_path.read_text())["pending_trade_dates"] == ["20260907"]


def test_pending_history_outside_bound_is_preserved_and_reported(tmp_path: Path) -> None:
    _seed(tmp_path, "20260803")
    state_path = _path(tmp_path).parent / "daily_poll_state.json"
    state_path.write_text(json.dumps({
        "schema_version": "top_list_daily_poll_v1", "pending_trade_dates": ["20130104"],
    }))
    pro = _HistoryPro()
    result = _refresh(tmp_path, pro)
    assert result["status"] == "success"
    assert _dataset(result)["pending_dates_outside_coverage"] == ["20130104"]
    assert min(pro.calls) == "20260511"
    assert json.loads(state_path.read_text())["pending_trade_dates"] == ["20130104"]


@pytest.mark.parametrize("column", ["net_amount", "amount", "net_rate", "pct_change"])
@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), float("-inf"), "invalid"])
def test_missing_or_nonfinite_numeric_inputs_never_replace_valid_partition(
    tmp_path: Path, column: str, value: object,
) -> None:
    path = _seed(tmp_path, TARGET)
    original = path.read_bytes()
    frame = _frame().assign(**{column: [value]})
    pro = _Pro(frame)
    result = _refresh(tmp_path, pro)
    assert result["status"] == "failed"
    assert result["data_missing"] is True
    assert "invalid numeric feature inputs" in result["error_summary"]
    assert column in result["error_summary"]
    assert pro.calls == [TARGET, TARGET]
    assert path.read_bytes() == original
    assert not _dataset(result)["polled_partitions"]
    assert not _dataset(result).get("polled_through")


def test_numeric_validation_failure_retries_fresh_provider_response(tmp_path: Path) -> None:
    pro = _Pro(_frame().assign(net_rate=[None]), _frame())
    result = _refresh(tmp_path, pro)
    assert result["status"] == "success"
    assert pro.calls == [TARGET, TARGET]
    assert pd.read_parquet(_path(tmp_path))["net_rate"].tolist() == [0.1]


@pytest.mark.parametrize("column", ["net_amount", "amount", "net_rate", "pct_change"])
def test_invalid_numeric_cached_history_is_repaired_outside_overlap(tmp_path: Path, column: str) -> None:
    _seed(tmp_path, "20260803", _frame("20260803").assign(**{column: [float("inf")]}))
    _seed(tmp_path, TARGET)
    pro = _HistoryPro()
    result = _refresh(tmp_path, pro)
    assert result["status"] == "success"
    assert _dataset(result)["repoll_from"] == "20260905"
    assert "20260803" in pro.calls
    assert pd.read_parquet(_path(tmp_path, "20260803"))[column].iloc[0] != float("inf")


def test_finite_numeric_strings_and_genuine_zero_rates_are_preserved(tmp_path: Path) -> None:
    frame = _frame().assign(net_amount=["0"], amount=["10.5"], net_rate=["0"], pct_change=["0"])
    result = _refresh(tmp_path, _Pro(frame))
    assert result["status"] == "success"
    stored = pd.read_parquet(_path(tmp_path))
    assert stored["amount"].tolist() == [10.5]
    assert stored[["net_amount", "net_rate", "pct_change"]].iloc[0].tolist() == [0, 0, 0]
