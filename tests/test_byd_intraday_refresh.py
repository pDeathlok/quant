from __future__ import annotations

import fcntl
import hashlib
import os
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

import quant.data.atomic_io as atomic_io
import quant.routine.byd_intraday_refresh as refresh_module
from quant.application.daily_dependencies import (
    DEFAULT_DAILY_DEPENDENCY_REGISTRY,
    state_is_current,
)
from quant.application.workspaces.byd import load_byd_intraday_validation_frame
from quant.data.byd_intraday import (
    PRICE_COLUMNS,
    SESSION_MINUTES,
    TRAINING_FILENAME,
    read_byd_intraday_training_evidence,
    read_byd_training_frame,
    validate_training_bars,
)
from quant.infrastructure.publication import (
    PublicationError,
    PublicationStore,
    publication_context,
)
from quant.routine.byd_intraday_refresh import refresh_byd_intraday_when_due
from quant.routine.daily_dependency_runtime import collect_node_states


def _bars(start: str = "2026-06-01", end: str = "2026-07-17") -> pd.DataFrame:
    stamps = pd.DatetimeIndex([
        day + pd.Timedelta(minutes=minute)
        for day in pd.bdate_range(start, end) for minute in SESSION_MINUTES
    ])
    return pd.DataFrame({
        "datetime": stamps, "date": stamps.normalize(),
        "code": "sz.002594", "adjustflag": "2",
        "open": 100.0, "high": 102.0, "low": 98.0, "close": 101.0,
        "volume": 1000.0, "amount": 100000.0,
    })


def _write(directory: Path, frame: pd.DataFrame, name: str = TRAINING_FILENAME) -> Path:
    path = directory / name
    frame.attrs.setdefault("provider", "test_source")
    frame.to_parquet(path, index=False)
    return path


class FakeSource:
    provider = "test_source"

    def __init__(self, bars: pd.DataFrame | None = None):
        self.bars = _bars(end="2026-09-15") if bars is None else bars
        self.calls: list[tuple[str, str, str]] = []
        self.daily_calls = 0
        self.suspended: set[pd.Timestamp] = set()
        self.holidays: set[pd.Timestamp] = set()
        self.mutate_minutes = lambda frame: frame
        self.mutate_daily = lambda frame: frame
        self.mutate_calendar = lambda frame: frame
        self.fail_exit = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.fail_exit:
            raise RuntimeError("provider exit failed")

    def calendar(self, start, end):
        self.calls.append(("calendar", start, end))
        dates = pd.date_range(start, end)
        return self.mutate_calendar(pd.DataFrame({
            "calendar_date": dates,
            "is_trading_day": [str(int(day.weekday() < 5 and day not in self.holidays)) for day in dates],
        }))

    def daily(self, start, end):
        self.calls.append(("daily", start, end))
        self.daily_calls += 1
        frame = self.bars.loc[self.bars.date.between(start, end)].groupby("date").agg(
            open=("open", "first"), high=("high", "max"),
            low=("low", "min"), close=("close", "last"),
        ).reset_index()
        frame = frame.loc[~frame.date.isin(self.holidays)]
        frame["code"], frame["adjustflag"] = "sz.002594", "2"
        frame["tradestatus"] = frame.date.map(lambda value: "0" if value in self.suspended else "1")
        return self.mutate_daily(frame)

    def minutes(self, start, end):
        self.calls.append(("minutes", start, end))
        frame = self.bars.loc[
            self.bars.date.between(start, end)
            & ~self.bars.date.isin(self.suspended | self.holidays)
        ].copy()
        return self.mutate_minutes(frame)


def _run(directory: Path, source: FakeSource, **kwargs):
    return refresh_byd_intraday_when_due(
        cache_dir=directory,
        expected_trade_date=kwargs.pop("expected_trade_date", "2026-09-15"),
        now=kwargs.pop("now", datetime(2026, 9, 15, 17)),
        source_factory=lambda: source,
        **kwargs,
    )


def test_complete_bars_normalize_sort_and_legacy_time():
    frame = _bars("2026-07-17", "2026-07-17")
    frame["time"] = frame.datetime.dt.strftime("%Y%m%d%H%M%S") + "000"
    pd.testing.assert_frame_equal(validate_training_bars(frame.iloc[::-1]), frame)


@pytest.mark.parametrize(("kind", "message"), [
    ("empty", "empty"), ("missing_column", "missing columns"),
    ("nat", "invalid local"), ("wrong_date", "date/datetime"),
    ("wrong_code", "requires"), ("unadjusted", "requires"),
    ("duplicate", "duplicate"), ("missing_bar", "incomplete session"),
    ("wrong_grid", "session grid"), ("seconds", "session grid"),
    ("negative_volume", "negative"), ("nan", "non-finite"),
    ("inf", "non-finite"), ("zero_price", "non-positive"),
    ("bad_ohlc", "OHLC"), ("bad_time", "malformed"),
    ("mismatched_time", "time/datetime"), ("timezone", "invalid local"),
])
def test_training_validation_rejects_bad_data(kind, message):
    frame = _bars("2026-07-17", "2026-07-17")
    if kind == "empty":
        frame = frame.iloc[:0]
    elif kind == "missing_column":
        frame = frame.drop(columns="amount")
    elif kind == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif kind == "missing_bar":
        frame = frame.iloc[:-1]
    elif kind == "timezone":
        frame["datetime"] = frame.datetime.dt.tz_localize("Asia/Shanghai")
    elif kind in {"bad_time", "mismatched_time"}:
        frame["time"] = frame.datetime.dt.strftime("%Y%m%d%H%M%S") + "000"
        frame.loc[0, "time"] = "bad" if kind == "bad_time" else "20260717094000000"
    else:
        column, value = {
            "nat": ("datetime", pd.NaT), "wrong_date": ("date", pd.Timestamp("2026-07-16")),
            "wrong_code": ("code", "sh.600000"), "unadjusted": ("adjustflag", "3"),
            "wrong_grid": ("datetime", pd.Timestamp("2026-07-17 09:30")),
            "seconds": ("datetime", pd.Timestamp("2026-07-17 09:35:01")),
            "negative_volume": ("volume", -1), "nan": ("close", float("nan")),
            "inf": ("close", float("inf")), "zero_price": ("low", 0),
            "bad_ohlc": ("high", 99),
        }[kind]
        frame.loc[0, column] = value
    with pytest.raises(ValueError, match=message):
        validate_training_bars(frame)


def test_canonical_wins_and_evidence_is_exact_consumed_file(tmp_path):
    _write(tmp_path, _bars(end="2026-09-15"), "baostock_002594_5min_20150101_20991231_qfq.parquet")
    canonical = _write(tmp_path, _bars("2026-07-17", "2026-07-17"))
    consumed = load_byd_intraday_validation_frame(tmp_path)
    date, checkpoint, fingerprint = read_byd_intraday_training_evidence(tmp_path)
    assert len(consumed) == 48
    assert consumed.attrs["intraday_training_source"] == str(canonical)
    assert date.isoformat() == "2026-07-17"
    assert checkpoint == datetime(2026, 7, 17)
    assert fingerprint == consumed.attrs["intraday_training_fingerprint"]
    assert fingerprint == "sha256:" + hashlib.sha256(canonical.read_bytes()).hexdigest()


def test_legacy_selection_uses_actual_coverage_not_size_or_filename(tmp_path):
    old = _bars(end="2026-07-16")
    old["padding"] = [str(i) * 100 for i in range(len(old))]
    old_path = _write(tmp_path, old, "baostock_002594_5min_20150101_20991231_qfq.parquet")
    path = _write(tmp_path, _bars(), "baostock_002594_5min_20240101_20260717_qfq.parquet")
    _write(tmp_path, _bars("2026-09-14", "2026-09-15"), "baostock_002594_5min_recent_qfq.parquet")
    assert old_path.stat().st_size > path.stat().st_size
    frame = read_byd_training_frame(tmp_path)
    assert frame.attrs["intraday_training_source"] == str(path)
    assert frame.date.max() == pd.Timestamp("2026-07-17")
    assert read_byd_intraday_training_evidence(tmp_path)[0].isoformat() == "2026-07-17"


def test_ambiguous_legacy_histories_fail_closed(tmp_path):
    _write(tmp_path, _bars(), "baostock_002594_5min_a_qfq.parquet")
    changed = _bars()
    changed[PRICE_COLUMNS] *= 0.9
    _write(tmp_path, changed, "baostock_002594_5min_b_qfq.parquet")
    with pytest.raises(ValueError, match="ambiguous"):
        read_byd_training_frame(tmp_path)


def test_corrupt_canonical_does_not_fallback(tmp_path):
    _write(tmp_path, _bars(), "baostock_002594_5min_old_qfq.parquet")
    _write(tmp_path, _bars().iloc[:-1])
    with pytest.raises(ValueError, match="incomplete"):
        read_byd_intraday_training_evidence(tmp_path)


def test_missing_data_has_no_evidence(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_byd_intraday_training_evidence(tmp_path)


def test_cache_replacement_same_size_mtime_and_mutation_isolation(tmp_path):
    path = _write(tmp_path, _bars())
    before = read_byd_training_frame(tmp_path)
    before.loc[0, "close"] = -100
    before.attrs["provider"] = "mutated"
    assert read_byd_training_frame(tmp_path).iloc[0].close == 101
    stat = path.stat()
    changed = _bars()
    changed["open"] = 99.0
    temporary = _write(tmp_path, changed, "replacement.parquet")
    # Parquet encoding keeps this replacement the same size; inode still changes.
    assert temporary.stat().st_size == stat.st_size
    os.utime(temporary, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    os.replace(temporary, path)
    after = read_byd_training_frame(tmp_path)
    assert after.iloc[0].open == 99
    assert after.attrs["provider"] != "mutated"
    assert before.attrs["intraday_training_fingerprint"] != after.attrs["intraday_training_fingerprint"]


def test_new_legacy_and_then_canonical_invalidate_reader_cache(tmp_path):
    _write(tmp_path, _bars(end="2026-07-16"), "baostock_002594_5min_a_qfq.parquet")
    assert read_byd_training_frame(tmp_path).date.max() == pd.Timestamp("2026-07-16")
    _write(tmp_path, _bars(), "baostock_002594_5min_b_qfq.parquet")
    assert read_byd_training_frame(tmp_path).date.max() == pd.Timestamp("2026-07-17")
    canonical = _write(tmp_path, _bars(end="2026-09-15"))
    assert read_byd_training_frame(tmp_path).attrs["intraday_training_source"] == str(canonical)


def test_fresh_skip_does_not_enter_provider(tmp_path):
    _write(tmp_path, _bars())

    def forbidden():
        pytest.fail("fresh data must not open a source")

    result = refresh_byd_intraday_when_due(
        cache_dir=tmp_path, expected_trade_date="2026-09-14",
        now=datetime(2026, 9, 14, 23, 59, 59, 999999), source_factory=forbidden,
    )
    assert result["status"] == "skipped"
    assert result["age_days"] < 60
    assert result["checked_at"] == "2026-07-17T00:00:00"


@pytest.mark.parametrize("now", [
    datetime(2026, 9, 15), datetime(2026, 9, 15, 17),
    pd.Timestamp("2026-09-15 09:00:00Z"),
])
def test_boundary_and_fractional_day_refresh_incrementally(tmp_path, now):
    old = _bars()
    legacy = _write(tmp_path, old, "baostock_002594_5min_old_qfq.parquet")
    original = legacy.read_bytes()
    source = FakeSource()
    result = _run(tmp_path, source, now=now, expected_trade_date="2026-09-14")
    assert result["status"] == "success"
    assert result["reason"] == "incremental"
    assert result["provider"] == "test_source"
    assert result["watermark"] == "2026-09-14"
    assert result["checked_at"] == "2026-09-14T00:00:00"
    assert result["intraday_is_current_feature"] is False
    assert legacy.read_bytes() == original
    assert result["source"] == str(tmp_path / TRAINING_FILENAME)
    calls = [call for call in source.calls if call[0] == "minutes"]
    assert calls[0][1] == "2026-07-13"
    assert all(pd.Timestamp(start).month == pd.Timestamp(end).month for _, start, end in calls)
    frame = read_byd_training_frame(tmp_path)
    assert not frame.datetime.duplicated().any()
    pd.testing.assert_frame_equal(frame.loc[frame.date < "2026-07-13"], old.loc[old.date < "2026-07-13"])
    assert result["fingerprint"] == read_byd_intraday_training_evidence(tmp_path)[2]
    source.calls.clear()
    assert _run(tmp_path, source)["status"] == "skipped"
    assert source.calls == []


def test_qfq_change_outside_overlap_rebuilds_whole_history(tmp_path):
    _write(tmp_path, _bars())
    updated = _bars(end="2026-09-15")
    # The recent overlap is unchanged: a tail-only check would miss this revision.
    updated.loc[updated.date < "2026-07-01", PRICE_COLUMNS] *= 0.8
    source = FakeSource(updated)
    result = _run(tmp_path, source)
    assert result["reason"] == "qfq_history_changed"
    assert [call for call in source.calls if call[0] == "minutes"][0][1] == "2026-06-01"
    pd.testing.assert_frame_equal(read_byd_training_frame(tmp_path), updated)


def test_provider_switch_rebuilds_history_without_mixing_sources(tmp_path):
    old = _bars()
    old["time"] = old.datetime.dt.strftime("%Y%m%d%H%M%S") + "000"
    old.attrs["provider"] = "baostock"
    _write(tmp_path, old)
    result = _run(tmp_path, FakeSource())
    assert result["reason"] == "provider_changed"
    assert result["provider"] == "test_source"
    actual = read_byd_training_frame(tmp_path)
    assert "time" not in actual
    pd.testing.assert_frame_equal(actual, _bars(end="2026-09-15"))


@pytest.mark.parametrize("kind", [
    "empty_minutes", "partial_day", "missing_day", "out_of_range", "bad_qfq",
    "missing_calendar_day", "missing_daily_day", "duplicate_daily", "bad_daily_status",
    "changed_reference", "exit_failure", "provider_error",
])
def test_failed_refresh_leaves_authority_and_cached_evidence_unchanged(tmp_path, kind):
    path = _write(tmp_path, _bars())
    original, evidence = path.read_bytes(), read_byd_intraday_training_evidence(tmp_path)
    source = FakeSource()
    if kind == "empty_minutes":
        source.mutate_minutes = lambda frame: frame.iloc[:0]
    elif kind == "partial_day":
        source.mutate_minutes = lambda frame: frame.iloc[:-1]
    elif kind == "missing_day":
        source.mutate_minutes = lambda frame: frame.loc[frame.date != frame.date.min()]
    elif kind == "out_of_range":
        source.mutate_minutes = lambda frame: pd.concat([frame, _bars("2026-09-16", "2026-09-16")])
    elif kind == "bad_qfq":
        def bad_prices(frame):
            frame[PRICE_COLUMNS] *= 0.8
            return frame
        source.mutate_minutes = bad_prices
    elif kind == "missing_calendar_day":
        source.mutate_calendar = lambda frame: frame.iloc[:-1]
    elif kind == "missing_daily_day":
        source.mutate_daily = lambda frame: frame.iloc[:-1]
    elif kind == "duplicate_daily":
        source.mutate_daily = lambda frame: pd.concat([frame, frame.iloc[:1]])
    elif kind == "bad_daily_status":
        source.mutate_daily = lambda frame: frame.assign(tradestatus="unknown")
    elif kind == "changed_reference":
        def changed_reference(frame):
            if source.daily_calls > 1:
                frame[PRICE_COLUMNS] *= 0.9
            return frame
        source.mutate_daily = changed_reference
    elif kind == "exit_failure":
        source.fail_exit = True
    else:
        def fail(frame):
            raise RuntimeError("provider failure")
        source.mutate_minutes = fail
    with pytest.raises((ValueError, RuntimeError)):
        _run(tmp_path, source)
    assert path.read_bytes() == original
    assert read_byd_intraday_training_evidence(tmp_path) == evidence


def test_atomic_replace_failure_preserves_old_file_and_cleans_temporary(tmp_path, monkeypatch):
    path = _write(tmp_path, _bars())
    original = path.read_bytes()

    def fail(temp, target):
        assert temp.exists()
        assert path.read_bytes() == original
        raise OSError("replace failed")

    monkeypatch.setattr(atomic_io, "_publish", fail)
    with pytest.raises(OSError, match="replace failed"):
        _run(tmp_path, FakeSource())
    assert path.read_bytes() == original
    assert list(tmp_path.glob(".*tmp.parquet")) == []


def test_cross_frequency_extrema_differences_are_audited_without_fabricating_prices(tmp_path):
    _write(tmp_path, _bars())
    source = FakeSource()
    def wider_daily(frame):
        frame["high"] += 0.2
        frame["low"] -= 0.2
        frame["close"] += 0.05
        return frame
    source.mutate_daily = wider_daily
    result = _run(tmp_path, source)
    assert result["status"] == "success"
    audit = result["cross_frequency_audit"]
    assert audit["cross_frequency_differing_sessions"] > 0
    assert audit["raw_minute_prices_modified"] is False
    actual = read_byd_training_frame(tmp_path)
    pd.testing.assert_frame_equal(actual[PRICE_COLUMNS], source.bars[PRICE_COLUMNS])


def test_minute_prices_outside_daily_envelope_are_rejected(tmp_path):
    path = _write(tmp_path, _bars())
    original = path.read_bytes()
    source = FakeSource()
    def outside_daily(frame):
        frame["high"] += 1
        return frame
    source.mutate_minutes = outside_daily
    with pytest.raises(ValueError, match="coverage/prices disagree"):
        _run(tmp_path, source)
    assert path.read_bytes() == original


def test_refresh_lock_rejects_concurrent_writer(tmp_path):
    _write(tmp_path, _bars())
    with (tmp_path / TRAINING_FILENAME).with_suffix(".lock").open("a") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="already running"):
            _run(tmp_path, FakeSource())


def test_holiday_and_verified_suspension_do_not_need_minute_rows(tmp_path):
    _write(tmp_path, _bars())
    source = FakeSource()
    source.holidays.add(pd.Timestamp("2026-09-15"))
    source.suspended.add(pd.Timestamp("2026-08-03"))
    result = _run(tmp_path, source)
    assert result["watermark"] == "2026-09-14"
    assert not read_byd_training_frame(tmp_path).date.eq("2026-08-03").any()


def test_still_stale_provider_cannot_publish_or_fake_requested_end(tmp_path):
    path = _write(tmp_path, _bars())
    original = path.read_bytes()
    source = FakeSource()
    source.suspended = set(pd.bdate_range("2026-07-20", "2026-09-15"))
    with pytest.raises(RuntimeError, match="60-day refresh boundary"):
        _run(tmp_path, source)
    assert path.read_bytes() == original


@pytest.mark.parametrize(("end", "now"), [
    ("2026-09-15", datetime(2026, 9, 15, 14, 59)),
    ("2026-09-16", datetime(2026, 9, 15, 17)),
    ("2026-09-15 17:00", datetime(2026, 9, 15, 17)),
])
def test_unclosed_or_invalid_decision_date_rejected_before_network(tmp_path, end, now):
    source = FakeSource()
    with pytest.raises(ValueError):
        _run(tmp_path, source, expected_trade_date=end, now=now)
    assert source.calls == []


def test_future_training_rows_rejected_before_network(tmp_path):
    _write(tmp_path, _bars(end="2026-09-16"))
    source = FakeSource()
    with pytest.raises(ValueError, match="beyond decision date"):
        _run(tmp_path, source)
    assert source.calls == []


def test_bootstrap_requires_full_requested_coverage(tmp_path):
    source = FakeSource()
    result = _run(tmp_path, source, history_start="2026-06-01")
    assert result["reason"] == "bootstrap"
    pd.testing.assert_frame_equal(read_byd_training_frame(tmp_path), source.bars)


def test_default_bootstrap_preserves_model_training_years(tmp_path):
    source = FakeSource(_bars(start="2020-01-01", end="2026-09-15"))
    result = _run(tmp_path, source)
    history = read_byd_training_frame(tmp_path)
    assert result["reason"] == "bootstrap"
    assert history.date.min() == pd.Timestamp("2020-01-01")
    assert {2022, 2023, 2024, 2025, 2026} <= set(history.date.dt.year)


def test_default_dispatch_uses_lazy_preferred_wrapper_without_network(tmp_path, monkeypatch):
    module = ModuleType("quant.routine.byd_intraday_sources")
    calls = []

    def wrapper(operation, **kwargs):
        calls.append((operation, kwargs))
        return {"status": "test_dispatch"}

    module.refresh_byd_with_preferred_sources = wrapper
    monkeypatch.setitem(sys.modules, module.__name__, module)
    now = datetime(2026, 9, 15, 17)
    result = refresh_byd_intraday_when_due(
        expected_trade_date="2026-09-15", cache_dir=tmp_path,
        now=now, history_start="2020-01-01",
    )
    assert result == {"status": "test_dispatch"}
    assert calls == [(refresh_module.refresh_byd_intraday_when_due, {
        "expected_trade_date": "2026-09-15", "cache_dir": tmp_path,
        "now": now, "history_start": "2020-01-01",
    })]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("canonical", [True, False])
def test_root_registry_consumes_same_evidence_and_ignores_newer_short_export(tmp_path, canonical):
    cache = tmp_path / "data/cache"
    cache.mkdir(parents=True)
    name = TRAINING_FILENAME if canonical else "baostock_002594_5min_history_qfq.parquet"
    _write(cache, _bars(), name)
    registry = DEFAULT_DAILY_DEPENDENCY_REGISTRY
    node = registry.nodes["data.byd_intraday_training"]
    original_state = collect_node_states(registry, tmp_path)[node.node_id]
    _write(cache, _bars("2026-09-14", "2026-09-15"), "baostock_002594_5min_recent_qfq.parquet")
    consumed = read_byd_training_frame(cache)
    state = collect_node_states(registry, tmp_path)[node.node_id]
    assert state == original_state
    assert state.watermark == consumed.date.max().date()
    assert state.checked_at == consumed.date.max().to_pydatetime()
    assert state.output_fingerprint == hashlib.sha256(
        consumed.attrs["intraday_training_fingerprint"].encode(),
    ).hexdigest()
    assert node.freshness.max_age_days == 60
    assert not state_is_current(node, state, datetime(2026, 9, 15).date(), datetime(2026, 9, 15, 17))[0]


@pytest.mark.parametrize(("now", "gate_current", "refresh_due"), [
    (datetime(2026, 9, 14, 23, 59, 59, 999999), True, False),
    (datetime(2026, 9, 15), True, True),
    (datetime(2026, 9, 15, 0, 0, 0, 1), False, True),
    (datetime(2026, 9, 15, 17), False, True),
])
def test_root_registry_60_day_boundary_and_routine_agree_without_relaxing_sla(
    tmp_path, now, gate_current, refresh_due,
):
    cache = tmp_path / "data/cache"
    cache.mkdir(parents=True)
    _write(cache, _bars())
    registry = DEFAULT_DAILY_DEPENDENCY_REGISTRY
    node = registry.nodes["data.byd_intraday_training"]
    state = collect_node_states(registry, tmp_path)[node.node_id]
    assert state.checked_at == datetime(2026, 7, 17)
    target = datetime(2026, 9, 14).date()
    assert state_is_current(node, state, target, now)[0] is gate_current
    source = FakeSource()
    result = _run(cache, source, expected_trade_date=target.isoformat(), now=now)
    assert (result["status"] == "success") is refresh_due
    assert bool(source.calls) is refresh_due
    after = collect_node_states(registry, tmp_path)[node.node_id]
    frame = read_byd_training_frame(cache)
    assert after.watermark == frame.date.max().date()
    assert after.checked_at == frame.date.max().to_pydatetime()
    assert after.output_fingerprint == hashlib.sha256(result["fingerprint"].encode()).hexdigest()
    assert state_is_current(node, after, target, now)[0] is True


def test_root_registry_rejects_corrupt_authority_despite_fresh_research_file(tmp_path):
    cache = tmp_path / "data/cache"
    cache.mkdir(parents=True)
    _write(cache, _bars().iloc[:-1])
    _write(cache, _bars("2026-09-14", "2026-09-15"), "baostock_002594_5min_recent_qfq.parquet")
    assert "data.byd_intraday_training" not in collect_node_states(DEFAULT_DAILY_DEPENDENCY_REGISTRY, tmp_path)


def _byd_publication_store(root: Path) -> PublicationStore:
    node = DEFAULT_DAILY_DEPENDENCY_REGISTRY.nodes["data.byd_intraday_training"]
    assert node.publication_outputs == (f"data/cache/{TRAINING_FILENAME}",)
    outputs = {
        path for node in DEFAULT_DAILY_DEPENDENCY_REGISTRY.nodes.values()
        for path in node.publication_outputs
    }
    return PublicationStore(root, tuple(sorted(outputs)))


def _training_publication_audit(root: Path, now: datetime) -> dict:
    registry = DEFAULT_DAILY_DEPENDENCY_REGISTRY
    node = registry.nodes["data.byd_intraday_training"]
    state = collect_node_states(registry, root)[node.node_id]
    assert state_is_current(node, state, now.date(), now)[0]
    return {
        "status": "success", "refresh_node_ids": [],
        "target_trade_date": now.date().isoformat(),
        "freshness_audit": {"status": "success", "failures": []},
    }


@pytest.mark.parametrize("abort_second", [True, False])
def test_publication_generations_isolate_training_refresh_and_rollback_evidence(
    tmp_path, abort_second,
):
    cache = tmp_path / "data/cache"
    cache.mkdir(parents=True)
    root_file = _write(cache, _bars())
    root_bytes = root_file.read_bytes()
    store = _byd_publication_store(tmp_path)
    with store.begin("byd-first") as staged:
        first_path = staged.resolve(root_file)
        first_bytes = first_path.read_bytes()
        first_evidence = read_byd_intraday_training_evidence(cache)
        assert read_byd_training_frame(cache).attrs["intraday_training_source"] == str(first_path)
        store.commit(staged, _training_publication_audit(tmp_path, datetime(2026, 7, 18)))
    first = store.view()
    source = FakeSource()
    second_evidence = None
    try:
        with store.begin("byd-second") as staged:
            assert read_byd_intraday_training_evidence(cache) == first_evidence
            result = _run(cache, source)
            second_evidence = read_byd_intraday_training_evidence(cache)
            second_path = staged.resolve(root_file)
            assert result["source"] == str(second_path)
            assert result["fingerprint"] == second_evidence[2]
            assert second_evidence[0].isoformat() == "2026-09-15"
            assert second_evidence[2] != first_evidence[2]
            assert root_file.read_bytes() == root_bytes
            assert first_path.read_bytes() == first_bytes
            # The parent can retain a warm first-generation cache while the
            # worker writes another generation; both reader and evidence agree.
            with publication_context(first):
                assert read_byd_intraday_training_evidence(cache) == first_evidence
                state = collect_node_states(DEFAULT_DAILY_DEPENDENCY_REGISTRY, tmp_path)[
                    "data.byd_intraday_training"
                ]
                assert state.checked_at == first_evidence[1]
                assert state.output_fingerprint == hashlib.sha256(first_evidence[2].encode()).hexdigest()
                with pytest.raises(PublicationError):
                    _run(cache, FakeSource())
            assert read_byd_intraday_training_evidence(cache) == second_evidence
            if abort_second:
                raise RuntimeError("downstream publication failed")
            store.commit(staged, _training_publication_audit(tmp_path, datetime(2026, 9, 15, 17)))
    except RuntimeError as exc:
        if not abort_second or str(exc) != "downstream publication failed":
            raise
    assert store.view().generation == ("byd-first" if abort_second else "byd-second")
    with publication_context(store.view()):
        assert read_byd_intraday_training_evidence(cache) == (
            first_evidence if abort_second else second_evidence
        )
    if abort_second:
        with pytest.raises(PublicationError):
            store.view("byd-second")
    # Selecting the prior committed generation is the existing rollback/read
    # interface; do not rewrite current.json or mutate committed tree files.
    with publication_context(store.view("byd-first")):
        assert read_byd_intraday_training_evidence(cache) == first_evidence
        assert read_byd_training_frame(cache).attrs["intraday_training_source"] == str(first_path)
    assert first_path.read_bytes() == first_bytes
    assert root_file.read_bytes() == root_bytes


@pytest.mark.parametrize("root_legacy_present", [False, True])
def test_old_generation_missing_new_training_path_cannot_borrow_root_evidence(
    tmp_path, root_legacy_present,
):
    cache = tmp_path / "data/cache"
    cache.mkdir(parents=True)
    root_file = _write(cache, _bars(end="2026-09-15"))
    root_bytes = root_file.read_bytes()
    if root_legacy_present:
        _write(cache, _bars(end="2026-09-15"), "baostock_002594_5min_research_qfq.parquet")
    # Build an actual committed generation using the pre-migration managed set.
    legacy_store = PublicationStore(tmp_path, ("data/workspace_snapshots",))
    with legacy_store.begin("byd-legacy") as staged:
        legacy_store.commit(staged, {
            "status": "success", "freshness_audit": {"status": "success", "failures": []},
        })
    store = _byd_publication_store(tmp_path)
    old = store.view()
    assert not old.resolve(root_file).exists()
    # Warm the root cache first: it must not satisfy the committed view.
    assert read_byd_intraday_training_evidence(cache)[0].isoformat() == "2026-09-15"
    with publication_context(old):
        with pytest.raises(FileNotFoundError):
            read_byd_training_frame(cache)
        with pytest.raises(FileNotFoundError):
            read_byd_intraday_training_evidence(cache)
        assert "data.byd_intraday_training" not in collect_node_states(
            DEFAULT_DAILY_DEPENDENCY_REGISTRY, tmp_path,
        )
        with pytest.raises(PublicationError):
            _run(cache, FakeSource(), history_start="2026-06-01")
    source = FakeSource()
    with store.begin("byd-migrated") as staged:
        assert not staged.resolve(root_file).exists()
        result = _run(cache, source, history_start="2026-06-01")
        assert result["reason"] == "bootstrap"
        assert source.calls
        assert result["source"] == str(staged.resolve(root_file))
        assert read_byd_intraday_training_evidence(cache)[2] == result["fingerprint"]
        store.commit(staged, _training_publication_audit(tmp_path, datetime(2026, 9, 15, 17)))
    assert store.view().generation == "byd-migrated"
    with publication_context(store.view("byd-legacy")):
        with pytest.raises(FileNotFoundError):
            read_byd_intraday_training_evidence(cache)
    assert not old.resolve(root_file).exists()
    assert root_file.read_bytes() == root_bytes


def test_publication_without_training_mapping_keeps_legacy_reader_compatibility(tmp_path):
    cache = tmp_path / "data/cache"
    cache.mkdir(parents=True)
    legacy_path = _write(cache, _bars(), "baostock_002594_5min_history_qfq.parquet")
    outside_evidence = read_byd_intraday_training_evidence(cache)
    store = PublicationStore(tmp_path, ("data/workspace_snapshots",))
    with store.begin("unmapped-training") as staged:
        assert read_byd_intraday_training_evidence(cache) == outside_evidence
        store.commit(staged, {
            "status": "success", "freshness_audit": {"status": "success", "failures": []},
        })
    with publication_context(store.view()):
        assert read_byd_intraday_training_evidence(cache) == outside_evidence
        assert read_byd_training_frame(cache).attrs["intraday_training_source"] == str(legacy_path)
