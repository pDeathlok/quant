"""Map contracts; timing demo: PYTHONPATH=src python tests/test_resource_scheduler.py."""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from statistics import median
import threading
import time

import pytest

from quant.infrastructure.publication import ContextThreadPoolExecutor
from quant.routine import resource_scheduler
from quant.routine.operation_contracts import ResourceClaim
from quant.routine.resource_scheduler import ResourceBudget, ResourceScheduler


PARENT = ResourceClaim(cpu_slots=2, memory_mb=256, requested_workers=2, max_workers=2)
CHILD = ResourceClaim(memory_mb=64)
TIMEOUT = 5


def test_map_refills_after_later_completion_without_waiting_for_slow_first():
    scheduler = ResourceScheduler(ResourceBudget(cpu_slots=2, memory_mb=256))
    first_started = threading.Event()
    release_first = threading.Event()
    third_started = threading.Event()

    def work(item, _grant):
        if item == 0:
            first_started.set()
            assert release_first.wait(TIMEOUT)
        elif item == 1:
            assert first_started.wait(TIMEOUT)
        elif item == 2:
            third_started.set()
        return item * 10

    with scheduler.reserve(PARENT) as parent, ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(parent.map, work, range(3), claim=CHILD, max_pending=2)
        try:
            assert first_started.wait(TIMEOUT)
            assert third_started.wait(TIMEOUT), "third job was blocked behind the slow first job"
            assert not future.done()
        finally:
            release_first.set()
        assert future.result(timeout=TIMEOUT) == [0, 10, 20]


@pytest.mark.parametrize("max_pending", [1, 2, 4])
def test_map_bounds_inflight_input_and_resources_while_preserving_results(max_pending):
    scheduler = ResourceScheduler(ResourceBudget(cpu_slots=2, memory_mb=256))
    release = threading.Event()
    filled = threading.Event()
    lock = threading.Lock()
    consumed = finished = active = peak_active = peak_inflight = 0

    def items():
        nonlocal consumed, peak_inflight
        for item in range(23):
            with lock:
                assert consumed - finished < max_pending
                consumed += 1
                peak_inflight = max(peak_inflight, consumed - finished)
                if consumed == max_pending:
                    filled.set()
            yield item

    def work(item, _grant):
        nonlocal finished, active, peak_active
        with lock:
            active += 1
            peak_active = max(peak_active, active)
        try:
            assert release.wait(TIMEOUT)
            return (item, item * item)
        finally:
            with lock:
                active -= 1
                finished += 1

    with scheduler.reserve(PARENT) as parent, ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(parent.map, work, items(), claim=CHILD, max_pending=max_pending)
        try:
            assert filled.wait(TIMEOUT)
            with lock:
                assert consumed == max_pending
                assert finished == 0
        finally:
            release.set()
        assert future.result(timeout=TIMEOUT) == [(item, item * item) for item in range(23)]
        with parent.child_scheduler() as children:
            usage = children.usage()
            assert usage["max_cpu_slots"] <= 2
            assert usage["max_memory_mb"] <= 128
    assert peak_inflight == max_pending
    assert peak_active <= min(2, max_pending)
    assert consumed == finished == 23
    assert active == 0


@pytest.mark.parametrize("values", [[], [None], [None, 0, False, "", 7]])
def test_map_preserves_empty_and_falsy_results(values):
    scheduler = ResourceScheduler(ResourceBudget(cpu_slots=2))
    with scheduler.reserve(PARENT) as parent:
        assert parent.map(lambda item, _grant: item, iter(values), claim=CHILD) == values


@pytest.mark.parametrize("max_pending", [0, -1, 5])
def test_map_does_not_relax_pending_limit(max_pending):
    scheduler = ResourceScheduler(ResourceBudget(cpu_slots=2))
    with scheduler.reserve(PARENT) as parent:
        with pytest.raises(ValueError, match="max_pending"):
            parent.map(lambda item, _grant: item, (), claim=CHILD, max_pending=max_pending)


def test_map_failure_stops_queued_work_and_joins_running_workers(monkeypatch):
    submitted = []
    submitted_all = threading.Event()
    shutting_down = threading.Event()
    first_started = threading.Event()
    second_started = threading.Event()
    release_first = threading.Event()
    fail_second = threading.Event()
    first_released = threading.Event()
    calls = []
    error = RuntimeError("second job failed")

    class ObservedExecutor(ContextThreadPoolExecutor):
        def submit(self, *args, **kwargs):
            future = super().submit(*args, **kwargs)
            submitted.append(future)
            if len(submitted) == 4:
                submitted_all.set()
            return future

        def shutdown(self, *args, **kwargs):
            shutting_down.set()
            return super().shutdown(*args, **kwargs)

    monkeypatch.setattr(resource_scheduler, "ContextThreadPoolExecutor", ObservedExecutor)

    def work(item, _grant):
        calls.append(item)
        if item == 0:
            first_started.set()
            try:
                assert release_first.wait(TIMEOUT)
            finally:
                first_released.set()
        elif item == 1:
            second_started.set()
            assert fail_second.wait(TIMEOUT)
            raise error
        return item

    scheduler = ResourceScheduler(ResourceBudget(cpu_slots=2, memory_mb=256))
    with scheduler.reserve(PARENT) as parent, ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(parent.map, work, range(20), claim=CHILD, max_pending=4)
        try:
            assert first_started.wait(TIMEOUT)
            assert second_started.wait(TIMEOUT)
            assert submitted_all.wait(TIMEOUT)
            fail_second.set()
            assert shutting_down.wait(TIMEOUT)
            assert not future.done()
            assert not first_released.is_set()
            assert set(calls) == {0, 1}
            assert len(submitted) == 4
        finally:
            fail_second.set()
            release_first.set()
        with pytest.raises(RuntimeError) as caught:
            future.result(timeout=TIMEOUT)
        assert caught.value is error
        assert first_released.is_set()
        assert all(future.done() for future in submitted)
        with parent.reserve(PARENT):
            pass
    with scheduler.reserve(PARENT):
        pass


def test_map_iterator_failure_cancels_queue_and_joins_running_worker(monkeypatch):
    running = threading.Event()
    release = threading.Event()
    shutting_down = threading.Event()
    error = ValueError("input iteration failed")

    class ObservedExecutor(ContextThreadPoolExecutor):
        def shutdown(self, *args, **kwargs):
            shutting_down.set()
            return super().shutdown(*args, **kwargs)

    monkeypatch.setattr(resource_scheduler, "ContextThreadPoolExecutor", ObservedExecutor)

    def items():
        yield 0
        assert running.wait(TIMEOUT)
        raise error

    def work(item, _grant):
        running.set()
        assert release.wait(TIMEOUT)
        return item

    scheduler = ResourceScheduler(ResourceBudget(cpu_slots=2, memory_mb=256))
    with scheduler.reserve(PARENT) as parent, ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(parent.map, work, items(), claim=CHILD, max_pending=2)
        try:
            assert shutting_down.wait(TIMEOUT)
            assert not future.done()
        finally:
            release.set()
        with pytest.raises(ValueError) as caught:
            future.result(timeout=TIMEOUT)
        assert caught.value is error
        with parent.reserve(PARENT):
            pass


def timing_comparison(repeats=5):
    """Synthetic skew only; no market data, production refresh, or speed assertion."""
    durations = (0.16, 0.01, 0.12, 0.01)

    def work(item, _grant):
        time.sleep(durations[item])
        return item

    def fifo(parent):
        def run(item):
            with parent.reserve(CHILD) as child:
                return work(item, child)

        result = []
        pending = deque()
        with ContextThreadPoolExecutor(max_workers=2) as pool:
            for item in range(len(durations)):
                if len(pending) == 2:
                    result.append(pending.popleft().result())
                pending.append(pool.submit(run, item))
            result.extend(future.result() for future in pending)
        return result

    samples = {"fifo": [], "completion_order": []}
    for repeat in range(repeats):
        for mode in (tuple(samples) if repeat % 2 == 0 else tuple(reversed(samples))):
            scheduler = ResourceScheduler(ResourceBudget(cpu_slots=2, memory_mb=256))
            with scheduler.reserve(PARENT) as parent:
                start = time.perf_counter()
                result = (fifo(parent) if mode == "fifo" else parent.map(
                    work, range(len(durations)), claim=CHILD, max_pending=2,
                ))
                samples[mode].append(time.perf_counter() - start)
                assert result == list(range(len(durations)))
    medians = {mode: median(values) for mode, values in samples.items()}
    return {
        "workers": 2, "max_pending": 2, "durations_seconds": durations,
        "samples_seconds": samples, "median_seconds": medians,
        "speedup": medians["fifo"] / medians["completion_order"],
    }


if __name__ == "__main__":
    import json

    print(json.dumps(timing_comparison(), indent=2))
