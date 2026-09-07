"""Cooperative resource leases shared by DAGs, external jobs and child workers.

Claims reserve an envelope, not measured RSS/CPU. Owners must route all work
through leases and pass the granted capacity to subprocesses/native libraries.
While children use a grant, their parent is an idle coordinator. Uninstrumented
thread pools, BLAS threads and remote processes cannot be bounded by this API.
"""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import os
import threading
from typing import Callable, Iterable, Iterator, Mapping, TypeVar

from quant.infrastructure.publication import ContextThreadPoolExecutor
from quant.routine.operation_contracts import ResourceClaim

T = TypeVar("T")
R = TypeVar("R")

_SCHEDULER: ContextVar[ResourceScheduler | None] = ContextVar("routine_scheduler", default=None)
_GRANT: ContextVar[ResourceGrant | None] = ContextVar("routine_resource_grant", default=None)


def current_resource_scheduler() -> ResourceScheduler | None:
    return _SCHEDULER.get()


def current_resource_grant() -> ResourceGrant | None:
    return _GRANT.get()


def run_with_resources(claim: ResourceClaim, function: Callable[..., R], *args, **kwargs) -> R:
    """Pool entry hook: nested calls subdivide the currently held envelope."""
    parent = current_resource_grant()
    scheduler = current_resource_scheduler()
    if parent is None and scheduler is None:
        return function(*args, **kwargs)
    owner = parent if parent is not None else scheduler
    with owner.reserve(claim):
        return function(*args, **kwargs)


@dataclass(frozen=True)
class ResourceBudget:
    cpu_slots: int
    io_slots: int = 4
    memory_mb: int = 4096
    rate_limit_concurrency: Mapping[str, int] | None = None
    db_connections: int = 8

    def __post_init__(self) -> None:
        if self.cpu_slots < 1 or min(self.io_slots, self.memory_mb, self.db_connections) < 0:
            raise ValueError("invalid resource budget")
        if any(value < 1 for value in (self.rate_limit_concurrency or {}).values()):
            raise ValueError("API concurrency limits must be positive")

    @classmethod
    def from_environment(cls) -> ResourceBudget:
        return cls(
            cpu_slots=max(1, int(os.getenv("ROUTINE_TOTAL_WORKERS", str(os.cpu_count() or 1)))),
            io_slots=max(1, int(os.getenv("ROUTINE_IO_SLOTS", "4"))),
            memory_mb=max(256, int(os.getenv("ROUTINE_MEMORY_BUDGET_MB", "4096"))),
            rate_limit_concurrency={"tushare": 1, "akshare": 2},
            db_connections=max(0, int(os.getenv("ROUTINE_DB_CONNECTIONS", "8"))),
        )


class ResourceScheduler:
    """Inject one instance into every stage and reserve external work on it."""

    def __init__(self, budget: ResourceBudget, *, allowed_rate_limit_groups: set[str] | None = None) -> None:
        self.budget = budget
        self._allowed_groups = allowed_rate_limit_groups
        self._used = {name: 0 for name in ("cpu_slots", "io_slots", "memory_mb", "db_connections")}
        self._peak = dict(self._used)
        self._groups: dict[str, int] = {}
        self._group_peaks: dict[str, int] = {}
        self._condition = threading.Condition()

    @contextmanager
    def activate(self) -> Iterator[ResourceScheduler]:
        """Carry the scheduler into context-aware web pools and later DAG stages."""
        token = _SCHEDULER.set(self)
        try:
            yield self
        finally:
            _SCHEDULER.reset(token)

    def validate(self, claim: ResourceClaim, operation_id: str = "external") -> None:
        if (claim.rate_limit_group and self._allowed_groups is not None
                and claim.rate_limit_group not in self._allowed_groups):
            raise ValueError("child API group was not reserved by parent")
        for name in self._used:
            if getattr(claim, name) > getattr(self.budget, name):
                label = {"cpu_slots": "CPU slots", "io_slots": "IO slots", "memory_mb": "MB"}.get(name, name)
                raise ValueError(
                    f"operation {operation_id} requests {getattr(claim, name)} {label} "
                    f"but budget is {getattr(self.budget, name)}"
                )
        if claim.rate_limit_group and claim.api_slots > self._group_limit(claim.rate_limit_group):
            raise ValueError(f"operation {operation_id} exceeds API concurrency budget")

    def _group_limit(self, group: str) -> int:
        return (self.budget.rate_limit_concurrency or {}).get(group, 1)

    def try_acquire(self, claim: ResourceClaim) -> ResourceGrant | None:
        self.validate(claim)
        with self._condition:
            if any(self._used[name] + getattr(claim, name) > getattr(self.budget, name) for name in self._used):
                return None
            group = claim.rate_limit_group
            if group and self._groups.get(group, 0) + claim.api_slots > self._group_limit(group):
                return None
            for name in self._used:
                self._used[name] += getattr(claim, name)
                self._peak[name] = max(self._peak[name], self._used[name])
            if group:
                self._groups[group] = self._groups.get(group, 0) + claim.api_slots
                self._group_peaks[group] = max(self._group_peaks.get(group, 0), self._groups[group])
            return ResourceGrant(self, claim)

    @contextmanager
    def reserve(self, claim: ResourceClaim) -> Iterator[ResourceGrant]:
        """Blocking external-work hook; reserve before starting a worker pool."""
        with self._condition:
            grant = self.try_acquire(claim)
            while grant is None:
                self._condition.wait()
                grant = self.try_acquire(claim)
        try:
            with grant.activate():
                yield grant
        finally:
            grant.close()

    def wait_for_release(self) -> None:
        # A timeout avoids a lost wakeup between a DAG's scan and this wait.
        with self._condition:
            self._condition.wait(timeout=0.05)

    def _release(self, claim: ResourceClaim) -> None:
        with self._condition:
            for name in self._used:
                self._used[name] -= getattr(claim, name)
            if claim.rate_limit_group:
                self._groups[claim.rate_limit_group] -= claim.api_slots
            self._condition.notify_all()

    def usage(self) -> dict:
        with self._condition:
            return {
                **{f"max_{name}": value for name, value in self._peak.items()},
                **{f"budget_{name}": getattr(self.budget, name) for name in self._used},
                "max_api_concurrency": dict(self._group_peaks),
                "accounting": "reserved_envelopes_not_measured_process_usage",
            }


class ResourceGrant:
    def __init__(self, scheduler: ResourceScheduler, claim: ResourceClaim) -> None:
        self.claim = claim
        self.granted_workers = min(claim.cpu_slots, claim.requested_workers, claim.max_workers)
        self._scheduler = scheduler
        self._children: ResourceScheduler | None = None
        self._active_children = 0
        self._closed = False
        self._condition = threading.Condition()

    @property
    def scheduler(self) -> ResourceScheduler:
        return self._scheduler

    @contextmanager
    def activate(self) -> Iterator[ResourceGrant]:
        token = _GRANT.set(self)
        try:
            yield self
        finally:
            _GRANT.reset(token)

    @contextmanager
    def child_scheduler(self) -> Iterator[ResourceScheduler]:
        """Scheduler for a nested DAG; parent capacity stays globally reserved."""
        with self._condition:
            if self._closed:
                raise RuntimeError("resource grant is closed")
            if self._children is None:
                self._children = ResourceScheduler(ResourceBudget(
                    cpu_slots=self.granted_workers,
                    io_slots=self.claim.io_slots,
                    memory_mb=self.claim.memory_mb,
                    db_connections=self.claim.db_connections,
                    rate_limit_concurrency=(
                        {self.claim.rate_limit_group: self.claim.api_slots}
                        if self.claim.rate_limit_group else {}
                    ),
                ), allowed_rate_limit_groups=(
                    {self.claim.rate_limit_group} if self.claim.rate_limit_group else set()
                ))
            self._active_children += 1
        try:
            yield self._children
        finally:
            with self._condition:
                self._active_children -= 1
                self._condition.notify_all()

    @contextmanager
    def reserve(self, claim: ResourceClaim) -> Iterator[ResourceGrant]:
        """Subdivide this envelope, never reacquire a held global CPU slot."""
        if claim.rate_limit_group and claim.rate_limit_group != self.claim.rate_limit_group:
            raise ValueError("child API group was not reserved by parent")
        with self.child_scheduler() as scheduler, scheduler.reserve(claim) as child:
            yield child

    def map(
        self,
        function: Callable[[T, ResourceGrant], R],
        items: Iterable[T],
        *,
        claim: ResourceClaim,
        max_pending: int | None = None,
    ) -> list[R]:
        """Ordered child work with bounded input consumption and outstanding jobs."""
        limit = self.granted_workers if max_pending is None else max_pending
        if not 1 <= limit <= 2 * self.granted_workers:
            raise ValueError("max_pending must be within [1, 2 * granted_workers]")

        def run(item: T) -> R:
            with self.reserve(claim) as child:
                return function(item, child)

        results: list[R] = []
        pending = deque()
        with ContextThreadPoolExecutor(max_workers=self.granted_workers) as executor:
            iterator = iter(items)
            while True:
                if len(pending) >= limit:
                    results.append(pending.popleft().result())
                try:
                    item = next(iterator)
                except StopIteration:
                    break
                pending.append(executor.submit(run, item))
            results.extend(future.result() for future in pending)
        return results

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            while self._active_children:
                self._condition.wait()
            self._scheduler._release(self.claim)


__all__ = [
    "ResourceBudget", "ResourceGrant", "ResourceScheduler",
    "current_resource_grant", "current_resource_scheduler",
    "run_with_resources",
]
