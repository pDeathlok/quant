"""Execution contracts shared by the declarative daily refresh DAG."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class CacheMode(str, Enum):
    NONE = "none"
    EXACT_DATE = "exact_date"
    APPEND_STATE = "append_state"
    PARTITION_REPLACE = "partition_replace"


class ExecutionMode(str, Enum):
    INLINE = "inline"
    THREAD = "thread"
    SUBPROCESS = "subprocess"


@dataclass(frozen=True)
class ResourceClaim:
    cpu_slots: int = 1
    io_slots: int = 0
    memory_mb: int = 256
    rate_limit_group: str | None = None
    requested_workers: int = 1
    max_workers: int = 1

    def __post_init__(self) -> None:
        if self.cpu_slots < 1:
            raise ValueError("operation cpu_slots must be >= 1")
        if self.io_slots < 0 or self.memory_mb < 0:
            raise ValueError("operation io_slots and memory_mb must be >= 0")
        if self.requested_workers < 1 or self.max_workers < 1:
            raise ValueError("operation worker counts must be >= 1")


@dataclass(frozen=True)
class CachePolicy:
    mode: CacheMode
    contract_version: str
    output_paths: tuple[str, ...] = ()
    state_path: str | None = None
    contract_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.contract_version.strip():
            raise ValueError("cache contract_version must not be empty")
        if self.mode != CacheMode.NONE and not self.output_paths:
            raise ValueError("cacheable operations must declare output_paths")


@dataclass(frozen=True)
class RetryPolicy:
    attempts: int = 1
    interval_seconds: float = 0.0
    retry_until: str | None = None
    retryable_categories: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.attempts < 1:
            raise ValueError("retry attempts must be >= 1")
        if self.interval_seconds < 0:
            raise ValueError("retry interval_seconds must be >= 0")


@dataclass(frozen=True)
class OperationDefinition:
    operation_id: str
    entrypoint: str
    produces: tuple[str, ...]
    execution_mode: ExecutionMode
    resources: ResourceClaim
    cache: CachePolicy
    retry: RetryPolicy = RetryPolicy()
    parameters: Mapping[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.operation_id.strip():
            raise ValueError("operation_id must not be empty")
        if not self.entrypoint.strip() or ":" not in self.entrypoint:
            raise ValueError(
                f"operation {self.operation_id} entrypoint must use module:function"
            )
        if not self.produces or len(set(self.produces)) != len(self.produces):
            raise ValueError(
                f"operation {self.operation_id} must declare unique produced nodes"
            )


@dataclass(frozen=True)
class OperationContext:
    target_trade_date: str
    scope: str
    granted_workers: int
    upstream_results: Mapping[str, Mapping[str, Any]]
    upstream_revisions: Mapping[str, int] = field(default_factory=dict)
    upstream_fingerprints: Mapping[str, str] = field(default_factory=dict)
    dirty_partitions: tuple[str, ...] = ()
    dirty_keys: tuple[str, ...] = ()
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OperationResult:
    status: str
    node_results: Mapping[str, Mapping[str, Any]]
    changed_partitions: tuple[str, ...] = ()
    changed_keys: tuple[str, ...] = ()
    output_fingerprints: Mapping[str, str] = field(default_factory=dict)
    dataset_revisions: Mapping[str, int] = field(default_factory=dict)
    metrics: Mapping[str, Any] = field(default_factory=dict)
    error_category: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"success", "failed", "cancelled"}:
            raise ValueError(f"unsupported operation status: {self.status}")
        if self.status == "success" and self.error:
            raise ValueError("successful operation cannot carry an error")


__all__ = [
    "CacheMode",
    "CachePolicy",
    "ExecutionMode",
    "OperationContext",
    "OperationDefinition",
    "OperationResult",
    "ResourceClaim",
    "RetryPolicy",
]
