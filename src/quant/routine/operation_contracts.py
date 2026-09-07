"""Execution contracts shared by the declarative daily refresh DAG."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping

if TYPE_CHECKING:
    from quant.routine.resource_scheduler import ResourceGrant


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
    db_connections: int = 0
    api_slots: int = 1

    def __post_init__(self) -> None:
        if self.cpu_slots < 1:
            raise ValueError("operation cpu_slots must be >= 1")
        if self.io_slots < 0 or self.memory_mb < 0:
            raise ValueError("operation io_slots and memory_mb must be >= 0")
        if self.requested_workers < 1 or self.max_workers < 1:
            raise ValueError("operation worker counts must be >= 1")
        if self.db_connections < 0 or self.api_slots < 1:
            raise ValueError("DB connections must be >= 0 and API slots >= 1")


@dataclass(frozen=True)
class CachePolicy:
    mode: CacheMode
    contract_version: str
    output_paths: tuple[str, ...] = ()
    state_path: str | None = None
    contract_paths: tuple[str, ...] = ()
    optional_contract_paths: tuple[str, ...] = ()
    environment_keys: tuple[str, ...] = ()
    track_python_imports: bool = False

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
    # None is legacy/unaudited; () explicitly declares a pure parameter-only op.
    input_ids: tuple[str, ...] | None = None
    production_ready: bool = True

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
        if self.input_ids is not None and (
            len(set(self.input_ids)) != len(self.input_ids)
            or any(not item.strip() for item in self.input_ids)
        ):
            raise ValueError("input_ids must be unique nonempty identities")


@dataclass(frozen=True)
class NodeChanges:
    partitions: tuple[str, ...] = ()
    keys: tuple[str, ...] = ()
    full_rebuild: bool = False

    def __post_init__(self) -> None:
        for name in ("partitions", "keys"):
            values = getattr(self, name)
            if isinstance(values, str) or any(not isinstance(value, str) or not value for value in values):
                raise ValueError(f"change {name} must be nonempty strings")
            object.__setattr__(self, name, tuple(sorted(set(values))))
        if type(self.full_rebuild) is not bool:
            raise ValueError("full_rebuild must be boolean")


@dataclass(frozen=True)
class InputSnapshot:
    """Pinned canonical content identity, never a date-only freshness marker.

    The caller must keep this source revision immutable during execution. A
    fingerprint includes source namespace and partition/key content revisions.
    """

    fingerprint: str
    revision: int | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)
    changes: NodeChanges = NodeChanges()

    def __post_init__(self) -> None:
        if not isinstance(self.fingerprint, str) or not self.fingerprint.strip():
            raise ValueError("input snapshot requires a nonempty fingerprint")
        if self.revision is not None and (
            type(self.revision) is not int or self.revision < 0
        ):
            raise ValueError("input snapshot revision must be a nonnegative integer")
        if not isinstance(self.changes, NodeChanges) or not isinstance(self.payload, Mapping):
            raise ValueError("snapshot requires a NodeChanges journal and mapping payload")
        if self.payload.get("status") in {"failed", "cancelled", "shadow", "shadow_only"}:
            raise ValueError("snapshot cannot represent unsuccessful output")


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
    required_input_ids: tuple[str, ...] = ()
    input_changes: Mapping[str, NodeChanges] = field(default_factory=dict)
    full_rebuild: bool = False
    resource_grant: ResourceGrant | None = field(default=None, compare=False, repr=False)
    identity_required: bool = False
    project_root: Path | None = None
    output_paths: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


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
    node_changes: Mapping[str, NodeChanges] = field(default_factory=dict)
    input_fingerprints: Mapping[str, str] = field(default_factory=dict)
    input_revisions: Mapping[str, int] = field(default_factory=dict)
    contract_identity: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"success", "failed", "cancelled"}:
            raise ValueError(f"unsupported operation status: {self.status}")
        if self.status == "success" and self.error:
            raise ValueError("successful operation cannot carry an error")
        if not isinstance(self.node_results, Mapping) or any(
            not isinstance(payload, Mapping) for payload in self.node_results.values()
        ):
            raise ValueError("node_results must contain payload mappings")
        for revisions in (self.dataset_revisions, self.input_revisions):
            if not isinstance(revisions, Mapping) or any(
                type(revision) is not int or revision < 0 for revision in revisions.values()
            ):
                raise ValueError("dataset revisions must be nonnegative integers")


OperationHandler = Callable[[OperationContext], OperationResult]


@dataclass(frozen=True)
class OperationBinding:
    """Composition-root binding; routine never imports its web-owned handler.

    input_ids includes hidden sources/configuration as well as graph inputs.
    cache.contract_version must cover code/schema/model/parameter semantics;
    contract_paths can additionally pin their on-disk bytes. CacheMode.NONE
    preserves the handler's existing manual cache without central reuse.
    """

    handler: OperationHandler
    input_ids: tuple[str, ...]
    cache: CachePolicy
    resources: ResourceClaim | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not callable(self.handler):
            raise ValueError("operation binding handler must be callable")


__all__ = [
    "CacheMode",
    "CachePolicy",
    "ExecutionMode",
    "InputSnapshot",
    "NodeChanges",
    "OperationBinding",
    "OperationContext",
    "OperationDefinition",
    "OperationHandler",
    "OperationResult",
    "ResourceClaim",
    "RetryPolicy",
]
