"""Composition-root API for staged production operations and shadow planning.

The web coordinator supplies OperationBinding objects for its owned operations,
InputSnapshot objects for omitted upstream nodes and hidden sources, and one
ResourceScheduler shared with its early/late work. Passing checkpoint_store
opts into complete, fail-closed input/contract/output identity validation. No
store is created automatically, preserving existing manual cache semantics.
"""

from __future__ import annotations

from pathlib import Path
from contextlib import contextmanager
from contextvars import ContextVar
import os
import json
from time import monotonic
from tempfile import TemporaryDirectory
from dataclasses import asdict
from typing import Any, Callable, Iterable, Iterator, Mapping

from quant.application.daily_dependencies import (
    DEFAULT_DAILY_DEPENDENCY_REGISTRY,
)
from quant.routine.dag_executor import DailyDagExecutor, ResourceBudget
from quant.routine.checkpoint_store import CheckpointStore
from quant.routine.default_operations import DEFAULT_DAILY_OPERATION_REGISTRY, build_default_operation_registry
from quant.routine.operation_contracts import InputSnapshot, NodeChanges, OperationBinding, OperationResult
from quant.routine.operation_identity import core_input_snapshots, snapshot_from_artifacts, snapshot_from_revision
from quant.routine.paths import PROJECT_ROOT
from quant.routine.resource_scheduler import ResourceScheduler


_PINNED_MARKET: ContextVar[InputSnapshot | None] = ContextVar("production_pinned_market", default=None)


def production_dag_mode(scope: str) -> str:
    """Core is the default for the migrated stages; explicit rollback stays available."""
    mode = os.getenv("ROUTINE_DAG_EXECUTOR", "core" if scope in {"all", "short"} else "legacy").strip().lower()
    if mode not in {"legacy", "shadow", "core", "enabled"}:
        raise ValueError("ROUTINE_DAG_EXECUTOR must be legacy, shadow, core, or enabled")
    if mode == "enabled":
        raise ValueError("Full identity-audited DAG cutover is not implemented; use core")
    return mode


@contextmanager
def sealed_core_market_inputs(project_root: Path) -> Iterator[InputSnapshot]:
    """Acquire only after source writes, retain bytes until all core workers finish."""
    from quant.data import MarketDataStore, MarketDataStoreConfig
    from quant.data.market_snapshot import export_market_snapshot, pinned_market_snapshot
    from quant.routine.checkpoint_store import canonical_fingerprint
    from quant.routine.operation_identity import dataset_manifest_changes
    from quant.infrastructure.publication import publication_path

    store = MarketDataStore(MarketDataStoreConfig.from_env(root=project_root / "data/raw"))
    with TemporaryDirectory(prefix="quant-core-market-") as directory:
        started = monotonic()
        sealed = export_market_snapshot(
            store, Path(directory) / "sealed", datasets=("daily", "daily_basic", "top_list"),
            supplemental_sources={
                "daily_basic": project_root / "data/raw/daily_basic",
                "top_list": project_root / "data/raw/top_list",
            },
        )
        manifest = json.loads(sealed.manifest_path.read_text())
        if (not isinstance(manifest, dict) or manifest.get("fingerprint") != sealed.fingerprint
                or canonical_fingerprint({key: value for key, value in manifest.items() if key != "fingerprint"})
                != sealed.fingerprint):
            raise ValueError("Sealed export manifest does not match the acquired descriptor")
        try:
            previous = json.loads(publication_path(project_root / CORE_SOURCE_MANIFEST).read_text())
            if (not isinstance(previous, dict) or not isinstance(previous.get("datasets"), dict)
                    or previous.get("fingerprint") != canonical_fingerprint({
                        key: value for key, value in previous.items() if key != "fingerprint"
                    })):
                previous = {}
        except (OSError, ValueError):
            previous = {}
        dataset_fingerprints = {
            dataset: canonical_fingerprint({"dataset": dataset, "content": entry})
            for dataset, entry in manifest["datasets"].items()
        }
        snapshot = InputSnapshot(
            fingerprint=sealed.fingerprint,
            payload={
                "manifest_path": str(sealed.manifest_path),
                "sealed_datasets": tuple(dataset_fingerprints),
                "dataset_fingerprints": dataset_fingerprints,
                "dataset_changes": {
                    dataset: asdict(dataset_manifest_changes(previous.get("datasets", {}).get(dataset), entry))
                    for dataset, entry in manifest["datasets"].items()
                },
                "export_seconds": round(monotonic() - started, 3),
                "capture_metrics": sealed.metrics,
                "export_bytes": sum(path.stat().st_size for path in sealed.root.rglob("*") if path.is_file()),
            },
            changes=NodeChanges(full_rebuild=True),
        )
        with pinned_market_snapshot(sealed.manifest_path) as acquired:
            if acquired.fingerprint != sealed.fingerprint:
                raise ValueError("Pinned market fingerprint changed after export acquisition")
            with pinned_market_inputs(snapshot):
                yield snapshot


CORE_SOURCE_MANIFEST = "data/cache/routine_source_manifests/core.json"


def save_core_source_manifest(project_root: Path, snapshot: InputSnapshot) -> None:
    """Publish only metadata, after successful consumers; never retain row exports."""
    from quant.data.atomic_io import atomic_write_json
    from quant.data.market_snapshot import current_market_snapshot
    from quant.infrastructure.publication import current_publication, publication_path

    view = current_publication()
    if view is None or not view.writable:
        raise ValueError("Source manifest baseline requires an active publication generation")
    reader = current_market_snapshot()
    if reader is None or reader.manifest["fingerprint"] != snapshot.fingerprint:
        raise ValueError("Source baseline fingerprint does not match the active sealed pin")
    atomic_write_json(reader.manifest, publication_path(project_root / CORE_SOURCE_MANIFEST))


@contextmanager
def pinned_market_inputs(snapshot: InputSnapshot) -> Iterator[None]:
    """Owner holds the canonical source immutable for the entire refresh call.

    This scope carries evidence, it does not acquire a database read lock. SQL
    owners must provide a revision-pinned reader before enabling core mode.
    """
    if not isinstance(snapshot, InputSnapshot):
        raise TypeError("a pinned canonical InputSnapshot is required")
    token = _PINNED_MARKET.set(snapshot)
    try:
        yield
    finally:
        _PINNED_MARKET.reset(token)


def require_pinned_market() -> InputSnapshot:
    snapshot = _PINNED_MARKET.get()
    if snapshot is None:
        raise ValueError(
            "core DAG requires pinned_market_inputs(InputSnapshot) around the refresh; "
            "the source owner must hold a canonical revision-pinned reader through completion"
        )
    return snapshot


def production_stage_inputs(
    project_root: Path, node_ids: Iterable[str], *,
    changes: Mapping[str, NodeChanges] | None = None,
    completed_snapshots: Mapping[str, InputSnapshot] | None = None,
) -> dict[str, InputSnapshot]:
    """Resolve only external inputs; never disguise stage outputs as inputs."""
    selected = {
        DEFAULT_DAILY_OPERATION_REGISTRY.operation_by_node[node] for node in node_ids
    }
    definitions = [DEFAULT_DAILY_OPERATION_REGISTRY.definitions[key] for key in selected]
    if any(definition.input_ids is None or not definition.production_ready for definition in definitions):
        raise ValueError("production stage includes an unaudited operation")
    produced = {node for definition in definitions for node in definition.produces}
    inputs = {node for definition in definitions for node in definition.input_ids or ()}
    inputs.update(
        edge.upstream for node in produced
        for edge in DEFAULT_DAILY_DEPENDENCY_REGISTRY.nodes[node].inputs
    )
    external = inputs - produced
    carried = {node: snapshot for node, snapshot in (completed_snapshots or {}).items() if node in external}
    if any(not isinstance(snapshot, InputSnapshot) for snapshot in carried.values()):
        raise ValueError("completed snapshots require verified InputSnapshot values")
    return {
        **core_input_snapshots(
            project_root, sorted(external - carried.keys()),
            canonical_market=require_pinned_market(), changes=changes,
        ),
        **carried,
    }


def build_daily_dag_plan(
    target_trade_date: str,
    scope: str,
    *,
    project_root: Path = PROJECT_ROOT,
    budget: ResourceBudget | None = None,
    bindings: Mapping[str, OperationBinding] | None = None,
) -> dict[str, Any]:
    executor = DailyDagExecutor(
        DEFAULT_DAILY_DEPENDENCY_REGISTRY,
        build_default_operation_registry(bindings=bindings) if bindings else DEFAULT_DAILY_OPERATION_REGISTRY,
        project_root=project_root,
        budget=budget,
    )
    return executor.plan(
        target_trade_date=target_trade_date,
        scope=scope,
    )


def execute_daily_operations(
    target_trade_date: str,
    scope: str,
    node_ids: Iterable[str],
    *,
    project_root: Path = PROJECT_ROOT,
    budget: ResourceBudget | None = None,
    bindings: Mapping[str, OperationBinding] | None = None,
    input_snapshots: Mapping[str, InputSnapshot] | None = None,
    checkpoint_store: CheckpointStore | None = None,
    scheduler: ResourceScheduler | None = None,
    require_identity: bool = False,
    dirty_partitions: Iterable[str] = (),
    dirty_keys: Iterable[str] = (),
    progress_callback: (
        Callable[[str, str, Mapping[str, Any]], None] | None
    ) = None,
) -> dict[str, Any]:
    executor = DailyDagExecutor(
        DEFAULT_DAILY_DEPENDENCY_REGISTRY,
        build_default_operation_registry(bindings=bindings) if bindings else DEFAULT_DAILY_OPERATION_REGISTRY,
        project_root=project_root,
        budget=budget,
        checkpoint_store=checkpoint_store,
        scheduler=scheduler,
        require_identity=require_identity,
        progress_callback=progress_callback,
    )
    return executor.execute(
        target_trade_date=target_trade_date,
        scope=scope,
        node_ids=node_ids,
        dirty_partitions=dirty_partitions,
        dirty_keys=dirty_keys,
        input_snapshots=input_snapshots,
    )


def validate_daily_operation_closure(
    target_trade_date: str,
    scope: str,
    *,
    node_ids: Iterable[str] | None = None,
    bindings: Mapping[str, OperationBinding] | None = None,
    project_root: Path = PROJECT_ROOT,
    budget: ResourceBudget | None = None,
    scheduler: ResourceScheduler | None = None,
    require_identity: bool = True,
) -> dict[str, Any]:
    """Fail before refresh if any active operation is missing/disabled/unaudited.

    Omit node_ids to validate the entire scope closure, not only a stage. The
    manifest's external_input_ids must be pinned at execution time. This checks
    declarations and callability, not execution or output availability.
    """
    return DailyDagExecutor(
        DEFAULT_DAILY_DEPENDENCY_REGISTRY,
        build_default_operation_registry(bindings=bindings),
        project_root=project_root,
        budget=budget,
        scheduler=scheduler,
        require_identity=require_identity,
    ).plan(
        target_trade_date=target_trade_date, scope=scope, node_ids=node_ids,
        production=True,
    )


def snapshots_from_results(results: Iterable[OperationResult]) -> dict[str, InputSnapshot]:
    """Carry verified outputs across separate stages, including manual-cache hits.

    Old date-only results cannot be upgraded to an identity by this function.
    The owning adapter must supply real output fingerprints and change journals.
    """
    snapshots: dict[str, InputSnapshot] = {}
    for result in results:
        if result.status != "success":
            raise ValueError("cannot use failed/cancelled operations as external inputs")
        for node_id, payload in result.node_results.items():
            if node_id in snapshots:
                raise ValueError(f"duplicate input snapshot: {node_id}")
            if node_id not in result.node_changes:
                raise ValueError(f"missing change journal: {node_id}")
            snapshots[node_id] = InputSnapshot(
                fingerprint=result.output_fingerprints.get(node_id, ""),
                revision=result.dataset_revisions.get(node_id),
                payload=payload,
                changes=result.node_changes[node_id],
            )
    return snapshots


__all__ = [
    "build_daily_dag_plan", "execute_daily_operations", "snapshots_from_results",
    "validate_daily_operation_closure",
    "core_input_snapshots", "snapshot_from_artifacts", "snapshot_from_revision",
    "pinned_market_inputs", "require_pinned_market", "production_stage_inputs",
    "production_dag_mode", "sealed_core_market_inputs",
    "save_core_source_manifest", "CORE_SOURCE_MANIFEST",
]
