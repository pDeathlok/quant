"""Local, read-only expansion of implementation contracts and source identities."""

from __future__ import annotations

import ast
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping

from quant.infrastructure.publication import publication_path
from quant.routine.operation_contracts import InputSnapshot, NodeChanges


def _complete_date_blocks(chunk: Mapping) -> dict[str, Mapping] | None:
    """Only trust monthly row hashes when metadata covers the complete chunk."""
    blocks = chunk.get("date_blocks")
    if (type(chunk.get("date_blocks_schema")) is not int or chunk["date_blocks_schema"] != 1
            or not isinstance(blocks, list) or not blocks):
        return None
    indexed: dict[str, Mapping] = {}
    try:
        for block in blocks:
            period = block["partition"]
            symbols = block.get("symbols")
            if symbols is None and len(chunk["symbols"]) == 1:
                symbols = chunk["symbols"]
            start, end = block["min_date"], block["max_date"]
            datetime.strptime(start, "%Y%m%d")
            datetime.strptime(end, "%Y%m%d")
            if (len(start) != 8 or len(end) != 8 or start > end
                    or start[:6] != period or end[:6] != period or period in indexed
                    or type(block["rows"]) is not int or block["rows"] <= 0
                    or not isinstance(block["sha256"], str) or len(block["sha256"]) != 64
                    or any(char not in "0123456789abcdef" for char in block["sha256"])
                    or not isinstance(symbols, list) or not symbols
                    or any(not isinstance(symbol, str) or not symbol for symbol in symbols)):
                return None
            indexed[period] = {**block, "symbols": symbols}
        if (type(chunk["rows"]) is not int
                or sum(block["rows"] for block in blocks) != chunk["rows"]
                or min(block["min_date"] for block in blocks) != chunk["min_date"]
                or max(block["max_date"] for block in blocks) != chunk["max_date"]
                or {symbol for block in indexed.values() for symbol in block["symbols"]} != set(chunk["symbols"])):
            return None
    except (KeyError, TypeError, ValueError):
        return None
    return indexed


def dataset_manifest_changes(previous: Mapping | None, current: Mapping) -> NodeChanges:
    """Compare monthly row hashes inside stable files, without reopening parquet."""
    if previous is None or any(previous.get(key) != current.get(key) for key in (
        "columns", "start_date", "symbols", "source_kind",
    )):
        return NodeChanges(full_rebuild=True)
    old = {chunk["path"]: chunk for chunk in previous.get("files", ())}
    new = {chunk["path"]: chunk for chunk in current.get("files", ())}
    changed: list[Mapping] = []
    for path in old.keys() | new.keys():
        before, after = old.get(path), new.get(path)
        if before == after:
            continue
        before_blocks = _complete_date_blocks(before) if before is not None else {}
        after_blocks = _complete_date_blocks(after) if after is not None else {}
        if before_blocks is not None and after_blocks is not None:
            changed.extend(
                block for period in before_blocks.keys() | after_blocks.keys()
                if before_blocks.get(period) != after_blocks.get(period)
                for block in (before_blocks.get(period), after_blocks.get(period)) if block is not None
            )
        else:
            # Older manifests cannot prove which history prefix is unchanged.
            changed.extend(chunk for chunk in (before, after) if chunk is not None)
    if any(not chunk.get("min_date") or not chunk.get("symbols") for chunk in changed):
        return NodeChanges(full_rebuild=True)
    return NodeChanges(
        partitions=tuple(sorted({chunk["min_date"] for chunk in changed})),
        keys=tuple(sorted({symbol for chunk in changed for symbol in chunk["symbols"]})),
    )


def implementation_paths(root: Path, declarations: Iterable[str]) -> tuple[str, ...]:
    """Follow static local imports, including thin script wrappers; never import code."""
    root = root.resolve()
    pending = [root / path for path in declarations]
    visited: set[Path] = set()

    def add(base: Path) -> None:
        for candidate in (base.with_suffix(".py"), base / "__init__.py"):
            if candidate.is_file() and candidate.resolve().is_relative_to(root):
                pending.append(candidate)

    while pending:
        path = pending.pop().resolve()
        path.relative_to(root)
        if path in visited:
            continue
        visited.add(path)
        if path.suffix != ".py" or not path.is_file():
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
                bases = [root / "src" if name.startswith("quant.") or name == "quant" else path.parent for name in names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
                bases = [path.parents[node.level - 1] if node.level else (
                    root / "src" if names[0].startswith("quant") else path.parent
                )]
            else:
                continue
            for name, base in zip(names, bases):
                target = base.joinpath(*name.split(".")) if name else base
                add(target)
                if isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        if alias.name != "*":
                            add(target / alias.name)
    return tuple(sorted(path.relative_to(root).as_posix() for path in visited))


def snapshot_from_revision(
    *, source_namespace: str, dataset: str, revision: int,
    changes: NodeChanges = NodeChanges(),
) -> InputSnapshot:
    """Use only monotonic canonical revisions that cover corrections and deletions."""
    from quant.routine.checkpoint_store import canonical_fingerprint

    if not source_namespace.strip() or not dataset.strip() or type(revision) is not int or revision < 0:
        raise ValueError("canonical snapshot requires source namespace, dataset and valid revision")
    return InputSnapshot(
        fingerprint=canonical_fingerprint({"source": source_namespace, "dataset": dataset, "revision": revision}),
        revision=revision, changes=changes,
    )


def snapshot_from_artifacts(
    project_root: Path, paths: Iterable[str], *,
    changes: NodeChanges = NodeChanges(),
) -> InputSnapshot:
    """Hash all declared bytes/history, not a date, mtime or latest-month shortcut."""
    from quant.routine.checkpoint_store import canonical_fingerprint, fingerprint_artifact

    hashes = {}
    for relative in paths:
        canonical = project_root.resolve() / relative
        canonical.resolve().relative_to(project_root.resolve())
        hashes[relative] = fingerprint_artifact(publication_path(canonical))
    if not hashes:
        raise ValueError("artifact snapshot requires at least one materialized input")
    return InputSnapshot(fingerprint=canonical_fingerprint(hashes), changes=changes)


def core_input_snapshots(
    project_root: Path, node_ids: Iterable[str], *,
    canonical_market: InputSnapshot | None = None,
    changes: Mapping[str, NodeChanges] | None = None,
) -> dict[str, InputSnapshot]:
    """Snapshot the concrete core-five reads; canonical SQL evidence is caller-owned.

    Keep source revisions/files pinned until the operation ends. Runtime
    collect_node_states fingerprints are freshness evidence, not sufficient for
    historical consumers: its market output hash covers only the latest month.
    """
    from quant.routine.default_operations import DEFAULT_DAILY_OPERATION_REGISTRY

    paths_by_node = {
        "data.daily_basic": ("data/raw/daily_basic",),
        "data.top_list": ("data/raw/top_list",),
        "source.market_daily_parquet": tuple(
            path for path in ("data/raw/daily", "data/raw/daily_partitioned")
            if (project_root / path).is_dir()
        ),
    }
    for definition in DEFAULT_DAILY_OPERATION_REGISTRY.definitions.values():
        if definition.input_ids is not None:
            for node in definition.produces:
                paths_by_node[node] = definition.cache.output_paths
    snapshots = {}
    for node in node_ids:
        dataset = {"data.market_daily": "daily", "source.market_daily_parquet": "daily",
                   "data.daily_basic": "daily_basic", "data.top_list": "top_list"}.get(node)
        if dataset and canonical_market is not None and "sealed_datasets" in canonical_market.payload:
            if dataset not in canonical_market.payload["sealed_datasets"]:
                raise ValueError(f"sealed market snapshot does not cover {dataset}")
            # Independent dataset identities avoid invalidating daily-only
            # signals when a daily-basic correction changes the shared export.
            fingerprint = canonical_market.payload.get("dataset_fingerprints", {}).get(dataset)
            snapshots[node] = InputSnapshot(
                fingerprint=fingerprint, payload=canonical_market.payload,
                changes=NodeChanges(**canonical_market.payload["dataset_changes"][dataset])
                if dataset in canonical_market.payload.get("dataset_changes", {}) else canonical_market.changes,
            ) if fingerprint else canonical_market
        elif node == "data.market_daily":
            if canonical_market is None:
                raise ValueError("data.market_daily requires a pinned canonical snapshot, not runtime freshness metadata")
            snapshots[node] = canonical_market
        elif node in paths_by_node:
            snapshots[node] = snapshot_from_artifacts(
                project_root, paths_by_node[node], changes=(changes or {}).get(node, NodeChanges()),
            )
        else:
            raise ValueError(f"no audited core input materialization for {node}")
    return snapshots


__all__ = ["core_input_snapshots", "implementation_paths", "snapshot_from_artifacts", "snapshot_from_revision"]
