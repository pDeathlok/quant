"""Central cache identity and checkpoint validation for routine operations."""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from quant.data.atomic_io import atomic_write_json
from quant.infrastructure.publication import current_publication, publication_path
from quant.routine.operation_contracts import (
    CacheMode,
    NodeChanges,
    OperationContext,
    OperationDefinition,
    OperationResult,
)
from quant.routine.operation_identity import implementation_paths


CHECKPOINT_SCHEMA_VERSION = "daily-operation-checkpoint-v2"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_fingerprint(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _contains_generation_path(value: Any, root: str) -> bool:
    if isinstance(value, str):
        return root in value
    if isinstance(value, Mapping):
        return any(_contains_generation_path(item, root) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_generation_path(item, root) for item in value)
    return False


def _publication_output_root() -> str | None:
    view = current_publication()
    if view is None or not view.generation:
        return None
    return str(view.directory / view.generation / "tree")


def fingerprint_artifact(path: Path) -> str:
    """Hash materialized bytes, including directory membership, not path mtimes."""
    if path.is_symlink():
        raise ValueError(f"artifact may not be a symlink: {path}")
    if path.is_file():
        return _sha256_file(path)
    if path.is_dir():
        entries = {}
        for child in sorted(path.rglob("*")):
            if child.is_symlink():
                raise ValueError(f"artifact may not contain symlinks: {child}")
            entries[child.relative_to(path).as_posix()] = (
                _sha256_file(child) if child.is_file() else "directory"
            )
        return canonical_fingerprint(entries)
    raise ValueError(f"missing materialized artifact: {path}")


def validate_result_identity(definition: OperationDefinition, result: OperationResult) -> None:
    missing = [
        node for node in definition.produces
        if not isinstance(result.output_fingerprints.get(node), str)
        or not result.output_fingerprints[node].strip()
        or node not in result.node_changes
        or not isinstance(result.node_changes[node], NodeChanges)
    ]
    if missing:
        raise ValueError(f"operation {definition.operation_id} has incomplete output identity/changes: {missing}")
    if (
        set(result.node_results) != set(definition.produces)
        or set(result.output_fingerprints) != set(definition.produces)
        or set(result.node_changes) != set(definition.produces)
        or not set(result.dataset_revisions) <= set(definition.produces)
    ):
        raise ValueError(f"operation {definition.operation_id} has mismatched node results")
    if result.status != "success" or any(
        payload.get("status") in {"failed", "cancelled", "shadow", "shadow_only"}
        for payload in result.node_results.values()
    ):
        raise ValueError("cannot checkpoint unsuccessful node results")


class CheckpointStore:
    def __init__(self, project_root: Path, checkpoint_dir: Path) -> None:
        self.project_root = project_root.resolve()
        self.checkpoint_dir = checkpoint_dir.resolve()

    def _safe_project_path(self, relative: str) -> Path:
        path = (self.project_root / relative).resolve()
        try:
            path.relative_to(self.project_root)
        except ValueError as exc:
            raise ValueError(f"operation path escapes project root: {relative}") from exc
        return path

    def _checkpoint_path(self, operation_id: str) -> Path:
        safe = "".join(
            character if character.isalnum() or character in {"-", "_"} else "_"
            for character in operation_id
        )
        suffix = hashlib.sha256(operation_id.encode()).hexdigest()[:12]
        return self.checkpoint_dir / f"{safe}-{suffix}.json"

    def _output_hashes(self, definition: OperationDefinition) -> dict[str, str]:
        paths = tuple(dict.fromkeys((
            *definition.cache.output_paths,
            *((definition.cache.state_path,) if definition.cache.state_path else ()),
        )))
        return {
            relative: fingerprint_artifact(publication_path(self._safe_project_path(relative)))
            for relative in paths
        }

    @staticmethod
    def _validate_identity_payload(definition: OperationDefinition, payload: Mapping[str, Any]) -> None:
        if definition.input_ids is None:
            raise ValueError("checkpoint operation has incomplete input declaration")
        expected = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "operation_id": definition.operation_id,
            "entrypoint": definition.entrypoint,
            "produces": list(definition.produces),
            "cache_mode": definition.cache.mode.value,
            "contract_version": definition.cache.contract_version,
            "output_paths": list(definition.cache.output_paths),
            "state_path": definition.cache.state_path,
            "optional_contract_paths": list(definition.cache.optional_contract_paths),
            "environment_keys": list(definition.cache.environment_keys),
            "track_python_imports": definition.cache.track_python_imports,
        }
        if any(payload.get(key) != value for key, value in expected.items()):
            raise ValueError("checkpoint identity does not match operation contract")
        inputs = payload.get("input_ids")
        fingerprints = payload.get("upstream_fingerprints")
        if (
            not isinstance(inputs, list)
            or not isinstance(fingerprints, dict)
            or not set(definition.input_ids) <= set(inputs)
            or any(not isinstance(fingerprints.get(node), str) or not fingerprints[node].strip() for node in inputs)
            or not isinstance(payload.get("upstream_revisions"), dict)
            or not isinstance(payload.get("parameters"), dict)
            or not isinstance(payload.get("contract_hashes"), dict)
            or not set(definition.cache.contract_paths) <= set(payload["contract_hashes"])
            or not isinstance(payload.get("optional_contract_hashes"), dict)
            or set(payload["optional_contract_hashes"]) != set(definition.cache.optional_contract_paths)
            or not isinstance(payload.get("environment"), dict)
            or set(payload["environment"]) != set(definition.cache.environment_keys)
            or not payload.get("target_trade_date")
            or not payload.get("scope")
        ):
            raise ValueError("checkpoint has incomplete source/contract identity")

    def build_identity(
        self,
        definition: OperationDefinition,
        context: OperationContext,
    ) -> tuple[str, dict[str, Any]]:
        if definition.input_ids is None:
            raise ValueError(f"operation {definition.operation_id} has incomplete input/contract identity")
        required = set(definition.input_ids) | set(context.required_input_ids)
        missing = sorted(
            node for node in required
            if not isinstance(context.upstream_fingerprints.get(node), str)
            or not context.upstream_fingerprints[node].strip()
        )
        if missing:
            raise ValueError(f"operation {definition.operation_id} missing input identities: {missing}")
        if not context.target_trade_date or not context.scope:
            raise ValueError("checkpoint requires target date and scope")
        paths = implementation_paths(self.project_root, definition.cache.contract_paths) if definition.cache.track_python_imports else definition.cache.contract_paths
        contract_hashes = {
            relative: _sha256_file(self._safe_project_path(relative))
            for relative in paths
        }
        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "operation_id": definition.operation_id,
            "entrypoint": definition.entrypoint,
            "produces": list(definition.produces),
            "input_ids": sorted(required),
            "cache_mode": definition.cache.mode.value,
            "contract_version": definition.cache.contract_version,
            "output_paths": list(definition.cache.output_paths),
            "state_path": definition.cache.state_path,
            "optional_contract_paths": list(definition.cache.optional_contract_paths),
            "optional_contract_hashes": {
                path: fingerprint_artifact(self._safe_project_path(path))
                if self._safe_project_path(path).exists() else None
                for path in definition.cache.optional_contract_paths
            },
            "environment_keys": list(definition.cache.environment_keys),
            "environment": {key: os.getenv(key) for key in definition.cache.environment_keys},
            "track_python_imports": definition.cache.track_python_imports,
            "target_trade_date": context.target_trade_date,
            "scope": context.scope,
            "upstream_revisions": dict(sorted(context.upstream_revisions.items())),
            "upstream_fingerprints": dict(
                sorted(context.upstream_fingerprints.items())
            ),
            # Dirty sets are work hints; authoritative revisions identify data.
            "parameters": dict(context.parameters),
            "contract_hashes": contract_hashes,
        }
        return canonical_fingerprint(payload), payload

    def requires_full_rebuild(
        self, definition: OperationDefinition, context: OperationContext,
        identity_payload: Mapping[str, Any],
    ) -> bool:
        """A changed contract or missing correction journal cannot use append-only work."""
        try:
            previous = json.loads(self._checkpoint_path(definition.operation_id).read_text())
            old = previous["identity_payload"]
            if not isinstance(old, dict):
                return True
            if canonical_fingerprint(old) != previous["identity"]:
                return True
        except (OSError, ValueError, KeyError, TypeError):
            return True
        source_fields = {"upstream_fingerprints", "upstream_revisions"}
        if any(not isinstance(old.get(field), dict) for field in source_fields):
            return True
        # A forward trading-date advance is not a code/model contract change.
        # Its input journal still has to prove a bounded repair below.
        old_date = str(old.get("target_trade_date", "")).replace("-", "")
        new_date = str(identity_payload.get("target_trade_date", "")).replace("-", "")
        if not old_date or new_date < old_date:
            return True
        work_fields = source_fields | {"target_trade_date"}
        if {key: value for key, value in old.items() if key not in work_fields} != {
            key: value for key, value in identity_payload.items() if key not in work_fields
        }:
            return True
        changed = {
            node for field in source_fields for node, value in identity_payload[field].items()
            if old.get(field, {}).get(node) != value
        }
        if not changed:
            # Matching identity with unusable/missing outputs needs integrity repair.
            return True
        return any(
            node not in context.input_changes
            or context.input_changes[node].full_rebuild
            or not (context.input_changes[node].partitions or context.input_changes[node].keys)
            for node in changed
        )

    def load(
        self,
        definition: OperationDefinition,
        identity: str,
    ) -> OperationResult | None:
        if definition.cache.mode == CacheMode.NONE:
            return None
        path = self._checkpoint_path(definition.operation_id)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, ValueError):
            return None
        if not isinstance(payload, dict) or (
            payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
            or payload.get("identity") != identity
            or payload.get("status") != "success"
            or payload.get("operation_id") != definition.operation_id
            or not isinstance(payload.get("identity_payload"), dict)
        ):
            return None
        try:
            if canonical_fingerprint(payload["identity_payload"]) != identity:
                return None
            self._validate_identity_payload(definition, payload["identity_payload"])
            if self._output_hashes(definition) != payload.get("output_hashes"):
                return None
        except (OSError, TypeError, ValueError):
            return None
        raw_result = payload.get("result")
        if not isinstance(raw_result, Mapping):
            return None
        old_root = payload.get("publication_output_root")
        if (isinstance(old_root, str) and old_root != _publication_output_root()
                and _contains_generation_path(raw_result, old_root)):
            # Artifact bytes can be cloned, but a result carrying staging paths
            # from another generation is not portable. Do not reuse those paths.
            return None
        try:
            raw_result = dict(raw_result)
            raw_result["node_changes"] = {
                node: NodeChanges(
                    partitions=tuple(change["partitions"]),
                    keys=tuple(change["keys"]),
                    full_rebuild=change["full_rebuild"],
                )
                for node, change in raw_result.get("node_changes", {}).items()
            }
            result = OperationResult(**raw_result)
            validate_result_identity(definition, result)
            identity_payload = payload["identity_payload"]
            if (
                result.contract_identity != identity
                or dict(result.input_fingerprints) != identity_payload["upstream_fingerprints"]
                or dict(result.input_revisions) != identity_payload["upstream_revisions"]
            ):
                return None
            return replace(
                result,
                changed_partitions=(),
                changed_keys=(),
                node_changes={node: NodeChanges() for node in definition.produces},
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            return None

    def save(
        self,
        definition: OperationDefinition,
        identity: str,
        identity_payload: Mapping[str, Any],
        result: OperationResult,
    ) -> None:
        if definition.cache.mode == CacheMode.NONE or result.status != "success":
            return
        validate_result_identity(definition, result)
        self._validate_identity_payload(definition, identity_payload)
        if (
            not identity_payload
            or canonical_fingerprint(identity_payload) != identity
            or result.contract_identity != identity
            or dict(result.input_fingerprints) != identity_payload.get("upstream_fingerprints")
            or dict(result.input_revisions) != identity_payload.get("upstream_revisions")
        ):
            raise ValueError("checkpoint identity and result lineage do not match")
        output_hashes = self._output_hashes(definition)
        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "operation_id": definition.operation_id,
            "status": "success",
            "identity": identity,
            "identity_payload": dict(identity_payload),
            "output_hashes": output_hashes,
            "result": asdict(result),
            "publication_output_root": _publication_output_root(),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        atomic_write_json(payload, self._checkpoint_path(definition.operation_id))


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION", "CheckpointStore", "canonical_fingerprint",
    "fingerprint_artifact", "validate_result_identity",
]
