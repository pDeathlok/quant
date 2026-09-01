"""Central cache identity and checkpoint validation for routine operations."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from quant.data.atomic_io import atomic_write_json
from quant.routine.operation_contracts import (
    CacheMode,
    OperationContext,
    OperationDefinition,
    OperationResult,
)


CHECKPOINT_SCHEMA_VERSION = "daily-operation-checkpoint-v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
        return self.checkpoint_dir / f"{safe}.json"

    def build_identity(
        self,
        definition: OperationDefinition,
        context: OperationContext,
    ) -> tuple[str, dict[str, Any]]:
        contract_hashes = {
            relative: _sha256_file(self._safe_project_path(relative))
            for relative in definition.cache.contract_paths
        }
        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "operation_id": definition.operation_id,
            "cache_mode": definition.cache.mode.value,
            "contract_version": definition.cache.contract_version,
            "target_trade_date": context.target_trade_date,
            "scope": context.scope,
            "upstream_revisions": dict(sorted(context.upstream_revisions.items())),
            "upstream_fingerprints": dict(
                sorted(context.upstream_fingerprints.items())
            ),
            "dirty_partitions": sorted(set(context.dirty_partitions)),
            "dirty_keys": sorted(set(context.dirty_keys)),
            "parameters": dict(definition.parameters),
            "contract_hashes": contract_hashes,
        }
        return _canonical_sha256(payload), payload

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
        if (
            payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
            or payload.get("identity") != identity
            or payload.get("status") != "success"
        ):
            return None
        output_hashes = payload.get("output_hashes") or {}
        for relative in definition.cache.output_paths:
            output = self._safe_project_path(relative)
            expected_hash = output_hashes.get(relative)
            if not output.is_file() or not expected_hash:
                return None
            if _sha256_file(output) != expected_hash:
                return None
        raw_result = payload.get("result")
        if not isinstance(raw_result, Mapping):
            return None
        try:
            return OperationResult(**dict(raw_result))
        except (TypeError, ValueError):
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
        output_hashes: dict[str, str] = {}
        for relative in definition.cache.output_paths:
            output = self._safe_project_path(relative)
            if not output.is_file():
                raise RuntimeError(
                    f"operation {definition.operation_id} did not publish {relative}"
                )
            output_hashes[relative] = _sha256_file(output)
        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "operation_id": definition.operation_id,
            "status": "success",
            "identity": identity,
            "identity_payload": dict(identity_payload),
            "output_hashes": output_hashes,
            "result": asdict(result),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        atomic_write_json(payload, self._checkpoint_path(definition.operation_id))


__all__ = ["CHECKPOINT_SCHEMA_VERSION", "CheckpointStore"]
