"""Durable workspace snapshot storage.

The repository keeps file and SQL persistence details outside the web delivery
layer. Project-specific environment loading stays at the composition root and
is injected through ``store_factory``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol


class _StoreConfig(Protocol):
    sql_url: str | None


class SnapshotStore(Protocol):
    config: _StoreConfig

    def _engine(self) -> Any: ...


def workspace_params_key(params: dict[str, Any] | None = None) -> str:
    """Return a stable key for semantically equivalent snapshot parameters."""

    raw = json.dumps(params or {}, ensure_ascii=True, sort_keys=True, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def canonical_snapshot_date(value: Any) -> str:
    """Normalize compact dates while preserving named or custom snapshots."""

    text_value = str(value or "latest").strip()
    if text_value == "latest":
        return text_value
    compact = text_value.replace("-", "")
    if len(compact) == 8 and compact.isdigit():
        return f"{compact[:4]}-{compact[4:6]}-{compact[6:]}"
    return text_value


@dataclass(frozen=True)
class WorkspaceSnapshotRepository:
    """Read and write versioned workspace payloads with optional SQL fallback."""

    directory: Path
    table_name: str
    schema_version: str = "workspace_payload_v1"
    store_factory: Callable[[], SnapshotStore] | None = None

    def snapshot_key(self, workspace: str, snapshot_date: str, params_key: str) -> str:
        raw = json.dumps(
            {
                "schema_version": self.schema_version,
                "workspace": workspace,
                "snapshot_date": canonical_snapshot_date(snapshot_date),
                "params_key": params_key,
            },
            ensure_ascii=True,
            sort_keys=True,
        )
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def file_path(self, workspace: str, params_key: str, snapshot_date: str) -> Path:
        safe_workspace = "".join(
            char for char in workspace if char.isalnum() or char in {"_", "-"}
        )
        date_key = canonical_snapshot_date(snapshot_date)
        return self.directory / safe_workspace / params_key / f"{date_key}.json"

    def read_filesystem(
        self,
        workspace: str,
        snapshot_date: str | None,
        params_key: str,
    ) -> dict[str, Any] | None:
        directory = self.file_path(workspace, params_key, "latest").parent
        requested = canonical_snapshot_date(snapshot_date) if snapshot_date else None
        candidates: list[Path] = []
        if requested:
            exact = self.file_path(workspace, params_key, requested)
            if exact.exists():
                candidates.append(exact)
            candidates.extend(
                sorted(
                    (
                        path
                        for path in directory.glob("*.json")
                        if path != exact and path.stem != "latest" and path.stem <= requested
                    ),
                    key=lambda path: path.stem,
                    reverse=True,
                )
            )
        elif directory.exists():
            candidates = sorted(
                (path for path in directory.glob("*.json") if path.stem != "latest"),
                key=lambda path: path.stem,
                reverse=True,
            )
        latest_path = directory / "latest.json"
        if latest_path.exists() and latest_path not in candidates:
            candidates.append(latest_path)

        for path in candidates:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            cached_date = canonical_snapshot_date(
                payload.get("trade_date") or payload.get("signal_date") or path.stem
            )
            if requested and cached_date != "latest" and cached_date > requested:
                continue
            payload["cache"] = {
                "hit": True,
                "backend": "filesystem",
                "workspace": workspace,
                "snapshot_date": cached_date,
                "requested_date": requested,
                "stale": bool(requested and cached_date != requested),
            }
            return payload
        return None

    def read(
        self,
        workspace: str,
        snapshot_date: str | None = None,
        params: dict[str, Any] | None = None,
        *,
        allow_sql: bool = True,
    ) -> dict[str, Any] | None:
        params_key = workspace_params_key(params)
        filesystem_payload = self.read_filesystem(workspace, snapshot_date, params_key)
        if filesystem_payload is not None:
            return filesystem_payload
        if not allow_sql:
            return None
        store = self._sql_store()
        if store is None:
            return None
        requested = canonical_snapshot_date(snapshot_date) if snapshot_date else None
        try:
            from sqlalchemy import text

            with store._engine().begin() as connection:
                rows = connection.execute(
                    text(
                        f"""
                        SELECT snapshot_key, snapshot_date, payload_json
                        FROM {self.table_name}
                        WHERE workspace = :workspace
                          AND params_key = :params_key
                        ORDER BY updated_at DESC
                        LIMIT 64
                        """
                    ),
                    {"workspace": workspace, "params_key": params_key},
                ).mappings().all()
            eligible: list[tuple[str, Any]] = []
            for row in rows:
                row_date = canonical_snapshot_date(row.get("snapshot_date"))
                if requested and row_date != "latest" and row_date > requested:
                    continue
                eligible.append((row_date, row))
            eligible.sort(
                key=lambda item: item[0] if item[0] != "latest" else "",
                reverse=True,
            )
            for row_date, row in eligible:
                if not row.get("payload_json"):
                    continue
                payload = json.loads(row["payload_json"])
                payload["cache"] = {
                    "hit": True,
                    "backend": "mysql",
                    "workspace": workspace,
                    "snapshot_key": str(row.get("snapshot_key") or ""),
                    "snapshot_date": row_date,
                    "requested_date": requested,
                    "stale": bool(requested and row_date != requested),
                }
                return payload
        except Exception:
            return None
        return None

    def write(
        self,
        workspace: str,
        snapshot_date: str | None,
        payload: dict[str, Any],
        params: dict[str, Any] | None = None,
        *,
        write_sql: bool = True,
    ) -> None:
        canonical_date = canonical_snapshot_date(snapshot_date)
        params = params or {}
        params_key = workspace_params_key(params)
        snapshot_key = self.snapshot_key(workspace, canonical_date, params_key)
        payload_to_store = dict(payload)
        payload_to_store["cache"] = {
            "hit": False,
            "backend": "generated",
            "workspace": workspace,
            "snapshot_key": snapshot_key,
            "snapshot_date": canonical_date,
        }
        payload_json = json.dumps(payload_to_store, ensure_ascii=False, default=str)
        self.write_filesystem(workspace, params_key, canonical_date, payload_json)
        if not write_sql:
            return
        store = self._sql_store()
        if store is None:
            return
        try:
            from sqlalchemy import text

            with store._engine().begin() as connection:
                connection.execute(
                    text(
                        f"""
                        CREATE TABLE IF NOT EXISTS {self.table_name} (
                            snapshot_key VARCHAR(64) PRIMARY KEY,
                            workspace VARCHAR(64) NOT NULL,
                            snapshot_date VARCHAR(32) NOT NULL,
                            params_key VARCHAR(64) NOT NULL,
                            params_json TEXT NOT NULL,
                            generated_at VARCHAR(32) NOT NULL,
                            payload_json LONGTEXT NOT NULL,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                            KEY idx_workspace_latest (workspace, params_key, snapshot_date)
                        )
                        """
                    )
                )
                connection.execute(
                    text(
                        f"""
                        INSERT INTO {self.table_name}
                            (snapshot_key, workspace, snapshot_date, params_key, params_json,
                             generated_at, payload_json)
                        VALUES
                            (:snapshot_key, :workspace, :snapshot_date, :params_key, :params_json,
                             :generated_at, :payload_json)
                        ON DUPLICATE KEY UPDATE
                            generated_at = VALUES(generated_at),
                            params_json = VALUES(params_json),
                            payload_json = VALUES(payload_json)
                        """
                    ),
                    {
                        "snapshot_key": snapshot_key,
                        "workspace": workspace,
                        "snapshot_date": canonical_date,
                        "params_key": params_key,
                        "params_json": json.dumps(
                            params,
                            ensure_ascii=False,
                            sort_keys=True,
                            default=str,
                        ),
                        "generated_at": str(
                            payload.get("generated_at")
                            or datetime.now().isoformat(timespec="seconds")
                        ),
                        "payload_json": payload_json,
                    },
                )
        except Exception:
            return

    def write_filesystem(
        self,
        workspace: str,
        params_key: str,
        snapshot_date: str,
        payload_json: str,
    ) -> None:
        snapshot_path = self.file_path(workspace, params_key, snapshot_date)
        latest_path = self.file_path(workspace, params_key, "latest")
        try:
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = snapshot_path.with_suffix(".json.tmp")
            temporary_path.write_text(payload_json, encoding="utf-8")
            temporary_path.replace(snapshot_path)
            latest_temporary_path = latest_path.with_suffix(".json.tmp")
            latest_temporary_path.write_text(payload_json, encoding="utf-8")
            latest_temporary_path.replace(latest_path)
        except Exception:
            return

    def dates(self, workspace: str, params: dict[str, Any] | None = None) -> set[str]:
        store = self._sql_store()
        if store is None:
            return set()
        params_key = workspace_params_key(params)
        try:
            from sqlalchemy import text

            with store._engine().begin() as connection:
                rows = connection.execute(
                    text(
                        f"""
                        SELECT snapshot_date
                        FROM {self.table_name}
                        WHERE workspace = :workspace
                          AND params_key = :params_key
                        """
                    ),
                    {"workspace": workspace, "params_key": params_key},
                ).scalars().all()
            return {str(item) for item in rows if item}
        except Exception:
            return set()

    def _sql_store(self) -> SnapshotStore | None:
        if self.store_factory is None:
            return None
        store = self.store_factory()
        if not store.config.sql_url:
            return None
        return store
