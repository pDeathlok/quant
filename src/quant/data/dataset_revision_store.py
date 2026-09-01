"""Durable dataset and logical-partition revisions for incremental routines."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import fcntl
import hashlib
import json
from pathlib import Path
from typing import Mapping

from quant.data.atomic_io import atomic_write_json


REVISION_SCHEMA_VERSION = "dataset-revisions-v1"


@dataclass(frozen=True)
class PartitionRevisionInput:
    row_count: int
    content_sha256: str


@dataclass(frozen=True)
class DatasetRevision:
    dataset_id: str
    revision: int
    watermark: str | None
    content_sha256: str
    changed_partitions: tuple[str, ...] = ()


class DatasetRevisionStore:
    def __init__(
        self,
        *,
        sql_url: str | None = None,
        metadata_path: Path | None = None,
    ) -> None:
        self.sql_url = sql_url
        self.metadata_path = metadata_path or Path(
            "data/raw/.dataset_revisions.json"
        )

    def get(self, dataset_id: str) -> DatasetRevision | None:
        if self.sql_url:
            return self._get_sql(dataset_id)
        payload = self._read_local()
        raw = (payload.get("datasets") or {}).get(dataset_id)
        if not isinstance(raw, dict):
            return None
        return DatasetRevision(
            dataset_id=dataset_id,
            revision=int(raw.get("revision") or 0),
            watermark=raw.get("watermark"),
            content_sha256=str(raw.get("content_sha256") or ""),
        )

    def commit(
        self,
        dataset_id: str,
        partitions: Mapping[str, PartitionRevisionInput],
        *,
        watermark: str | None,
    ) -> DatasetRevision:
        if not partitions:
            current = self.get(dataset_id)
            return current or DatasetRevision(dataset_id, 0, watermark, "")
        if self.sql_url:
            return self._commit_sql(dataset_id, partitions, watermark=watermark)
        return self._commit_local(dataset_id, partitions, watermark=watermark)

    def _engine(self):
        from sqlalchemy import create_engine

        return create_engine(self.sql_url, pool_pre_ping=True, pool_recycle=300)

    @staticmethod
    def _tables(metadata):
        from sqlalchemy import BigInteger, Column, DateTime, Integer, String, Table

        datasets = Table(
            "routine_dataset_revisions",
            metadata,
            Column("dataset_id", String(128), primary_key=True),
            Column("revision", BigInteger, nullable=False),
            Column("watermark", String(32), nullable=True),
            Column("content_sha256", String(64), nullable=False),
            Column("updated_at", DateTime, nullable=False),
        )
        partitions = Table(
            "routine_partition_revisions",
            metadata,
            Column("dataset_id", String(128), primary_key=True),
            Column("partition_key", String(64), primary_key=True),
            Column("revision", BigInteger, nullable=False),
            Column("row_count", Integer, nullable=False),
            Column("content_sha256", String(64), nullable=False),
            Column("updated_at", DateTime, nullable=False),
        )
        return datasets, partitions

    def _get_sql(self, dataset_id: str) -> DatasetRevision | None:
        from sqlalchemy import MetaData, select

        engine = self._engine()
        try:
            metadata = MetaData()
            datasets, _ = self._tables(metadata)
            metadata.create_all(engine)
            with engine.connect() as connection:
                row = connection.execute(
                    select(datasets).where(datasets.c.dataset_id == dataset_id)
                ).mappings().first()
            if row is None:
                return None
            return DatasetRevision(
                dataset_id=dataset_id,
                revision=int(row["revision"]),
                watermark=row["watermark"],
                content_sha256=str(row["content_sha256"]),
            )
        finally:
            engine.dispose()

    def _commit_sql(
        self,
        dataset_id: str,
        inputs: Mapping[str, PartitionRevisionInput],
        *,
        watermark: str | None,
    ) -> DatasetRevision:
        from sqlalchemy import MetaData, and_, select

        engine = self._engine()
        now = datetime.now()
        try:
            metadata = MetaData()
            datasets, partitions = self._tables(metadata)
            metadata.create_all(engine)
            with engine.begin() as connection:
                current_rows = connection.execute(
                    select(partitions).where(
                        and_(
                            partitions.c.dataset_id == dataset_id,
                            partitions.c.partition_key.in_(list(inputs)),
                        )
                    )
                ).mappings()
                current = {
                    str(row["partition_key"]): row for row in current_rows
                }
                changed = sorted(
                    key
                    for key, incoming in inputs.items()
                    if key not in current
                    or int(current[key]["row_count"]) != incoming.row_count
                    or str(current[key]["content_sha256"])
                    != incoming.content_sha256
                )
                dataset_row = connection.execute(
                    select(datasets).where(datasets.c.dataset_id == dataset_id)
                ).mappings().first()
                revision = int(dataset_row["revision"]) if dataset_row else 0
                content_sha256 = (
                    str(dataset_row["content_sha256"]) if dataset_row else ""
                )
                if changed:
                    revision += 1
                    for key in changed:
                        incoming = inputs[key]
                        row = current.get(key)
                        values = {
                            "revision": (int(row["revision"]) if row else 0) + 1,
                            "row_count": incoming.row_count,
                            "content_sha256": incoming.content_sha256,
                            "updated_at": now,
                        }
                        if row:
                            connection.execute(
                                partitions.update()
                                .where(
                                    and_(
                                        partitions.c.dataset_id == dataset_id,
                                        partitions.c.partition_key == key,
                                    )
                                )
                                .values(**values)
                            )
                        else:
                            connection.execute(
                                partitions.insert().values(
                                    dataset_id=dataset_id,
                                    partition_key=key,
                                    **values,
                                )
                            )
                    content_sha256 = self._next_dataset_hash(
                        content_sha256,
                        {key: inputs[key] for key in changed},
                    )
                dataset_values = {
                    "revision": revision,
                    "watermark": watermark,
                    "content_sha256": content_sha256,
                    "updated_at": now,
                }
                if dataset_row:
                    connection.execute(
                        datasets.update()
                        .where(datasets.c.dataset_id == dataset_id)
                        .values(**dataset_values)
                    )
                else:
                    connection.execute(
                        datasets.insert().values(
                            dataset_id=dataset_id,
                            **dataset_values,
                        )
                    )
            return DatasetRevision(
                dataset_id,
                revision,
                watermark,
                content_sha256,
                tuple(changed),
            )
        finally:
            engine.dispose()

    def _read_local(self) -> dict:
        try:
            payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, ValueError):
            return {
                "schema_version": REVISION_SCHEMA_VERSION,
                "datasets": {},
                "partitions": {},
            }
        if payload.get("schema_version") != REVISION_SCHEMA_VERSION:
            return {
                "schema_version": REVISION_SCHEMA_VERSION,
                "datasets": {},
                "partitions": {},
            }
        return payload

    def _commit_local(
        self,
        dataset_id: str,
        inputs: Mapping[str, PartitionRevisionInput],
        *,
        watermark: str | None,
    ) -> DatasetRevision:
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.metadata_path.with_suffix(self.metadata_path.suffix + ".lock")
        with lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            payload = self._read_local()
            datasets = payload.setdefault("datasets", {})
            all_partitions = payload.setdefault("partitions", {})
            current_dataset = datasets.get(dataset_id) or {}
            current_partitions = all_partitions.setdefault(dataset_id, {})
            changed = sorted(
                key
                for key, incoming in inputs.items()
                if key not in current_partitions
                or int(current_partitions[key].get("row_count") or 0)
                != incoming.row_count
                or str(current_partitions[key].get("content_sha256") or "")
                != incoming.content_sha256
            )
            revision = int(current_dataset.get("revision") or 0)
            content_sha256 = str(current_dataset.get("content_sha256") or "")
            now = datetime.now().isoformat(timespec="seconds")
            if changed:
                revision += 1
                for key in changed:
                    incoming = inputs[key]
                    previous = current_partitions.get(key) or {}
                    current_partitions[key] = {
                        "revision": int(previous.get("revision") or 0) + 1,
                        "row_count": incoming.row_count,
                        "content_sha256": incoming.content_sha256,
                        "updated_at": now,
                    }
                content_sha256 = self._next_dataset_hash(
                    content_sha256,
                    {key: inputs[key] for key in changed},
                )
            datasets[dataset_id] = {
                "revision": revision,
                "watermark": watermark,
                "content_sha256": content_sha256,
                "updated_at": now,
            }
            atomic_write_json(payload, self.metadata_path)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        return DatasetRevision(
            dataset_id,
            revision,
            watermark,
            content_sha256,
            tuple(changed),
        )

    @staticmethod
    def _next_dataset_hash(
        previous_hash: str,
        changed: Mapping[str, PartitionRevisionInput],
    ) -> str:
        payload = {
            "previous": previous_hash,
            "changed": {
                key: {
                    "row_count": value.row_count,
                    "content_sha256": value.content_sha256,
                }
                for key, value in sorted(changed.items())
            },
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()


__all__ = [
    "DatasetRevision",
    "DatasetRevisionStore",
    "PartitionRevisionInput",
    "REVISION_SCHEMA_VERSION",
]
