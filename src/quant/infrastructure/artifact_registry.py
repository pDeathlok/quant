"""Local artifact ownership and reachability, with process-held read/build leases.

Producers register directory ownership (including its contents), pin active and
previous committed outputs, and explicitly retire disposable artifacts. Unknown
ownership never authorizes deletion. Leases must cover the entire read/build,
including memmap consumption and worker completion. Locks are local POSIX locks;
this is not a distributed/NFS lease service. Lock failure is deliberately fatal.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, Mapping, TypeVar
from uuid import uuid4

from quant.data.atomic_io import atomic_write_json

T = TypeVar("T")

if TYPE_CHECKING:
    from quant.infrastructure.publication import PublicationView


def _overlaps(left: str, right: str) -> bool:
    return left == right or left.startswith(right + "/") or right.startswith(left + "/")


def _storage(paths: Iterable[Path]) -> dict[str, int]:
    files: dict[Path, os.stat_result] = {}
    for path in paths:
        if path.is_symlink():
            continue
        for child in path.rglob("*") if path.is_dir() else [path]:
            if child.is_file() and not child.is_symlink():
                files[child] = child.stat()
    inodes = {(stat.st_dev, stat.st_ino): stat for stat in files.values()}
    return {
        "files": len(files),
        "logical_bytes": sum(stat.st_size for stat in files.values()),
        "allocated_bytes": sum(stat.st_blocks * 512 for stat in inodes.values()),
    }


class ArtifactRegistry:
    """Metadata updates, lease acquisition, and guarded deletion share one lock.

    Constructing/inspecting the registry does not create files. The project-root
    directory inode is the stable lock, never a deletable cache inode. Do not
    nest registry methods inside ``deletion_guard``.
    """

    def __init__(self, project_root: Path) -> None:
        self.root = project_root.resolve(strict=True)
        self.directory = self.root / "data/artifact_registry"
        self.state_path = self.directory / "registry.json"

    def _key(self, path: Path | str, *, inventory_scope: bool = False) -> str:
        candidate = Path(path)
        candidate = candidate if candidate.is_absolute() else self.root / candidate
        relative = candidate.relative_to(self.root)
        if not relative.parts or ".." in relative.parts:
            raise ValueError("artifact path must be strictly inside project root")
        current = self.root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError(f"symlink artifact path: {candidate}")
        if not inventory_scope and _overlaps(relative.as_posix(), "data/artifact_registry"):
            raise ValueError("registry metadata cannot be an artifact")
        return relative.as_posix()

    @contextmanager
    def _locked(self) -> Iterator[None]:
        descriptor = os.open(self.root, os.O_RDONLY)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            os.close(descriptor)

    def _read(self) -> dict[str, Any]:
        if any(path.is_symlink() for path in (
            self.root / "data", self.directory, self.state_path,
        )):
            raise ValueError("symlink registry metadata")
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"schema_version": 1, "artifacts": {}, "references": {}, "leases": {}}
        if not isinstance(state, dict) or state.get("schema_version") != 1:
            raise ValueError("unsupported artifact registry")
        for field in ("artifacts", "references", "leases"):
            if not isinstance(state.get(field), dict):
                raise ValueError(f"invalid registry {field}")
        for key, record in state["artifacts"].items():
            if self._key(key) != key:
                raise ValueError("registry paths must be project-relative")
            self._validate_record(record)
        for paths in state["references"].values():
            self._validate_paths(paths)
        for token, lease in state["leases"].items():
            if len(token) != 32 or any(char not in "0123456789abcdef" for char in token):
                raise ValueError("invalid lease token")
            self._validate_paths(lease["paths"])
            if not lease["paths"] or not isinstance(lease.get("owner"), str) or not lease["owner"].strip():
                raise ValueError("invalid lease ownership")
            if lease["kind"] not in {"read", "build"}:
                raise ValueError("invalid lease kind")
        return state

    def _save(self, registry: dict[str, Any]) -> None:
        # Called under the registry lock. A crashed process cannot reacquire its
        # unique token; missing/unreadable lock files remain conservative pins.
        dead_tokens = [token for token in registry["leases"] if not self._live_lease(token)]
        for token in dead_tokens:
            registry["leases"].pop(token)
        atomic_write_json(registry, self.state_path)
        for token in dead_tokens:
            (self.directory / (token + ".lease")).unlink(missing_ok=True)

    def _validate_paths(self, paths: Any) -> None:
        if not isinstance(paths, list) or any(not isinstance(path, str) for path in paths):
            raise ValueError("invalid artifact paths")
        for path in paths:
            if self._key(path) != path:
                raise ValueError("registry paths must be project-relative")

    def _validate_record(self, record: Any) -> None:
        if not isinstance(record, dict) or not isinstance(record.get("producer"), str):
            raise ValueError("artifact producer is required")
        if not record["producer"].strip():
            raise ValueError("artifact producer is required")
        if record.get("retention_class") not in {"rebuildable", "temporary", "raw", "evidence"}:
            raise ValueError("invalid retention class")
        if record.get("state") not in {"building", "committed", "retired"}:
            raise ValueError("invalid artifact state")
        if not isinstance(record.get("protected"), bool):
            raise ValueError("invalid protected flag")
        if not isinstance(record.get("ownership_boundary", False), bool):
            raise ValueError("invalid ownership boundary")
        versions = record.get("input_versions")
        if not isinstance(versions, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in versions.items()
        ):
            raise ValueError("input versions must be a string mapping")
        self._validate_paths(record.get("dependencies"))
        for field in ("last_access", "rebuild_cost_seconds"):
            value = record.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid {field}")

    def register(
        self,
        path: Path | str,
        *,
        producer: str,
        input_versions: Mapping[str, str],
        retention_class: str = "evidence",
        state: str = "building",
        dependencies: Iterable[Path | str] = (),
        protected: bool = False,
        ownership_boundary: bool = False,
        rebuild_cost_seconds: float = 0,
        last_access: float | None = None,
    ) -> None:
        """Declare ownership, optionally separating this subtree from its parent.

        An independent boundary opts out of a same-producer rebuildable parent's
        committed/reference retention, not its live leases or protected/raw state.
        Pin independently owned current/previous generations explicitly. Omitted
        boundaries retain the original whole-subtree protection semantics.
        """
        key = self._key(path)
        record = {
            "producer": producer,
            "input_versions": dict(input_versions),
            "retention_class": retention_class,
            "state": state,
            "dependencies": [self._key(item) for item in dependencies],
            "protected": protected,
            "ownership_boundary": ownership_boundary,
            "last_access": time.time() if last_access is None else last_access,
            "rebuild_cost_seconds": rebuild_cost_seconds,
        }
        self._validate_record(record)
        with self._locked():
            registry = self._read()
            previous = registry["artifacts"].get(key)
            if previous and previous["producer"] != producer:
                raise ValueError("artifact is already owned by another producer")
            registry["artifacts"][key] = record
            self._save(registry)

    def set_references(self, consumer: str, paths: Iterable[Path | str]) -> None:
        """Pin active config/model/report inputs, even before they exist on disk."""
        if not consumer.strip():
            raise ValueError("consumer is required")
        keys = [self._key(path) for path in paths]
        with self._locked():
            registry = self._read()
            registry["references"][consumer] = keys
            self._save(registry)

    def referenced_paths(self, consumer: str) -> tuple[Path, ...]:
        with self._locked():
            return tuple(self.root / key for key in self._read()["references"].get(consumer, []))

    def commit(self, consumer: str, path: Path | str) -> tuple[Path, ...]:
        """Pin current/previous output and return displaced previous references."""
        key = self._key(path)
        if not consumer.strip():
            raise ValueError("consumer is required")
        with self._locked():
            registry = self._read()
            if not (self.root / key).exists():
                raise FileNotFoundError(key)
            record = registry["artifacts"][key]
            record["state"] = "committed"
            old = registry["references"].get(consumer, [])
            displaced = registry["references"].get(consumer + ":previous", []) if old != [key] else []
            if old != [key]:
                registry["references"][consumer + ":previous"] = old
            registry["references"][consumer] = [key]
            self._save(registry)
            return tuple(self.root / item for item in displaced)

    def retire(self, path: Path | str) -> None:
        """Opt into collection; references and leases still override retirement."""
        with self._locked():
            registry = self._read()
            registry["artifacts"][self._key(path)]["state"] = "retired"
            self._save(registry)

    def retire_owned_tree(
        self, path: Path | str, *, producer: str, consumers: Iterable[str],
        protected_consumers: Iterable[str] = (),
    ) -> dict[str, Any]:
        """Retire an obsolete producer namespace, never unrelated consumer pins.

        This changes ownership metadata only. Conflicting ownership, unfinished
        builds, protected artifacts, and current/previous consumer roots fail
        closed. Actual deletion still requires the normal reference/lease guard.
        """
        key = self._key(path)
        consumers = tuple(consumers)
        with self._locked():
            registry = self._read()
            records = {
                item: record for item, record in registry["artifacts"].items()
                if item == key or item.startswith(key + "/")
            }
            keep = None
            if key not in records or any(record["producer"] != producer for record in records.values()):
                keep = "unknown_or_conflicting_ownership"
            elif any(
                record["state"] == "building" or record["protected"]
                or record["retention_class"] not in {"rebuildable", "temporary"}
                for record in records.values()
            ):
                keep = "building_or_protected_artifact"
            elif any(
                _overlaps(key, target) for consumer in protected_consumers
                for target in registry["references"].get(consumer, [])
            ):
                keep = "active_or_previous_reference"
            elif any(
                target != key and not target.startswith(key + "/") for consumer in consumers
                for target in registry["references"].get(consumer, [])
            ):
                keep = "consumer_has_external_references"
            if keep:
                return {"retired_paths": [], "removed_consumers": [], "keep_reason": keep}
            for record in records.values():
                record["state"] = "retired"
            removed = [consumer for consumer in consumers if consumer in registry["references"]]
            for consumer in removed:
                registry["references"].pop(consumer)
            self._save(registry)
            return {"retired_paths": sorted(records), "removed_consumers": removed, "keep_reason": None}

    @contextmanager
    def lease(
        self, paths: Iterable[Path | str], *, owner: str, kind: str = "read"
    ) -> Iterator[None]:
        paths = tuple(paths)
        if not paths:
            raise ValueError("lease needs paths, owner, and read/build kind")
        with self.lease_selection(lambda: (None, paths), owner=owner, kind=kind):
            yield

    @contextmanager
    def lease_selection(
        self, select: Callable[[], tuple[T, Iterable[Path | str]]], *,
        owner: str, kind: str = "read",
    ) -> Iterator[T]:
        """Select a view and acquire its lease atomically against deletion.

        ``select`` runs under the registry lock, must not call registry methods,
        and returns (view, paths). Empty paths are permitted for a legacy view
        with no managed generation. Readers do not wait on publication writers.
        """
        if not owner.strip() or kind not in {"read", "build"}:
            raise ValueError("lease needs owner and read/build kind")
        token = uuid4().hex
        lease_path = self.directory / (token + ".lease")
        handle = None
        with self._locked():
            registry = self._read()
            selected, paths = select()
            keys = [self._key(path) for path in paths]
            if kind == "read" and any(not (self.root / key).exists() for key in keys):
                raise FileNotFoundError("cannot lease a missing artifact for reading")
            if keys:
                self.directory.mkdir(parents=True, exist_ok=True)
                handle = lease_path.open("xb")
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                    registry["leases"][token] = {
                        "paths": keys, "owner": owner, "kind": kind, "pid": os.getpid(),
                    }
                    for key, record in registry["artifacts"].items():
                        if any(_overlaps(key, target) for target in keys):
                            record["last_access"] = time.time()
                    self._save(registry)
                except BaseException:
                    handle.close()
                    lease_path.unlink(missing_ok=True)
                    raise
        try:
            yield selected
        finally:
            if handle is not None:
                try:
                    with self._locked():
                        registry = self._read()
                        registry["leases"].pop(token, None)
                        for key, record in registry["artifacts"].items():
                            if any(_overlaps(key, target) for target in keys):
                                record["last_access"] = time.time()
                        self._save(registry)
                        lease_path.unlink(missing_ok=True)
                finally:
                    handle.close()

    def _live_lease(self, token: str) -> bool:
        path = self.directory / (token + ".lease")
        if path.is_symlink():
            return True
        try:
            with path.open("rb") as handle:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return True
            return False
        except OSError:
            # Missing/unreadable lease metadata is not evidence of a dead owner.
            return True

    @staticmethod
    def _protection_covers(
        target: str, candidate: str, registry: dict[str, Any], *, reason: str,
    ) -> bool:
        if not _overlaps(target, candidate):
            return False
        if target == candidate or target.startswith(candidate + "/") or reason == "live_lease":
            return True
        owner = registry["artifacts"].get(target)
        if (
            not owner or owner["protected"] or owner["state"] == "building"
            or owner["retention_class"] not in {"rebuildable", "temporary"}
        ):
            return True
        # Only an explicitly independent, same-producer descendant may exclude
        # soft ancestor protection. Unknown ownership never establishes a boundary.
        return not any(
            record.get("ownership_boundary", False)
            and record["producer"] == owner["producer"]
            and boundary.startswith(target + "/")
            and (candidate == boundary or candidate.startswith(boundary + "/"))
            for boundary, record in registry["artifacts"].items()
        )

    def _protected(self, registry: dict[str, Any]) -> dict[str, str]:
        protected = {
            key: "artifact_not_retired"
            for key, record in registry["artifacts"].items()
            if record["protected"] or record["state"] != "retired"
            or record["retention_class"] not in {"rebuildable", "temporary"}
        }
        for paths in registry["references"].values():
            protected.update({key: "active_reference" for key in paths})
        for token, lease in registry["leases"].items():
            if self._live_lease(token):
                protected.update({key: "live_lease" for key in lease["paths"]})
        # A protected descendant also retains the owning directory's inputs.
        visited: set[str] = set()
        while True:
            reachable = {
                key for key in registry["artifacts"] if key not in visited
                and any(
                    self._protection_covers(target, key, registry, reason=reason)
                    for target, reason in protected.items()
                )
            }
            if not reachable:
                break
            for key in reachable:
                visited.add(key)
                for dependency in registry["artifacts"][key]["dependencies"]:
                    protected.setdefault(dependency, "reachable_dependency")
        return protected

    def _reason(
        self, key: str, registry: dict[str, Any], *, require_owned: bool = True
    ) -> str | None:
        for protected, reason in self._protected(registry).items():
            if self._protection_covers(protected, key, registry, reason=reason):
                return reason
        if require_owned and key not in registry["artifacts"]:
            return "unknown_ownership"
        return None

    @contextmanager
    def deletion_guard(
        self, path: Path | str, *, require_owned: bool = True, older_than: float | None = None,
        protected_paths: Iterable[Path | str] = (),
    ) -> Iterator[str | None]:
        """Yield a keep reason or None; hold the lock through the actual deletion.

        ``require_owned=False`` is only for existing, narrowly owned legacy rules.
        Registered references/leases still override those rules. Never use it to
        collect unclassified research/raw trees.
        """
        with self._locked():
            try:
                key = self._key(path)
                registry = self._read()
                reason = self._reason(key, registry, require_owned=require_owned)
                if any(_overlaps(key, self._key(item)) for item in protected_paths):
                    reason = "explicit_reference"
                if older_than is not None:
                    record = registry["artifacts"].get(key)
                    if not record or record["last_access"] >= older_than:
                        reason = reason or "not_expired"
            except (OSError, ValueError, KeyError, TypeError) as exc:
                reason = f"registry_unavailable:{exc}"
            yield reason

    def delete_if_eligible(
        self, path: Path | str, *, protected_paths: Iterable[Path | str] = (),
        older_than: float | None = None,
    ) -> dict[str, Any]:
        with self.deletion_guard(
            path, protected_paths=protected_paths, older_than=older_than,
        ) as reason:
            if reason:
                return {"deleted": False, "reason": reason, "reclaimed_bytes": 0}
            return self._delete_locked(path)

    def _delete_locked(self, path: Path | str) -> dict[str, Any]:
        target = self.root / self._key(path)
        exists = target.exists()
        size = _storage([target])["logical_bytes"]
        if target.is_dir():
            shutil.rmtree(target)
        elif exists:
            target.unlink()
        registry = self._read()
        deleted_key = self._key(path)
        registry["artifacts"] = {
            key: value for key, value in registry["artifacts"].items()
            if key != deleted_key and not key.startswith(deleted_key + "/")
        }
        self._save(registry)
        return {
            "deleted": exists, "reason": None if exists else "missing", "reclaimed_bytes": size,
            "forgotten": not exists,
        }

    def collect_retired_children(
        self, directory: Path | str, *, dry_run: bool = True,
        protected_paths: Iterable[Path | str] = (),
    ) -> dict[str, Any]:
        """Preview/collect only directly owned, independent retired children.

        Useful for compiled generations inside a still-active config. Producers
        commit current/previous refs and explicitly retire superseded generations
        first. Unknown or non-boundary children never qualify. Call after releasing
        config-wide leases; leases always win, including those held by the caller.
        Inventory is advisory; deletion and its boundary/lease recheck are atomic.
        """
        root = self.root / self._key(directory)
        protected_paths = tuple(protected_paths)
        result: dict[str, Any] = {
            "dry_run": dry_run, "candidates": [], "deleted_paths": [], "kept": {},
            "forgotten_paths": [],
            "reclaimed_bytes": 0, "errors": [], "inventory": self.inventory([root]),
        }
        paths = set(root.iterdir()) if root.is_dir() else set()
        paths.update(
            self.root / item["path"] for item in result["inventory"]["entries"]
            if (self.root / item["path"]).parent == root
        )
        for path in sorted(paths):
            try:
                with self.deletion_guard(path, protected_paths=protected_paths) as reason:
                    key = path.relative_to(self.root).as_posix()
                    if reason:
                        result["kept"][key] = reason
                        continue
                    record = self._read()["artifacts"].get(key, {})
                    if not record.get("ownership_boundary", False):
                        result["kept"][key] = "ownership_boundary_not_declared"
                        continue
                    result["candidates"].append(key)
                    if not dry_run:
                        deletion = self._delete_locked(path)
                        if deletion["deleted"]:
                            result["deleted_paths"].append(key)
                            result["reclaimed_bytes"] += deletion["reclaimed_bytes"]
                        elif deletion.get("forgotten"):
                            result["forgotten_paths"].append(key)
                        else:
                            result["kept"][key] = deletion["reason"]
            except (OSError, ValueError, KeyError, TypeError) as exc:
                result["errors"].append(f"artifact_collection:{path.name}:{exc}")
        return result

    @contextmanager
    def build_mutation_guard(self, path: Path | str) -> Iterator[None]:
        """Allow in-process maintenance of an unpublished, locally leased build.

        Caller must also verify the producer's staging/publication metadata while
        holding this guard. Read leases and durable references forbid mutation.
        """
        with self._locked():
            key = self._key(path)
            registry = self._read()
            record = registry["artifacts"].get(key)
            if not record or record["state"] != "building":
                raise ValueError("mutation requires registered building ownership")
            if any(
                _overlaps(key, target) for paths in registry["references"].values() for target in paths
            ):
                raise ValueError("referenced artifacts cannot be mutated")
            live = [
                lease for token, lease in registry["leases"].items() if self._live_lease(token)
            ]
            if any(
                lease["kind"] == "read" and any(_overlaps(key, target) for target in lease["paths"])
                for lease in live
            ):
                raise ValueError("read-leased artifacts cannot be mutated")
            if not any(
                lease["kind"] == "build" and lease.get("pid") == os.getpid()
                and any(key == target or key.startswith(target + "/") for target in lease["paths"])
                for lease in live
            ):
                raise ValueError("mutation requires a live build lease in this process")
            yield

    def inventory(
        self, paths: Iterable[Path | str], *, budgets: Mapping[str, int] | None = None
    ) -> dict[str, Any]:
        """Read-only preview. Budgets are allocated-byte limits per requested scope.

        Totals deduplicate overlapping paths and hardlink inodes; logical bytes
        count each pathname. Allocated bytes describe storage, not guaranteed
        reclaim (other hardlinks/open readers may retain blocks). Budgets never
        authorize deletion and unregistered contents remain unknown.
        """
        keys = sorted({self._key(path, inventory_scope=True) for path in paths})
        limits = {
            self._key(key, inventory_scope=True): value for key, value in (budgets or {}).items()
        }
        if any(type(value) is not int or value < 0 for value in limits.values()):
            raise ValueError("budgets must be nonnegative allocated-byte limits")
        with self._locked():
            errors: list[str] = []
            try:
                registry = self._read()
            except (OSError, ValueError, KeyError, TypeError) as exc:
                errors.append(f"registry_unavailable:{exc}")
                registry = {"artifacts": {}, "references": {}, "leases": {}}
            protected = self._protected(registry)
        # Inventory is advisory: do not hold up lease acquisition while scanning
        # large research trees. Apply always rechecks live reachability under lock.
        entries = []
        for key in sorted(set(keys) | {
            artifact for artifact in registry["artifacts"]
            if any(artifact.startswith(scope + "/") for scope in keys)
        }):
            record = registry["artifacts"].get(key, {})
            reason = errors[0] if errors else next(
                (reason for target, reason in protected.items()
                 if self._protection_covers(target, key, registry, reason=reason)),
                None if record else "unknown_ownership",
            )
            entries.append({
                "path": key, **record, **_storage([self.root / key]),
                "consumers": [
                    consumer for consumer, targets in registry["references"].items()
                    if any(
                        self._protection_covers(target, key, registry, reason="active_reference")
                        for target in targets
                    )
                ],
                "deletion_candidate": reason is None,
                "keep_reason": reason,
            })
        budget_report = {}
        for key, limit in limits.items():
            usage = _storage([self.root / key])["allocated_bytes"]
            budget_report[key] = {
                "limit_bytes": limit, "allocated_bytes": usage,
                "over_budget_bytes": max(0, usage - limit),
            }
        return {
            "dry_run": True, "entries": entries,
            "totals": _storage(self.root / key for key in keys),
            "budgets": budget_report, "errors": errors,
            "missing_references": sorted({
                target for targets in registry["references"].values()
                for target in targets if not (self.root / target).exists()
            }),
        }


@contextmanager
def publication_read_lease(
    project_root: Path, select_view: Callable[[], PublicationView], *,
    owner: str = "publication:reader",
) -> Iterator[PublicationView]:
    """Pin the view chosen by ``store.view(requested_generation)`` for a request.

    Usage: with publication_read_lease(root, lambda: store.view(header)) as view:
        with publication_context(view): ... serialize/stream the complete result

    Selection happens under the deletion lock, not the long-lived writer lock.
    A legacy view has no generation lease. Do not call store.view() again inside
    the request; use the returned view, including its generation response header.
    """
    def select() -> tuple[PublicationView, tuple[Path, ...]]:
        view = select_view()
        paths = (view.directory / view.generation,) if view.generation else ()
        return view, paths

    with ArtifactRegistry(project_root).lease_selection(select, owner=owner) as view:
        yield view
