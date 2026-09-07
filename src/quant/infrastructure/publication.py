"""Immutable output generations with one atomic publication pointer."""

from __future__ import annotations

import contextvars
import fcntl
import json
import os
import re
import shutil
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


class PublicationError(RuntimeError):
    """A result cannot be read or published as a committed generation."""


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".publish-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@dataclass(frozen=True)
class PublicationView:
    root: Path
    directory: Path
    generation: str | None
    managed: tuple[str, ...]
    writable: bool = False

    def resolve(self, path: Path) -> Path:
        try:
            relative = path.absolute().relative_to(self.root.absolute())
        except ValueError:
            return path
        if self.generation and any(
            relative == Path(item) or Path(item) in relative.parents
            for item in self.managed
        ):
            return self.directory / self.generation / "tree" / relative
        return path


_VIEW: contextvars.ContextVar[PublicationView | None] = contextvars.ContextVar(
    "quant_publication_view", default=None
)


def current_publication() -> PublicationView | None:
    return _VIEW.get()


def publication_path(path: Path) -> Path:
    view = current_publication()
    return view.resolve(path) if view is not None else path


def publication_write_path(path: Path) -> Path:
    resolved = publication_path(path)
    view = current_publication()
    if view and view.generation and (
        resolved != path or view.directory / view.generation in path.parents
    ):
        assert_publication_writable()
    return resolved


def assert_publication_writable() -> None:
    view = current_publication()
    if view is not None and view.generation and not view.writable:
        raise PublicationError(
            "No committed result for this request; run the daily workspace refresh "
            "instead of computing against unpublished inputs"
        )
    if view and view.writable and view.generation:
        state_path = view.directory / view.generation / "state.json"
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            raise PublicationError("Publication state is unavailable") from exc
        if not isinstance(state, dict) or state.get("status") != "staging":
            raise PublicationError("Committed generation is immutable")


def publication_sql_key(key: str) -> str:
    view = current_publication()
    if view is None or not view.generation:
        return key
    import hashlib

    return hashlib.sha256(f"{view.generation}:{key}".encode()).hexdigest()


@contextmanager
def publication_context(view: PublicationView) -> Iterator[PublicationView]:
    token = _VIEW.set(view)
    try:
        yield view
    finally:
        _VIEW.reset(token)


class ContextThreadPoolExecutor(ThreadPoolExecutor):
    """Carry publication and resource contexts into every submitted task."""

    def submit(self, fn, /, *args, **kwargs):
        context = contextvars.copy_context()
        return super().submit(context.run, fn, *args, **kwargs)


@dataclass(frozen=True)
class PublicationStore:
    root: Path
    managed: tuple[str, ...]

    def __post_init__(self) -> None:
        paths = tuple(Path(item) for item in self.managed)
        if any(path.is_absolute() or ".." in path.parts or not path.parts for path in paths):
            raise PublicationError("Publication outputs must be relative paths within the project")
        if any(left != right and left in right.parents for left in paths for right in paths):
            raise PublicationError("Publication outputs must not overlap")

    @property
    def directory(self) -> Path:
        return self.root / "data/publications"

    def view(self, requested_generation: str | None = None) -> PublicationView:
        pointer = self.directory / "current.json"
        if not pointer.exists():
            return PublicationView(self.root, self.directory, None, self.managed)
        try:
            data = json.loads(pointer.read_text(encoding="utf-8"))
            generation = self._valid_id(data["generation"])
            if requested_generation and requested_generation != generation:
                generation = self._valid_id(requested_generation)
                if generation not in data.get("committed_generations", [data.get("previous")]):
                    raise ValueError("generation was never published")
            state = json.loads((self.directory / generation / "state.json").read_text(encoding="utf-8"))
            if state.get("status") != "validated":
                raise ValueError("generation is not committed")
            if not (self.directory / generation / "tree").is_dir():
                raise ValueError("missing generation tree")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise PublicationError("Committed publication is unavailable") from exc
        return PublicationView(self.root, self.directory, generation, self.managed)

    @staticmethod
    def _valid_id(value: str) -> str:
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
            raise PublicationError("Invalid publication generation")
        return value

    def _clone(self, source: PublicationView, generation: str) -> PublicationView:
        from quant.infrastructure.artifact_registry import ArtifactRegistry

        if (self.directory / generation).exists():
            raise PublicationError("Publication generation already exists")
        registry = ArtifactRegistry(self.root)
        registry.register(
            self.directory / generation, producer="publication",
            input_versions={"previous": source.generation or "legacy"},
            retention_class="rebuildable", state="building",
        )
        target = self.directory / generation / "tree"
        target.mkdir(parents=True, exist_ok=False)
        try:
            for relative in self.managed:
                origin = source.resolve(self.root / relative)
                destination = target / relative
                if origin.is_symlink():
                    raise PublicationError(f"Publication source cannot be a symlink: {relative}")
                if origin.is_dir():
                    if any(child.is_symlink() for child in origin.rglob("*")):
                        raise PublicationError(f"Publication source contains symlinks: {relative}")
                    # Copies isolate legacy writers that bypass atomic rename.
                    shutil.copytree(origin, destination)
                elif origin.is_file():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(origin, destination)
        except BaseException:
            _atomic_json(target.parent / "state.json", {"status": "aborted"})
            registry.retire(target.parent)
            raise
        return PublicationView(self.root, self.directory, generation, self.managed, True)

    @contextmanager
    def begin(self, run_id: str) -> Iterator[PublicationView]:
        from quant.infrastructure.artifact_registry import ArtifactRegistry

        generation = self._valid_id(run_id)
        registry = ArtifactRegistry(self.root)
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / "writer.lock").open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise PublicationError("Another publication is already running") from exc
            try:
                previous = self.view()
                if previous.generation is None:
                    baseline = self._clone(previous, f"baseline-{uuid.uuid4().hex}")
                    _atomic_json(self.directory / baseline.generation / "state.json", {
                        "status": "validated", "kind": "legacy_baseline",
                    })
                    registry.commit("publication:current", self.directory / baseline.generation)
                    _atomic_json(self.directory / "current.json", {
                        "generation": baseline.generation, "kind": "legacy_baseline",
                    })
                    previous = self.view()
                staged = self._clone(previous, generation)
                _atomic_json(self.directory / generation / "state.json", {
                    "status": "staging", "previous": previous.generation,
                })
                try:
                    with registry.lease((self.directory / generation,), owner=f"publication:{generation}", kind="build"), publication_context(staged):
                        yield staged
                finally:
                    if self.view().generation != staged.generation:
                        _atomic_json(self.directory / generation / "state.json", {
                            "status": "aborted", "previous": previous.generation,
                        })
                        registry.retire(self.directory / generation)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def commit(self, view: PublicationView, audit: dict[str, Any]) -> None:
        from quant.infrastructure.artifact_registry import ArtifactRegistry

        if not view.writable or view.directory != self.directory or not view.generation:
            raise PublicationError("Invalid publication owner")
        freshness = audit.get("freshness_audit") or {}
        if (
            audit.get("status") != "success"
            or audit.get("refresh_node_ids")
            or freshness.get("status") != "success"
            or freshness.get("failures")
        ):
            raise PublicationError("Publication requires a successful strict freshness audit")
        _atomic_json(self.directory / view.generation / "state.json", {
            "status": "validated", "audit": audit,
        })
        pointer_path = self.directory / "current.json"
        previous = json.loads(pointer_path.read_text(encoding="utf-8"))
        registry = ArtifactRegistry(self.root)
        registry.commit("publication:current", self.directory / view.generation)
        if previous.get("generation"):
            registry.retire(self.directory / previous["generation"])
        _atomic_json(self.directory / "current.json", {
            "generation": view.generation,
            "previous": previous.get("generation"),
            "committed_generations": sorted(set(previous.get("committed_generations", [])) | {previous["generation"], view.generation}),
            "target_trade_date": audit.get("target_trade_date"),
            "kind": "validated",
        })
