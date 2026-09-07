"""Lazy process-owned SQL engines; callers close connections, not engines."""

from __future__ import annotations

import atexit
import os
import threading
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine


_engine_lock = threading.Lock()
_engine_pid = os.getpid()
_engines: dict[tuple[str, tuple[tuple[str, int], ...]], Engine] = {}


def _reset_lock_after_fork() -> None:
    global _engine_lock
    # A different parent thread may have owned the inherited lock at fork time.
    _engine_lock = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_lock_after_fork)


def _check_process() -> None:
    global _engine_pid
    pid = os.getpid()
    if pid != _engine_pid:
        # Detach inherited pools without closing the parent's DBAPI connections.
        for engine in _engines.values():
            engine.dispose(close=False)
        _engines.clear()
        _engine_pid = pid


def get_sql_engine(
    sql_url: str, *, connect_args: Mapping[str, int] | None = None
) -> Engine:
    """Reuse an engine for this process and configuration, never across forks.

    Use ``engine.connect()`` or ``engine.begin()`` as a context manager. Do not
    call ``dispose()`` per query; the registry owns the engine's lifetime.
    """
    from sqlalchemy import create_engine

    arguments = dict(connect_args or {})
    key = (sql_url, tuple(sorted(arguments.items())))
    with _engine_lock:
        _check_process()
        engine = _engines.get(key)
        if engine is None:
            engine = create_engine(
                sql_url,
                connect_args=arguments,
                pool_pre_ping=True,
                pool_recycle=300,
            )
            _engines[key] = engine
        return engine


def dispose_sql_engines() -> None:
    """Release process-owned pools at shutdown or isolated-test teardown only."""
    with _engine_lock:
        _check_process()
        for engine in _engines.values():
            engine.dispose()
        _engines.clear()


atexit.register(dispose_sql_engines)
