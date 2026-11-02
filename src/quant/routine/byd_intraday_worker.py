"""Bound BYD provider I/O without sharing vendor SDK sockets with web threads."""
from __future__ import annotations

import multiprocessing as mp
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from quant.data.byd_intraday import DEFAULT_CACHE_DIR
from quant.infrastructure.publication import current_publication, publication_context


def _worker(connection: Any, trade_date: str, cache_dir: Path, publication: Any) -> None:
    from quant.routine.byd_intraday_refresh import refresh_byd_intraday_when_due

    try:
        with publication_context(publication) if publication else nullcontext():
            result = refresh_byd_intraday_when_due(expected_trade_date=trade_date, cache_dir=cache_dir)
        connection.send({"ok": True, "result": result})
    except Exception as exc:
        connection.send({"ok": False, "error": str(exc)})
    finally:
        connection.close()


def refresh_byd_training_isolated(
    *, expected_trade_date: str, cache_dir: Path = DEFAULT_CACHE_DIR, timeout_seconds: float = 900,
) -> dict[str, Any]:
    if timeout_seconds <= 0:
        raise ValueError("BYD refresh timeout must be positive")
    context = mp.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_worker, args=(sender, expected_trade_date, cache_dir, current_publication()),
    )
    process.start()
    sender.close()
    try:
        if not receiver.poll(timeout_seconds):
            raise TimeoutError(f"BYD training refresh exceeded {timeout_seconds}s")
        try:
            message = receiver.recv()
        except EOFError as exc:
            raise RuntimeError("BYD training worker exited without a result") from exc
        if not message["ok"]:
            raise RuntimeError(message["error"])
        return message["result"]
    finally:
        receiver.close()
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
