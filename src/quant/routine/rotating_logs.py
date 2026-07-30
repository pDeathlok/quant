from __future__ import annotations

import io
import logging
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path


DEFAULT_WEBAPP_LOG_MAX_BYTES = 5 * 1024 * 1024
DEFAULT_WEBAPP_LOG_BACKUP_COUNT = 2


class LineLoggingStream(io.TextIOBase):
    """Send print-style stdout/stderr writes through a rotating logger."""

    def __init__(self, logger: logging.Logger, level: int) -> None:
        self._logger = logger
        self._level = level
        self._buffer = ""
        self._lock = threading.Lock()

    @property
    def encoding(self) -> str:
        return "utf-8"

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return False

    def write(self, value: str) -> int:
        if not value:
            return 0
        with self._lock:
            self._buffer += value
            lines = self._buffer.split("\n")
            self._buffer = lines.pop()
            for line in lines:
                if line:
                    self._logger.log(self._level, line.rstrip("\r"))
        return len(value)

    def flush(self) -> None:
        with self._lock:
            if self._buffer:
                self._logger.log(self._level, self._buffer.rstrip("\r"))
                self._buffer = ""
        for handler in self._logger.handlers:
            handler.flush()


def configure_rotating_logging(
    log_path: Path,
    *,
    max_bytes: int = DEFAULT_WEBAPP_LOG_MAX_BYTES,
    backup_count: int = DEFAULT_WEBAPP_LOG_BACKUP_COUNT,
    redirect_streams: bool = True,
) -> RotatingFileHandler:
    """Configure one process-wide rotating file for app, uvicorn, and print output."""

    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    if backup_count <= 0:
        raise ValueError("backup_count must be positive")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )

    root_logger = logging.getLogger()
    for existing in list(root_logger.handlers):
        root_logger.removeHandler(existing)
        if getattr(existing, "_quant_rotating_handler", False):
            existing.close()
    handler._quant_rotating_handler = True
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        logger.propagate = True
        logger.setLevel(logging.INFO)

    if redirect_streams:
        stdout_logger = logging.getLogger("stdout")
        sys.stdout = LineLoggingStream(stdout_logger, logging.INFO)
    return handler
