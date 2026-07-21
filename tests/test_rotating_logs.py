from __future__ import annotations

import logging
from pathlib import Path

import pytest

from quant.routine.rotating_logs import configure_rotating_logging


def test_configure_rotating_logging_caps_files_and_routes_uvicorn(tmp_path: Path) -> None:
    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    original_level = root_logger.level
    log_path = tmp_path / "webapp.log"

    handler = configure_rotating_logging(
        log_path,
        max_bytes=180,
        backup_count=2,
        redirect_streams=False,
    )
    try:
        for index in range(30):
            logging.getLogger("uvicorn.access").info("request-%02d %s", index, "x" * 30)
        handler.flush()

        assert log_path.exists()
        assert (tmp_path / "webapp.log.1").exists()
        assert (tmp_path / "webapp.log.2").exists()
        assert not (tmp_path / "webapp.log.3").exists()
        combined = "".join(path.read_text(encoding="utf-8") for path in tmp_path.glob("webapp.log*"))
        assert "uvicorn.access" in combined
    finally:
        root_logger.removeHandler(handler)
        handler.close()
        for original in original_handlers:
            root_logger.addHandler(original)
        root_logger.setLevel(original_level)


@pytest.mark.parametrize(
    ("max_bytes", "backup_count"),
    [(0, 2), (100, 0), (100, -1)],
)
def test_configure_rotating_logging_rejects_invalid_limits(
    tmp_path: Path,
    max_bytes: int,
    backup_count: int,
) -> None:
    with pytest.raises(ValueError):
        configure_rotating_logging(
            tmp_path / "webapp.log",
            max_bytes=max_bytes,
            backup_count=backup_count,
            redirect_streams=False,
        )


def test_configure_rotating_logging_closes_previous_owned_handler(tmp_path: Path) -> None:
    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    original_level = root_logger.level
    first = configure_rotating_logging(tmp_path / "first.log", redirect_streams=False)
    second = configure_rotating_logging(tmp_path / "second.log", redirect_streams=False)
    try:
        assert first.stream is None
        assert second.stream is not None
    finally:
        root_logger.removeHandler(second)
        second.close()
        for original in original_handlers:
            root_logger.addHandler(original)
        root_logger.setLevel(original_level)
