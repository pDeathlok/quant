from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

import pandas as pd


def _temporary_path(target: Path) -> Path:
    suffix = target.suffix
    return target.with_name(f".{target.stem}.{os.getpid()}.{uuid.uuid4().hex}.tmp{suffix}")


def _publish(temp_path: Path, target: Path) -> None:
    os.replace(temp_path, target)


def atomic_write_parquet(frame: pd.DataFrame, target: Path, **kwargs: Any) -> Path:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temporary_path(target)
    try:
        frame.to_parquet(temp_path, **kwargs)
        _publish(temp_path, target)
    finally:
        temp_path.unlink(missing_ok=True)
    return target


def atomic_write_csv(frame: pd.DataFrame, target: Path, **kwargs: Any) -> Path:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temporary_path(target)
    try:
        frame.to_csv(temp_path, **kwargs)
        _publish(temp_path, target)
    finally:
        temp_path.unlink(missing_ok=True)
    return target


def atomic_write_json(payload: Any, target: Path, *, indent: int = 2) -> Path:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temporary_path(target)
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=indent, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _publish(temp_path, target)
    finally:
        temp_path.unlink(missing_ok=True)
    return target


def atomic_write_text(text: str, target: Path) -> Path:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temporary_path(target)
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        _publish(temp_path, target)
    finally:
        temp_path.unlink(missing_ok=True)
    return target


def atomic_link_or_copy(source: Path, target: Path) -> Path:
    """Atomically publish an alias, preferring a zero-copy hard link."""

    source = Path(source)
    target = Path(target)
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temporary_path(target)
    try:
        try:
            os.link(source, temp_path)
        except OSError:
            shutil.copy2(source, temp_path)
        _publish(temp_path, target)
    finally:
        temp_path.unlink(missing_ok=True)
    return target
