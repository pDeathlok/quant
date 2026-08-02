"""Auditable and reproducible metadata for research and backtest runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from quant.data.atomic_io import atomic_write_json


def file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, project_root: Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    try:
        display_path = str(resolved.relative_to(project_root.resolve()))
    except ValueError:
        display_path = str(resolved)
    stat = resolved.stat()
    return {
        "path": display_path,
        "bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "sha256": file_sha256(resolved),
    }


def _git_state(project_root: Path) -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        return {"revision": revision, "dirty": bool(status.strip())}
    except (OSError, subprocess.SubprocessError):
        return {"revision": None, "dirty": None}


def _package_versions(names: Iterable[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def build_research_manifest(
    *,
    strategy_name: str,
    parameters: Mapping[str, Any],
    data_paths: Iterable[Path | str],
    start_date: str,
    end_date: str,
    random_seed: int,
    project_root: Path | str,
    code_paths: Iterable[Path | str] = (),
    run_id: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    start = str(start_date).replace("-", "")
    end = str(end_date).replace("-", "")
    if len(start) != 8 or not start.isdigit() or len(end) != 8 or not end.isdigit():
        raise ValueError("start_date and end_date must be YYYYMMDD")
    if start > end:
        raise ValueError("start_date must be on or before end_date")
    if not strategy_name.strip():
        raise ValueError("strategy_name is required")
    return {
        "schema_version": "research-manifest/v1",
        "run_id": run_id or f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "strategy_name": strategy_name,
        "parameters": dict(parameters),
        "sample": {"start_date": start, "end_date": end},
        "random_seed": int(random_seed),
        "inputs": [_file_record(Path(path), root) for path in data_paths],
        "code_inputs": [_file_record(Path(path), root) for path in code_paths],
        "git": _git_state(root),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": _package_versions(
                ["numpy", "pandas", "scipy", "scikit-learn", "akquant", "tushare"]
            ),
        },
        "extra": dict(extra or {}),
    }


def write_research_manifest(
    output_dir: Path | str,
    **kwargs: Any,
) -> Path:
    manifest = build_research_manifest(**kwargs)
    return atomic_write_json(manifest, Path(output_dir) / "research_manifest.json")
