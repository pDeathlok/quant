"""Configuration boundary for factor governance and execution policy."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FACTOR_GOVERNANCE_CONFIG = (
    PROJECT_ROOT / "configs" / "factors" / "governance.json"
)


def factor_governance_config_path() -> Path:
    configured = os.getenv("FACTOR_GOVERNANCE_CONFIG")
    return Path(configured).expanduser() if configured else DEFAULT_FACTOR_GOVERNANCE_CONFIG


@lru_cache(maxsize=4)
def load_factor_governance_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the declarative policy used by the registry and execution planner."""

    resolved = Path(path) if path is not None else factor_governance_config_path()
    if not resolved.is_file():
        raise FileNotFoundError(f"factor governance config is missing: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "factor_governance_v1":
        raise ValueError("factor governance config schema_version must be factor_governance_v1")
    for key, expected_type in (
        ("execution", dict),
        ("calculators", dict),
        ("factor_overrides", dict),
        ("factor_extensions", list),
    ):
        if not isinstance(payload.get(key), expected_type):
            raise ValueError(f"factor governance config {key} must be {expected_type.__name__}")
    return payload


def factor_governance_config_sha256(
    config: dict[str, Any] | None = None,
) -> str:
    payload = config or load_factor_governance_config()
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

