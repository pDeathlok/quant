"""Versioned production scoring bridge for the shared ETF context."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant.features.short_etf_rank import (
    ARTIFACT, MACRO_SCALES, VERSION, apply_residual, market_matrix, stock_matrix,
)
from quant.infrastructure.publication import publication_path
from quant.routine.paths import PROJECT_ROOT
from quant.routine.short_etf_flow import SNAPSHOT_DIRECTORY, read_short_etf_flow


@lru_cache(maxsize=8)
def _model(path: Path, modified_ns: int, size: int) -> tuple[dict[str, Any], str]:
    content = path.read_bytes()
    data = json.loads(content)
    if data.get("schema") != VERSION or data.get("market_features") != list(MACRO_SCALES):
        raise ValueError("ETF ranker feature contract mismatch")
    return data["sides"], hashlib.sha256(content).hexdigest()


@lru_cache(maxsize=16)
def _context(path: Path, modified_ns: int, size: int, signal_date: str, root: Path) -> dict:
    return read_short_etf_flow(signal_date, project_root=root)


def load_market_context(signal_date: str, project_root: Path = PROJECT_ROOT) -> dict:
    target = pd.Timestamp(signal_date)
    path = publication_path(project_root / SNAPSHOT_DIRECTORY / f"{target:%Y%m%d}.json")
    try:
        stat = path.stat()
    except FileNotFoundError:
        return {}
    source = _context(path, stat.st_mtime_ns, stat.st_size, target.date().isoformat(), project_root)
    return source.get("features") or {}


def market_context_identity(signal_date: str, project_root: Path = PROJECT_ROOT) -> str:
    features = load_market_context(signal_date, project_root)
    values = {key: features.get(key) for key in MACRO_SCALES}
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def model_identity(project_root: Path = PROJECT_ROOT) -> str | None:
    path = project_root / ARTIFACT
    if not path.is_file():
        return None
    stat = path.stat()
    return _model(path, stat.st_mtime_ns, stat.st_size)[1]


def rank_with_etf(
    side: str, frame: pd.DataFrame, base: np.ndarray, signal_date: str, base_sha256: str,
    *, project_root: Path = PROJECT_ROOT,
) -> tuple[np.ndarray, dict]:
    path = project_root / ARTIFACT
    meta = {"schema": VERSION, "status": "model_unavailable"}
    if not path.is_file():
        return base, meta
    stat = path.stat()
    sides, digest = _model(path, stat.st_mtime_ns, stat.st_size)
    model = sides[side]
    meta["artifact_sha256"] = digest
    if model["base_sha256"] != base_sha256:
        return base, {**meta, "status": "base_model_mismatch"}
    if signal_date <= model["trained_through"]:
        return base, {**meta, "status": "before_release_window"}
    if not model["active"]:
        return base, {**meta, "status": "not_admitted"}
    features = load_market_context(signal_date, project_root)
    market, ready = market_matrix(pd.DataFrame([features]))
    if not ready[0]:
        return base, {**meta, "status": "etf_context_unavailable"}
    signals = tuple(model["signals"])
    strength = float(model["strength"])
    coeff = np.asarray(model["coefficients"], dtype=float)
    if (coeff.shape != (2 + 2 * len(signals), len(MACRO_SCALES))
            or not np.isfinite(coeff).all() or not 0 < strength <= 1):
        raise ValueError("ETF ranker coefficients are invalid")
    if not len(base):
        return base, {**meta, "status": "no_candidates"}
    # One market row is shared by every symbol. Only the small stock/signal
    # matrix varies across candidates.
    adjusted = apply_residual(stock_matrix(frame, base, signals),
                              np.broadcast_to(market, (len(base), market.shape[1])),
                              base, coeff, strength)
    return adjusted, {**meta, "status": "applied", "strength": strength,
                      "adjusted_candidates": int(np.count_nonzero(adjusted != base)),
                      "max_abs_adjustment": float(np.max(np.abs(adjusted - base)))}
