"""Release contract and selector adapter for the unified left-side ranker."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from quant.features.canonical_factor_names import find_forbidden_aliases_in_payload
from quant.features.left_side_factor_contract import (
    LEFT_SIDE_ARTIFACT_SCHEMA_VERSION,
    LEFT_SIDE_FACTOR_CONTRACT_SHA256,
    LEFT_SIDE_FEATURE_SCHEMA_VERSION,
    LEFT_SIDE_MODEL_INPUT_CONTRACT_SHA256,
    LEFT_SIDE_SCORE_SCHEMA_VERSION,
    LEFT_SIDE_SCORING_INPUT_CONTRACT_SHA256,
)
from quant.research.left_side_unified_features import (
    LEFT_SIDE_RULE_FEATURE_SCHEMA_VERSION,
    LEFT_SIDE_SIGNALS,
)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LEFT_SIDE_RANKING_CONFIG_PATH = (
    PROJECT_ROOT / "configs/strategies/left_side_unified.yaml"
)


@dataclass(frozen=True)
class LeftSideRankingPaths:
    artifact: Path
    artifact_manifest: Path
    project_feature_cache: Path
    project_feature_manifest: Path
    signal_cache: Path
    b1_gate_cache: Path
    family_signal_cache: Path
    market_data_root: Path
    feature_output: Path
    feature_manifest: Path
    score_output: Path
    score_manifest: Path
    ranking_decision: Path


@dataclass(frozen=True)
class LeftSideRankingConfig:
    release_id: str
    enabled: bool
    strategy_keys: tuple[str, ...]
    overlap_policy: str
    failure_policy: str
    score_field: str
    normalized_score_field: str
    score_normalization: str
    factor_workers: int
    paths: LeftSideRankingPaths


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"left-side ranking config {field} must be a mapping")
    return value


def _safe_path(project_root: Path, value: object) -> Path:
    path = Path(str(value or ""))
    resolved = path.resolve() if path.is_absolute() else (project_root / path).resolve()
    try:
        resolved.relative_to(project_root.resolve())
    except ValueError as exc:
        raise ValueError(f"left-side ranking path escapes project root: {path}") from exc
    return resolved


def load_left_side_ranking_config(
    project_root: Path = PROJECT_ROOT,
    config_path: Path | str = DEFAULT_LEFT_SIDE_RANKING_CONFIG_PATH,
) -> LeftSideRankingConfig:
    source = Path(config_path)
    if not source.is_absolute():
        source = project_root / source
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if find_forbidden_aliases_in_payload(payload):
        raise ValueError("left-side ranking config contains forbidden factor aliases")
    release = _mapping(payload.get("release"), field="release")
    selector = _mapping(payload.get("selector"), field="selector")
    model = _mapping(payload.get("model"), field="model")
    paths = _mapping(payload.get("paths"), field="paths")
    expected = {
        "artifact_schema_version": LEFT_SIDE_ARTIFACT_SCHEMA_VERSION,
        "feature_schema_version": LEFT_SIDE_FEATURE_SCHEMA_VERSION,
        "rule_factor_schema_version": LEFT_SIDE_RULE_FEATURE_SCHEMA_VERSION,
        "score_schema_version": LEFT_SIDE_SCORE_SCHEMA_VERSION,
        "factor_contract_sha256": LEFT_SIDE_FACTOR_CONTRACT_SHA256,
        "model_input_contract_sha256": LEFT_SIDE_MODEL_INPUT_CONTRACT_SHA256,
        "scoring_input_contract_sha256": LEFT_SIDE_SCORING_INPUT_CONTRACT_SHA256,
    }
    drift = {
        key: (model.get(key), expected_value)
        for key, expected_value in expected.items()
        if model.get(key) != expected_value
    }
    if drift:
        raise ValueError(f"left-side ranking model contract drifted: {drift}")
    strategy_keys = tuple(str(value) for value in (selector.get("strategy_keys") or ()))
    if strategy_keys != LEFT_SIDE_SIGNALS:
        raise ValueError("left-side ranking strategy order drifted")
    if selector.get("overlap_policy") != "right_side_precedence":
        raise ValueError("left-side ranking overlap policy is unsupported")
    if selector.get("failure_policy") != "fail_closed":
        raise ValueError("left-side ranking must fail closed")
    if selector.get("normalization") != "daily_cross_section_percentile_v1":
        raise ValueError("left-side ranking normalization drifted")
    factor_workers = int(selector.get("factor_workers") or 1)
    if not 1 <= factor_workers <= 32:
        raise ValueError("left-side ranking factor_workers must be in [1, 32]")
    return LeftSideRankingConfig(
        release_id=str(release.get("id") or ""),
        enabled=selector.get("enabled") is True,
        strategy_keys=strategy_keys,
        overlap_policy=str(selector["overlap_policy"]),
        failure_policy=str(selector["failure_policy"]),
        score_field=str(selector.get("score_field") or ""),
        normalized_score_field=str(selector.get("normalized_score_field") or ""),
        score_normalization=str(selector.get("normalization") or ""),
        factor_workers=factor_workers,
        paths=LeftSideRankingPaths(
            artifact=_safe_path(project_root, model.get("artifact")),
            artifact_manifest=_safe_path(project_root, model.get("artifact_manifest")),
            project_feature_cache=_safe_path(project_root, paths.get("project_feature_cache")),
            project_feature_manifest=_safe_path(project_root, paths.get("project_feature_manifest")),
            signal_cache=_safe_path(project_root, paths.get("signal_cache")),
            b1_gate_cache=_safe_path(project_root, paths.get("b1_gate_cache")),
            family_signal_cache=_safe_path(
                project_root, paths.get("family_signal_cache")
            ),
            market_data_root=_safe_path(project_root, paths.get("market_data_root")),
            feature_output=_safe_path(project_root, paths.get("feature_output")),
            feature_manifest=_safe_path(project_root, paths.get("feature_manifest")),
            score_output=_safe_path(project_root, paths.get("score_output")),
            score_manifest=_safe_path(project_root, paths.get("score_manifest")),
            ranking_decision=_safe_path(project_root, paths.get("ranking_decision")),
        ),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_left_side_ranking_frame(
    signal_date: str | pd.Timestamp,
    *,
    config: LeftSideRankingConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not config.enabled:
        raise RuntimeError("left-side unified ranking is disabled")
    try:
        manifest = json.loads(config.paths.score_manifest.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise RuntimeError("left-side score manifest is unavailable") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError("left-side score manifest must be a mapping")
    target = pd.Timestamp(signal_date).date().isoformat()
    if manifest.get("schema_version") != LEFT_SIDE_SCORE_SCHEMA_VERSION:
        raise RuntimeError("left-side score schema mismatch")
    if manifest.get("target_date") != target:
        raise RuntimeError("left-side score manifest is stale")
    if manifest.get("factor_contract_sha256") != LEFT_SIDE_FACTOR_CONTRACT_SHA256:
        raise RuntimeError("left-side score factor contract mismatch")
    if find_forbidden_aliases_in_payload(manifest):
        raise RuntimeError("left-side score manifest contains forbidden aliases")
    if manifest.get("output_sha256") != _sha256(config.paths.score_output):
        raise RuntimeError("left-side score parquet checksum mismatch")
    frame = pd.read_parquet(config.paths.score_output)
    required = {
        "symbol",
        "date",
        config.score_field,
        config.normalized_score_field,
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"left-side score output missing columns: {sorted(missing)}")
    dates = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype(str)
    if not dates.eq(target).all() or frame["symbol"].astype(str).duplicated().any():
        raise RuntimeError("left-side score keys are stale or duplicated")
    raw = pd.to_numeric(frame[config.score_field], errors="coerce").to_numpy(float)
    normalized = pd.to_numeric(
        frame[config.normalized_score_field], errors="coerce"
    ).to_numpy(float)
    if not np.isfinite(raw).all() or not np.isfinite(normalized).all():
        raise RuntimeError("left-side scores must be finite")
    if ((raw < 0.0) | (raw > 1.0)).any() or (
        (normalized < 0.0) | (normalized > 100.0)
    ).any():
        raise RuntimeError("left-side scores are outside their contracts")
    return frame, manifest


def load_left_side_ranking_candidates(
    signal_date: str | pd.Timestamp,
    *,
    config: LeftSideRankingConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the validated score rows needed to materialize selector candidates."""

    frame, manifest = _load_left_side_ranking_frame(signal_date, config=config)
    missing = set(LEFT_SIDE_SIGNALS) - set(frame.columns)
    if missing:
        raise RuntimeError(
            f"left-side score output missing strategy flags: {sorted(missing)}"
        )
    candidates = frame[
        [
            "symbol",
            "date",
            *LEFT_SIDE_SIGNALS,
            config.score_field,
            config.normalized_score_field,
        ]
    ].copy()
    candidates[list(LEFT_SIDE_SIGNALS)] = (
        candidates[list(LEFT_SIDE_SIGNALS)].fillna(False).astype(bool)
    )
    if not candidates[list(LEFT_SIDE_SIGNALS)].any(axis=1).all():
        raise RuntimeError("left-side score output contains candidates without a strategy")
    return candidates, manifest


def load_left_side_ranking_scores(
    signal_date: str | pd.Timestamp,
    *,
    config: LeftSideRankingConfig,
) -> tuple[dict[str, tuple[float, float]], dict[str, Any]]:
    frame, manifest = _load_left_side_ranking_frame(signal_date, config=config)
    raw = pd.to_numeric(frame[config.score_field], errors="coerce").to_numpy(float)
    normalized = pd.to_numeric(
        frame[config.normalized_score_field], errors="coerce"
    ).to_numpy(float)
    return dict(
        zip(frame["symbol"].astype(str), zip(raw, normalized), strict=True)
    ), manifest


DEFAULT_LEFT_SIDE_RANKING_CONFIG = load_left_side_ranking_config()


__all__ = [
    "DEFAULT_LEFT_SIDE_RANKING_CONFIG",
    "DEFAULT_LEFT_SIDE_RANKING_CONFIG_PATH",
    "LeftSideRankingConfig",
    "LeftSideRankingPaths",
    "load_left_side_ranking_config",
    "load_left_side_ranking_candidates",
    "load_left_side_ranking_scores",
]
