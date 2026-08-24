"""Production adapter for the unified right-side ranking score.

This module intentionally reuses the already-audited causal feature and score
builders from the isolated shadow routine, while publishing to distinct
production paths and enforcing a production-only artifact schema.  It is
dormant until the selector source switch is explicitly promoted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from quant.application.selector_ranking import (
    NORMALIZED_RANKING_SCORE_FIELD,
    RANKING_NORMALIZATION_SCHEMA_VERSION,
    RIGHT_SIDE_PRODUCTION_ARTIFACT_SCHEMA_VERSION,
    RIGHT_SIDE_PRODUCTION_SCORE_SCHEMA_VERSION,
    SelectorRankingConfig,
    SelectorRankingSource,
    load_right_side_ranking_scores,
    load_selector_ranking_config,
    validate_selector_promotion_approval,
)
from quant.data.atomic_io import atomic_write_json, atomic_write_parquet
from quant.features.canonical_factor_names import (
    assert_no_forbidden_factor_names,
    find_forbidden_aliases_in_payload,
)
from quant.features.right_side_factor_contract import (
    RIGHT_SIDE_SHADOW_FACTOR_CONTRACT_SHA256,
    RIGHT_SIDE_SHADOW_IDENTITY_COLUMNS,
    RIGHT_SIDE_SHADOW_MODEL_INPUT_COLUMNS,
    factor_contract_sha256,
)
from quant.research.right_side_unified_features import (
    RULE_FEATURE_COLUMNS,
    RULE_FEATURE_COLUMNS_SHA256,
    rule_feature_columns_sha256,
)
from quant.routine.paths import PROJECT_ROOT
from quant.routine.right_side_unified_shadow import (
    DEFAULT_CONFIG_PATH as DEFAULT_SHADOW_CONFIG_PATH,
    ShadowPaths,
    ShadowReleaseConfig,
    _load_shadow_bundle,
    build_right_side_shadow_features,
    load_shadow_release_config,
)


DEFAULT_PRODUCTION_SOURCE_SHADOW = (
    PROJECT_ROOT
    / "models/research/right_side_unified_canonical_v5_rule113/shadow/ranking.joblib"
)
DEFAULT_NORMALIZATION_REFERENCE = (
    PROJECT_ROOT
    / "reports/research/right_side_unified_canonical_v5_rule113/test_predictions.parquet"
)
NORMALIZATION_REFERENCE_PREDICTION_COLUMN = (
    "pred_unified_long_task_deep"
)
NORMALIZATION_QUANTILE_COUNT = 1001


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise RuntimeError(f"right-side production manifest is unavailable: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("right-side production manifest must be a mapping")
    return payload


def _atomic_dump_joblib(value: object, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        joblib.dump(value, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _promoted_config_for_staging(
    project_root: Path = PROJECT_ROOT,
) -> SelectorRankingConfig:
    config = load_selector_ranking_config(project_root)
    return replace(
        config,
        source=SelectorRankingSource.RIGHT_SIDE_UNIFIED,
        promotion_enabled=True,
    )


def _build_frozen_score_normalization(
    reference_path: Path = DEFAULT_NORMALIZATION_REFERENCE,
) -> dict[str, Any]:
    columns = [
        "fold",
        "entry_mode",
        "horizon",
        "label",
        NORMALIZATION_REFERENCE_PREDICTION_COLUMN,
    ]
    frame = pd.read_parquet(reference_path, columns=columns)
    mask = (
        frame["fold"].astype(str).eq("B")
        & frame["entry_mode"].astype(str).eq("next_close")
        & pd.to_numeric(frame["horizon"], errors="coerce").eq(5)
        & frame["label"].astype(str).eq("good_path5")
    )
    values = pd.to_numeric(
        frame.loc[mask, NORMALIZATION_REFERENCE_PREDICTION_COLUMN],
        errors="coerce",
    ).to_numpy(float)
    values = values[np.isfinite(values)]
    if len(values) < 1000 or ((values < 0.0) | (values > 1.0)).any():
        raise RuntimeError("ranking normalization reference is missing or invalid")
    percentiles = np.linspace(0.0, 100.0, NORMALIZATION_QUANTILE_COUNT)
    quantiles = np.quantile(values, percentiles / 100.0)
    return {
        "schema_version": RANKING_NORMALIZATION_SCHEMA_VERSION,
        "source_path": reference_path.resolve()
        .relative_to(PROJECT_ROOT.resolve())
        .as_posix(),
        "source_sha256": _sha256(reference_path),
        "source_fold": "B",
        "entry_mode": "next_close",
        "horizon": 5,
        "label": "good_path5",
        "prediction_column": NORMALIZATION_REFERENCE_PREDICTION_COLUMN,
        "reference_rows": int(len(values)),
        "percentiles": percentiles.tolist(),
        "quantiles": quantiles.tolist(),
        "production_threshold_mode": "none_rank_only",
        "research_threshold_reference": 0.3490536110991524,
    }


def _normalize_ranking_scores(
    probabilities: np.ndarray,
    contract: Mapping[str, Any],
) -> np.ndarray:
    if contract.get("schema_version") != RANKING_NORMALIZATION_SCHEMA_VERSION:
        raise RuntimeError("ranking normalization schema mismatch")
    quantiles = np.asarray(contract.get("quantiles") or (), dtype=float)
    percentiles = np.asarray(contract.get("percentiles") or (), dtype=float)
    if (
        len(quantiles) != NORMALIZATION_QUANTILE_COUNT
        or len(percentiles) != NORMALIZATION_QUANTILE_COUNT
        or not np.isfinite(quantiles).all()
        or not np.isfinite(percentiles).all()
        or np.any(np.diff(quantiles) < 0.0)
        or np.any(np.diff(percentiles) <= 0.0)
    ):
        raise RuntimeError("ranking normalization quantile contract is invalid")
    unique_quantiles, first = np.unique(quantiles[::-1], return_index=True)
    matching_percentiles = percentiles[::-1][first]
    order = np.argsort(unique_quantiles, kind="stable")
    normalized = np.interp(
        probabilities,
        unique_quantiles[order],
        matching_percentiles[order],
        left=0.0,
        right=100.0,
    )
    return np.clip(normalized, 0.0, 100.0)


def _normalization_summary(contract: Mapping[str, Any]) -> dict[str, Any]:
    encoded = json.dumps(
        dict(contract), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        key: value
        for key, value in contract.items()
        if key not in {"percentiles", "quantiles"}
    } | {
        "quantile_count": len(contract.get("quantiles") or ()),
        "contract_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def stage_production_ranking_release(
    *,
    source_shadow: Path | str = DEFAULT_PRODUCTION_SOURCE_SHADOW,
    project_root: Path = PROJECT_ROOT,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create a checksum-pinned production bundle before the source cutover.

    The live config may still point to ``legacy_z_skill`` while this green
    artifact is staged and validated.  Only the later atomic config change
    exposes it to the selector.
    """

    config = _promoted_config_for_staging(project_root)
    approval = validate_selector_promotion_approval(config)
    source_path = Path(source_shadow)
    if not source_path.is_absolute():
        source_path = project_root / source_path
    source_path = source_path.resolve()
    shadow_config = load_shadow_release_config(project_root, DEFAULT_SHADOW_CONFIG_PATH)
    if source_path != shadow_config.paths.artifact.resolve():
        raise RuntimeError("production staging source must be the accepted shadow artifact")
    shadow_bundle, shadow_sha = _load_shadow_bundle(
        shadow_config,
        project_root=project_root,
    )
    selected = str(approval.get("selected_research_candidate") or "")
    if shadow_bundle.get("selected_candidate") != selected:
        raise RuntimeError("production staging candidate differs from shadow artifact")
    research = approval.get("research_decision") or {}
    if shadow_bundle.get("ranking_decision_sha256") != research.get("sha256"):
        raise RuntimeError("production staging shadow is not bound to approved research")
    if config.paths.artifact.exists() and not overwrite:
        raise FileExistsError(
            f"production artifact already exists; pass overwrite: {config.paths.artifact}"
        )
    features = tuple(str(value) for value in (shadow_bundle.get("features") or ()))
    selected_rules = tuple(
        str(value) for value in (shadow_bundle.get("selected_rule_factor_columns") or ())
    )
    bundle = {
        **shadow_bundle,
        "schema_version": RIGHT_SIDE_PRODUCTION_ARTIFACT_SCHEMA_VERSION,
        "score_field": "ranking_score",
        "lifecycle": "production",
        "playbook_coupling": "independent",
        "model_input_contract_sha256": factor_contract_sha256(
            features,
            schema_version=RIGHT_SIDE_PRODUCTION_ARTIFACT_SCHEMA_VERSION,
        ),
        "promotion_approval_sha256": _sha256(config.paths.promotion_approval),
        "source_shadow_artifact_sha256": shadow_sha,
        "score_normalization": _build_frozen_score_normalization(),
        "probability_calibration": "embedded_event_level_platt",
        "production_threshold_mode": "none_rank_only",
    }
    _atomic_dump_joblib(bundle, config.paths.artifact)
    artifact_sha = _sha256(config.paths.artifact)
    manifest = {
        "status": "success",
        "schema_version": RIGHT_SIDE_PRODUCTION_ARTIFACT_SCHEMA_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "lifecycle": "production",
        "score_field": "ranking_score",
        "playbook_coupling": "independent",
        "selected_candidate": selected,
        "selected_rule_factor_count": len(selected_rules),
        "selected_rule_factor_columns": list(selected_rules),
        "selected_rule_factor_columns_sha256": bundle.get(
            "selected_rule_factor_columns_sha256"
        ),
        "factor_contract_sha256": RIGHT_SIDE_SHADOW_FACTOR_CONTRACT_SHA256,
        "promotion_approval_sha256": _sha256(config.paths.promotion_approval),
        "source_shadow_artifact_sha256": shadow_sha,
        "score_normalization": _normalization_summary(
            bundle["score_normalization"]
        ),
        "probability_calibration": bundle["probability_calibration"],
        "production_threshold_mode": bundle["production_threshold_mode"],
        "replaced_signals": list(config.replaced_signals),
        "preserved_legacy_signals": list(config.preserved_legacy_signals),
        "models": {
            "ranking": {
                "path": config.paths.artifact.resolve()
                .relative_to(project_root.resolve())
                .as_posix(),
                "sha256": artifact_sha,
            }
        },
    }
    atomic_write_json(manifest, config.paths.artifact_manifest)
    validate_production_ranking_artifact(config, project_root=project_root)
    return {
        "status": "success",
        "artifact": str(config.paths.artifact),
        "artifact_sha256": artifact_sha,
        "manifest": str(config.paths.artifact_manifest),
        "source_shadow_artifact_sha256": shadow_sha,
        "selector_source_changed": False,
    }


def _production_input_snapshot(
    config: SelectorRankingConfig,
    target: pd.Timestamp,
    *,
    project_root: Path,
) -> tuple[str, dict[str, Any]]:
    month_partition = (
        config.paths.market_data_root
        / "daily_partitioned"
        / f"year_month={target.strftime('%Y%m')}"
        / "data.parquet"
    )
    required_files = {
        "artifact": config.paths.artifact,
        "approval": config.paths.promotion_approval,
        "z_signal_cache": config.paths.z_signal_cache,
        "family_signal_cache": config.paths.family_signal_cache,
        "market_month_partition": month_partition,
    }
    contract_files = (
        "configs/strategies/right_side_ranking_selector.yaml",
        "src/quant/application/selector_ranking.py",
        "src/quant/features/right_side_factor_contract.py",
        "src/quant/features/project_factor_layer.py",
        "src/quant/research/right_side_unified_features.py",
        "src/quant/research/right_side_unified_signals.py",
        "src/quant/routine/right_side_unified_shadow.py",
        "src/quant/routine/right_side_unified_production.py",
    )
    for relative in contract_files:
        required_files[f"contract:{relative}"] = project_root / relative
    missing = [name for name, path in required_files.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"production ranking inputs are missing: {missing}")
    payload = {
        "target_date": target.date().isoformat(),
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in required_files.items()
        },
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), payload


def validate_production_ranking_artifact(
    config: SelectorRankingConfig,
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Validate checksum, semantic output, and frozen model-input contracts."""

    if config.source != SelectorRankingSource.RIGHT_SIDE_UNIFIED:
        raise RuntimeError("unified production ranker is not the configured selector source")
    if not config.promotion_enabled:
        raise RuntimeError("unified production ranker promotion is disabled")
    approval = validate_selector_promotion_approval(config)
    manifest = _load_json(config.paths.artifact_manifest)
    if manifest.get("schema_version") != RIGHT_SIDE_PRODUCTION_ARTIFACT_SCHEMA_VERSION:
        raise RuntimeError("right-side production artifact manifest schema mismatch")
    if manifest.get("lifecycle") != "production":
        raise RuntimeError("right-side production artifact lifecycle mismatch")
    if manifest.get("score_field") != "ranking_score":
        raise RuntimeError("right-side production artifact must publish ranking_score")
    if manifest.get("playbook_coupling") != "independent":
        raise RuntimeError("right-side production artifact must be playbook-independent")
    if tuple(manifest.get("replaced_signals") or ()) != config.replaced_signals:
        raise RuntimeError("right-side production replaced_signals mismatch")
    if tuple(manifest.get("preserved_legacy_signals") or ()) != (
        config.preserved_legacy_signals
    ):
        raise RuntimeError("right-side production preserved_legacy_signals mismatch")
    forbidden_manifest = find_forbidden_aliases_in_payload(manifest)
    if forbidden_manifest:
        raise RuntimeError(
            "right-side production manifest contains forbidden factor aliases: "
            f"{forbidden_manifest}"
        )
    if manifest.get("factor_contract_sha256") != RIGHT_SIDE_SHADOW_FACTOR_CONTRACT_SHA256:
        raise RuntimeError("right-side production artifact factor contract mismatch")
    if manifest.get("selected_candidate") != approval.get(
        "selected_research_candidate"
    ):
        raise RuntimeError("right-side production artifact differs from promotion approval")
    if manifest.get("promotion_approval_sha256") != _sha256(
        config.paths.promotion_approval
    ):
        raise RuntimeError("right-side production approval checksum mismatch")
    model_item = (manifest.get("models") or {}).get("ranking") or {}
    relative = config.paths.artifact.resolve().relative_to(project_root.resolve()).as_posix()
    if model_item.get("path") != relative:
        raise RuntimeError("right-side production artifact path mismatch")
    if not config.paths.artifact.is_file():
        raise RuntimeError("right-side production artifact is missing")
    digest = _sha256(config.paths.artifact)
    if model_item.get("sha256") != digest:
        raise RuntimeError("right-side production artifact checksum mismatch")
    bundle = joblib.load(config.paths.artifact)
    if not isinstance(bundle, dict):
        raise RuntimeError("right-side production artifact must be a bundle mapping")
    if bundle.get("schema_version") != RIGHT_SIDE_PRODUCTION_ARTIFACT_SCHEMA_VERSION:
        raise RuntimeError("right-side production bundle schema mismatch")
    if bundle.get("score_field") != "ranking_score":
        raise RuntimeError("right-side production bundle output semantics mismatch")
    if bundle.get("probability_calibration") != config.probability_calibration:
        raise RuntimeError("right-side production probability calibration mismatch")
    if bundle.get("production_threshold_mode") != config.production_threshold_mode:
        raise RuntimeError("right-side production threshold policy mismatch")
    normalization = bundle.get("score_normalization")
    if not isinstance(normalization, Mapping):
        raise RuntimeError("right-side production score normalization is missing")
    _normalize_ranking_scores(np.asarray([0.0, 0.5, 1.0]), normalization)
    forbidden = {"pred_up5", "pred_up8", "pred_down3"} & set(bundle)
    if forbidden:
        raise RuntimeError(f"right-side ranker contains forbidden output aliases: {sorted(forbidden)}")
    forbidden_bundle = find_forbidden_aliases_in_payload(bundle)
    if forbidden_bundle:
        raise RuntimeError(
            "right-side production bundle contains forbidden factor aliases: "
            f"{forbidden_bundle}"
        )
    if bundle.get("factor_contract_sha256") != RIGHT_SIDE_SHADOW_FACTOR_CONTRACT_SHA256:
        raise RuntimeError("right-side production bundle factor contract mismatch")
    if manifest.get("selected_candidate") != bundle.get("selected_candidate"):
        raise RuntimeError("right-side production selected candidate mismatch")
    if bundle.get("materialized_rule_factor_columns_sha256") != RULE_FEATURE_COLUMNS_SHA256:
        raise RuntimeError("right-side production materialized rule-factor hash mismatch")
    features = tuple(str(value) for value in (bundle.get("features") or ()))
    if not features or len(features) != len(set(features)):
        raise RuntimeError("right-side production model inputs are empty or duplicated")
    assert_no_forbidden_factor_names(
        features,
        context="right-side production artifact inputs",
    )
    unknown = sorted(set(features) - set(RIGHT_SIDE_SHADOW_MODEL_INPUT_COLUMNS))
    missing_identity = sorted(
        set(RIGHT_SIDE_SHADOW_IDENTITY_COLUMNS) - set(features)
    )
    selected_rules = tuple(
        str(value) for value in (bundle.get("selected_rule_factor_columns") or ())
    )
    if (
        int(bundle.get("selected_rule_factor_count") or -1) != len(selected_rules)
        or bundle.get("selected_rule_factor_columns_sha256")
        != rule_feature_columns_sha256(selected_rules)
    ):
        raise RuntimeError("right-side production selected rule-factor contract mismatch")
    if manifest.get("selected_rule_factor_columns") != list(selected_rules):
        raise RuntimeError("right-side production manifest rule-factor list mismatch")
    valid_rules = bool(
        bundle.get("selected_candidate") == "unified_long_task_deep"
        and selected_rules == tuple(RULE_FEATURE_COLUMNS)
    )
    missing_rules = sorted(set(selected_rules) - set(features))
    if unknown or missing_identity or missing_rules or not valid_rules:
        raise RuntimeError(
            "right-side production model input contract mismatch; "
            f"unknown={unknown} missing_identity={missing_identity} "
            f"missing_rules={missing_rules} valid_rules={valid_rules}"
        )
    expected_input_hash = factor_contract_sha256(
        features,
        schema_version=RIGHT_SIDE_PRODUCTION_ARTIFACT_SCHEMA_VERSION,
    )
    if bundle.get("model_input_contract_sha256") != expected_input_hash:
        raise RuntimeError("right-side production model-input hash mismatch")
    if bundle.get("model") is None:
        raise RuntimeError("right-side production bundle has no model")
    model = bundle["model"]
    assert_no_forbidden_factor_names(
        getattr(model, "feature_names_in_", ()),
        context="right-side production model feature_names_in_",
    )
    assert_no_forbidden_factor_names(
        getattr(model, "selected_features_", ()),
        context="right-side production model selected_features_",
    )
    return bundle


def _shadow_compatible_config(config: SelectorRankingConfig) -> ShadowReleaseConfig:
    """Translate production paths for audited shared builders."""

    return ShadowReleaseConfig(
        enabled=True,
        top_n=50,
        history_years=config.history_years,
        paths=ShadowPaths(
            artifact=config.paths.artifact,
            artifact_manifest=config.paths.artifact_manifest,
            ranking_decision=config.paths.artifact_manifest,
            feature_output=config.paths.feature_output,
            feature_manifest=config.paths.feature_manifest,
            score_output=config.paths.score_output,
            score_manifest=config.paths.score_manifest,
            product_output=config.paths.score_output,
            product_manifest=config.paths.score_manifest,
            run_status=config.paths.score_manifest,
            z_signal_cache=config.paths.z_signal_cache,
            family_signal_cache=config.paths.family_signal_cache,
            market_data_root=config.paths.market_data_root,
        ),
        decision_field="status",
        accepted_decisions=("success",),
        selected_candidate_field="selected_candidate",
        eligible_candidates=("unified_long_task_deep",),
        decision_schema_version="right-side-production-replacement-decision-v1",
        production_replacement_field="replace_online",
        factor_workers=config.factor_workers,
    )


def build_right_side_unified_production_features(
    target_date: str,
    *,
    config: SelectorRankingConfig,
) -> dict[str, Any]:
    """Build the exact-date canonical project-v5/rule-v4 production sidecar."""

    return build_right_side_shadow_features(
        target_date,
        config=_shadow_compatible_config(config),
    )


def score_right_side_unified_production(
    target_date: str,
    *,
    config: SelectorRankingConfig,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Publish ranking_score from the validated production bundle."""

    bundle = validate_production_ranking_artifact(config, project_root=project_root)
    target = pd.to_datetime(target_date, errors="raise").normalize()
    feature_manifest = _load_json(config.paths.feature_manifest)
    if (
        feature_manifest.get("status") != "success"
        or feature_manifest.get("target_date") != target.date().isoformat()
        or feature_manifest.get("factor_contract_sha256")
        != RIGHT_SIDE_SHADOW_FACTOR_CONTRACT_SHA256
    ):
        raise RuntimeError("right-side production features are stale or incompatible")
    if feature_manifest.get("output_sha256") != _sha256(config.paths.feature_output):
        raise RuntimeError("right-side production feature checksum mismatch")
    frame = pd.read_parquet(config.paths.feature_output)
    features = tuple(str(value) for value in (bundle.get("features") or ()))
    missing = sorted(set(features) - set(frame.columns))
    if missing:
        raise RuntimeError(f"right-side production features missing model inputs: {missing}")
    if frame.empty:
        probabilities = np.asarray([], dtype=float)
    else:
        predicted = np.asarray(
            bundle["model"].predict_proba(frame[list(features)]),
            dtype=float,
        )
        if predicted.ndim != 2 or predicted.shape != (len(frame), 2):
            raise RuntimeError("right-side production ranker returned invalid probabilities")
        probabilities = predicted[:, 1]
        if (
            not np.isfinite(probabilities).all()
            or ((probabilities < 0.0) | (probabilities > 1.0)).any()
        ):
            raise RuntimeError("right-side production ranker returned invalid scores")
    output_columns = [
        "ts_code",
        "symbol",
        "trade_date",
        "date",
        *RIGHT_SIDE_SHADOW_IDENTITY_COLUMNS,
    ]
    scored = frame[output_columns].copy()
    scored["ranking_score"] = probabilities
    scored[NORMALIZED_RANKING_SCORE_FIELD] = _normalize_ranking_scores(
        probabilities,
        bundle["score_normalization"],
    )
    scored["model_artifact_sha256"] = _sha256(config.paths.artifact)
    atomic_write_parquet(scored, config.paths.score_output, index=False)
    score_manifest = {
        "status": "success",
        "schema_version": RIGHT_SIDE_PRODUCTION_SCORE_SCHEMA_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "target_date": target.date().isoformat(),
        "artifact_schema_version": RIGHT_SIDE_PRODUCTION_ARTIFACT_SCHEMA_VERSION,
        "artifact_path": config.paths.artifact.resolve()
        .relative_to(project_root.resolve())
        .as_posix(),
        "artifact_sha256": _sha256(config.paths.artifact),
        "factor_contract_sha256": RIGHT_SIDE_SHADOW_FACTOR_CONTRACT_SHA256,
        "feature_output_sha256": feature_manifest.get("output_sha256"),
        "source_input_fingerprint": feature_manifest.get(
            "source_input_fingerprint"
        ),
        "candidate_count": int(len(scored)),
        "score_field": "ranking_score",
        "normalized_score_field": NORMALIZED_RANKING_SCORE_FIELD,
        "probability_calibration": config.probability_calibration,
        "score_normalization": _normalization_summary(
            bundle["score_normalization"]
        ),
        "production_threshold_mode": config.production_threshold_mode,
        "selection_policy": config.selection_policy,
        "research_threshold_reference": config.research_threshold_reference,
        "selector_adapter_status": "ready",
        "playbook_coupling": "independent",
        "replaced_signals": list(config.replaced_signals),
        "preserved_legacy_signals": list(config.preserved_legacy_signals),
        "promotion_approval_sha256": _sha256(config.paths.promotion_approval),
        "output": str(config.paths.score_output),
        "output_sha256": _sha256(config.paths.score_output),
    }
    atomic_write_json(score_manifest, config.paths.score_manifest)
    return score_manifest


def validate_right_side_unified_selector_adapter(
    target_date: str,
    *,
    config: SelectorRankingConfig,
) -> dict[str, Any]:
    """Re-open the released score through the selector's strict adapter."""

    scores, manifest = load_right_side_ranking_scores(target_date, config=config)
    return {
        "status": "success",
        "target_date": pd.Timestamp(target_date).date().isoformat(),
        "candidate_count": len(scores),
        "score_field": "ranking_score",
        "artifact_sha256": manifest.get("artifact_sha256"),
        "playbook_coupling": "independent",
    }


def run_right_side_unified_production(
    target_date: str,
    *,
    project_root: Path = PROJECT_ROOT,
    config: SelectorRankingConfig | None = None,
    factor_workers: int | None = None,
) -> dict[str, Any]:
    """Build and publish one exact-date score, reusing an unchanged checkpoint."""

    config = config or load_selector_ranking_config(project_root)
    if factor_workers is not None:
        if not 1 <= factor_workers <= 32:
            raise ValueError("right-side production factor_workers must be in [1, 32]")
        config = replace(config, factor_workers=factor_workers)
    validate_production_ranking_artifact(config, project_root=project_root)
    target = pd.to_datetime(target_date, errors="raise").normalize()
    input_fingerprint, input_snapshot = _production_input_snapshot(
        config,
        target,
        project_root=project_root,
    )
    try:
        feature_manifest = _load_json(config.paths.feature_manifest)
        score_manifest = _load_json(config.paths.score_manifest)
        if (
            feature_manifest.get("status") == "success"
            and feature_manifest.get("target_date") == target.date().isoformat()
            and feature_manifest.get("source_input_fingerprint") == input_fingerprint
            and feature_manifest.get("output_sha256")
            == _sha256(config.paths.feature_output)
            and score_manifest.get("source_input_fingerprint") == input_fingerprint
        ):
            scores, validated_manifest = load_right_side_ranking_scores(
                target.date().isoformat(),
                config=config,
            )
            adapter = validate_right_side_unified_selector_adapter(
                target.date().isoformat(),
                config=config,
            )
            return {
                **validated_manifest,
                "status": "success",
                "checkpoint_reused": True,
                "refresh_reason": "same_target_and_identical_inputs",
                "candidate_count": len(scores),
                "selector_adapter": adapter,
            }
    except (FileNotFoundError, OSError, RuntimeError, ValueError):
        pass
    feature_manifest = build_right_side_unified_production_features(
        target.date().isoformat(),
        config=config,
    )
    feature_manifest = {
        **feature_manifest,
        "source_input_fingerprint": input_fingerprint,
        "source_inputs": input_snapshot,
    }
    atomic_write_json(feature_manifest, config.paths.feature_manifest)
    score_manifest = score_right_side_unified_production(
        target.date().isoformat(),
        config=config,
        project_root=project_root,
    )
    adapter = validate_right_side_unified_selector_adapter(
        target.date().isoformat(),
        config=config,
    )
    return {
        **score_manifest,
        "checkpoint_reused": False,
        "refresh_reason": "new_target_or_changed_inputs",
        "selector_adapter": adapter,
    }


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Unified right-side production ranker")
    subparsers = parser.add_subparsers(dest="command", required=True)
    stage = subparsers.add_parser(
        "stage-production-release",
        help="stage and validate the green production artifact before cutover",
    )
    stage.add_argument(
        "--source-shadow",
        type=Path,
        default=DEFAULT_PRODUCTION_SOURCE_SHADOW,
    )
    stage.add_argument("--overwrite", action="store_true")
    run = subparsers.add_parser("run", help="build or reuse one production date")
    run.add_argument("--target-date", required=True)
    run.add_argument(
        "--precutover",
        action="store_true",
        help="validate production outputs with the staged source before routing changes",
    )
    args = parser.parse_args()
    if args.command == "stage-production-release":
        payload = stage_production_ranking_release(
            source_shadow=args.source_shadow,
            overwrite=args.overwrite,
        )
    else:
        config = _promoted_config_for_staging() if args.precutover else None
        payload = run_right_side_unified_production(
            args.target_date,
            config=config,
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0 if payload.get("status") == "success" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())


__all__ = [
    "build_right_side_unified_production_features",
    "run_right_side_unified_production",
    "score_right_side_unified_production",
    "stage_production_ranking_release",
    "validate_right_side_unified_selector_adapter",
    "validate_production_ranking_artifact",
]
