"""Isolated daily shadow routine for the unified right-side ranking model.

The routine deliberately has no selector integration.  It builds a current
causal feature sidecar, scores a checksum-pinned research bundle, and publishes
an independently versioned candidate product.  Any contract failure blocks the
shadow run while preserving the production short/all workflow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import yaml

from quant.data import MarketDataStore, MarketDataStoreConfig
from quant.data.atomic_io import atomic_link_or_copy, atomic_write_json, atomic_write_parquet
from quant.data.source_merge import normalize_tushare_daily
from quant.features.canonical_factor_names import (
    assert_no_forbidden_factor_names,
    find_forbidden_aliases_in_payload,
    stable_canonical_feature_union,
)
from quant.features.project_factor_layer import (
    PROJECT_FACTOR_SCHEMA_VERSION,
    calculate_project_market_factors,
)
from quant.features.right_side_factor_contract import (
    RIGHT_SIDE_SHADOW_ARTIFACT_SCHEMA_VERSION,
    RIGHT_SIDE_SHADOW_FACTOR_COLUMNS,
    RIGHT_SIDE_SHADOW_FACTOR_CONTRACT_SHA256,
    RIGHT_SIDE_SHADOW_FEATURE_SCHEMA_VERSION,
    RIGHT_SIDE_SHADOW_IDENTITY_COLUMNS,
    factor_contract_sha256,
    right_side_shadow_contract_payload,
    validate_right_side_shadow_factor_contract,
)
from quant.features.variable_library import PROJECT_FACTOR_COLUMNS
from quant.research.right_side_unified import load_signal_universe
from quant.research.right_side_unified_features import (
    RULE_FEATURE_COLUMNS,
    RULE_FEATURE_COLUMNS_SHA256,
    RULE_FEATURE_SCHEMA_VERSION,
    compute_right_side_rule_features,
    rule_feature_columns_sha256,
)
from quant.research.right_side_unified_signals import CANONICAL_SIGNAL_SCHEMA_VERSION
from quant.routine.paths import PROJECT_ROOT


DEFAULT_CONFIG_PATH = Path("configs/strategies/right_side_unified.yaml")
SHADOW_SCOPE = "rightSideShadow"


@dataclass(frozen=True)
class ShadowPaths:
    artifact: Path
    artifact_manifest: Path
    ranking_decision: Path
    feature_output: Path
    feature_manifest: Path
    score_output: Path
    score_manifest: Path
    product_output: Path
    product_manifest: Path
    run_status: Path
    z_signal_cache: Path
    family_signal_cache: Path
    market_data_root: Path


@dataclass(frozen=True)
class ShadowReleaseConfig:
    enabled: bool
    top_n: int
    history_years: int
    paths: ShadowPaths
    decision_field: str
    accepted_decisions: tuple[object, ...]
    selected_candidate_field: str
    eligible_candidates: tuple[str, ...]
    decision_schema_version: str = "right-side-production-replacement-decision-v1"
    production_replacement_field: str = "replace_online"
    factor_workers: int = 1


def _under_root(project_root: Path, value: str) -> Path:
    root = project_root.resolve()
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"right-side shadow path escapes project root: {value}") from exc
    return path


def _read_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"right-side shadow config is missing: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"right-side shadow config must be a mapping: {path}")
    return payload


def load_shadow_release_config(
    project_root: Path = PROJECT_ROOT,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
) -> ShadowReleaseConfig:
    """Load and validate the research-only shadow release configuration."""

    root = project_root.resolve()
    config = Path(config_path)
    if not config.is_absolute():
        config = root / config
    payload = _read_mapping(config)
    release = payload.get("release") or {}
    routine = payload.get("routine") or {}
    gate = payload.get("ranking_gate") or {}
    promotion = payload.get("promotion") or {}
    if not all(isinstance(item, Mapping) for item in (release, routine, gate, promotion)):
        raise ValueError("right-side shadow release/routine/gate/promotion must be mappings")
    if release.get("lifecycle") != "research_only":
        raise ValueError("right-side shadow release.lifecycle must remain research_only")
    if release.get("scope") != SHADOW_SCOPE:
        raise ValueError(f"right-side shadow release.scope must be {SHADOW_SCOPE}")
    expected_nodes = {
        "feature.right_side_unified_shadow",
        "score.right_side_unified_shadow",
        "product.right_side_unified_shadow",
    }
    if set(release.get("dependency_nodes") or ()) != expected_nodes:
        raise ValueError("right-side shadow dependency_nodes do not match the four-layer registry")
    if promotion.get("enabled") is not False:
        raise ValueError("production promotion must remain explicitly disabled in shadow config")
    if release.get("artifact_schema_version") != RIGHT_SIDE_SHADOW_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("right-side shadow artifact schema drifted")
    if release.get("feature_schema_version") != RIGHT_SIDE_SHADOW_FEATURE_SCHEMA_VERSION:
        raise ValueError("right-side shadow feature schema drifted")
    if release.get("project_factor_schema_version") != PROJECT_FACTOR_SCHEMA_VERSION:
        raise ValueError("right-side shadow project-factor schema drifted")
    if release.get("rule_factor_schema_version") != RULE_FEATURE_SCHEMA_VERSION:
        raise ValueError("right-side shadow rule-factor schema drifted")
    if int(release.get("rule_factor_count") or -1) != len(RULE_FEATURE_COLUMNS):
        raise ValueError("right-side shadow rule-factor count drifted")
    if int(release.get("factor_count") or -1) != len(RIGHT_SIDE_SHADOW_FACTOR_COLUMNS):
        raise ValueError("right-side shadow factor count drifted")
    if release.get("factor_contract_sha256") != RIGHT_SIDE_SHADOW_FACTOR_CONTRACT_SHA256:
        raise ValueError("right-side shadow factor hash drifted")

    top_n = int(routine.get("target_top_n") or 0)
    history_years = int(routine.get("history_years") or 0)
    factor_workers = int(routine.get("factor_workers") or 1)
    if top_n <= 0 or history_years < 6:
        raise ValueError("right-side shadow requires target_top_n > 0 and history_years >= 6")
    if not 1 <= factor_workers <= 32:
        raise ValueError("right-side shadow factor_workers must be in [1, 32]")
    if routine.get("production_failure_policy") != "isolate_shadow":
        raise ValueError("right-side shadow failure policy must isolate production")

    def resolve(section: Mapping[str, Any], key: str) -> Path:
        value = str(section.get(key) or "")
        if not value:
            raise ValueError(f"right-side shadow config missing path: {key}")
        return _under_root(root, value)

    accepted = tuple(gate.get("accepted_values") or ())
    if not accepted:
        raise ValueError("right-side shadow ranking gate has no accepted values")
    selected_candidate_field = str(
        gate.get("selected_candidate_field") or "selected_candidate"
    )
    eligible_candidates = tuple(
        str(value) for value in (gate.get("eligible_candidates") or ())
    )
    supported_candidates = {"unified_long_task_deep"}
    if not selected_candidate_field or not eligible_candidates:
        raise ValueError("right-side shadow ranking gate must declare selected candidates")
    if not set(eligible_candidates) <= supported_candidates:
        raise ValueError("right-side shadow ranking gate declares an unsupported candidate")
    decision_schema = str(gate.get("decision_schema_version") or "")
    if decision_schema != "right-side-production-replacement-decision-v1":
        raise ValueError("right-side shadow must use the composite replacement decision")
    production_replacement_field = str(
        gate.get("production_replacement_field") or ""
    )
    if production_replacement_field != "replace_online":
        raise ValueError("right-side shadow production authorization field drifted")
    paths = ShadowPaths(
        artifact=resolve(release, "artifact"),
        artifact_manifest=resolve(release, "artifact_manifest"),
        ranking_decision=resolve(release, "ranking_decision"),
        feature_output=resolve(routine, "feature_output"),
        feature_manifest=resolve(routine, "feature_manifest"),
        score_output=resolve(routine, "score_output"),
        score_manifest=resolve(routine, "score_manifest"),
        product_output=resolve(routine, "product_output"),
        product_manifest=resolve(routine, "product_manifest"),
        run_status=resolve(routine, "run_status"),
        z_signal_cache=resolve(routine, "z_signal_cache"),
        family_signal_cache=resolve(routine, "family_signal_cache"),
        market_data_root=resolve(routine, "market_data_root"),
    )
    return ShadowReleaseConfig(
        enabled=bool(routine.get("enabled", False)),
        top_n=top_n,
        history_years=history_years,
        paths=paths,
        decision_field=str(gate.get("decision_field") or ""),
        accepted_decisions=accepted,
        selected_candidate_field=selected_candidate_field,
        eligible_candidates=eligible_candidates,
        decision_schema_version=decision_schema,
        production_replacement_field=production_replacement_field,
        factor_workers=factor_workers,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(project_root: Path, path: Path) -> str:
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def _lookup(payload: Mapping[str, Any], dotted: str) -> Any:
    current: Any = payload
    for part in dotted.split(".") if dotted else ():
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _accepted_ranking_decision(
    config: ShadowReleaseConfig,
) -> tuple[dict[str, Any], str]:
    payload = _load_json(config.paths.ranking_decision)
    if payload.get("schema_version") != config.decision_schema_version:
        raise RuntimeError(
            "right-side shadow requires the composite production replacement decision"
        )
    value = _lookup(payload, config.decision_field)
    normalized = str(value).strip().lower() if not isinstance(value, bool) else value
    accepted = {
        item if isinstance(item, bool) else str(item).strip().lower()
        for item in config.accepted_decisions
    }
    if normalized not in accepted:
        raise RuntimeError("right-side ranking gate has not accepted shadow staging")
    selected = str(_lookup(payload, config.selected_candidate_field) or "")
    if not selected:
        raise RuntimeError("right-side ranking decision has no selected_candidate")
    if config.eligible_candidates and selected not in config.eligible_candidates:
        raise RuntimeError(
            f"right-side ranking selected an ineligible shadow candidate: {selected}"
        )
    if selected == "unified_long_task_deep" and (
        _lookup(payload, "canonical_factor_gate.passed") is not True
    ):
        raise RuntimeError("canonical unified shadow candidate has not passed factor gate")
    online_authorized = _lookup(payload, config.production_replacement_field)
    if not isinstance(online_authorized, bool):
        raise RuntimeError("composite decision has no explicit production authorization")
    return payload, selected


def _load_source_training_manifest(source: Path) -> dict[str, Any]:
    path = source.with_suffix(".manifest.json")
    payload = _load_json(path)
    if not payload:
        raise FileNotFoundError(f"shadow source model manifest is missing: {path}")
    return payload


def _validated_source_rule_contract(
    *,
    selected_candidate: str,
    source_model: object,
    source_manifest: Mapping[str, Any],
) -> tuple[tuple[str, ...], str]:
    forbidden_manifest = find_forbidden_aliases_in_payload(source_manifest)
    if forbidden_manifest:
        raise ValueError(
            "shadow source manifest contains forbidden factor aliases: "
            f"{forbidden_manifest}"
        )
    if source_manifest.get("experiment") != selected_candidate:
        raise ValueError(
            "shadow source manifest experiment does not match "
            f"decision.selected_candidate={selected_candidate}"
        )
    common_features = tuple(
        str(value) for value in (getattr(source_model, "common_features", None) or ())
    )
    if not common_features:
        raise TypeError("shadow source model must expose non-empty common_features")
    assert_no_forbidden_factor_names(
        common_features,
        context="shadow source model common_features",
    )
    raw_feature_names = getattr(source_model, "feature_names_in_", None)
    model_feature_names = tuple(
        str(value) for value in (() if raw_feature_names is None else raw_feature_names)
    )
    if model_feature_names:
        assert_no_forbidden_factor_names(
            model_feature_names,
            context="shadow source model feature_names_in_",
        )
    manifest_common = tuple(
        str(value) for value in (source_manifest.get("common_features") or ())
    )
    if common_features != manifest_common:
        raise ValueError("shadow source model common_features differ from its manifest")
    manifest_rules = tuple(
        str(value) for value in (source_manifest.get("rule_feature_columns") or ())
    )
    actual_rules = tuple(
        feature for feature in common_features if feature in set(RULE_FEATURE_COLUMNS)
    )
    if actual_rules != manifest_rules:
        raise ValueError(
            "shadow source model rule columns do not exactly match its manifest"
        )
    expected_digest = rule_feature_columns_sha256(manifest_rules)
    if source_manifest.get("rule_feature_columns_sha256") != expected_digest:
        raise ValueError("shadow source manifest rule-factor hash mismatch")
    if int(source_manifest.get("rule_feature_count") or -1) != len(manifest_rules):
        raise ValueError("shadow source manifest rule-factor count mismatch")

    if selected_candidate == "unified_long_task_deep":
        expected_rules = tuple(RULE_FEATURE_COLUMNS)
        expected_schema = RULE_FEATURE_SCHEMA_VERSION
    else:
        raise ValueError(f"unsupported selected right-side candidate: {selected_candidate}")
    if manifest_rules != expected_rules:
        raise ValueError(
            "shadow source rule contract does not equal the frozen candidate contract"
        )
    if source_manifest.get("rule_feature_schema_version") != expected_schema:
        raise ValueError("shadow source rule-factor schema mismatch")
    unknown_common = sorted(
        set(common_features) - set(RIGHT_SIDE_SHADOW_FACTOR_COLUMNS)
    )
    if unknown_common:
        raise ValueError(f"shadow source model contains unregistered factors: {unknown_common}")
    return manifest_rules, expected_digest


def _atomic_dump_joblib(value: object, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        joblib.dump(value, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def stage_shadow_release(
    source_model_path: Path | str,
    *,
    project_root: Path = PROJECT_ROOT,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Wrap one ranking model in the pinned shadow artifact contract.

    This is a shadow-stage operation only.  It refuses to run until the A/B
    ranking decision says replace/pass and never writes to models/production.
    """

    config = load_shadow_release_config(project_root, config_path)
    validate_right_side_shadow_factor_contract()
    _decision, selected_candidate = _accepted_ranking_decision(config)
    source = Path(source_model_path)
    if not source.is_absolute():
        source = project_root.resolve() / source
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if config.paths.artifact.exists() and not overwrite:
        raise FileExistsError(
            f"shadow artifact already exists; pass overwrite explicitly: {config.paths.artifact}"
        )
    model = joblib.load(source)
    source_manifest = _load_source_training_manifest(source)
    selected_rule_features, selected_rule_hash = _validated_source_rule_contract(
        selected_candidate=selected_candidate,
        source_model=model,
        source_manifest=source_manifest,
    )
    common_features = tuple(
        str(value) for value in (getattr(model, "common_features", None) or ())
    )
    external_features = stable_canonical_feature_union(
        common_features,
        RIGHT_SIDE_SHADOW_IDENTITY_COLUMNS,
    )
    model_input_contract_sha256 = factor_contract_sha256(
        external_features,
        schema_version=RIGHT_SIDE_SHADOW_ARTIFACT_SCHEMA_VERSION,
    )
    bundle = {
        "schema_version": RIGHT_SIDE_SHADOW_ARTIFACT_SCHEMA_VERSION,
        "model": model,
        "features": external_features,
        "factor_contract_sha256": RIGHT_SIDE_SHADOW_FACTOR_CONTRACT_SHA256,
        "model_input_contract_sha256": model_input_contract_sha256,
        "project_factor_schema_version": PROJECT_FACTOR_SCHEMA_VERSION,
        "rule_factor_schema_version": source_manifest[
            "rule_feature_schema_version"
        ],
        "materialized_rule_factor_schema_version": RULE_FEATURE_SCHEMA_VERSION,
        "materialized_rule_factor_columns_sha256": RULE_FEATURE_COLUMNS_SHA256,
        "selected_candidate": selected_candidate,
        "selected_rule_factor_count": len(selected_rule_features),
        "selected_rule_factor_columns": selected_rule_features,
        "selected_rule_factor_columns_sha256": selected_rule_hash,
        "beam_search": None,
        "ranking_decision_sha256": _sha256(config.paths.ranking_decision),
        "source_model_manifest_path": _relative(
            project_root,
            source.with_suffix(".manifest.json"),
        ),
        "source_model_manifest_sha256": _sha256(
            source.with_suffix(".manifest.json")
        ),
        "source_model_path": _relative(project_root, source),
        "source_model_sha256": _sha256(source),
    }
    _atomic_dump_joblib(bundle, config.paths.artifact)
    artifact_sha = _sha256(config.paths.artifact)
    manifest = {
        "status": "success",
        "schema_version": RIGHT_SIDE_SHADOW_ARTIFACT_SCHEMA_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "lifecycle": "research_only",
        "scope": SHADOW_SCOPE,
        "selector_consumer": None,
        "source_model_path": bundle["source_model_path"],
        "source_model_sha256": bundle["source_model_sha256"],
        "selected_candidate": selected_candidate,
        "ranking_decision_sha256": bundle["ranking_decision_sha256"],
        "selected_rule_factor_count": len(selected_rule_features),
        "selected_rule_factor_columns": list(selected_rule_features),
        "selected_rule_factor_columns_sha256": selected_rule_hash,
        "selected_model_input_count": len(external_features),
        "selected_model_input_contract_sha256": model_input_contract_sha256,
        "feature_count": len(external_features),
        **right_side_shadow_contract_payload(),
        "models": {
            "ranking": {
                "path": _relative(project_root, config.paths.artifact),
                "sha256": artifact_sha,
            }
        },
    }
    atomic_write_json(manifest, config.paths.artifact_manifest)
    return {
        "status": "success",
        "artifact": str(config.paths.artifact),
        "artifact_sha256": artifact_sha,
        "manifest": str(config.paths.artifact_manifest),
        "production_changed": False,
    }


def _target_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    target = pd.to_datetime(value, errors="coerce")
    if pd.isna(target):
        raise ValueError(f"invalid right-side shadow target date: {value}")
    return pd.Timestamp(target).normalize()


def _empty_feature_frame() -> pd.DataFrame:
    columns = [
        "ts_code",
        "symbol",
        "trade_date",
        "date",
        *RIGHT_SIDE_SHADOW_FACTOR_COLUMNS,
        "factor_schema_version",
        "right_side_feature_schema_version",
        *RIGHT_SIDE_SHADOW_IDENTITY_COLUMNS,
    ]
    return pd.DataFrame(columns=columns)


def _build_shadow_symbol_feature(
    task: tuple[str, pd.DataFrame, dict[str, Any], pd.Timestamp],
) -> tuple[str, pd.DataFrame, str | None]:
    """Build one symbol in an isolated worker using the training-time contract."""

    symbol, daily, signal_values, target = task
    try:
        if daily.empty:
            raise ValueError("target market row is missing")
        normalized = normalize_tushare_daily(daily, symbol).sort_values(
            "date", kind="stable"
        ).reset_index(drop=True)
        if normalized.empty or not normalized["date"].dt.normalize().eq(target).any():
            raise ValueError("target market row is missing")
        project = calculate_project_market_factors(
            normalized,
            symbol=symbol,
            factor_schema_version=PROJECT_FACTOR_SCHEMA_VERSION,
        ).reset_index(drop=True)
        if not project["factor_schema_version"].eq(
            PROJECT_FACTOR_SCHEMA_VERSION
        ).all():
            raise ValueError("project factor calculator returned the wrong schema")
        rules = compute_right_side_rule_features(
            normalized,
            canonical_factors=project,
        ).reset_index(drop=True)
        missing_rules = set(RULE_FEATURE_COLUMNS) - set(rules.columns)
        if missing_rules:
            raise ValueError(
                f"rule factor calculator contract incomplete: {sorted(missing_rules)}"
            )

        # The frozen training build deliberately materialized 112 causal OHLCV
        # project factors and represented the 35 daily_basic factors as null.
        # Keep that exact semantic contract here: publish all 147 registered
        # columns, but never synthesize unavailable daily_basic values.
        for column in PROJECT_FACTOR_COLUMNS:
            if column not in project.columns:
                project[column] = np.nan
        base = pd.concat(
            [
                project[
                    [
                        "ts_code",
                        "symbol",
                        "trade_date",
                        "date",
                        *PROJECT_FACTOR_COLUMNS,
                        "factor_schema_version",
                    ]
                ],
                rules[list(RULE_FEATURE_COLUMNS)],
            ],
            axis=1,
        )
        base["date"] = pd.to_datetime(base["date"], errors="coerce").dt.normalize()
        current = base[base["date"].eq(target)].tail(1).copy()
        if len(current) != 1:
            raise ValueError("target feature row is not unique")
        current["right_side_feature_schema_version"] = (
            RIGHT_SIDE_SHADOW_FEATURE_SCHEMA_VERSION
        )
        for name in RIGHT_SIDE_SHADOW_IDENTITY_COLUMNS:
            current[name] = bool(signal_values[name])
        return symbol, current, None
    except Exception as exc:
        return symbol, pd.DataFrame(), str(exc)


def build_right_side_shadow_feature_frame(
    market: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    target_date: str | pd.Timestamp,
    workers: int = 1,
) -> pd.DataFrame:
    """Build one exact-date project-v5/rule-v4-113 frame from loaded inputs."""

    validate_right_side_shadow_factor_contract()
    target = _target_timestamp(target_date)
    required_signal_columns = {
        "symbol",
        "date",
        *RIGHT_SIDE_SHADOW_IDENTITY_COLUMNS,
    }
    missing_signals = required_signal_columns - set(signals.columns)
    if missing_signals:
        raise ValueError(f"right-side shadow signals missing columns: {sorted(missing_signals)}")
    signal_rows = signals.copy()
    signal_rows["date"] = pd.to_datetime(signal_rows["date"], errors="coerce").dt.normalize()
    signal_rows = signal_rows[signal_rows["date"].eq(target)].copy()
    signal_rows = signal_rows[
        signal_rows[list(RIGHT_SIDE_SHADOW_IDENTITY_COLUMNS)]
        .fillna(False)
        .astype(bool)
        .any(axis=1)
    ]
    if signal_rows.empty:
        return _empty_feature_frame()
    if signal_rows.duplicated(["symbol", "date"]).any():
        raise ValueError("right-side shadow signal keys are not unique")
    if market.empty:
        raise ValueError("right-side shadow market history is empty")
    if workers < 1:
        raise ValueError("right-side shadow factor workers must be positive")

    frames: list[pd.DataFrame] = []
    failures: list[str] = []
    market_by_symbol = {
        str(symbol): frame.copy()
        for symbol, frame in market.assign(
            _shadow_symbol=market["ts_code"].astype(str)
        ).groupby("_shadow_symbol", sort=False)
    }
    tasks = [
        (
            str(symbol),
            market_by_symbol.get(str(symbol), pd.DataFrame()).drop(
                columns=["_shadow_symbol"], errors="ignore"
            ),
            signal.iloc[-1].to_dict(),
            target,
        )
        for symbol, signal in signal_rows.groupby("symbol", sort=True)
    ]

    def collect(result: tuple[str, pd.DataFrame, str | None]) -> None:
        symbol, frame, error = result
        if error is not None:
            failures.append(f"{symbol}: {error}")
        else:
            frames.append(frame)

    if workers == 1 or len(tasks) <= 1:
        for task in tasks:
            collect(_build_shadow_symbol_feature(task))
    else:
        task_iterator = iter(tasks)
        max_workers = min(workers, len(tasks))
        max_pending = max_workers * 2
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            pending = set()
            for _ in range(min(max_pending, len(tasks))):
                pending.add(executor.submit(_build_shadow_symbol_feature, next(task_iterator)))
            while pending:
                completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    collect(future.result())
                    try:
                        pending.add(
                            executor.submit(
                                _build_shadow_symbol_feature,
                                next(task_iterator),
                            )
                        )
                    except StopIteration:
                        pass
    if failures:
        raise RuntimeError("right-side shadow feature build failed: " + " | ".join(failures))
    result = pd.concat(frames, ignore_index=True, sort=False) if frames else _empty_feature_frame()
    expected_columns = list(_empty_feature_frame().columns)
    missing = set(expected_columns) - set(result.columns)
    if missing:
        raise RuntimeError(f"right-side shadow output missing columns: {sorted(missing)}")
    result = result[expected_columns].sort_values(["date", "symbol"], kind="stable")
    if result.duplicated(["symbol", "date"]).any():
        raise RuntimeError("right-side shadow feature output has duplicate keys")
    if len(result) != len(signal_rows):
        raise RuntimeError(
            "right-side shadow candidate coverage is incomplete: "
            f"signals={len(signal_rows)} features={len(result)}"
        )
    if not result["factor_schema_version"].eq(PROJECT_FACTOR_SCHEMA_VERSION).all():
        raise RuntimeError("right-side shadow project-factor rows have wrong schema")
    return result.reset_index(drop=True)


def build_right_side_shadow_features(
    target_date: str | pd.Timestamp,
    *,
    config: ShadowReleaseConfig,
) -> dict[str, Any]:
    target = _target_timestamp(target_date)
    if not config.paths.z_signal_cache.is_file():
        raise FileNotFoundError(config.paths.z_signal_cache)
    if not config.paths.family_signal_cache.is_file():
        raise FileNotFoundError(config.paths.family_signal_cache)
    signals = load_signal_universe(
        config.paths.z_signal_cache,
        config.paths.family_signal_cache,
        start_date=target,
        end_date=target,
    )
    symbols = sorted(signals["symbol"].astype(str).unique()) if not signals.empty else []
    start = target - pd.DateOffset(years=config.history_years)
    store = MarketDataStore(
        MarketDataStoreConfig(
            backend="parquet",
            root=config.paths.market_data_root,
            mirror_parquet=True,
        )
    )
    market = store.read_market_range(
        "daily",
        start_date=start.strftime("%Y%m%d"),
        end_date=target.strftime("%Y%m%d"),
        symbols=symbols,
        columns=(
            "ts_code",
            "trade_date",
            "date",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "pre_close",
            "pct_chg",
            "vol",
            "volume",
            "name",
        ),
    ) if symbols else pd.DataFrame()
    frame = build_right_side_shadow_feature_frame(
        market,
        signals,
        target_date=target,
        workers=config.factor_workers,
    )
    atomic_write_parquet(frame, config.paths.feature_output, index=False)
    manifest = {
        "status": "success",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "target_date": target.date().isoformat(),
        "candidate_coverage_status": "complete",
        "signal_candidate_count": int(len(signals)),
        "computed_candidate_count": int(len(frame)),
        "empty_candidate_set": bool(frame.empty),
        "output": str(config.paths.feature_output),
        "output_sha256": _sha256(config.paths.feature_output),
        **right_side_shadow_contract_payload(),
    }
    atomic_write_json(manifest, config.paths.feature_manifest)
    dated_path = config.paths.feature_output.with_name(f"{target:%Y%m%d}_features.parquet")
    atomic_link_or_copy(config.paths.feature_output, dated_path)
    atomic_write_json(manifest, dated_path.with_suffix(".json"))
    return manifest


def _load_shadow_bundle(
    config: ShadowReleaseConfig,
    *,
    project_root: Path = PROJECT_ROOT,
) -> tuple[dict[str, Any], str]:
    if not config.paths.artifact.is_file():
        raise FileNotFoundError(config.paths.artifact)
    manifest = _load_json(config.paths.artifact_manifest)
    if manifest.get("schema_version") != RIGHT_SIDE_SHADOW_ARTIFACT_SCHEMA_VERSION:
        raise RuntimeError("right-side shadow artifact manifest schema mismatch")
    model_item = (manifest.get("models") or {}).get("ranking") or {}
    expected_path = _relative(project_root, config.paths.artifact)
    if model_item.get("path") != expected_path:
        raise RuntimeError("right-side shadow artifact manifest path mismatch")
    digest = _sha256(config.paths.artifact)
    if model_item.get("sha256") != digest:
        raise RuntimeError("right-side shadow artifact checksum mismatch")
    bundle = joblib.load(config.paths.artifact)
    if not isinstance(bundle, dict):
        raise TypeError("right-side shadow artifact must be a bundle mapping")
    if bundle.get("schema_version") != RIGHT_SIDE_SHADOW_ARTIFACT_SCHEMA_VERSION:
        raise RuntimeError("right-side shadow bundle schema mismatch")
    if bundle.get("factor_contract_sha256") != RIGHT_SIDE_SHADOW_FACTOR_CONTRACT_SHA256:
        raise RuntimeError("right-side shadow bundle factor hash mismatch")
    if bundle.get("materialized_rule_factor_columns_sha256") != RULE_FEATURE_COLUMNS_SHA256:
        raise RuntimeError("right-side shadow bundle materialized rule-factor hash mismatch")
    forbidden_payload = find_forbidden_aliases_in_payload(bundle)
    if forbidden_payload:
        raise RuntimeError(
            "right-side shadow bundle contains forbidden factor aliases: "
            f"{forbidden_payload}"
        )
    features = tuple(str(value) for value in (bundle.get("features") or ()))
    if len(features) != len(set(features)) or not features:
        raise RuntimeError("right-side shadow bundle features are empty or duplicated")
    selected_candidate = str(bundle.get("selected_candidate") or "")
    selected_rules = tuple(
        str(value) for value in (bundle.get("selected_rule_factor_columns") or ())
    )
    selected_hash = rule_feature_columns_sha256(selected_rules)
    if bundle.get("selected_rule_factor_columns_sha256") != selected_hash:
        raise RuntimeError("right-side shadow bundle selected rule-factor hash mismatch")
    if int(bundle.get("selected_rule_factor_count") or -1) != len(selected_rules):
        raise RuntimeError("right-side shadow bundle selected rule-factor count mismatch")
    valid_rule_contract = bool(
        selected_candidate == "unified_long_task_deep"
        and selected_rules == tuple(RULE_FEATURE_COLUMNS)
    )
    if not valid_rule_contract:
        raise RuntimeError("right-side shadow bundle has an unsupported rule subset")
    missing_rules = sorted(set(selected_rules) - set(features))
    missing_identity = sorted(set(RIGHT_SIDE_SHADOW_IDENTITY_COLUMNS) - set(features))
    unknown = sorted(
        set(features)
        - set(RIGHT_SIDE_SHADOW_FACTOR_COLUMNS)
        - set(RIGHT_SIDE_SHADOW_IDENTITY_COLUMNS)
    )
    expected_input_hash = factor_contract_sha256(
        features,
        schema_version=RIGHT_SIDE_SHADOW_ARTIFACT_SCHEMA_VERSION,
    )
    if bundle.get("model_input_contract_sha256") != expected_input_hash:
        raise RuntimeError("right-side shadow bundle model-input hash mismatch")
    if missing_rules or missing_identity or unknown:
        raise RuntimeError(
            "right-side shadow bundle input contract mismatch; "
            f"missing_rules={missing_rules} missing_identity={missing_identity} "
            f"unknown={unknown}"
        )
    if bundle.get("model") is None:
        raise RuntimeError("right-side shadow bundle has no model")
    model = bundle["model"]
    assert_no_forbidden_factor_names(
        getattr(model, "feature_names_in_", ()),
        context="right-side shadow model feature_names_in_",
    )
    assert_no_forbidden_factor_names(
        getattr(model, "selected_features_", ()),
        context="right-side shadow model selected_features_",
    )
    return bundle, digest


def validate_shadow_release_preflight(
    config: ShadowReleaseConfig,
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Pin one run to the current decision, artifact, and staged manifest."""

    if not config.paths.artifact.is_file():
        raise FileNotFoundError(f"missing artifact: {config.paths.artifact}")
    if not config.paths.artifact_manifest.is_file():
        raise FileNotFoundError(
            f"missing artifact manifest: {config.paths.artifact_manifest}"
        )
    bundle, artifact_sha = _load_shadow_bundle(config, project_root=project_root)
    decision, selected = _accepted_ranking_decision(config)
    manifest = _load_json(config.paths.artifact_manifest)
    decision_sha = _sha256(config.paths.ranking_decision)
    if bundle.get("selected_candidate") != selected:
        raise RuntimeError("right-side shadow artifact differs from the active decision")
    if bundle.get("ranking_decision_sha256") != decision_sha:
        raise RuntimeError("right-side shadow artifact decision checksum is stale")
    if manifest.get("ranking_decision_sha256") != decision_sha:
        raise RuntimeError("right-side shadow manifest decision checksum is stale")
    if manifest.get("selected_candidate") != selected:
        raise RuntimeError("right-side shadow manifest differs from the active decision")
    return {
        "status": "success",
        "selected_candidate": selected,
        "artifact_sha256": artifact_sha,
        "ranking_decision_sha256": decision_sha,
        "replace_online": decision.get(config.production_replacement_field),
    }


def score_right_side_shadow(
    target_date: str | pd.Timestamp,
    *,
    config: ShadowReleaseConfig,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    target = _target_timestamp(target_date)
    feature_manifest = _load_json(config.paths.feature_manifest)
    if (
        feature_manifest.get("status") != "success"
        or feature_manifest.get("target_date") != target.date().isoformat()
        or feature_manifest.get("factor_contract_sha256")
        != RIGHT_SIDE_SHADOW_FACTOR_CONTRACT_SHA256
    ):
        raise RuntimeError("right-side shadow feature manifest is stale or incompatible")
    if not config.paths.feature_output.is_file():
        raise RuntimeError("right-side shadow feature sidecar is missing")
    if feature_manifest.get("output_sha256") != _sha256(config.paths.feature_output):
        raise RuntimeError("right-side shadow feature sidecar checksum mismatch")
    frame = pd.read_parquet(config.paths.feature_output)
    frame_dates = pd.to_datetime(frame.get("date"), errors="coerce").dt.normalize()
    if frame_dates.isna().any() or not frame_dates.eq(target).all():
        raise RuntimeError("right-side shadow feature sidecar is not exact-date")
    if frame.assign(date=frame_dates).duplicated(["symbol", "date"]).any():
        raise RuntimeError("right-side shadow feature sidecar has duplicate keys")
    bundle, artifact_sha = _load_shadow_bundle(config, project_root=project_root)
    features = tuple(str(value) for value in bundle["features"])
    missing = set(features) - set(frame.columns)
    if missing:
        raise RuntimeError(f"right-side shadow feature sidecar missing model inputs: {sorted(missing)}")
    if frame.empty:
        probabilities = np.asarray([], dtype=float)
    else:
        predicted = np.asarray(bundle["model"].predict_proba(frame[list(features)]), dtype=float)
        if predicted.ndim != 2 or predicted.shape != (len(frame), 2):
            raise RuntimeError("right-side shadow model returned an invalid probability matrix")
        probabilities = predicted[:, 1]
        if (
            not np.isfinite(probabilities).all()
            or ((probabilities < 0.0) | (probabilities > 1.0)).any()
        ):
            raise RuntimeError("right-side shadow model returned invalid scores")
    output_columns = [
        "ts_code",
        "symbol",
        "trade_date",
        "date",
        *RIGHT_SIDE_SHADOW_IDENTITY_COLUMNS,
    ]
    scored = frame[output_columns].copy()
    scored["ranking_score"] = probabilities
    scored["model_artifact_sha256"] = artifact_sha
    atomic_write_parquet(scored, config.paths.score_output, index=False)
    manifest = {
        "status": "success",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "target_date": target.date().isoformat(),
        "artifact_schema_version": RIGHT_SIDE_SHADOW_ARTIFACT_SCHEMA_VERSION,
        "artifact_path": _relative(project_root, config.paths.artifact),
        "artifact_sha256": artifact_sha,
        "factor_contract_sha256": RIGHT_SIDE_SHADOW_FACTOR_CONTRACT_SHA256,
        "feature_output_sha256": feature_manifest.get("output_sha256"),
        "candidate_count": int(len(scored)),
        "output": str(config.paths.score_output),
        "output_sha256": _sha256(config.paths.score_output),
    }
    atomic_write_json(manifest, config.paths.score_manifest)
    return manifest


def publish_right_side_shadow_product(
    target_date: str | pd.Timestamp,
    *,
    config: ShadowReleaseConfig,
) -> dict[str, Any]:
    target = _target_timestamp(target_date)
    score_manifest = _load_json(config.paths.score_manifest)
    if (
        score_manifest.get("status") != "success"
        or score_manifest.get("target_date") != target.date().isoformat()
        or score_manifest.get("factor_contract_sha256")
        != RIGHT_SIDE_SHADOW_FACTOR_CONTRACT_SHA256
    ):
        raise RuntimeError("right-side shadow score manifest is stale or incompatible")
    if not config.paths.score_output.is_file():
        raise RuntimeError("right-side shadow score snapshot is missing")
    if score_manifest.get("output_sha256") != _sha256(config.paths.score_output):
        raise RuntimeError("right-side shadow score snapshot checksum mismatch")
    scored = pd.read_parquet(config.paths.score_output)
    score_dates = pd.to_datetime(scored.get("date"), errors="coerce").dt.normalize()
    if score_dates.isna().any() or not score_dates.eq(target).all():
        raise RuntimeError("right-side shadow score snapshot is not exact-date")
    if scored.assign(date=score_dates).duplicated(["symbol", "date"]).any():
        raise RuntimeError("right-side shadow score snapshot has duplicate keys")
    product = scored.sort_values(
        ["ranking_score", "symbol"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)
    product["daily_rank"] = np.arange(1, len(product) + 1, dtype=np.int32)
    product["daily_percentile"] = (
        1.0 - (product["daily_rank"] - 1) / max(len(product), 1)
    ).astype(float)
    product["selected_for_shadow"] = product["daily_rank"].le(config.top_n)
    product["consumer"] = "research_shadow_only"
    atomic_write_parquet(product, config.paths.product_output, index=False)
    manifest = {
        "status": "success",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "target_date": target.date().isoformat(),
        "candidate_coverage_status": "complete",
        "consumer": "research_shadow_only",
        "selector_published": False,
        "candidate_count": int(len(product)),
        "selected_count": int(product["selected_for_shadow"].sum()),
        "top_n": config.top_n,
        "score_output_sha256": score_manifest.get("output_sha256"),
        "artifact_sha256": score_manifest.get("artifact_sha256"),
        "output": str(config.paths.product_output),
        "output_sha256": _sha256(config.paths.product_output),
    }
    atomic_write_json(manifest, config.paths.product_manifest)
    return manifest


def _local_market_evidence(
    config: ShadowReleaseConfig,
    target_date: pd.Timestamp | None = None,
    *,
    allow_historical_replay: bool = False,
) -> dict[str, Any]:
    store = MarketDataStore(
        MarketDataStoreConfig(
            backend="parquet",
            root=config.paths.market_data_root,
            mirror_parquet=True,
        )
    )
    latest = store.latest_dataset_trade_date("daily")
    selected = latest
    replay = False
    if allow_historical_replay and target_date is not None and latest is not None:
        target = _target_timestamp(target_date)
        exact = store.read_market_range(
            "daily",
            start_date=target.strftime("%Y%m%d"),
            end_date=target.strftime("%Y%m%d"),
            columns=("trade_date",),
        )
        exact_dates = pd.to_datetime(
            exact.get("trade_date"),
            format="%Y%m%d",
            errors="coerce",
        )
        if exact.empty or not exact_dates.eq(target).any():
            return {
                "status": "failed",
                "dataset_trade_date": None,
                "source_latest_trade_date": latest.date().isoformat(),
                "historical_replay": True,
                "error": "requested historical market partition is unavailable",
            }
        selected = target
        replay = target != latest.normalize()
    return {
        "status": "success" if selected is not None else "failed",
        "dataset_trade_date": selected.date().isoformat() if selected is not None else None,
        "source_latest_trade_date": latest.date().isoformat() if latest is not None else None,
        "historical_replay": replay,
    }


def run_configured_right_side_shadow(
    target_date: str | pd.Timestamp,
    *,
    project_root: Path = PROJECT_ROOT,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    upstream_results: Mapping[str, Any] | None = None,
    force_enabled: bool = False,
) -> dict[str, Any]:
    """Execute the isolated shadow chain and its dependency final gate."""

    config = load_shadow_release_config(project_root, config_path)
    target = _target_timestamp(target_date)
    if not config.enabled and not force_enabled:
        return {
            "status": "skipped",
            "scope": SHADOW_SCOPE,
            "target_date": target.date().isoformat(),
            "reason": "ranking gate has not enabled the research shadow routine",
            "production_affected": False,
        }
    results = dict(upstream_results or {})
    results.setdefault(
        "refresh_data",
        _local_market_evidence(
            config,
            target,
            allow_historical_replay=force_enabled,
        ),
    )
    started = datetime.now()

    def publish_status(status: str, **extra: Any) -> dict[str, Any]:
        payload = {
            "status": status,
            "scope": SHADOW_SCOPE,
            "target_date": target.date().isoformat(),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "production_affected": False,
            "selector_published": False,
            "elapsed_seconds": round((datetime.now() - started).total_seconds(), 3),
            **extra,
        }
        atomic_write_json(payload, config.paths.run_status)
        return payload

    try:
        from quant.routine.daily_dependency_runtime import (
            publish_daily_dependency_snapshot,
        )

        release_preflight = validate_shadow_release_preflight(
            config,
            project_root=project_root,
        )
        preflight = publish_daily_dependency_snapshot(
            project_root,
            target,
            scope=SHADOW_SCOPE,
            results=results,
            phase="preflight",
            strict_models=True,
            strict_freshness=False,
            raise_on_failure=False,
        )
    except Exception as exc:
        return publish_status(
            "blocked",
            stage="preflight",
            error=str(exc),
            reason="missing or incompatible shadow artifact/contract",
        )

    try:
        feature = build_right_side_shadow_features(target, config=config)
        score = score_right_side_shadow(
            target,
            config=config,
            project_root=project_root,
        )
        product = publish_right_side_shadow_product(target, config=config)
        postflight_results = {
            **results,
            "dependency_preflight": preflight,
            "right_side_shadow_features": feature,
            "right_side_shadow_scores": score,
            "right_side_shadow_product": product,
            "right_side_shadow_release_preflight": release_preflight,
        }
        postflight = publish_daily_dependency_snapshot(
            project_root,
            target,
            scope=SHADOW_SCOPE,
            results=postflight_results,
            phase="postflight",
            strict_models=True,
            strict_freshness=True,
            raise_on_failure=False,
        )
        if postflight.get("status") != "success":
            raise RuntimeError(
                "right-side shadow dependency final gate failed: "
                f"{postflight.get('freshness_audit')}"
            )
        return publish_status(
            "success",
            stage="complete",
            feature_candidate_count=feature["computed_candidate_count"],
            scored_candidate_count=score["candidate_count"],
            selected_candidate_count=product["selected_count"],
            release_preflight=release_preflight,
            dependency_preflight=preflight,
            dependency_postflight=postflight,
        )
    except Exception as exc:
        return publish_status("failed", stage="execute", error=str(exc))


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Unified right-side research shadow routine")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="build, score, and publish one shadow date")
    run.add_argument("--target-date", required=True)
    run.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    run.add_argument("--force-enabled", action="store_true")
    stage = subparsers.add_parser("stage-shadow-release", help="wrap a passed model for shadow use")
    stage.add_argument("--source-model", type=Path, required=True)
    stage.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    stage.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.command == "stage-shadow-release":
        payload = stage_shadow_release(
            args.source_model,
            config_path=args.config,
            overwrite=args.overwrite,
        )
    else:
        payload = run_configured_right_side_shadow(
            args.target_date,
            config_path=args.config,
            force_enabled=args.force_enabled,
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0 if payload.get("status") in {"success", "skipped"} else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "SHADOW_SCOPE",
    "ShadowPaths",
    "ShadowReleaseConfig",
    "build_right_side_shadow_feature_frame",
    "build_right_side_shadow_features",
    "load_shadow_release_config",
    "publish_right_side_shadow_product",
    "run_configured_right_side_shadow",
    "score_right_side_shadow",
    "stage_shadow_release",
    "validate_shadow_release_preflight",
]
