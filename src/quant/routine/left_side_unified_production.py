"""Production build, score, and postflight for the unified left-side ranker."""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
import yaml

from quant.application.left_side_ranking import (
    DEFAULT_LEFT_SIDE_RANKING_CONFIG,
    LeftSideRankingConfig,
    load_left_side_ranking_scores,
)
from quant.data.atomic_io import atomic_write_json, atomic_write_parquet
from quant.data.market_data_store import MarketDataStore, MarketDataStoreConfig
from quant.data.source_merge import normalize_tushare_daily
from quant.features.canonical_factor_names import (
    assert_no_forbidden_factor_names,
    find_forbidden_aliases_in_payload,
)
from quant.features.left_side_factor_contract import (
    LEFT_SIDE_ARTIFACT_SCHEMA_VERSION,
    LEFT_SIDE_FACTOR_COLUMNS,
    LEFT_SIDE_FACTOR_CONTRACT_SHA256,
    LEFT_SIDE_FEATURE_SCHEMA_VERSION,
    LEFT_SIDE_MODEL_INPUT_COLUMNS,
    LEFT_SIDE_MODEL_INPUT_CONTRACT_SHA256,
    LEFT_SIDE_SCORE_SCHEMA_VERSION,
    LEFT_SIDE_SCORING_INPUT_COLUMNS,
    LEFT_SIDE_SCORING_INPUT_CONTRACT_SHA256,
    left_side_contract_payload,
)
from quant.features.project_factor_layer import (
    PROJECT_FACTOR_SCHEMA_VERSION,
    calculate_project_market_factors,
)
from quant.features.variable_library import PROJECT_FACTOR_COLUMNS
from quant.research.left_side_unified_features import (
    LEFT_SIDE_RULE_FEATURE_COLUMNS,
    LEFT_SIDE_RULE_FEATURE_SCHEMA_VERSION,
    LEFT_SIDE_SHARED_RULE_REQUIREMENTS,
    LEFT_SIDE_SIGNALS,
    compute_left_side_rule_features,
)
from quant.routine.paths import PROJECT_ROOT


LEFT_SIDE_NORMALIZATION_SCHEMA_VERSION = "daily-cross-section-percentile-v1"
LEFT_SIDE_PRODUCTION_FEATURE_BUILDER_VERSION = "left-side-production-feature-builder-v3"
LEFT_SIDE_PRODUCTION_SCORE_BUILDER_VERSION = "left-side-production-score-builder-v2"
DEFAULT_RESEARCH_MODEL = (
    PROJECT_ROOT
    / "models/research/left_side_unified_v3_group4_input_parity/next_close/h5/"
    "good_path5/C/unified_left_long_task_deep.joblib"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON contract must be a mapping: {path}")
    return payload


def _atomic_dump_joblib(value: object, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        joblib.dump(value, temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def normalize_daily_percentile(scores: np.ndarray) -> np.ndarray:
    """Return a stable monotonic [0, 100] cross-sectional percentile."""

    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("left-side ranking scores must be one-dimensional and finite")
    if len(values) == 0:
        return values.copy()
    if len(values) == 1:
        return np.asarray([50.0], dtype=float)
    ranks = pd.Series(values).rank(method="average").to_numpy(dtype=float)
    normalized = (ranks - 1.0) * (100.0 / float(len(values) - 1))
    return np.clip(normalized, 0.0, 100.0)


def _read_exact_date(path: Path, target: pd.Timestamp) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    if not {"symbol", "date"}.issubset(frame.columns):
        raise RuntimeError(f"signal cache has no symbol/date keys: {path}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    return frame[frame["date"].eq(target)].copy()


def load_left_side_signal_universe(
    target_date: str | pd.Timestamp,
    *,
    config: LeftSideRankingConfig,
) -> pd.DataFrame:
    """Aggregate the current raw caches into the four stable left groups."""

    target = pd.to_datetime(target_date, errors="raise").normalize()
    gate = _read_exact_date(config.paths.b1_gate_cache, target)
    family = _read_exact_date(config.paths.family_signal_cache, target)
    low = _read_exact_date(config.paths.signal_cache, target)
    keys = pd.concat(
        [
            frame[["symbol", "date"]]
            for frame in (gate, family, low)
            if not frame.empty
        ],
        ignore_index=True,
    ).drop_duplicates()
    if keys.empty:
        return pd.DataFrame(columns=["symbol", "date", *LEFT_SIDE_SIGNALS])
    keys["symbol"] = keys["symbol"].astype(str)
    output = keys.copy()
    gate_keys = set(zip(gate["symbol"].astype(str), gate["date"], strict=False))
    output["B1"] = [
        (symbol, date) in gate_keys
        for symbol, date in zip(output["symbol"], output["date"], strict=True)
    ]
    family_flags = family[["symbol", "date"]].copy()
    family_flags["symbol"] = family_flags["symbol"].astype(str)
    sb1_columns = [
        column for column in family.columns if column.startswith("sb1_")
    ]
    super_columns = [
        column for column in family.columns if column.startswith("super_washout_")
    ]
    if not sb1_columns or not super_columns:
        raise RuntimeError("left-side family cache lacks SB1/SUPER_B1 rule flags")
    family_flags["SB1"] = family[sb1_columns].fillna(False).astype(bool).any(axis=1)
    family_flags["SUPER_B1"] = (
        family[super_columns].fillna(False).astype(bool).any(axis=1)
    )
    output = output.merge(
        family_flags[["symbol", "date", "SB1", "SUPER_B1"]],
        on=["symbol", "date"],
        how="left",
        validate="one_to_one",
    )
    low_columns = ("YIDONG_DILIAN", "NANA", "DUICHEN_VA")
    missing_low = set(low_columns) - set(low.columns)
    if missing_low:
        raise RuntimeError(f"left-side low-pullback cache missing: {sorted(missing_low)}")
    low_flags = low[["symbol", "date"]].copy()
    low_flags["symbol"] = low_flags["symbol"].astype(str)
    low_flags["LOW_PULLBACK"] = (
        low[list(low_columns)].fillna(False).astype(bool).any(axis=1)
    )
    output = output.merge(
        low_flags[["symbol", "date", "LOW_PULLBACK"]],
        on=["symbol", "date"],
        how="left",
        validate="one_to_one",
    )
    output[list(LEFT_SIDE_SIGNALS)] = (
        output[list(LEFT_SIDE_SIGNALS)].fillna(False).astype(bool)
    )
    output = output[output[list(LEFT_SIDE_SIGNALS)].any(axis=1)]
    if output.duplicated(["symbol", "date"]).any():
        raise RuntimeError("left-side signal universe contains duplicate keys")
    return output.sort_values(["date", "symbol"], kind="stable").reset_index(drop=True)


def _build_symbol_feature(
    symbol: str,
    daily: pd.DataFrame,
    signal_values: Mapping[str, Any],
    target: pd.Timestamp,
) -> pd.DataFrame:
    normalized = normalize_tushare_daily(daily, symbol).sort_values(
        "date", kind="stable"
    ).reset_index(drop=True)
    if normalized.empty or not normalized["date"].dt.normalize().eq(target).any():
        raise RuntimeError(f"left-side market target row missing: {symbol}")
    project = calculate_project_market_factors(
        normalized,
        symbol=symbol,
        factor_schema_version=PROJECT_FACTOR_SCHEMA_VERSION,
    ).reset_index(drop=True)
    for column in PROJECT_FACTOR_COLUMNS:
        if column not in project:
            project[column] = np.nan
    # The 27 unique left rules need at most 60 sessions.  Keep a generous
    # 260-session causal window instead of re-running their Python state loop
    # over the six-year project-factor history.  The two shared right fields
    # are direct row-level definitions, so computing all 113 right rules here
    # would be a pure daily-update waste.
    rule_history = normalized.tail(260).reset_index(drop=True)
    left_current = compute_left_side_rule_features(rule_history).tail(1).reset_index(
        drop=True
    )
    range_ = (rule_history["high"] - rule_history["low"]).replace(0.0, np.nan)
    shared_current = pd.DataFrame(
        {
            "rs_is_rise": (rule_history["close"] > rule_history["open"]).tail(1).to_numpy(),
            "rs_close_pos": (
                (rule_history["close"] - rule_history["low"]) / range_
            ).tail(1).to_numpy(),
        }
    )
    project["date"] = pd.to_datetime(project["date"], errors="coerce").dt.normalize()
    project_current = project[project["date"].eq(target)].tail(1).reset_index(drop=True)
    base = pd.concat(
        [
            project_current[
                [
                    "ts_code",
                    "symbol",
                    "trade_date",
                    "date",
                    *PROJECT_FACTOR_COLUMNS,
                    "factor_schema_version",
                ]
            ],
            shared_current[list(LEFT_SIDE_SHARED_RULE_REQUIREMENTS)],
            left_current[list(LEFT_SIDE_RULE_FEATURE_COLUMNS)],
        ],
        axis=1,
    )
    base["date"] = pd.to_datetime(base["date"], errors="coerce").dt.normalize()
    current = base.copy()
    if len(current) != 1:
        raise RuntimeError(f"left-side feature target row is not unique: {symbol}")
    current["left_side_feature_schema_version"] = LEFT_SIDE_FEATURE_SCHEMA_VERSION
    for signal in LEFT_SIDE_SIGNALS:
        current[signal] = bool(signal_values[signal])
    return current


def build_left_side_feature_frame(
    market: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    target_date: str | pd.Timestamp,
    workers: int = 1,
) -> pd.DataFrame:
    target = pd.to_datetime(target_date, errors="raise").normalize()
    expected = [
        "ts_code",
        "symbol",
        "trade_date",
        "date",
        *LEFT_SIDE_FACTOR_COLUMNS,
        "factor_schema_version",
        "left_side_feature_schema_version",
        *LEFT_SIDE_SIGNALS,
    ]
    if signals.empty:
        return pd.DataFrame(columns=expected)
    by_symbol = {
        str(symbol): frame.drop(columns=["_symbol"], errors="ignore")
        for symbol, frame in market.assign(_symbol=market["ts_code"].astype(str)).groupby(
            "_symbol", sort=False
        )
    }
    if not 1 <= workers <= 32:
        raise ValueError("left-side production factor workers must be in [1, 32]")

    def build_one(signal: pd.Series) -> tuple[pd.DataFrame | None, str | None]:
        symbol = str(signal["symbol"])
        try:
            return (
                _build_symbol_feature(
                    symbol,
                    by_symbol.get(symbol, pd.DataFrame()),
                    signal.to_dict(),
                    target,
                ),
                None,
            )
        except Exception as exc:
            return None, f"{symbol}: {exc}"

    signal_rows = [signal for _, signal in signals.iterrows()]
    if workers == 1 or len(signal_rows) <= 1:
        results = list(map(build_one, signal_rows))
    else:
        with ThreadPoolExecutor(
            max_workers=min(workers, len(signal_rows))
        ) as executor:
            results = list(executor.map(build_one, signal_rows))
    frames = []
    failures = []
    for frame, error in results:
        if error is not None:
            failures.append(error)
        elif frame is not None:
            frames.append(frame)
    if failures:
        raise RuntimeError("left-side feature build failed: " + " | ".join(failures[:20]))
    result = pd.concat(frames, ignore_index=True, sort=False)
    missing = set(expected) - set(result.columns)
    if missing:
        raise RuntimeError(f"left-side feature output missing: {sorted(missing)}")
    result = result[expected].sort_values(["date", "symbol"], kind="stable")
    assert_no_forbidden_factor_names(result.columns, context="left production features")
    if len(result) != len(signals) or result.duplicated(["symbol", "date"]).any():
        raise RuntimeError("left-side feature coverage is incomplete or duplicated")
    return result.reset_index(drop=True)


def _fingerprint(paths: tuple[Path, ...], payload: Mapping[str, Any]) -> str:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"left-side production inputs missing: {missing}")
    document = {
        **payload,
        "inputs": [(str(path), _sha256(path)) for path in paths],
    }
    return hashlib.sha256(
        json.dumps(document, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _feature_input_fingerprint(
    target: pd.Timestamp,
    config: LeftSideRankingConfig,
) -> str:
    month = (
        config.paths.market_data_root
        / "daily_partitioned"
        / f"year_month={target.strftime('%Y%m')}"
        / "data.parquet"
    )
    paths = (
        config.paths.b1_gate_cache,
        config.paths.family_signal_cache,
        config.paths.signal_cache,
        month,
        PROJECT_ROOT / "src/quant/data/source_merge.py",
        PROJECT_ROOT / "src/quant/features/project_factor_layer.py",
        PROJECT_ROOT / "src/quant/features/variable_library.py",
        PROJECT_ROOT / "src/quant/research/left_side_unified_features.py",
    )
    return _fingerprint(paths, {
        "target_date": target.date().isoformat(),
        "builder_version": LEFT_SIDE_PRODUCTION_FEATURE_BUILDER_VERSION,
        "feature_schema_version": LEFT_SIDE_FEATURE_SCHEMA_VERSION,
        "factor_contract_sha256": LEFT_SIDE_FACTOR_CONTRACT_SHA256,
    })


def _score_input_fingerprint(
    target: pd.Timestamp,
    config: LeftSideRankingConfig,
) -> str:
    paths = (
        config.paths.artifact,
        config.paths.artifact_manifest,
        config.paths.feature_output,
        config.paths.feature_manifest,
        PROJECT_ROOT / "configs/strategies/left_side_unified.yaml",
    )
    return _fingerprint(paths, {
        "target_date": target.date().isoformat(),
        "builder_version": LEFT_SIDE_PRODUCTION_SCORE_BUILDER_VERSION,
        "score_schema_version": LEFT_SIDE_SCORE_SCHEMA_VERSION,
        "normalization_schema_version": LEFT_SIDE_NORMALIZATION_SCHEMA_VERSION,
    })


def stage_left_side_ranking_release(
    *,
    config: LeftSideRankingConfig = DEFAULT_LEFT_SIDE_RANKING_CONFIG,
    research_model: Path = DEFAULT_RESEARCH_MODEL,
) -> dict[str, Any]:
    """Create the new production bundle without touching rollback artifacts."""

    decision = _load_json(config.paths.ranking_decision)
    if (
        decision.get("schema_version")
        != "left-side-ranking-replacement-decision-v2-group4"
        or decision.get("replace_online") is not True
    ):
        raise RuntimeError("left-side ranking decision does not authorize promotion")
    expected_research_model = Path(
        str(decision.get("production_candidate_artifact") or "")
    ).resolve()
    if decision.get("production_candidate_fold") != "C":
        raise RuntimeError("left-side production candidate must be confirmed fold C")
    if research_model.resolve() != expected_research_model:
        raise RuntimeError("left-side production candidate artifact drifted from decision")
    if config.paths.artifact.exists() or config.paths.artifact_manifest.exists():
        raise FileExistsError("left-side production artifact path already exists")
    model = joblib.load(research_model)
    model_inputs = tuple(str(value) for value in model.feature_names_in_)
    if model_inputs != LEFT_SIDE_MODEL_INPUT_COLUMNS:
        raise RuntimeError("left-side research model input contract drifted")
    assert_no_forbidden_factor_names(model_inputs, context="left production model")
    normalization = {
        "schema_version": LEFT_SIDE_NORMALIZATION_SCHEMA_VERSION,
        "method": "stable_average_rank",
        "range": [0.0, 100.0],
        "single_candidate_value": 50.0,
        "changes_order": False,
    }
    strategy_thresholds = {
        signal: {
            "mode": "none_rank_only",
            "selection_policy": "downstream_top_n_ordering",
            "fixed_probability_threshold": None,
        }
        for signal in LEFT_SIDE_SIGNALS
    }
    bundle = {
        "schema_version": LEFT_SIDE_ARTIFACT_SCHEMA_VERSION,
        "lifecycle": "production",
        "release_id": config.release_id,
        "model": model,
        "features": list(LEFT_SIDE_SCORING_INPUT_COLUMNS),
        "feature_names_in": list(model_inputs),
        "selected_feature_columns": list(model_inputs),
        "factor_contract_sha256": LEFT_SIDE_FACTOR_CONTRACT_SHA256,
        "model_input_contract_sha256": LEFT_SIDE_MODEL_INPUT_CONTRACT_SHA256,
        "scoring_input_contract_sha256": LEFT_SIDE_SCORING_INPUT_CONTRACT_SHA256,
        "score_field": "ranking_score",
        "normalized_score_field": "ranking_score_normalized",
        "score_normalization": normalization,
        "production_threshold_mode": "none_rank_only",
        "strategy_thresholds": strategy_thresholds,
        "strategy_keys": list(LEFT_SIDE_SIGNALS),
        "source_research_artifact_sha256": _sha256(research_model),
        "ranking_decision_sha256": _sha256(config.paths.ranking_decision),
    }
    if find_forbidden_aliases_in_payload(bundle):
        raise RuntimeError("left-side production bundle contains forbidden aliases")
    _atomic_dump_joblib(bundle, config.paths.artifact)
    manifest = {
        "status": "success",
        "schema_version": LEFT_SIDE_ARTIFACT_SCHEMA_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "lifecycle": "production",
        "release_id": config.release_id,
        "score_field": "ranking_score",
        "normalized_score_field": "ranking_score_normalized",
        "features": list(LEFT_SIDE_SCORING_INPUT_COLUMNS),
        "feature_names_in": list(model_inputs),
        "selected_feature_columns": list(model_inputs),
        "score_normalization": normalization,
        "production_threshold_mode": "none_rank_only",
        "strategy_thresholds": strategy_thresholds,
        "strategy_keys": list(LEFT_SIDE_SIGNALS),
        "artifact_sha256": _sha256(config.paths.artifact),
        "source_research_artifact": str(research_model.relative_to(PROJECT_ROOT)),
        "source_research_artifact_sha256": _sha256(research_model),
        "ranking_decision_sha256": _sha256(config.paths.ranking_decision),
        "rollback_artifacts": (
            yaml.safe_load(
                (PROJECT_ROOT / "configs/strategies/left_side_unified.yaml").read_text(
                    encoding="utf-8"
                )
            )
            .get("promotion", {})
            .get("preserve_rollback_artifacts", [])
        ),
        **left_side_contract_payload(),
    }
    if find_forbidden_aliases_in_payload(manifest):
        raise RuntimeError("left-side production manifest contains forbidden aliases")
    atomic_write_json(manifest, config.paths.artifact_manifest)
    validate_left_side_production_artifact(config)
    return manifest


def validate_left_side_production_artifact(
    config: LeftSideRankingConfig = DEFAULT_LEFT_SIDE_RANKING_CONFIG,
) -> dict[str, Any]:
    manifest = _load_json(config.paths.artifact_manifest)
    if manifest.get("schema_version") != LEFT_SIDE_ARTIFACT_SCHEMA_VERSION:
        raise RuntimeError("left-side artifact manifest schema mismatch")
    if manifest.get("artifact_sha256") != _sha256(config.paths.artifact):
        raise RuntimeError("left-side artifact checksum mismatch")
    bundle = joblib.load(config.paths.artifact)
    if not isinstance(bundle, dict):
        raise RuntimeError("left-side artifact must be a bundle mapping")
    if bundle.get("schema_version") != LEFT_SIDE_ARTIFACT_SCHEMA_VERSION:
        raise RuntimeError("left-side production bundle schema mismatch")
    if tuple(bundle.get("features") or ()) != LEFT_SIDE_SCORING_INPUT_COLUMNS:
        raise RuntimeError("left-side scoring input contract mismatch")
    model_inputs = tuple(bundle["model"].feature_names_in_)
    if model_inputs != LEFT_SIDE_MODEL_INPUT_COLUMNS:
        raise RuntimeError("left-side model feature_names_in_ contract mismatch")
    if bundle.get("factor_contract_sha256") != LEFT_SIDE_FACTOR_CONTRACT_SHA256:
        raise RuntimeError("left-side factor contract hash mismatch")
    if (
        bundle.get("model_input_contract_sha256")
        != LEFT_SIDE_MODEL_INPUT_CONTRACT_SHA256
        or bundle.get("scoring_input_contract_sha256")
        != LEFT_SIDE_SCORING_INPUT_CONTRACT_SHA256
    ):
        raise RuntimeError("left-side model/scoring input hash mismatch")
    if bundle.get("score_normalization", {}).get("schema_version") != (
        LEFT_SIDE_NORMALIZATION_SCHEMA_VERSION
    ):
        raise RuntimeError("left-side score normalization contract mismatch")
    forbidden = find_forbidden_aliases_in_payload({"manifest": manifest, "bundle": bundle})
    if forbidden:
        raise RuntimeError(f"left-side production artifact contains forbidden aliases: {forbidden}")
    return bundle


def build_left_side_production_features(
    target_date: str,
    *,
    config: LeftSideRankingConfig = DEFAULT_LEFT_SIDE_RANKING_CONFIG,
) -> dict[str, Any]:
    target = pd.to_datetime(target_date, errors="raise").normalize()
    signals = load_left_side_signal_universe(target, config=config)
    symbols = sorted(signals["symbol"].astype(str).unique()) if not signals.empty else []
    store = MarketDataStore(
        MarketDataStoreConfig(backend="parquet", root=config.paths.market_data_root)
    )
    market = (
        store.read_market_range(
            "daily",
            start_date=(target - pd.DateOffset(years=6)).strftime("%Y%m%d"),
            end_date=target.strftime("%Y%m%d"),
            symbols=symbols,
            columns=(
                "ts_code",
                "trade_date",
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
        )
        if symbols
        else pd.DataFrame()
    )
    frame = build_left_side_feature_frame(
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
        "signal_candidate_count": len(signals),
        "computed_candidate_count": len(frame),
        "output_sha256": _sha256(config.paths.feature_output),
        "source_input_fingerprint": _feature_input_fingerprint(target, config),
        **left_side_contract_payload(),
    }
    atomic_write_json(manifest, config.paths.feature_manifest)
    return manifest


def score_left_side_production(
    target_date: str,
    *,
    config: LeftSideRankingConfig = DEFAULT_LEFT_SIDE_RANKING_CONFIG,
) -> dict[str, Any]:
    bundle = validate_left_side_production_artifact(config)
    target = pd.to_datetime(target_date, errors="raise").normalize()
    feature_manifest = _load_json(config.paths.feature_manifest)
    if feature_manifest.get("target_date") != target.date().isoformat():
        raise RuntimeError("left-side production features are stale")
    if feature_manifest.get("output_sha256") != _sha256(config.paths.feature_output):
        raise RuntimeError("left-side feature checksum mismatch")
    frame = pd.read_parquet(config.paths.feature_output)
    missing = set(LEFT_SIDE_SCORING_INPUT_COLUMNS) - set(frame.columns)
    if missing:
        raise RuntimeError(f"left-side production features missing: {sorted(missing)}")
    probabilities = (
        np.asarray(bundle["model"].predict_proba(frame), dtype=float)[:, 1]
        if not frame.empty
        else np.asarray([], dtype=float)
    )
    if not np.isfinite(probabilities).all():
        raise RuntimeError("left-side ranker returned non-finite probabilities")
    scored = frame[["ts_code", "symbol", "trade_date", "date", *LEFT_SIDE_SIGNALS]].copy()
    scored["ranking_score"] = probabilities
    scored["ranking_score_normalized"] = normalize_daily_percentile(probabilities)
    scored["model_artifact_sha256"] = _sha256(config.paths.artifact)
    atomic_write_parquet(scored, config.paths.score_output, index=False)
    manifest = {
        "status": "success",
        "schema_version": LEFT_SIDE_SCORE_SCHEMA_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "target_date": target.date().isoformat(),
        "artifact_schema_version": LEFT_SIDE_ARTIFACT_SCHEMA_VERSION,
        "artifact_sha256": _sha256(config.paths.artifact),
        "factor_contract_sha256": LEFT_SIDE_FACTOR_CONTRACT_SHA256,
        "source_input_fingerprint": _score_input_fingerprint(target, config),
        "candidate_count": len(scored),
        "score_field": "ranking_score",
        "normalized_score_field": "ranking_score_normalized",
        "score_normalization": bundle["score_normalization"],
        "production_threshold_mode": "none_rank_only",
        "selection_policy": "downstream_top_n_ordering",
        "strategy_thresholds": bundle["strategy_thresholds"],
        "selector_adapter_status": "ready",
        "output_sha256": _sha256(config.paths.score_output),
    }
    if find_forbidden_aliases_in_payload(manifest):
        raise RuntimeError("left-side score manifest contains forbidden aliases")
    atomic_write_json(manifest, config.paths.score_manifest)
    return manifest


def validate_left_side_selector_adapter(
    target_date: str,
    *,
    config: LeftSideRankingConfig = DEFAULT_LEFT_SIDE_RANKING_CONFIG,
) -> dict[str, Any]:
    scores, manifest = load_left_side_ranking_scores(target_date, config=config)
    return {
        "status": "success",
        "target_date": pd.Timestamp(target_date).date().isoformat(),
        "candidate_count": len(scores),
        "artifact_sha256": manifest.get("artifact_sha256"),
    }


def run_left_side_production(
    target_date: str,
    *,
    config: LeftSideRankingConfig = DEFAULT_LEFT_SIDE_RANKING_CONFIG,
) -> dict[str, Any]:
    if not config.enabled:
        raise RuntimeError("left-side unified ranking is disabled")
    target = pd.to_datetime(target_date, errors="raise").normalize()
    feature_fingerprint = _feature_input_fingerprint(target, config)
    feature_checkpoint_reused = False
    feature: dict[str, Any]
    try:
        feature = _load_json(config.paths.feature_manifest)
        feature_checkpoint_reused = (
            feature.get("status") == "success"
            and feature.get("target_date") == target.date().isoformat()
            and feature.get("candidate_coverage_status") == "complete"
            and feature.get("factor_contract_sha256")
            == LEFT_SIDE_FACTOR_CONTRACT_SHA256
            and feature.get("source_input_fingerprint") == feature_fingerprint
            and feature.get("output_sha256") == _sha256(config.paths.feature_output)
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError):
        feature = {}
        feature_checkpoint_reused = False
    if not feature_checkpoint_reused:
        feature = build_left_side_production_features(target_date, config=config)

    score_fingerprint = _score_input_fingerprint(target, config)
    if config.paths.score_manifest.is_file() and config.paths.score_output.is_file():
        existing = _load_json(config.paths.score_manifest)
        if (
            existing.get("target_date") == target.date().isoformat()
            and existing.get("source_input_fingerprint") == score_fingerprint
            and existing.get("output_sha256") == _sha256(config.paths.score_output)
        ):
            validate_left_side_selector_adapter(target_date, config=config)
            return {
                "status": "success",
                "target_date": target.date().isoformat(),
                "checkpoint_reused": True,
                "feature_checkpoint_reused": feature_checkpoint_reused,
                "candidate_count": existing.get("candidate_count"),
            }
    score = score_left_side_production(target_date, config=config)
    adapter = validate_left_side_selector_adapter(target_date, config=config)
    return {
        "status": "success",
        "target_date": target.date().isoformat(),
        "checkpoint_reused": False,
        "feature_checkpoint_reused": feature_checkpoint_reused,
        "feature": feature,
        "score": score,
        "adapter": adapter,
    }


__all__ = [
    "LEFT_SIDE_NORMALIZATION_SCHEMA_VERSION",
    "build_left_side_feature_frame",
    "build_left_side_production_features",
    "load_left_side_signal_universe",
    "normalize_daily_percentile",
    "run_left_side_production",
    "score_left_side_production",
    "stage_left_side_ranking_release",
    "validate_left_side_production_artifact",
    "validate_left_side_selector_adapter",
]
