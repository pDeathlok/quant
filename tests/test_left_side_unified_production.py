from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from quant.application.left_side_ranking import DEFAULT_LEFT_SIDE_RANKING_CONFIG
from quant.application.daily_dependencies import (
    DEFAULT_DAILY_DEPENDENCY_REGISTRY,
    Lifecycle,
)
from quant.features.project_factor_layer import PROJECT_FACTOR_SCHEMA_VERSION
from quant.features.variable_library import PROJECT_FACTOR_COLUMNS
from quant.research.left_side_unified_features import (
    LEFT_SIDE_RULE_FEATURE_COLUMNS,
    LEFT_SIDE_SIGNALS,
)
from quant.features.candlestick_context import CANDLE_CONTEXT_FEATURE_COLUMNS
from quant.routine import left_side_unified_production as production
from quant.routine.left_side_unified_production import (
    normalize_daily_percentile,
    validate_left_side_production_artifact,
)


def test_daily_percentile_normalization_is_monotonic_and_tie_stable() -> None:
    scores = np.asarray([0.2, 0.1, 0.2, 0.9], dtype=float)
    normalized = normalize_daily_percentile(scores)

    assert normalized.tolist() == [50.0, 0.0, 50.0, 100.0]
    assert normalize_daily_percentile(np.asarray([0.4])).tolist() == [50.0]
    assert normalize_daily_percentile(np.asarray([])).tolist() == []


def test_promoted_left_artifact_and_daily_registry_are_consistent() -> None:
    bundle = validate_left_side_production_artifact()
    registry = DEFAULT_DAILY_DEPENDENCY_REGISTRY
    short = set(registry.required_node_ids("short"))

    assert bundle["production_threshold_mode"] == "none_rank_only"
    assert "score.left_side_unified" in short
    assert "score.z_skill" not in short
    assert "score.b1" not in short
    assert registry.nodes["score.z_skill"].lifecycle == Lifecycle.RETIRED
    assert registry.nodes["score.b1"].lifecycle == Lifecycle.RETIRED


def _project_row(symbol: str, target: pd.Timestamp) -> dict[str, object]:
    row: dict[str, object] = {column: 1.0 for column in PROJECT_FACTOR_COLUMNS}
    row.update(
        ts_code=symbol,
        symbol=symbol,
        trade_date=target.strftime("%Y%m%d"),
        date=target,
        factor_schema_version=PROJECT_FACTOR_SCHEMA_VERSION,
    )
    return row


def test_left_feature_builder_consumes_shared_project_factor_row(monkeypatch) -> None:
    target = pd.Timestamp("2026-08-31")
    symbol = "000001.SZ"
    project = pd.DataFrame([_project_row(symbol, target)])
    project.loc[0, "turnover_rate"] = 7.5
    daily = pd.DataFrame(
        {
            "ts_code": [symbol],
            "trade_date": ["20260831"],
            "open": [10.0],
            "high": [10.8],
            "low": [9.8],
            "close": [10.5],
            "pre_close": [10.0],
            "pct_chg": [5.0],
            "vol": [1_000.0],
        }
    )
    monkeypatch.setattr(
        production,
        "compute_left_side_rule_features",
        lambda _: pd.DataFrame(
            {column: [0.0] for column in LEFT_SIDE_RULE_FEATURE_COLUMNS}
        ),
    )
    monkeypatch.setattr(
        production,
        "compute_candlestick_context_features",
        lambda _: pd.DataFrame(
            {column: [0.0] for column in CANDLE_CONTEXT_FEATURE_COLUMNS}
        ),
    )

    result = production._build_symbol_feature(
        symbol,
        daily,
        project,
        {signal: True for signal in LEFT_SIDE_SIGNALS},
        target,
    )

    assert result.loc[0, "turnover_rate"] == 7.5
    assert result.loc[0, "factor_schema_version"] == PROJECT_FACTOR_SCHEMA_VERSION


def test_project_feature_cache_requires_explained_candidate_coverage(
    tmp_path,
) -> None:
    target = pd.Timestamp("2026-08-31")
    cache = tmp_path / "project.parquet"
    manifest_path = tmp_path / "project.json"
    pd.DataFrame([_project_row("000001.SZ", target)]).to_parquet(
        cache,
        index=False,
    )
    paths = replace(
        DEFAULT_LEFT_SIDE_RANKING_CONFIG.paths,
        project_feature_cache=cache,
        project_feature_manifest=manifest_path,
    )
    config = replace(DEFAULT_LEFT_SIDE_RANKING_CONFIG, paths=paths)
    signals = pd.DataFrame(
        {
            "symbol": ["000001.SZ", "000002.SZ"],
            "date": [target, target],
        }
    )
    manifest = {
        "status": "success",
        "target_date": "2026-08-31",
        "candidate_coverage_status": "complete",
        "factor_schema_version": PROJECT_FACTOR_SCHEMA_VERSION,
        "output_sha256": production._sha256(cache),
        "policy_excluded_candidate_symbols": ["000002.SZ"],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    features, eligible, excluded = production._load_project_feature_cache(
        target,
        signals,
        config,
    )
    assert features["symbol"].tolist() == ["000001.SZ"]
    assert eligible["symbol"].tolist() == ["000001.SZ"]
    assert excluded == ["000002.SZ"]

    manifest["policy_excluded_candidate_symbols"] = []
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="unexplained missing candidates"):
        production._load_project_feature_cache(target, signals, config)

    broken = pd.DataFrame(
        [
            _project_row("000001.SZ", target),
            _project_row("000003.SZ", target),
        ]
    )
    broken.loc[0, "ts_volume_ratio"] = np.nan
    broken.to_parquet(cache, index=False)
    manifest["output_sha256"] = production._sha256(cache)
    manifest["policy_excluded_candidate_symbols"] = ["000002.SZ"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="incomplete daily_basic values"):
        production._load_project_feature_cache(target, signals, config)
