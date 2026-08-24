from __future__ import annotations

from pathlib import Path

import yaml

from quant.application.left_side_ranking import load_left_side_ranking_config
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


def test_repository_left_side_release_is_canonical_and_promoted_after_gate() -> None:
    config = load_left_side_ranking_config()

    assert config.enabled is True
    assert config.strategy_keys == LEFT_SIDE_SIGNALS
    assert config.paths.artifact.name == "ranking.joblib"
    assert "left_side_unified_canonical_v4_group4" in str(config.paths.artifact)
    assert config.score_normalization == "daily_cross_section_percentile_v1"


def test_left_side_release_rejects_contract_drift(tmp_path: Path) -> None:
    payload = {
        "release": {"id": "test"},
        "selector": {
            "enabled": False,
            "failure_policy": "fail_closed",
            "score_field": "ranking_score",
            "normalized_score_field": "ranking_score_normalized",
            "overlap_policy": "right_side_precedence",
            "normalization": "daily_cross_section_percentile_v1",
            "strategy_keys": list(LEFT_SIDE_SIGNALS),
        },
        "model": {
            "artifact": "models/test/ranking.joblib",
            "artifact_manifest": "models/test/manifest.json",
            "artifact_schema_version": LEFT_SIDE_ARTIFACT_SCHEMA_VERSION,
            "feature_schema_version": LEFT_SIDE_FEATURE_SCHEMA_VERSION,
            "rule_factor_schema_version": LEFT_SIDE_RULE_FEATURE_SCHEMA_VERSION,
            "score_schema_version": LEFT_SIDE_SCORE_SCHEMA_VERSION,
            "factor_contract_sha256": LEFT_SIDE_FACTOR_CONTRACT_SHA256,
            "model_input_contract_sha256": LEFT_SIDE_MODEL_INPUT_CONTRACT_SHA256,
            "scoring_input_contract_sha256": "drifted",
        },
        "paths": {
            "project_feature_cache": "data/project.parquet",
            "project_feature_manifest": "data/project.json",
            "signal_cache": "data/signals.parquet",
            "b1_gate_cache": "data/b1.parquet",
            "family_signal_cache": "data/family.parquet",
            "market_data_root": "data/raw",
            "feature_output": "data/left/features.parquet",
            "feature_manifest": "data/left/features.json",
            "score_output": "data/left/scores.parquet",
            "score_manifest": "data/left/scores.json",
            "ranking_decision": "reports/left/decision.json",
        },
    }
    path = tmp_path / "left.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    try:
        load_left_side_ranking_config(tmp_path, path)
    except ValueError as exc:
        assert "contract drifted" in str(exc)
    else:
        raise AssertionError("drifted scoring input hash was accepted")

    payload["model"]["scoring_input_contract_sha256"] = (
        LEFT_SIDE_SCORING_INPUT_CONTRACT_SHA256
    )
