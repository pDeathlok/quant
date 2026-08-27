from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
import yaml
import joblib
import numpy as np

from quant.application import selector_ranking as selector_ranking_module
from quant.application.daily_dependencies import (
    build_default_daily_dependency_registry,
)
from quant.application.left_side_ranking import DEFAULT_LEFT_SIDE_RANKING_CONFIG
from quant.application.selector_ranking import (
    NORMALIZED_RANKING_SCORE_FIELD,
    RANKING_NORMALIZATION_SCHEMA_VERSION,
    RIGHT_SIDE_PRODUCTION_APPROVAL_SCHEMA_VERSION,
    RIGHT_SIDE_PRODUCTION_ARTIFACT_SCHEMA_VERSION,
    RIGHT_SIDE_PRODUCTION_SCORE_SCHEMA_VERSION,
    SelectorRankingPaths,
    SelectorRankingSource,
    apply_selector_ranking_source,
    load_right_side_ranking_scores,
    load_selector_ranking_config,
)
from quant.features.right_side_factor_contract import (
    RIGHT_SIDE_SHADOW_FACTOR_CONTRACT_SHA256,
    RIGHT_SIDE_SHADOW_IDENTITY_COLUMNS,
    factor_contract_sha256,
)
from quant.features.variable_library import PROJECT_FACTOR_COLUMNS
from quant.research.right_side_beam_feature_selection import (
    BEAM_SCHEMA_VERSION,
    feature_columns_sha256 as beam_feature_columns_sha256,
)
from quant.research.right_side_unified_features import (
    ADDED_RULE_FEATURE_COLUMNS_V2,
    LEGACY_RULE_FEATURE_COLUMNS_V1,
    RIGHT_SIDE_SIGNALS,
    RULE_FEATURE_COLUMNS_SHA256,
    rule_feature_columns_sha256,
)
from quant.routine.right_side_unified_production import (
    validate_production_ranking_artifact,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _promoted_config(tmp_path: Path):
    base = load_selector_ranking_config()
    score_output = tmp_path / "data/features/right_side_unified/latest_scores.parquet"
    return replace(
        base,
        source=SelectorRankingSource.RIGHT_SIDE_UNIFIED,
        promotion_enabled=True,
        paths=SelectorRankingPaths(
            artifact=tmp_path / "models/production/right_side_unified/ranking.joblib",
            artifact_manifest=tmp_path / "models/production/right_side_unified/manifest.json",
            feature_output=tmp_path / "data/features/right_side_unified/latest_features.parquet",
            feature_manifest=tmp_path / "data/features/right_side_unified/feature_manifest.json",
            score_output=score_output,
            score_manifest=score_output.with_name("score_manifest.json"),
            z_signal_cache=tmp_path / "data/features/z_skill_daily_candidates.parquet",
            family_signal_cache=tmp_path / "data/features/b1/b1_family_rule_candidates.parquet",
            market_data_root=tmp_path / "data/raw",
            promotion_approval=tmp_path / "reports/production_rollout_approval.json",
        ),
    )


def _write_promotion_approval(
    config,
    *,
    selected_candidate: str,
    approve_online: bool = True,
) -> None:
    config.paths.promotion_approval.parent.mkdir(parents=True, exist_ok=True)
    evidence_dir = config.paths.promotion_approval.parent
    research_path = evidence_dir / "research_decision.json"
    shadow_path = evidence_dir / "shadow_acceptance.json"
    research_path.write_text(
        json.dumps(
            {
                "selected_research_candidate": selected_candidate,
                "replace_online": True,
            }
        ),
        encoding="utf-8",
    )
    shadow_path.write_text(
        json.dumps(
            {
                "status": "success",
                "production_affected": False,
                "selector_published": False,
            }
        ),
        encoding="utf-8",
    )
    config.paths.promotion_approval.write_text(
        json.dumps(
            {
                "schema_version": RIGHT_SIDE_PRODUCTION_APPROVAL_SCHEMA_VERSION,
                "selected_research_candidate": selected_candidate,
                "approve_online": approve_online,
                "deployment_mode": "reversible_unified_ranking_cutover",
                "playbook_promoted": False,
                "rollback_ranking_source": "legacy_z_skill",
                "research_decision": {
                    "path": str(research_path),
                    "sha256": _sha256(research_path),
                },
                "shadow_acceptance": {
                    "path": str(shadow_path),
                    "sha256": _sha256(shadow_path),
                },
                "acknowledged_risks": [
                    "canonical_alias_free_retrain_completed",
                    "legacy_artifact_preserved_for_rollback",
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_scores(config, rows: list[dict[str, object]]) -> None:
    selected_candidate = "unified_long_task_deep"
    _write_promotion_approval(
        config,
        selected_candidate=selected_candidate,
    )
    config.paths.artifact.parent.mkdir(parents=True, exist_ok=True)
    config.paths.artifact.write_bytes(b"test-production-ranking-artifact")
    artifact_sha = _sha256(config.paths.artifact)
    config.paths.artifact_manifest.write_text(
        json.dumps(
            {
                "schema_version": RIGHT_SIDE_PRODUCTION_ARTIFACT_SCHEMA_VERSION,
                "lifecycle": "production",
                "score_field": "ranking_score",
                "playbook_coupling": "independent",
                "probability_calibration": config.probability_calibration,
                "production_threshold_mode": config.production_threshold_mode,
                "score_normalization": {
                    "schema_version": RANKING_NORMALIZATION_SCHEMA_VERSION
                },
                "replaced_signals": list(config.replaced_signals),
                "preserved_legacy_signals": list(
                    config.preserved_legacy_signals
                ),
                "selected_candidate": selected_candidate,
                "promotion_approval_sha256": _sha256(
                    config.paths.promotion_approval
                ),
                "models": {"ranking": {"sha256": artifact_sha}},
            }
        ),
        encoding="utf-8",
    )
    config.paths.score_output.parent.mkdir(parents=True, exist_ok=True)
    score_rows = [dict(row) for row in rows]
    for row in score_rows:
        row.setdefault(
            NORMALIZED_RANKING_SCORE_FIELD,
            float(row.get("ranking_score") or 0.0) * 100.0,
        )
    pd.DataFrame(score_rows).to_parquet(config.paths.score_output, index=False)
    config.paths.score_manifest.write_text(
        json.dumps(
            {
                "status": "success",
                "schema_version": RIGHT_SIDE_PRODUCTION_SCORE_SCHEMA_VERSION,
                "target_date": "2026-08-12",
                "score_field": "ranking_score",
                "normalized_score_field": NORMALIZED_RANKING_SCORE_FIELD,
                "probability_calibration": config.probability_calibration,
                "production_threshold_mode": config.production_threshold_mode,
                "selection_policy": config.selection_policy,
                "score_normalization": {
                    "schema_version": RANKING_NORMALIZATION_SCHEMA_VERSION
                },
                "artifact_schema_version": RIGHT_SIDE_PRODUCTION_ARTIFACT_SCHEMA_VERSION,
                "factor_contract_sha256": RIGHT_SIDE_SHADOW_FACTOR_CONTRACT_SHA256,
                "playbook_coupling": "independent",
                "replaced_signals": list(config.replaced_signals),
                "preserved_legacy_signals": list(
                    config.preserved_legacy_signals
                ),
                "promotion_approval_sha256": _sha256(
                    config.paths.promotion_approval
                ),
                "candidate_count": len(rows),
                "artifact_sha256": artifact_sha,
                "output_sha256": _sha256(config.paths.score_output),
            }
        ),
        encoding="utf-8",
    )


def test_repository_selector_ranking_source_is_unified_and_promotion_enabled() -> None:
    config = load_selector_ranking_config()

    assert config.source == SelectorRankingSource.RIGHT_SIDE_UNIFIED
    assert config.promotion_enabled is True
    assert config.preserved_legacy_signals == (
        "DUICHEN_VA",
        "NANA",
        "YIDONG_DILIAN",
    )


def test_config_rejects_unified_source_without_explicit_promotion(tmp_path: Path) -> None:
    source = Path("configs/strategies/right_side_ranking_selector.yaml")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["selector"]["ranking_source"] = "right_side_unified"
    payload["promotion"]["enabled"] = False
    target = tmp_path / "selector.yaml"
    target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="promotion.enabled=true"):
        load_selector_ranking_config(tmp_path, target)


def test_unified_adapter_uses_only_ranking_score_and_keeps_playbook_scores(
    tmp_path: Path,
) -> None:
    config = _promoted_config(tmp_path)
    _write_scores(
        config,
        [
            {"symbol": "000001.SZ", "date": "2026-08-12", "ranking_score": 0.91},
            {"symbol": "000002.SZ", "date": "2026-08-12", "ranking_score": 0.22},
        ],
    )
    rows = [
        {
            "symbol": "000001.SZ",
            "selector_score": 40.0,
            "opportunity_score": 40.0,
            "holding_score": 55.0,
            "signals": [{"strategy_key": "B2", "buy_plan": "A", "sell_plan": "B"}],
        },
        {
            "symbol": "000002.SZ",
            "selector_score": 80.0,
            "opportunity_score": 80.0,
            "holding_score": 65.0,
            "signals": [{"strategy_key": "NANA", "buy_plan": "C", "sell_plan": "D"}],
        },
    ]

    result = apply_selector_ranking_source(
        rows,
        "2026-08-12",
        config=config,
    )

    assert result[0]["ranking_score"] == pytest.approx(0.91)
    assert result[0]["selector_score"] == pytest.approx(91.0)
    assert result[0]["opportunity_score"] == 40.0
    assert result[0]["holding_score"] == 55.0
    assert result[0]["signals"][0]["buy_plan"] == "A"
    assert result[0]["signals"][0]["sell_plan"] == "B"
    assert not {"pred_up5", "pred_up8", "pred_down3"} & set(result[0])
    assert result[1]["selector_score"] == 80.0
    assert result[1]["ranking_source"] == "unified_ranker_not_applicable"


def test_unified_adapter_fails_closed_on_missing_eligible_score(tmp_path: Path) -> None:
    config = _promoted_config(tmp_path)
    _write_scores(
        config,
        [{"symbol": "000001.SZ", "date": "2026-08-12", "ranking_score": 0.91}],
    )
    rows = [
        {
            "symbol": "000099.SZ",
            "selector_score": 40.0,
            "signals": [{"strategy_key": "B3"}],
        }
    ]

    with pytest.raises(RuntimeError, match="coverage is incomplete"):
        apply_selector_ranking_source(rows, "2026-08-12", config=config)


def test_full_materialization_accepts_left_candidate_consumed_by_right_precedence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _promoted_config(tmp_path)
    monkeypatch.setattr(
        selector_ranking_module,
        "load_right_side_ranking_scores",
        lambda *args, **kwargs: (
            {"000001.SZ": (0.9, 90.0)},
            {"artifact_sha256": "right"},
        ),
    )
    monkeypatch.setattr(
        selector_ranking_module,
        "load_left_side_ranking_scores",
        lambda *args, **kwargs: (
            {
                "000001.SZ": (0.8, 80.0),
                "000002.SZ": (0.7, 70.0),
            },
            {"artifact_sha256": "left"},
        ),
    )
    rows = [
        {
            "symbol": "000001.SZ",
            "selector_score": 0.0,
            "signals": [
                {"strategy_key": "B2"},
                {"strategy_key": "B1"},
            ],
        },
        {
            "symbol": "000002.SZ",
            "selector_score": 0.0,
            "signals": [
                {
                    "strategy_key": "LOW_PULLBACK",
                    "strategy_group": "LOW_PULLBACK",
                }
            ],
        },
    ]

    result = apply_selector_ranking_source(
        rows,
        "2026-08-12",
        config=config,
        left_config=DEFAULT_LEFT_SIDE_RANKING_CONFIG,
        require_all_ranked_candidates=True,
    )

    assert result[0]["ranking_source"] == "right_side_unified"
    assert result[1]["ranking_source"] == "left_side_unified"


def test_full_materialization_still_rejects_absent_left_candidate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _promoted_config(tmp_path)
    monkeypatch.setattr(
        selector_ranking_module,
        "load_right_side_ranking_scores",
        lambda *args, **kwargs: ({}, {"artifact_sha256": "right"}),
    )
    monkeypatch.setattr(
        selector_ranking_module,
        "load_left_side_ranking_scores",
        lambda *args, **kwargs: (
            {
                "000001.SZ": (0.8, 80.0),
                "000099.SZ": (0.7, 70.0),
            },
            {"artifact_sha256": "left"},
        ),
    )

    with pytest.raises(RuntimeError, match="did not materialize"):
        apply_selector_ranking_source(
            [
                {
                    "symbol": "000001.SZ",
                    "selector_score": 0.0,
                    "signals": [{"strategy_key": "B1"}],
                }
            ],
            "2026-08-12",
            config=config,
            left_config=DEFAULT_LEFT_SIDE_RANKING_CONFIG,
            require_all_ranked_candidates=True,
        )


def test_unified_adapter_rejects_missing_operator_approval(
    tmp_path: Path,
) -> None:
    config = _promoted_config(tmp_path)
    _write_scores(
        config,
        [{"symbol": "000001.SZ", "date": "2026-08-12", "ranking_score": 0.91}],
    )
    _write_promotion_approval(
        config,
        selected_candidate="unified_long_task_deep_rule105",
        approve_online=False,
    )

    with pytest.raises(RuntimeError, match="approve_online approval"):
        apply_selector_ranking_source(
            [
                {
                    "symbol": "000001.SZ",
                    "signals": [{"strategy_key": "B2"}],
                }
            ],
            "2026-08-12",
            config=config,
        )


def test_score_snapshot_rejects_legacy_probability_aliases(tmp_path: Path) -> None:
    config = _promoted_config(tmp_path)
    _write_scores(
        config,
        [
            {
                "symbol": "000001.SZ",
                "date": "2026-08-12",
                "ranking_score": 0.91,
                "pred_up5": 0.91,
            }
        ],
    )

    with pytest.raises(RuntimeError, match="must not impersonate legacy targets"):
        load_right_side_ranking_scores("2026-08-12", config=config)


def test_registry_source_switch_is_explicit_and_legacy_is_rollback_default() -> None:
    default = build_default_daily_dependency_registry(
        SelectorRankingSource.LEGACY_Z_SKILL
    )
    promoted = build_default_daily_dependency_registry(
        SelectorRankingSource.RIGHT_SIDE_UNIFIED
    )
    default_inputs = {edge.upstream for edge in default.nodes["score.selector"].inputs}
    promoted_inputs = {edge.upstream for edge in promoted.nodes["score.selector"].inputs}

    assert "score.z_skill" in default_inputs
    assert "score.right_side_unified" not in default_inputs
    assert "score.right_side_unified" in promoted_inputs
    assert "score.z_skill" not in promoted_inputs
    assert "score.left_side_unified" in promoted_inputs
    assert "score.right_side_unified" not in set(default.required_node_ids("short"))
    assert "score.right_side_unified" in set(promoted.required_node_ids("short"))
    assert default.required_node_ids("rightSideRankingCandidate")[-1] == (
        "product.right_side_unified_adapter"
    )
    config = load_selector_ranking_config()
    assert tuple(RIGHT_SIDE_SIGNALS) == config.supported_strategy_keys
    assert config.replaced_signals == tuple(RIGHT_SIDE_SIGNALS)
    assert config.preserved_legacy_signals == (
        "DUICHEN_VA",
        "NANA",
        "YIDONG_DILIAN",
    )

    default_z = default.nodes["score.z_skill"].artifact
    promoted_z = promoted.nodes["score.z_skill"].artifact
    assert default_z is not None and len(default_z.artifact_paths) == 30
    assert promoted_z is not None and len(promoted_z.artifact_paths) == 0
    assert promoted_z.artifact_paths == ()
    assert not any(
        f"/{signal}_" in path
        for signal in ("B2", "BREATHING", "GOLDEN_BOWL", "KEY_K", "VIOLENCE_K", "YUEYUE", "ZAIHOU")
        for path in promoted_z.artifact_paths
    )


def test_selector_snapshot_cache_cannot_cross_ranking_source_cutover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quant.webapp import services

    monkeypatch.setattr(
        services,
        "DEFAULT_SELECTOR_RANKING_CONFIG",
        _promoted_config(Path("/tmp/right-side-selector-cache-contract")),
    )
    assert services._selector_snapshot_matches_current_ranking_source(
        {"ranking_source": "right_side_unified"}
    )
    assert not services._selector_snapshot_matches_current_ranking_source(
        {"ranking_source": "legacy_z_skill"}
    )
    assert not services._selector_snapshot_matches_current_ranking_source({})


def test_latest_candidate_date_does_not_fall_back_to_an_old_selector_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quant.webapp import services

    monkeypatch.setattr(
        services,
        "_selector_snapshot_dates",
        lambda strategies, include_extended: ["2026-08-20", "2026-08-21"],
    )
    monkeypatch.setattr(
        services,
        "_latest_candidate_signal_date",
        lambda: "2026-08-24",
    )

    assert services._resolve_selector_signal_date(
        "2026-08-24", None, False
    ) == "2026-08-24"
    assert services._resolve_selector_signal_date(
        "2026-08-23", None, False
    ) == "2026-08-21"


def test_two_unified_rankers_retire_legacy_strategy_model_score_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quant.webapp import services

    scored = tmp_path / "legacy_z_scores.parquet"
    pd.DataFrame(
        [
            {"symbol": "000001.SZ", "signal": "B2", "model_pass": True},
            {"symbol": "000002.SZ", "signal": "NANA", "model_pass": True},
            {"symbol": "000003.SZ", "signal": "YIDONG_DILIAN", "model_pass": True},
            {"symbol": "000004.SZ", "signal": "DUICHEN_VA", "model_pass": True},
            {"symbol": "000005.SZ", "signal": "KEY_K", "model_pass": True},
        ]
    ).to_parquet(scored, index=False)
    monkeypatch.setattr(services, "EXTENDED_MODEL_SCORED", scored)
    monkeypatch.setattr(
        services,
        "DEFAULT_SELECTOR_RANKING_CONFIG",
        _promoted_config(tmp_path),
    )
    services._model_scored_candidates_for_date.cache_clear()
    try:
        result = services._model_scored_candidates_for_date(None)
    finally:
        services._model_scored_candidates_for_date.cache_clear()

    assert result == {}


class _ProductionModel:
    def predict_proba(self, values: pd.DataFrame) -> np.ndarray:
        score = np.full(len(values), 0.5)
        return np.column_stack([1.0 - score, score])


def _write_production_bundle(
    config,
    *,
    permutation_gate_passed: bool,
    pipeline_select_gate_passed: bool = True,
    max_remove: int = 10,
) -> None:
    _write_promotion_approval(
        config,
        selected_candidate="unified_long_task_deep_beam",
    )
    selected_increment = tuple(ADDED_RULE_FEATURE_COLUMNS_V2[:6])
    selected_rules = (*LEGACY_RULE_FEATURE_COLUMNS_V1, *selected_increment)
    features = (
        *PROJECT_FACTOR_COLUMNS,
        *selected_rules,
        *RIGHT_SIDE_SHADOW_IDENTITY_COLUMNS,
    )
    beam = {
        "schema_version": BEAM_SCHEMA_VERSION,
        "test_data_used": False,
        "test_data_used_for_search": False,
        "ranking_objective": "median_delta_pr_auc_primary_no_returns",
        "settings": {"width": 4, "min_features": 6, "max_remove": max_remove},
        "candidate_features": list(ADDED_RULE_FEATURE_COLUMNS_V2),
        "candidate_features_sha256": beam_feature_columns_sha256(
            ADDED_RULE_FEATURE_COLUMNS_V2
        ),
        "selected_features": list(selected_increment),
        "selected_features_sha256": beam_feature_columns_sha256(selected_increment),
        "all_increment_candidates": list(ADDED_RULE_FEATURE_COLUMNS_V2),
        "prefilter_method": "none_full_13_candidate_universe",
        "visited_combinations": 37,
        "all_accessed_combinations": 37,
        "permutation_gate_passed": permutation_gate_passed,
        "pipeline_select": {
            "gate_passed": pipeline_select_gate_passed,
            "test_data_used": False,
        },
        "pipeline_select_gate_passed": pipeline_select_gate_passed,
        "development_gates_passed": bool(
            permutation_gate_passed and pipeline_select_gate_passed
        ),
    }
    bundle = {
        "schema_version": RIGHT_SIDE_PRODUCTION_ARTIFACT_SCHEMA_VERSION,
        "score_field": "ranking_score",
        "probability_calibration": config.probability_calibration,
        "production_threshold_mode": config.production_threshold_mode,
        "score_normalization": {
            "schema_version": RANKING_NORMALIZATION_SCHEMA_VERSION,
            "quantiles": np.linspace(0.0, 1.0, 1001).tolist(),
            "percentiles": np.linspace(0.0, 100.0, 1001).tolist(),
        },
        "model": _ProductionModel(),
        "features": features,
        "factor_contract_sha256": RIGHT_SIDE_SHADOW_FACTOR_CONTRACT_SHA256,
        "model_input_contract_sha256": factor_contract_sha256(
            features,
            schema_version=RIGHT_SIDE_PRODUCTION_ARTIFACT_SCHEMA_VERSION,
        ),
        "materialized_rule_factor_columns_sha256": RULE_FEATURE_COLUMNS_SHA256,
        "selected_candidate": "unified_long_task_deep_beam",
        "selected_rule_factor_count": len(selected_rules),
        "selected_rule_factor_columns": selected_rules,
        "selected_rule_factor_columns_sha256": rule_feature_columns_sha256(
            selected_rules
        ),
        "beam_search": beam,
    }
    config.paths.artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, config.paths.artifact)
    config.paths.artifact_manifest.write_text(
        json.dumps(
            {
                "schema_version": RIGHT_SIDE_PRODUCTION_ARTIFACT_SCHEMA_VERSION,
                "lifecycle": "production",
                "score_field": "ranking_score",
                "playbook_coupling": "independent",
                "probability_calibration": config.probability_calibration,
                "production_threshold_mode": config.production_threshold_mode,
                "score_normalization": {
                    "schema_version": RANKING_NORMALIZATION_SCHEMA_VERSION
                },
                "replaced_signals": list(config.replaced_signals),
                "preserved_legacy_signals": list(
                    config.preserved_legacy_signals
                ),
                "factor_contract_sha256": RIGHT_SIDE_SHADOW_FACTOR_CONTRACT_SHA256,
                "selected_candidate": "unified_long_task_deep_beam",
                "promotion_approval_sha256": _sha256(
                    config.paths.promotion_approval
                ),
                "selected_rule_factor_columns": list(selected_rules),
                "models": {
                    "ranking": {
                        "path": config.paths.artifact.relative_to(
                            config.paths.artifact.parents[3]
                        ).as_posix(),
                        "sha256": _sha256(config.paths.artifact),
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_production_artifact_blocks_retired_beam_candidate(
    tmp_path: Path,
) -> None:
    config = _promoted_config(tmp_path)
    _write_production_bundle(config, permutation_gate_passed=False)

    with pytest.raises(RuntimeError, match="supported candidate"):
        validate_production_ranking_artifact(config, project_root=tmp_path)


def test_production_artifact_does_not_accept_precanonical_beam_contract(tmp_path: Path) -> None:
    config = _promoted_config(tmp_path)
    _write_production_bundle(config, permutation_gate_passed=True)

    with pytest.raises(RuntimeError, match="supported candidate"):
        validate_production_ranking_artifact(config, project_root=tmp_path)


def test_production_artifact_rejects_old_budget_limited_beam_contract(
    tmp_path: Path,
) -> None:
    config = _promoted_config(tmp_path)
    _write_production_bundle(
        config,
        permutation_gate_passed=True,
        max_remove=2,
    )

    with pytest.raises(RuntimeError, match="supported candidate"):
        validate_production_ranking_artifact(config, project_root=tmp_path)


def test_production_artifact_requires_independent_pipeline_select_gate(
    tmp_path: Path,
) -> None:
    config = _promoted_config(tmp_path)
    _write_production_bundle(
        config,
        permutation_gate_passed=True,
        pipeline_select_gate_passed=False,
    )

    with pytest.raises(RuntimeError, match="supported candidate"):
        validate_production_ranking_artifact(config, project_root=tmp_path)
