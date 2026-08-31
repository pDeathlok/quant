from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from quant.features.candlestick_context import (
    CANDLE_CONTEXT_RESEARCH_FEATURE_COLUMNS,
)
from quant.features.project_factor_layer import PROJECT_FACTOR_SCHEMA_VERSION
from quant.features.right_side_factor_contract import (
    RIGHT_SIDE_SHADOW_ARTIFACT_SCHEMA_VERSION,
    RIGHT_SIDE_SHADOW_FACTOR_COLUMNS,
    RIGHT_SIDE_SHADOW_FACTOR_CONTRACT_SHA256,
    RIGHT_SIDE_SHADOW_FEATURE_SCHEMA_VERSION,
    RIGHT_SIDE_SHADOW_IDENTITY_COLUMNS,
    RIGHT_SIDE_SHADOW_MODEL_INPUT_COLUMNS,
    factor_contract_sha256,
)
from quant.features.variable_library import PROJECT_FACTOR_COLUMNS
from quant.research.right_side_unified_features import (
    ADDED_RULE_FEATURE_COLUMNS_V2,
    LEGACY_RULE_FEATURE_COLUMNS_V1,
    LEGACY_RULE_FEATURE_SCHEMA_VERSION_V1,
    RULE_FEATURE_COLUMNS,
    RULE_FEATURE_COLUMNS_SHA256,
    RULE_FEATURE_SCHEMA_VERSION,
)
from quant.research.right_side_beam_feature_selection import (
    BEAM_SCHEMA_VERSION,
    feature_columns_sha256 as beam_feature_columns_sha256,
)
from quant.routine import right_side_unified_shadow as shadow


class _FixedProbabilityModel:
    def predict_proba(self, values: pd.DataFrame) -> np.ndarray:
        probability = np.full(len(values), 0.70, dtype=float)
        return np.column_stack([1.0 - probability, probability])


class _StageableModel(_FixedProbabilityModel):
    common_features = (*PROJECT_FACTOR_COLUMNS, *RULE_FEATURE_COLUMNS)


class _Rule105StageableModel(_FixedProbabilityModel):
    common_features = (*PROJECT_FACTOR_COLUMNS, *LEGACY_RULE_FEATURE_COLUMNS_V1)


_BEAM_SELECTED_INCREMENT = tuple(ADDED_RULE_FEATURE_COLUMNS_V2[:6])
_BEAM_RULE_FEATURES = (
    *LEGACY_RULE_FEATURE_COLUMNS_V1,
    *_BEAM_SELECTED_INCREMENT,
)


class _BeamStageableModel(_FixedProbabilityModel):
    common_features = (*PROJECT_FACTOR_COLUMNS, *_BEAM_RULE_FEATURES)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config(tmp_path: Path, *, enabled: bool = True) -> shadow.ShadowReleaseConfig:
    return shadow.ShadowReleaseConfig(
        enabled=enabled,
        top_n=50,
        history_years=6,
        paths=shadow.ShadowPaths(
            artifact=tmp_path / "models/research/shadow/ranking.joblib",
            artifact_manifest=tmp_path / "models/research/shadow/manifest.json",
            ranking_decision=tmp_path / "reports/ranking_decision.json",
            feature_output=tmp_path / "data/features/shadow/features.parquet",
            feature_manifest=tmp_path / "data/features/shadow/feature_manifest.json",
            score_output=tmp_path / "data/features/shadow/scores.parquet",
            score_manifest=tmp_path / "data/features/shadow/score_manifest.json",
            product_output=tmp_path / "reports/shadow/candidates.parquet",
            product_manifest=tmp_path / "reports/shadow/product_manifest.json",
            run_status=tmp_path / "reports/shadow/run_status.json",
            z_signal_cache=tmp_path / "data/features/z.parquet",
            family_signal_cache=tmp_path / "data/features/family.parquet",
            market_data_root=tmp_path / "data/raw",
        ),
        decision_field="shadow_candidate",
        accepted_decisions=(True, "pass", "replace"),
        selected_candidate_field="selected_research_candidate",
        eligible_candidates=(
            "unified_long_task_deep_rule105",
            "unified_long_task_deep",
            "unified_long_task_deep_beam",
        ),
    )


def _write_source_manifest(
    source: Path,
    *,
    experiment: str,
    common_features: tuple[str, ...],
    rule_features: tuple[str, ...],
    beam_selected: tuple[str, ...] = (),
) -> None:
    payload: dict[str, object] = {
        "schema_version": "right-side-long-task-v1-event-max",
        "experiment": experiment,
        "common_features": list(common_features),
        "rule_feature_schema_version": {
            "unified_long_task_deep_rule105": LEGACY_RULE_FEATURE_SCHEMA_VERSION_V1,
            "unified_long_task_deep": RULE_FEATURE_SCHEMA_VERSION,
            "unified_long_task_deep_beam": (
                "right_side_rule_features_v1_105_plus_beam_v2_increment"
            ),
        }[experiment],
        "rule_feature_count": len(rule_features),
        "rule_feature_columns": list(rule_features),
        "rule_feature_columns_sha256": shadow.rule_feature_columns_sha256(
            rule_features
        ),
    }
    if experiment == "unified_long_task_deep_beam":
        payload["beam_search"] = {
            "schema_version": BEAM_SCHEMA_VERSION,
            "test_data_used": False,
            "test_data_used_for_search": False,
            "ranking_objective": "median_delta_pr_auc_primary_no_returns",
            "settings": {
                "width": 4,
                "min_features": 6,
                "max_remove": 10,
            },
            "candidate_features": list(ADDED_RULE_FEATURE_COLUMNS_V2),
            "candidate_features_sha256": beam_feature_columns_sha256(
                ADDED_RULE_FEATURE_COLUMNS_V2
            ),
            "selected_features": list(beam_selected),
            "selected_features_sha256": beam_feature_columns_sha256(beam_selected),
            "all_increment_candidates": list(ADDED_RULE_FEATURE_COLUMNS_V2),
            "prefilter_method": "none_full_13_candidate_universe",
            "visited_combinations": 37,
            "all_accessed_combinations": 37,
            "permutation_gate_passed": True,
            "pipeline_select": {"gate_passed": True, "test_data_used": False},
            "pipeline_select_gate_passed": True,
            "development_gates_passed": True,
        }
    source.with_suffix(".manifest.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def _composite_decision(
    selected: str,
    *,
    shadow_candidate: bool = True,
    replace_online: bool = False,
) -> dict[str, object]:
    return {
        "schema_version": "right-side-production-replacement-decision-v1",
        "selected_research_candidate": selected,
        "shadow_candidate": shadow_candidate,
        "replace_online": replace_online,
        "architecture_gate": {"passed": True},
        "canonical_factor_gate": {
            "passed": selected == "unified_long_task_deep"
        },
        "full_118_factor_increment_gate": {
            "passed": selected == "unified_long_task_deep"
        },
        "legacy_online_artifact_gate": {"passed": replace_online},
    }


def test_repository_shadow_config_is_disabled_and_promotion_is_not_automatic() -> None:
    config = shadow.load_shadow_release_config()
    payload = shadow._read_mapping(Path(shadow.DEFAULT_CONFIG_PATH))

    assert config.enabled is False
    assert payload["release"]["lifecycle"] == "research_only"
    assert payload["release"]["scope"] == "rightSideShadow"
    assert payload["promotion"]["enabled"] is False
    assert payload["promotion"]["requires_new_unseen_shadow_period"] is False
    assert payload["promotion"]["engineering_acceptance"] == {
        "minimum_complete_shadow_runs": 1,
        "require_schema_freshness_artifact_final_gate": True,
    }
    assert config.paths.ranking_decision.name == (
        "production_replacement_decision_ab.json"
    )
    assert config.decision_field == "shadow_candidate"
    assert config.production_replacement_field == "replace_online"
    assert config.eligible_candidates == ("unified_long_task_deep",)


def test_disabled_shadow_returns_without_accessing_artifacts() -> None:
    result = shadow.run_configured_right_side_shadow("2026-08-12")

    assert result["status"] == "skipped"
    assert result["scope"] == "rightSideShadow"
    assert result["production_affected"] is False


def test_shadow_feature_frame_requires_complete_exact_date_candidate_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dates = pd.bdate_range("2026-08-10", periods=3)
    market = pd.DataFrame(
        {
            "ts_code": "000001.SZ",
            "symbol": "000001.SZ",
            "trade_date": dates.strftime("%Y%m%d"),
            "date": dates,
            "open": [10.0, 10.2, 10.4],
            "high": [10.3, 10.5, 10.8],
            "low": [9.9, 10.1, 10.3],
            "close": [10.2, 10.4, 10.7],
            "pre_close": [10.0, 10.2, 10.4],
            "vol": [1000.0, 1100.0, 1200.0],
        }
    )
    signals = pd.DataFrame(
        {
            "symbol": ["000001.SZ"],
            "date": [dates[-1]],
            **{
                name: [name == "B2"]
                for name in RIGHT_SIDE_SHADOW_IDENTITY_COLUMNS
            },
        }
    )

    def fake_project(daily, *, symbol, factor_schema_version):
        assert factor_schema_version == PROJECT_FACTOR_SCHEMA_VERSION
        return pd.DataFrame(
            {
                "ts_code": symbol,
                "symbol": symbol,
                "trade_date": daily["trade_date"].astype(str).to_numpy(),
                "date": pd.to_datetime(daily["date"]).to_numpy(),
                **{name: np.arange(len(daily), dtype=float) for name in PROJECT_FACTOR_COLUMNS},
                "factor_schema_version": factor_schema_version,
            }
        )

    def fake_rules(daily, *, canonical_factors=None):
        assert canonical_factors is not None
        return pd.DataFrame(
            {
                name: np.arange(len(daily), dtype=float)
                for name in RULE_FEATURE_COLUMNS
            }
        )

    monkeypatch.setattr(shadow, "calculate_project_market_factors", fake_project)
    monkeypatch.setattr(shadow, "compute_right_side_rule_features", fake_rules)

    result = shadow.build_right_side_shadow_feature_frame(
        market,
        signals,
        target_date=dates[-1],
    )

    assert len(result) == 1
    assert set(RIGHT_SIDE_SHADOW_FACTOR_COLUMNS) <= set(result.columns)
    assert set(CANDLE_CONTEXT_RESEARCH_FEATURE_COLUMNS) <= set(result.columns)
    assert result["B2"].tolist() == [True]
    assert result["right_side_feature_schema_version"].eq(
        RIGHT_SIDE_SHADOW_FEATURE_SCHEMA_VERSION
    ).all()

    second = signals.copy()
    second["symbol"] = "000002.SZ"
    with pytest.raises(RuntimeError, match="target market row is missing"):
        shadow.build_right_side_shadow_feature_frame(
            market,
            pd.concat([signals, second], ignore_index=True),
            target_date=dates[-1],
        )


def test_shadow_score_and_product_are_checksum_pinned_and_selector_is_not_published(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    features = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "symbol": ["000001.SZ"],
            "trade_date": ["20260812"],
            "date": [pd.Timestamp("2026-08-12")],
            **{name: [0.0] for name in RIGHT_SIDE_SHADOW_FACTOR_COLUMNS},
            **{
                name: [name == "B2"]
                for name in RIGHT_SIDE_SHADOW_IDENTITY_COLUMNS
            },
        }
    )
    config.paths.feature_output.parent.mkdir(parents=True)
    features.to_parquet(config.paths.feature_output, index=False)
    config.paths.feature_manifest.write_text(
        json.dumps(
            {
                "status": "success",
                "target_date": "2026-08-12",
                "factor_contract_sha256": RIGHT_SIDE_SHADOW_FACTOR_CONTRACT_SHA256,
                "output_sha256": _sha256(config.paths.feature_output),
            }
        ),
        encoding="utf-8",
    )

    config.paths.artifact.parent.mkdir(parents=True)
    joblib.dump(
        {
            "schema_version": RIGHT_SIDE_SHADOW_ARTIFACT_SCHEMA_VERSION,
            "model": _FixedProbabilityModel(),
            "features": RIGHT_SIDE_SHADOW_MODEL_INPUT_COLUMNS,
            "factor_contract_sha256": RIGHT_SIDE_SHADOW_FACTOR_CONTRACT_SHA256,
            "model_input_contract_sha256": factor_contract_sha256(
                RIGHT_SIDE_SHADOW_MODEL_INPUT_COLUMNS,
                schema_version=RIGHT_SIDE_SHADOW_ARTIFACT_SCHEMA_VERSION,
            ),
            "materialized_rule_factor_columns_sha256": RULE_FEATURE_COLUMNS_SHA256,
            "selected_candidate": "unified_long_task_deep",
            "selected_rule_factor_count": len(RULE_FEATURE_COLUMNS),
            "selected_rule_factor_columns": RULE_FEATURE_COLUMNS,
            "selected_rule_factor_columns_sha256": RULE_FEATURE_COLUMNS_SHA256,
        },
        config.paths.artifact,
    )
    config.paths.artifact_manifest.write_text(
        json.dumps(
            {
                "schema_version": RIGHT_SIDE_SHADOW_ARTIFACT_SCHEMA_VERSION,
                "models": {
                    "ranking": {
                        "path": config.paths.artifact.relative_to(tmp_path).as_posix(),
                        "sha256": _sha256(config.paths.artifact),
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    score = shadow.score_right_side_shadow(
        "2026-08-12",
        config=config,
        project_root=tmp_path,
    )
    product = shadow.publish_right_side_shadow_product(
        "2026-08-12",
        config=config,
    )
    candidates = pd.read_parquet(config.paths.product_output)

    assert score["candidate_count"] == 1
    assert product["candidate_count"] == 1
    assert product["selector_published"] is False
    assert candidates["ranking_score"].tolist() == [0.70]
    assert candidates["consumer"].tolist() == ["research_shadow_only"]
    assert candidates["daily_rank"].tolist() == [1]

    features.assign(close=1.0).to_parquet(config.paths.feature_output, index=False)
    with pytest.raises(RuntimeError, match="feature sidecar checksum mismatch"):
        shadow.score_right_side_shadow(
            "2026-08-12",
            config=config,
            project_root=tmp_path,
        )

    pd.read_parquet(config.paths.score_output).assign(ranking_score=0.1).to_parquet(
        config.paths.score_output,
        index=False,
    )
    with pytest.raises(RuntimeError, match="score snapshot checksum mismatch"):
        shadow.publish_right_side_shadow_product(
            "2026-08-12",
            config=config,
        )


def test_missing_shadow_artifact_is_observable_and_isolated(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, enabled=True)
    monkeypatch.setattr(shadow, "load_shadow_release_config", lambda *args: config)

    result = shadow.run_configured_right_side_shadow(
        "2026-08-12",
        project_root=tmp_path,
    )

    assert result["status"] == "blocked"
    assert result["stage"] == "preflight"
    assert "missing artifact" in result["error"]
    assert result["production_affected"] is False
    assert result["selector_published"] is False
    assert config.paths.run_status.is_file()


def test_shadow_stage_refuses_unaccepted_ranking_decision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    source = tmp_path / "model.joblib"
    joblib.dump(_StageableModel(), source)
    config.paths.ranking_decision.parent.mkdir(parents=True)
    config.paths.ranking_decision.write_text(
        json.dumps(
            _composite_decision(
                "unified_long_task_deep_rule105",
                shadow_candidate=False,
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(shadow, "load_shadow_release_config", lambda *args: config)

    with pytest.raises(RuntimeError, match="ranking gate has not accepted"):
        shadow.stage_shadow_release(source, project_root=tmp_path)
    assert not config.paths.artifact.exists()


def test_architecture_decision_cannot_authorize_shadow_or_production(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    source = tmp_path / "model.joblib"
    joblib.dump(_Rule105StageableModel(), source)
    config.paths.ranking_decision.parent.mkdir(parents=True)
    config.paths.ranking_decision.write_text(
        json.dumps(
            {
                "schema_version": "right-side-ranking-promotion-decision-v1",
                "replace_online": True,
                "selected_candidate": "unified_long_task_deep_rule105",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(shadow, "load_shadow_release_config", lambda *args: config)

    with pytest.raises(RuntimeError, match="composite production replacement"):
        shadow.stage_shadow_release(source, project_root=tmp_path)
    assert not config.paths.artifact.exists()


def test_shadow_stage_wraps_passed_model_without_writing_production(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    source = tmp_path / "model.joblib"
    joblib.dump(_StageableModel(), source)
    config.paths.ranking_decision.parent.mkdir(parents=True)
    config.paths.ranking_decision.write_text(
        json.dumps(
            _composite_decision("unified_long_task_deep")
        ),
        encoding="utf-8",
    )
    _write_source_manifest(
        source,
        experiment="unified_long_task_deep",
        common_features=_StageableModel.common_features,
        rule_features=tuple(RULE_FEATURE_COLUMNS),
    )
    monkeypatch.setattr(shadow, "load_shadow_release_config", lambda *args: config)

    result = shadow.stage_shadow_release(source, project_root=tmp_path)
    manifest = json.loads(config.paths.artifact_manifest.read_text(encoding="utf-8"))

    assert result["status"] == "success"
    assert result["production_changed"] is False
    assert manifest["lifecycle"] == "research_only"
    assert manifest["selected_rule_factor_count"] == len(RULE_FEATURE_COLUMNS)
    assert manifest["rule_factor_count"] == len(RULE_FEATURE_COLUMNS)
    assert manifest["factor_contract_sha256"] == RIGHT_SIDE_SHADOW_FACTOR_CONTRACT_SHA256
    assert not (tmp_path / "models/production/right_side_unified").exists()

    decision = json.loads(config.paths.ranking_decision.read_text(encoding="utf-8"))
    decision["audit_note"] = "decision changed after staging"
    config.paths.ranking_decision.write_text(json.dumps(decision), encoding="utf-8")
    with pytest.raises(RuntimeError, match="decision checksum is stale"):
        shadow.validate_shadow_release_preflight(config, project_root=tmp_path)


def test_shadow_stage_rejects_pre_canonical_beam_candidate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    source = tmp_path / "beam.joblib"
    joblib.dump(_BeamStageableModel(), source)
    _write_source_manifest(
        source,
        experiment="unified_long_task_deep_beam",
        common_features=_BeamStageableModel.common_features,
        rule_features=_BEAM_RULE_FEATURES,
        beam_selected=_BEAM_SELECTED_INCREMENT,
    )
    config.paths.ranking_decision.parent.mkdir(parents=True)
    config.paths.ranking_decision.write_text(
        json.dumps(
            _composite_decision("unified_long_task_deep_beam")
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(shadow, "load_shadow_release_config", lambda *args: config)

    with pytest.raises(ValueError, match="unsupported selected right-side candidate"):
        shadow.stage_shadow_release(source, project_root=tmp_path)
    assert not config.paths.artifact.exists()
