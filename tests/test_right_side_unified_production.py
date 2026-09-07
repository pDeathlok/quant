from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pandas as pd
import pytest

from quant.application.selector_ranking import load_selector_ranking_config
from quant.features.project_factor_layer import PROJECT_FACTOR_SCHEMA_VERSION
from quant.features.right_side_factor_contract import (
    RIGHT_SIDE_SHADOW_IDENTITY_COLUMNS,
)
from quant.features.variable_library import (
    DAILY_BASIC_PROJECT_FACTOR_COLUMNS,
    PROJECT_FACTOR_COLUMNS,
)
from quant.routine import right_side_unified_production as production
from quant.routine.project_feature_cache import ProjectFeatureCacheSnapshot
from quant.infrastructure.publication import PublicationStore, publication_path
from quant.routine.operation_adapters import run_right_side_unified
from quant.routine.operation_contracts import OperationContext


def test_production_features_reuse_shared_cache_and_preserve_training_nulls(
    monkeypatch,
    tmp_path,
) -> None:
    target = pd.Timestamp("2026-09-01")
    symbol = "000001.SZ"
    z_cache = tmp_path / "z.parquet"
    family_cache = tmp_path / "family.parquet"
    z_cache.touch()
    family_cache.touch()
    base = load_selector_ranking_config()
    config = replace(
        base,
        paths=replace(
            base.paths,
            z_signal_cache=z_cache,
            family_signal_cache=family_cache,
        ),
    )
    signals = pd.DataFrame(
        {
            "symbol": [symbol],
            "date": [target],
            **{
                name: [name == "B2"]
                for name in RIGHT_SIDE_SHADOW_IDENTITY_COLUMNS
            },
        }
    )
    project_row = {name: 1.0 for name in PROJECT_FACTOR_COLUMNS}
    project_row.update(
        ts_code=symbol,
        symbol=symbol,
        trade_date="20260901",
        date=target,
        factor_schema_version=PROJECT_FACTOR_SCHEMA_VERSION,
    )
    snapshot = ProjectFeatureCacheSnapshot(
        features=pd.DataFrame([project_row]),
        eligible_signals=signals,
        policy_excluded_symbols=(),
        manifest={},
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(production, "load_signal_universe", lambda *a, **k: signals)
    monkeypatch.setattr(
        production,
        "load_exact_date_project_feature_cache",
        lambda *a, **k: snapshot,
    )
    monkeypatch.setattr(production, "_sha256", lambda path: "cache-sha")

    def capture_build(*args, **kwargs):
        captured.update(kwargs)
        return {"status": "success"}

    monkeypatch.setattr(production, "build_right_side_shadow_features", capture_build)

    result = production.build_right_side_unified_production_features(
        target.date().isoformat(),
        config=config,
    )

    assert result["status"] == "success"
    reused = captured["project_features"]
    assert isinstance(reused, pd.DataFrame)
    assert reused.loc[0, "alpha003"] == 1.0
    assert reused[list(DAILY_BASIC_PROJECT_FACTOR_COLUMNS)].isna().all().all()
    assert captured["source_signal_candidate_count"] == 1
    assert captured["project_feature_cache_sha256"] == "cache-sha"


def test_daily_basic_project_factor_family_is_explicit_and_registered() -> None:
    assert len(DAILY_BASIC_PROJECT_FACTOR_COLUMNS) == 35
    assert set(DAILY_BASIC_PROJECT_FACTOR_COLUMNS) <= set(PROJECT_FACTOR_COLUMNS)
    assert len(DAILY_BASIC_PROJECT_FACTOR_COLUMNS) == len(
        set(DAILY_BASIC_PROJECT_FACTOR_COLUMNS)
    )


@pytest.fixture
def production_inputs(tmp_path):
    base = load_selector_ranking_config()
    config = replace(base, paths=replace(base.paths, **{
        field: tmp_path / getattr(base.paths, field).relative_to(production.PROJECT_ROOT)
        for field in base.paths.__dataclass_fields__
    }))
    files = [
        config.paths.artifact, config.paths.promotion_approval,
        config.paths.z_signal_cache, config.paths.family_signal_cache,
        config.paths.market_data_root / "daily_partitioned/year_month=202609/data.parquet",
    ]
    contracts = (
        "configs/strategies/right_side_ranking_selector.yaml",
        "src/quant/application/selector_ranking.py",
        "src/quant/features/right_side_factor_contract.py",
        "src/quant/features/project_factor_layer.py",
        "src/quant/features/variable_library.py",
        "src/quant/research/right_side_unified_features.py",
        "src/quant/research/right_side_unified_signals.py",
        "src/quant/routine/project_feature_cache.py",
        "src/quant/routine/right_side_unified_shadow.py",
        "src/quant/routine/right_side_unified_production.py",
    )
    for path in [*files, *(tmp_path / relative for relative in contracts)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture input")
    for path in (config.paths.feature_output, config.paths.score_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    feature = tmp_path / production.ACTIVE_PROJECT_FEATURE_PATH.relative_to(production.PROJECT_ROOT)
    manifest = tmp_path / production.ACTIVE_PROJECT_FEATURE_MANIFEST_PATH.relative_to(production.PROJECT_ROOT)
    _write_shared_features(feature, manifest, value=1.0)
    return config, feature, manifest


def _write_shared_features(feature: Path, manifest: Path, *, value: float) -> None:
    feature.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        **{name: value for name in PROJECT_FACTOR_COLUMNS},
        "ts_code": "000001.SZ", "symbol": "000001.SZ", "trade_date": "20260901",
        "date": pd.Timestamp("2026-09-01"), "factor_schema_version": PROJECT_FACTOR_SCHEMA_VERSION,
    }]).to_parquet(feature, index=False)
    manifest.write_text(json.dumps({
        "status": "success", "target_date": "2026-09-01",
        "candidate_coverage_status": "complete", "factor_schema_version": PROJECT_FACTOR_SCHEMA_VERSION,
        "output_sha256": production._sha256(feature),
    }))


@pytest.mark.parametrize("explicit_paths", [False, True])
def test_shared_feature_reads_and_hashes_use_staged_generation(
    monkeypatch, tmp_path, production_inputs, explicit_paths,
) -> None:
    config, feature, manifest = production_inputs
    signals = pd.DataFrame({"symbol": ["000001.SZ"], "date": [pd.Timestamp("2026-09-01")]})
    monkeypatch.setattr(production, "load_signal_universe", lambda *a, **k: signals)
    captured = {}

    def capture_build(*args, **kwargs):
        captured.update(kwargs)
        return {"status": "success"}

    monkeypatch.setattr(production, "build_right_side_shadow_features", capture_build)
    original_sha = production._sha256(feature)
    store = PublicationStore(tmp_path, ("data/features",))
    with store.begin("staged-shared-features"):
        staged_feature, staged_manifest = publication_path(feature), publication_path(manifest)
        _write_shared_features(staged_feature, staged_manifest, value=9.0)
        kwargs = {
            "project_feature_path": staged_feature,
            "project_feature_manifest_path": staged_manifest,
        } if explicit_paths else {}
        fingerprint, snapshot = production._production_input_snapshot(
            config, pd.Timestamp("2026-09-01"), project_root=tmp_path, **kwargs,
        )
        assert snapshot["inputs"]["project_feature_cache"] == {
            "path": str(staged_feature), "sha256": production._sha256(staged_feature),
        }
        assert snapshot["inputs"]["project_feature_manifest"]["path"] == str(staged_manifest)
        production.build_right_side_unified_production_features(
            "2026-09-01", config=config, project_root=tmp_path, **kwargs,
        )
        assert captured["project_features"].loc[0, "alpha003"] == 9.0
        assert captured["project_feature_cache_sha256"] == production._sha256(staged_feature)
        assert production._sha256(feature) == original_sha
        # Mutating canonical bytes must not affect the generation being consumed.
        _write_shared_features(feature, manifest, value=3.0)
        assert production._production_input_snapshot(
            config, pd.Timestamp("2026-09-01"), project_root=tmp_path, **kwargs,
        )[0] == fingerprint
        _write_shared_features(staged_feature, staged_manifest, value=8.0)
        assert production._production_input_snapshot(
            config, pd.Timestamp("2026-09-01"), project_root=tmp_path, **kwargs,
        )[0] != fingerprint


@pytest.mark.parametrize("broken", ["feature", "manifest", "checksum"])
def test_shared_feature_staging_fails_closed_without_canonical_fallback(
    monkeypatch, tmp_path, production_inputs, broken,
) -> None:
    config, feature, manifest = production_inputs
    signals = pd.DataFrame({"symbol": ["000001.SZ"], "date": [pd.Timestamp("2026-09-01")]})
    monkeypatch.setattr(production, "load_signal_universe", lambda *a, **k: signals)
    monkeypatch.setattr(production, "build_right_side_shadow_features", lambda *a, **k: pytest.fail("builder must not run"))
    store = PublicationStore(tmp_path, ("data/features",))
    with pytest.raises(RuntimeError):
        with store.begin(f"broken-{broken}"):
            if broken == "checksum":
                publication_path(feature).write_bytes(b"interrupted cache write")
            else:
                publication_path(feature if broken == "feature" else manifest).unlink()
            production.build_right_side_unified_production_features(
                "2026-09-01", config=config, project_root=tmp_path,
            )
    # An interrupted generation never replaces the committed shared features.
    committed = store.view()
    assert committed.generation != f"broken-{broken}"
    assert production._sha256(committed.resolve(feature)) == production._sha256(feature)


def test_strict_adapter_passes_same_staged_paths_to_snapshot_and_builder(
    monkeypatch, tmp_path, production_inputs,
) -> None:
    from quant.application import selector_ranking

    config, feature, manifest = production_inputs
    monkeypatch.setattr(selector_ranking, "load_selector_ranking_config", lambda root: config)
    captured = {}

    def snapshot(config, target, **kwargs):
        captured["snapshot"] = kwargs
        return "fixture-fingerprint", {}

    def build(target, **kwargs):
        captured["build"] = kwargs
        return {"status": "success"}

    monkeypatch.setattr(production, "_production_input_snapshot", snapshot)
    monkeypatch.setattr(production, "build_right_side_unified_production_features", build)
    monkeypatch.setattr(production, "score_right_side_unified_production", lambda *a, **k: {"status": "success"})
    monkeypatch.setattr(production, "validate_right_side_unified_selector_adapter", lambda *a, **k: {"status": "success"})
    context = OperationContext(
        target_trade_date="20260901", scope="short", granted_workers=2,
        upstream_results={}, identity_required=True, project_root=tmp_path,
    )
    with PublicationStore(tmp_path, ("data/features",)).begin("strict-right"):
        assert run_right_side_unified(context).status == "success"
        for call in ("snapshot", "build"):
            assert captured[call]["project_feature_path"] == publication_path(feature)
            assert captured[call]["project_feature_manifest_path"] == publication_path(manifest)
            assert captured[call]["project_root"] == tmp_path
        built_config = captured["build"]["config"]
        assert built_config.factor_workers == 2
        for field in ("z_signal_cache", "family_signal_cache", "feature_output", "feature_manifest", "score_output", "score_manifest"):
            assert getattr(built_config.paths, field) == publication_path(getattr(config.paths, field))


@pytest.mark.parametrize("changed", ["feature", "manifest", "source"])
def test_same_date_checkpoint_invalidated_by_consumed_input_changes(
    monkeypatch, tmp_path, production_inputs, changed,
) -> None:
    config, feature, manifest = production_inputs
    calls = []
    monkeypatch.setattr(production, "validate_production_ranking_artifact", lambda *a, **k: {})

    def build(target, **kwargs):
        calls.append(kwargs)
        config.paths.feature_output.write_bytes(b"fixture output")
        return {
            "status": "success", "target_date": target,
            "output_sha256": production._sha256(config.paths.feature_output),
        }

    def score(*args, **kwargs):
        payload = json.loads(config.paths.feature_manifest.read_text())
        config.paths.score_manifest.write_text(json.dumps(payload))
        return payload

    monkeypatch.setattr(production, "build_right_side_unified_production_features", build)
    monkeypatch.setattr(production, "score_right_side_unified_production", score)
    monkeypatch.setattr(production, "load_right_side_ranking_scores", lambda *a, **k: (pd.DataFrame(), json.loads(config.paths.score_manifest.read_text())))
    monkeypatch.setattr(production, "validate_right_side_unified_selector_adapter", lambda *a, **k: {"status": "success"})

    def run():
        return production.run_right_side_unified_production(
            "2026-09-01", config=config, project_root=tmp_path,
            project_feature_path=feature, project_feature_manifest_path=manifest,
        )

    assert run()["checkpoint_reused"] is False
    assert run()["checkpoint_reused"] is True
    assert len(calls) == 1
    if changed == "feature":
        _write_shared_features(feature, manifest, value=2.0)
    elif changed == "manifest":
        payload = json.loads(manifest.read_text())
        manifest.write_text(json.dumps({**payload, "source_revision": 2}))
    else:
        (tmp_path / "src/quant/routine/project_feature_cache.py").write_text("changed reader contract")
    assert run()["checkpoint_reused"] is False
    assert len(calls) == 2
    assert calls[-1]["project_feature_path"] == feature
    assert calls[-1]["project_feature_manifest_path"] == manifest
