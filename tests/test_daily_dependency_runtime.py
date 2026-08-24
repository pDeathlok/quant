from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
import pytest

from quant.application.daily_dependencies import (
    DEFAULT_DAILY_DEPENDENCY_REGISTRY,
    PRODUCTION_PROJECT_FACTOR_SCHEMA,
    ArtifactSpec,
    Cadence,
    DependencyEdge,
    DependencyNode,
    DependencyRegistry,
    EvidenceSpec,
    FreshnessMode,
    FreshnessPolicy,
    IncrementalPolicy,
    Layer,
    Lifecycle,
    ModelContract,
    NodeState,
)
from quant.features.right_side_factor_contract import (
    RIGHT_SIDE_SHADOW_MODEL_INPUT_COLUMNS,
)
from quant.features.variable_library import PROJECT_FACTOR_COLUMNS
from quant.routine import daily_dependency_runtime as runtime


class _SklearnArtifact:
    feature_names_in_ = ("close", "volume", "top_list_count")
    selected_features_ = ("close", "volume")


class _ImportanceModel:
    feature_importances_ = (0.8, 0.0, 0.2)


def test_right_side_shadow_catalog_matches_complete_model_input_contract() -> None:
    features = tuple(RIGHT_SIDE_SHADOW_MODEL_INPUT_COLUMNS)
    contract = ModelContract(
        node_id="score.right_side_unified_shadow",
        artifact_hashes=(("ranking.joblib", "abc"),),
        features_by_artifact={"ranking.joblib": features},
        effective_features_by_artifact={"ranking.joblib": features},
        required_feature_union=features,
        effective_feature_union=features,
        combined_hash="abc",
    )
    contracts = {"score.right_side_unified_shadow": contract}

    catalogs = runtime._feature_catalogs(contracts)
    usage = {
        item.feature_node_id: item
        for item in runtime.classify_feature_usage(
            DEFAULT_DAILY_DEPENDENCY_REGISTRY,
            "rightSideShadow",
            contracts,
            catalogs,
        )
    }["feature.right_side_unified_shadow"]

    assert catalogs["feature.right_side_unified_shadow"] == features
    assert usage.required == tuple(sorted(features))
    assert usage.unknown == ()
    assert usage.projection_safe is False


def _node(
    node_id: str,
    layer: Layer,
    *,
    inputs: tuple[DependencyEdge, ...] = (),
    freshness: FreshnessPolicy | None = None,
    cadence: Cadence = Cadence.TRADE_DAILY,
    artifact: ArtifactSpec | None = None,
    final_gate: bool = False,
) -> DependencyNode:
    return DependencyNode(
        node_id=node_id,
        layer=layer,
        owner="tests",
        lifecycle=Lifecycle.PRODUCTION,
        cadence=cadence,
        inputs=inputs,
        freshness=freshness
        or FreshnessPolicy(FreshnessMode.EXACT_TRADE_DATE),
        incremental=IncrementalPolicy(partition_key="date"),
        operation_id=f"run_{node_id}",
        artifact=artifact,
        final_gate=final_gate,
    )


def _complete_snapshot_cycle(
    tmp_path: Path,
    target: date,
    registry: DependencyRegistry,
    results: dict[str, object],
    *,
    scope: str = "demo",
) -> tuple[dict[str, object], dict[str, object]]:
    preflight = runtime.publish_daily_dependency_snapshot(
        tmp_path,
        target,
        scope=scope,
        results=results,
        phase="preflight",
        registry=registry,
    )
    postflight_results = {**results, "dependency_preflight": preflight}
    postflight = runtime.publish_daily_dependency_snapshot(
        tmp_path,
        target,
        scope=scope,
        results=postflight_results,
        phase="postflight",
        strict_freshness=True,
        registry=registry,
    )
    return preflight, postflight


def test_freshness_audit_aggregates_exact_and_poll_failures() -> None:
    exact = _node("data.exact", Layer.DATA_SOURCE, final_gate=True)
    polled = _node(
        "data.polled",
        Layer.DATA_SOURCE,
        cadence=Cadence.EVENT_POLL_DAILY,
        freshness=FreshnessPolicy(FreshnessMode.POLLED_THROUGH),
        final_gate=True,
    )
    product = _node(
        "product.output",
        Layer.PRODUCT,
        inputs=(DependencyEdge("data.exact"), DependencyEdge("data.polled")),
        final_gate=True,
    )
    registry = DependencyRegistry((exact, polled, product), {"demo": (product.node_id,)})
    target = date(2026, 8, 12)
    states = {
        "data.exact": NodeState("data.exact", watermark=date(2026, 8, 11)),
        "product.output": NodeState("product.output", watermark=target),
    }

    audit = runtime.audit_required_freshness(registry, "demo", target, states)

    assert audit["status"] == "failed"
    assert audit["checked_nodes"] == [
        "data.exact",
        "data.polled",
        "product.output",
    ]
    failures = {item["node_id"]: item for item in audit["failures"]}
    assert set(failures) == {"data.exact", "data.polled"}
    assert failures["data.exact"] == {
        "node_id": "data.exact",
        "mode": "exact_trade_date",
        "expected": "2026-08-12",
        "actual": "2026-08-11",
    }
    assert failures["data.polled"]["actual"] is None


def test_required_json_predicate_is_part_of_freshness_evidence(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "target_date": "2026-08-12",
                "candidate_coverage_status": "incomplete",
            }
        ),
        encoding="utf-8",
    )
    node = _node(
        "feature.sidecar",
        Layer.FEATURE,
        freshness=FreshnessPolicy(
            FreshnessMode.EXACT_TRADE_DATE,
            (
                EvidenceSpec(
                    "json",
                    "manifest.json",
                    "target_date",
                    predicate_field="candidate_coverage_status",
                    expected_value="complete",
                ),
            ),
        ),
        final_gate=True,
    )
    product = _node(
        "product.output",
        Layer.PRODUCT,
        inputs=(DependencyEdge("feature.sidecar"),),
    )
    registry = DependencyRegistry(
        (node, product),
        {"demo": ("product.output",)},
    )

    assert runtime.collect_node_states(registry, tmp_path) == {}

    manifest.write_text(
        json.dumps(
            {
                "target_date": "2026-08-12",
                "candidate_coverage_status": "complete",
            }
        ),
        encoding="utf-8",
    )

    states = runtime.collect_node_states(registry, tmp_path)
    assert states["feature.sidecar"].watermark == date(2026, 8, 12)


def test_strategy_signal_manifest_proves_completion_when_union_is_empty(
    tmp_path: Path,
) -> None:
    feature_dir = tmp_path / "data/features/b1"
    feature_dir.mkdir(parents=True)
    for relative_path in (
        "b1/b1_gate_candidates.parquet",
        "b1/b1_family_rule_candidates.parquet",
        "z_skill_daily_candidates.parquet",
    ):
        path = feature_dir.parent / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["symbol", "date"]).to_parquet(path, index=False)
    (feature_dir / "b1_gate_manifest.json").write_text(
        json.dumps(
            {
                "status": "success",
                "processed_through_date": "2026-08-12",
                "candidate_rows": 0,
                "z_candidate_count": 0,
                "union_candidate_count": 0,
            }
        ),
        encoding="utf-8",
    )

    states = runtime.collect_node_states(
        DEFAULT_DAILY_DEPENDENCY_REGISTRY,
        tmp_path,
    )

    assert states["feature.strategy_signals"].watermark == date(2026, 8, 12)


@pytest.mark.parametrize(
    "reference_poll,expected_watermark",
    (
        ({"status": "success", "polled_through": "20260812"}, date(2026, 8, 12)),
        ({"status": "failed", "polled_through": None}, None),
    ),
    ids=("successful_empty_poll", "failed_poll_fallback"),
)
def test_convertible_bond_reference_nested_poll_evidence_is_fail_closed(
    tmp_path: Path,
    reference_poll: dict[str, str | None],
    expected_watermark: date | None,
) -> None:
    results = {
        "convertible_bond_plan": {
            "status": "success",
            "data_refresh": {
                "status": "no_data",
                "reference_poll": reference_poll,
            },
        }
    }

    states = runtime.collect_node_states(
        DEFAULT_DAILY_DEPENDENCY_REGISTRY,
        tmp_path,
        results,
    )

    if expected_watermark is None:
        assert "data.cb_reference" not in states
    else:
        assert states["data.cb_reference"].polled_through == expected_watermark


def test_strategy_signal_manifest_is_exact_truth_when_b1_is_empty_and_z_has_hits(
    tmp_path: Path,
) -> None:
    feature_dir = tmp_path / "data/features/b1"
    feature_dir.mkdir(parents=True)
    pd.DataFrame(columns=["symbol", "date", "b1_signal"]).to_parquet(
        feature_dir / "b1_gate_candidates.parquet",
        index=False,
    )
    pd.DataFrame(columns=["symbol", "date", "b1_signal"]).to_parquet(
        feature_dir / "b1_family_rule_candidates.parquet",
        index=False,
    )
    pd.DataFrame(
        {
            "symbol": ["000001.SZ"],
            "date": [pd.Timestamp("2026-08-12")],
            "z_signal": [True],
        }
    ).to_parquet(feature_dir.parent / "z_skill_daily_candidates.parquet", index=False)
    (feature_dir / "b1_gate_manifest.json").write_text(
        json.dumps(
            {
                "status": "success",
                "processed_through_date": "2026-08-12",
                "signal_scan_status": "complete",
                "b1_candidate_count": 0,
                "z_candidate_count": 1,
                "union_candidate_count": 1,
                "empty_candidate_set": False,
            }
        ),
        encoding="utf-8",
    )

    states = runtime.collect_node_states(
        DEFAULT_DAILY_DEPENDENCY_REGISTRY,
        tmp_path,
    )

    assert states["feature.strategy_signals"].watermark == date(2026, 8, 12)


def test_active_project_manifest_is_exact_truth_when_candidate_union_is_empty(
    tmp_path: Path,
) -> None:
    feature_dir = tmp_path / "data/features/b1"
    feature_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "symbol": ["000001.SZ"],
            "date": [pd.Timestamp("2026-08-11")],
            "close": [10.0],
        }
    ).to_parquet(feature_dir / "training_xgb_project_vars.parquet", index=False)
    pd.DataFrame(columns=["symbol", "date", "factor_schema_version"]).to_parquet(
        feature_dir / "active_candidate_project_features.parquet",
        index=False,
    )
    (feature_dir / "active_candidate_project_features_manifest.json").write_text(
        json.dumps(
            {
                "status": "success",
                "target_date": "2026-08-12",
                "candidate_coverage_status": "complete",
                "b1_candidate_count": 0,
                "z_candidate_count": 0,
                "union_candidate_count": 0,
                "computed_candidate_count": 0,
                "empty_candidate_set": True,
                "factor_count": len(PROJECT_FACTOR_COLUMNS),
                "factor_schema_version": PRODUCTION_PROJECT_FACTOR_SCHEMA,
            }
        ),
        encoding="utf-8",
    )

    states = runtime.collect_node_states(
        DEFAULT_DAILY_DEPENDENCY_REGISTRY,
        tmp_path,
    )

    assert states["feature.project_daily"].watermark == date(2026, 8, 12)


def test_artifact_extractors_keep_shape_contract_and_remove_zero_importance(
    tmp_path: Path,
) -> None:
    sklearn_path = tmp_path / "sklearn.joblib"
    bundle_path = tmp_path / "bundle.joblib"
    joblib.dump(_SklearnArtifact(), sklearn_path)
    joblib.dump(
        {
            "features": ("close", "top_list_count", "volume"),
            "model": _ImportanceModel(),
        },
        bundle_path,
    )

    sklearn = runtime._extract_artifact(sklearn_path, "sklearn_feature_names")
    bundle = runtime._extract_artifact(bundle_path, "bundle_features")

    assert sklearn == {
        "features": ["close", "volume", "top_list_count"],
        "effective_features": ["close", "volume"],
        "schema_version": None,
    }
    assert bundle == {
        "features": ["close", "top_list_count", "volume"],
        "effective_features": ["close", "volume"],
        "schema_version": None,
    }


def test_model_contract_cache_avoids_reextracting_unchanged_artifact(
    monkeypatch,
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "models/demo.joblib"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(b"stable-test-artifact")
    feature = _node("feature.demo", Layer.FEATURE)
    score = _node(
        "score.demo",
        Layer.MODEL_SCORE,
        inputs=(DependencyEdge("feature.demo"),),
        artifact=ArtifactSpec(
            artifact_paths=("models/demo.joblib",),
            extractor="bundle_features",
            feature_node_id="feature.demo",
        ),
    )
    product = _node(
        "product.demo",
        Layer.PRODUCT,
        inputs=(DependencyEdge("score.demo"),),
    )
    registry = DependencyRegistry(
        (feature, score, product),
        {"demo": ("product.demo",)},
    )
    extraction_calls: list[tuple[Path, ...]] = []

    def fake_extract(project_root, extractor, paths):
        materialized = tuple(paths)
        extraction_calls.append(materialized)
        return {
            str(path): {
                "features": ["close", "top_list_count"],
                "effective_features": ["close"],
            }
            for path in materialized
        }

    monkeypatch.setattr(runtime, "_extract_artifacts_subprocess", fake_extract)
    cache_path = tmp_path / "model_contract_cache.json"

    first, first_audit = runtime.resolve_model_contracts(
        registry,
        tmp_path,
        "demo",
        cache_path=cache_path,
    )
    second, second_audit = runtime.resolve_model_contracts(
        registry,
        tmp_path,
        "demo",
        cache_path=cache_path,
    )

    assert first["score.demo"].required_feature_union == (
        "close",
        "top_list_count",
    )
    assert first["score.demo"].effective_feature_union == ("close",)
    assert second["score.demo"].combined_hash == first["score.demo"].combined_hash
    assert first_audit["artifacts_loaded"] == 1
    assert first_audit["artifact_cache_hits"] == 0
    assert second_audit["artifacts_loaded"] == 0
    assert second_audit["artifact_cache_hits"] == 1
    assert [len(paths) for paths in extraction_calls] == [1, 0]


def test_model_contract_rejects_declared_schema_mismatch(
    monkeypatch,
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "models/demo.joblib"
    artifact_path.parent.mkdir(parents=True)
    joblib.dump(
        {
            "features": ["close"],
            "model": None,
            "schema_version": "demo_schema_v1",
        },
        artifact_path,
    )
    feature = _node("feature.demo", Layer.FEATURE)
    score = _node(
        "score.demo",
        Layer.MODEL_SCORE,
        inputs=(DependencyEdge("feature.demo"),),
        artifact=ArtifactSpec(
            artifact_paths=("models/demo.joblib",),
            extractor="bundle_features",
            feature_node_id="feature.demo",
            expected_schema="demo_schema_v2",
        ),
    )
    product = _node(
        "product.demo",
        Layer.PRODUCT,
        inputs=(DependencyEdge("score.demo"),),
    )
    registry = DependencyRegistry(
        (feature, score, product),
        {"demo": ("product.demo",)},
    )
    monkeypatch.setattr(
        runtime,
        "_extract_artifacts_subprocess",
        lambda project_root, extractor, paths: {
            str(path): runtime._extract_artifact(path, extractor)
            for path in paths
        },
    )

    with pytest.raises(RuntimeError, match="schema mismatch"):
        runtime.resolve_model_contracts(
            registry,
            tmp_path,
            "demo",
            cache_path=tmp_path / "cache.json",
        )


def test_dependency_snapshot_has_stable_contract_and_plan_fields(
    tmp_path: Path,
) -> None:
    target = date(2026, 8, 12)
    source = _node(
        "data.source",
        Layer.DATA_SOURCE,
        freshness=FreshnessPolicy(
            FreshnessMode.EXACT_TRADE_DATE,
            (EvidenceSpec("result", "source", "trade_date"),),
        ),
        final_gate=True,
    )
    product = _node(
        "product.output",
        Layer.PRODUCT,
        inputs=(DependencyEdge("data.source"),),
        freshness=FreshnessPolicy(
            FreshnessMode.EXACT_TRADE_DATE,
            (EvidenceSpec("result", "output", "trade_date"),),
        ),
        final_gate=True,
    )
    registry = DependencyRegistry(
        (product, source),
        {"demo": ("product.output",)},
        schema_version="test_registry_v1",
    )
    results = {
        "source": {"status": "success", "trade_date": "20260812"},
        "output": {"status": "success", "trade_date": "2026-08-12"},
    }

    preflight, summary = _complete_snapshot_cycle(
        tmp_path,
        target,
        registry,
        results,
    )
    payload = json.loads(Path(summary["path"]).read_text(encoding="utf-8"))

    assert preflight["comparison_identity"] == "missing"
    assert preflight["baseline_committed"] is False
    assert set(preflight["refresh_node_ids"]) == {
        "data.source",
        "product.output",
    }
    assert summary["status"] == "success"
    assert summary["comparison_identity"] == "preflight"
    assert summary["baseline_committed"] is True
    assert payload["schema_version"] == runtime.SNAPSHOT_SCHEMA_VERSION
    assert payload["registry_schema_version"] == "test_registry_v1"
    assert payload["target_trade_date"] == "2026-08-12"
    assert payload["scope"] == "demo"
    assert payload["phase"] == "postflight"
    assert payload["active_nodes"] == ["data.source", "product.output"]
    assert payload["inactive_nodes"] == []
    assert payload["layer_counts"] == {
        "data_source": 1,
        "feature": 0,
        "model_score": 0,
        "product": 1,
    }
    assert [entry["node_id"] for entry in payload["incremental_plan"]] == [
        "data.source",
        "product.output",
    ]
    assert {entry["action"] for entry in payload["incremental_plan"]} == {"reuse"}
    assert payload["freshness_audit"] == {
        "status": "success",
        "checked_nodes": ["data.source", "product.output"],
        "failures": [],
    }
    assert payload["registry"]["schema_version"] == "test_registry_v1"
    assert Path(summary["dated_path"]).name == "2026-08-12-demo-postflight.json"


def test_contract_source_change_dirties_only_declared_node_and_downstream(
    tmp_path: Path,
) -> None:
    target = date(2026, 8, 12)
    config_path = tmp_path / "configs/demo.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("threshold: 1\n", encoding="utf-8")
    source = _node(
        "data.source",
        Layer.DATA_SOURCE,
        freshness=FreshnessPolicy(
            FreshnessMode.EXACT_TRADE_DATE,
            (EvidenceSpec("result", "source", "trade_date"),),
        ),
        final_gate=True,
    )
    product = DependencyNode(
        **{
            **_node(
                "product.output",
                Layer.PRODUCT,
                inputs=(DependencyEdge("data.source"),),
                freshness=FreshnessPolicy(
                    FreshnessMode.EXACT_TRADE_DATE,
                    (EvidenceSpec("result", "output", "trade_date"),),
                ),
                final_gate=True,
            ).__dict__,
            "contract_sources": ("configs/demo.yaml",),
        }
    )
    registry = DependencyRegistry(
        (source, product),
        {"demo": ("product.output",)},
    )
    results = {
        "source": {"status": "success", "trade_date": "2026-08-12"},
        "output": {"status": "success", "trade_date": "2026-08-12"},
    }

    _complete_snapshot_cycle(tmp_path, target, registry, results)
    config_path.write_text("threshold: 2\n", encoding="utf-8")
    summary = runtime.publish_daily_dependency_snapshot(
        tmp_path,
        target,
        scope="demo",
        results=results,
        registry=registry,
    )
    payload = json.loads(Path(summary["path"]).read_text(encoding="utf-8"))
    entries = {
        entry["node_id"]: entry for entry in payload["incremental_plan"]
    }

    assert summary["changed_contract_nodes"] == ["product.output"]
    assert payload["refresh_node_ids"] == ["product.output"]
    assert entries["data.source"]["action"] == "reuse"
    assert entries["product.output"]["action"] == "refresh"


@pytest.mark.parametrize(
    ("changed_output", "expected_dirty"),
    [
        (
            "data/raw/daily_partitioned/year_month=202608/data.parquet",
            {
                "data.market",
                "data.daily_basic",
                "feature.sidecar",
                "product.output",
            },
        ),
        (
            "data/raw/daily_basic/20260812.parquet",
            {"data.daily_basic", "feature.sidecar", "product.output"},
        ),
        (
            "data/features/active.parquet",
            {"feature.sidecar", "product.output"},
        ),
        (
            "data/research/watchlist.json",
            {"data.watchlist", "product.output"},
        ),
    ],
)
def test_output_content_change_dirties_node_and_downstream_without_advancing_baseline(
    tmp_path: Path,
    changed_output: str,
    expected_dirty: set[str],
) -> None:
    target = date(2026, 8, 12)
    market = replace(
        _node(
            "data.market",
            Layer.DATA_SOURCE,
            freshness=FreshnessPolicy(
                FreshnessMode.EXACT_TRADE_DATE,
                (EvidenceSpec("result", "market", "trade_date"),),
            ),
        ),
        outputs=("data/raw/daily_partitioned",),
    )
    daily_basic = replace(
        _node(
            "data.daily_basic",
            Layer.DATA_SOURCE,
            inputs=(DependencyEdge("data.market"),),
            freshness=FreshnessPolicy(
                FreshnessMode.EXACT_TRADE_DATE,
                (EvidenceSpec("result", "daily_basic", "trade_date"),),
            ),
        ),
        outputs=("data/raw/daily_basic/YYYYMMDD.parquet",),
    )
    sidecar = replace(
        _node(
            "feature.sidecar",
            Layer.FEATURE,
            inputs=(DependencyEdge("data.daily_basic"),),
            freshness=FreshnessPolicy(
                FreshnessMode.EXACT_TRADE_DATE,
                (EvidenceSpec("result", "sidecar", "target_date"),),
            ),
        ),
        outputs=("data/features/active.parquet",),
    )
    watchlist = replace(
        _node(
            "data.watchlist",
            Layer.DATA_SOURCE,
            cadence=Cadence.ON_DEMAND,
            freshness=FreshnessPolicy(
                FreshnessMode.IMMUTABLE,
                (EvidenceSpec("file_fingerprint", "data/research/watchlist.json"),),
            ),
        ),
        incremental=IncrementalPolicy(),
        outputs=("data/research/watchlist.json",),
    )
    product = _node(
        "product.output",
        Layer.PRODUCT,
        inputs=(DependencyEdge("feature.sidecar"), DependencyEdge("data.watchlist")),
        freshness=FreshnessPolicy(
            FreshnessMode.EXACT_TRADE_DATE,
            (EvidenceSpec("result", "product", "trade_date"),),
        ),
        final_gate=True,
    )
    registry = DependencyRegistry(
        (market, daily_basic, sidecar, watchlist, product),
        {"demo": ("product.output",)},
    )
    output_paths = (
        "data/raw/daily_partitioned/year_month=202608/data.parquet",
        "data/raw/daily_basic/20260812.parquet",
        "data/features/active.parquet",
        "data/research/watchlist.json",
    )
    for relative in output_paths:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"base")
    results = {
        "market": {"status": "success", "trade_date": "2026-08-12", "rows": 10},
        "daily_basic": {"status": "success", "trade_date": "2026-08-12", "rows": 10},
        "sidecar": {"status": "success", "target_date": "2026-08-12", "rows": 2},
        "product": {"status": "success", "trade_date": "2026-08-12", "rows": 2},
    }

    _, postflight = _complete_snapshot_cycle(tmp_path, target, registry, results)
    committed_path = Path(postflight["committed_path"])
    committed_before = committed_path.read_bytes()
    (tmp_path / changed_output).write_bytes(b"next")

    preflight = runtime.publish_daily_dependency_snapshot(
        tmp_path,
        target,
        scope="demo",
        results=results,
        phase="preflight",
        registry=registry,
    )

    assert set(preflight["changed_state_nodes"]) == {changed_output_to_node(changed_output)}
    assert set(preflight["refresh_node_ids"]) == expected_dirty
    assert committed_path.read_bytes() == committed_before


def changed_output_to_node(path: str) -> str:
    return {
        "data/raw/daily_partitioned/year_month=202608/data.parquet": "data.market",
        "data/raw/daily_basic/20260812.parquet": "data.daily_basic",
        "data/features/active.parquet": "feature.sidecar",
        "data/research/watchlist.json": "data.watchlist",
    }[path]


def test_identical_file_rewrite_keeps_stable_content_fingerprint(
    tmp_path: Path,
) -> None:
    target = date(2026, 8, 12)
    path = tmp_path / "data/watchlist.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"stable")
    watchlist = replace(
        _node(
            "data.watchlist",
            Layer.DATA_SOURCE,
            cadence=Cadence.ON_DEMAND,
            freshness=FreshnessPolicy(
                FreshnessMode.IMMUTABLE,
                (EvidenceSpec("file_fingerprint", "data/watchlist.json"),),
            ),
            final_gate=True,
        ),
        incremental=IncrementalPolicy(),
        outputs=("data/watchlist.json",),
    )
    product = _node(
        "product.output",
        Layer.PRODUCT,
        inputs=(DependencyEdge("data.watchlist"),),
        freshness=FreshnessPolicy(
            FreshnessMode.EXACT_TRADE_DATE,
            (EvidenceSpec("result", "product", "trade_date"),),
        ),
        final_gate=True,
    )
    registry = DependencyRegistry(
        (watchlist, product),
        {"demo": ("product.output",)},
    )
    results = {
        "product": {"status": "success", "trade_date": "2026-08-12"},
    }
    _complete_snapshot_cycle(tmp_path, target, registry, results)
    path.write_bytes(b"stable")

    preflight = runtime.publish_daily_dependency_snapshot(
        tmp_path,
        target,
        scope="demo",
        results=results,
        phase="preflight",
        registry=registry,
    )

    assert preflight["changed_state_nodes"] == []
    assert preflight["refresh_node_ids"] == []


def test_missing_checkpoint_identity_fails_closed_until_preflight_is_carried_forward(
    tmp_path: Path,
) -> None:
    target = date(2026, 8, 12)
    source = _node(
        "data.source",
        Layer.DATA_SOURCE,
        freshness=FreshnessPolicy(
            FreshnessMode.EXACT_TRADE_DATE,
            (EvidenceSpec("result", "source", "trade_date"),),
        ),
        final_gate=True,
    )
    product = _node(
        "product.output",
        Layer.PRODUCT,
        inputs=(DependencyEdge("data.source"),),
        freshness=FreshnessPolicy(
            FreshnessMode.EXACT_TRADE_DATE,
            (EvidenceSpec("result", "product", "trade_date"),),
        ),
        final_gate=True,
    )
    registry = DependencyRegistry((source, product), {"demo": ("product.output",)})
    results = {
        "source": {"status": "success", "trade_date": "2026-08-12"},
        "product": {"status": "success", "trade_date": "2026-08-12"},
    }

    preflight = runtime.publish_daily_dependency_snapshot(
        tmp_path,
        target,
        scope="demo",
        results=results,
        phase="preflight",
        registry=registry,
    )

    assert preflight["comparison_identity"] == "missing"
    assert preflight["baseline_committed"] is False
    assert set(preflight["refresh_node_ids"]) == {"data.source", "product.output"}
    assert not Path(preflight["committed_path"]).exists()
    with pytest.raises(RuntimeError, match="unresolved=data.source,product.output"):
        runtime.publish_daily_dependency_snapshot(
            tmp_path,
            target,
            scope="demo",
            results=results,
            phase="postflight",
            strict_freshness=True,
            registry=registry,
        )

    postflight = runtime.publish_daily_dependency_snapshot(
        tmp_path,
        target,
        scope="demo",
        results={**results, "dependency_preflight": preflight},
        phase="postflight",
        strict_freshness=True,
        registry=registry,
    )

    assert postflight["status"] == "success"
    assert postflight["comparison_identity"] == "preflight"
    assert postflight["baseline_committed"] is True
    assert postflight["refresh_node_ids"] == []
    assert Path(postflight["committed_path"]).is_file()


@pytest.mark.parametrize(
    "invalid_identity",
    ("missing", "incomplete", "wrong_target", "wrong_scope"),
)
def test_strict_postflight_requires_its_own_valid_preflight_even_with_committed_baseline(
    tmp_path: Path,
    invalid_identity: str,
) -> None:
    target = date(2026, 8, 12)
    source = _node(
        "data.source",
        Layer.DATA_SOURCE,
        freshness=FreshnessPolicy(
            FreshnessMode.EXACT_TRADE_DATE,
            (EvidenceSpec("result", "source", "trade_date"),),
        ),
        final_gate=True,
    )
    product = _node(
        "product.output",
        Layer.PRODUCT,
        inputs=(DependencyEdge("data.source"),),
        freshness=FreshnessPolicy(
            FreshnessMode.EXACT_TRADE_DATE,
            (EvidenceSpec("result", "product", "trade_date"),),
        ),
        final_gate=True,
    )
    registry = DependencyRegistry((source, product), {"demo": ("product.output",)})
    results = {
        "source": {"status": "success", "trade_date": "2026-08-12"},
        "product": {"status": "success", "trade_date": "2026-08-12"},
    }
    preflight, committed = _complete_snapshot_cycle(
        tmp_path,
        target,
        registry,
        results,
    )
    committed_path = Path(committed["committed_path"])
    committed_before = committed_path.read_bytes()

    postflight_results = dict(results)
    if invalid_identity != "missing":
        identity = dict(preflight)
        if invalid_identity == "incomplete":
            identity.pop("node_contract_hashes")
        elif invalid_identity == "wrong_target":
            identity["target_trade_date"] = "2026-08-11"
        elif invalid_identity == "wrong_scope":
            identity["scope"] = "other"
        postflight_results["dependency_preflight"] = identity

    with pytest.raises(
        RuntimeError,
        match="unresolved=data.source,product.output",
    ):
        runtime.publish_daily_dependency_snapshot(
            tmp_path,
            target,
            scope="demo",
            results=postflight_results,
            phase="postflight",
            strict_freshness=True,
            registry=registry,
        )

    assert committed_path.read_bytes() == committed_before


def test_model_artifact_content_change_dirties_score_and_product(
    monkeypatch,
    tmp_path: Path,
) -> None:
    target = date(2026, 8, 12)
    artifact_path = tmp_path / "models/demo.joblib"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(b"model-v1")
    feature = _node(
        "feature.input",
        Layer.FEATURE,
        freshness=FreshnessPolicy(
            FreshnessMode.EXACT_TRADE_DATE,
            (EvidenceSpec("result", "feature", "trade_date"),),
        ),
    )
    score = _node(
        "score.demo",
        Layer.MODEL_SCORE,
        inputs=(DependencyEdge("feature.input"),),
        freshness=FreshnessPolicy(
            FreshnessMode.EXACT_TRADE_DATE,
            (EvidenceSpec("result", "score", "trade_date"),),
        ),
        artifact=ArtifactSpec(
            artifact_paths=("models/demo.joblib",),
            extractor="sklearn_feature_names",
            feature_node_id="feature.input",
        ),
    )
    product = _node(
        "product.output",
        Layer.PRODUCT,
        inputs=(DependencyEdge("score.demo"),),
        freshness=FreshnessPolicy(
            FreshnessMode.EXACT_TRADE_DATE,
            (EvidenceSpec("result", "product", "trade_date"),),
        ),
        final_gate=True,
    )
    registry = DependencyRegistry(
        (feature, score, product),
        {"demo": ("product.output",)},
    )
    results = {
        key: {"status": "success", "trade_date": "2026-08-12"}
        for key in ("feature", "score", "product")
    }
    monkeypatch.setattr(
        runtime,
        "_extract_artifacts_subprocess",
        lambda project_root, extractor, paths: {
            str(path): {
                "features": ["close"],
                "effective_features": ["close"],
            }
            for path in paths
        },
    )
    _complete_snapshot_cycle(tmp_path, target, registry, results)
    artifact_path.write_bytes(b"model-v2")

    preflight = runtime.publish_daily_dependency_snapshot(
        tmp_path,
        target,
        scope="demo",
        results=results,
        phase="preflight",
        registry=registry,
    )

    assert preflight["changed_model_nodes"] == ["score.demo"]
    assert set(preflight["refresh_node_ids"]) == {"score.demo", "product.output"}


def test_registry_definition_change_alters_node_contract_hash(
    tmp_path: Path,
) -> None:
    exact = _node("data.source", Layer.DATA_SOURCE)
    ttl = _node(
        "data.source",
        Layer.DATA_SOURCE,
        cadence=Cadence.STATIC,
        freshness=FreshnessPolicy(FreshnessMode.TTL, max_age_days=7),
    )
    product = _node(
        "product.output",
        Layer.PRODUCT,
        inputs=(DependencyEdge("data.source"),),
    )
    exact_registry = DependencyRegistry(
        (exact, product),
        {"demo": ("product.output",)},
    )
    ttl_registry = DependencyRegistry(
        (ttl, product),
        {"demo": ("product.output",)},
    )

    exact_hash = runtime._node_contract_hashes(
        exact_registry,
        tmp_path,
        ("data.source",),
    )
    ttl_hash = runtime._node_contract_hashes(
        ttl_registry,
        tmp_path,
        ("data.source",),
    )

    assert exact_hash["data.source"] != ttl_hash["data.source"]
