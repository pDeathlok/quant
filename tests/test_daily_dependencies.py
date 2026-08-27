from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import pytest
import yaml

from quant.application.daily_dependencies import (
    DEFAULT_DAILY_DEPENDENCY_REGISTRY,
    ArtifactSpec,
    Cadence,
    ColumnMode,
    DependencyEdge,
    DependencyNode,
    DependencyRegistry,
    FreshnessMode,
    FreshnessPolicy,
    IncrementalPolicy,
    Layer,
    Lifecycle,
    ModelContract,
    NodeState,
    build_dependency_plan,
    classify_feature_usage,
)
from quant.routine.daily_dependency_runtime import collect_node_states


def _node(
    node_id: str,
    layer: Layer,
    *,
    inputs: tuple[DependencyEdge, ...] = (),
    freshness: FreshnessPolicy | None = None,
    cadence: Cadence = Cadence.TRADE_DAILY,
    lifecycle: Lifecycle = Lifecycle.PRODUCTION,
    partitioned: bool = True,
    artifact: ArtifactSpec | None = None,
) -> DependencyNode:
    return DependencyNode(
        node_id=node_id,
        layer=layer,
        owner="tests",
        lifecycle=lifecycle,
        cadence=cadence,
        inputs=inputs,
        freshness=freshness
        or FreshnessPolicy(FreshnessMode.EXACT_TRADE_DATE),
        incremental=IncrementalPolicy(
            partition_key="date" if partitioned else None,
        ),
        operation_id=f"run_{node_id}",
        artifact=artifact,
    )


def _planning_registry() -> DependencyRegistry:
    return DependencyRegistry(
        (
            _node("data.exact", Layer.DATA_SOURCE),
            _node(
                "data.poll",
                Layer.DATA_SOURCE,
                cadence=Cadence.EVENT_POLL_DAILY,
                freshness=FreshnessPolicy(FreshnessMode.POLLED_THROUGH),
                partitioned=False,
            ),
            _node(
                "data.ttl",
                Layer.DATA_SOURCE,
                cadence=Cadence.WEEKLY,
                freshness=FreshnessPolicy(
                    FreshnessMode.TTL,
                    max_age_days=7,
                ),
                partitioned=False,
            ),
            _node(
                "feature.live",
                Layer.FEATURE,
                inputs=(
                    DependencyEdge("data.exact"),
                    DependencyEdge("data.poll"),
                    DependencyEdge("data.ttl"),
                ),
            ),
            _node(
                "product.live",
                Layer.PRODUCT,
                inputs=(DependencyEdge("feature.live"),),
            ),
        ),
        {"demo": ("product.live",)},
    )


def _entries_by_id(entries):
    return {entry.node_id: entry for entry in entries}


def test_registry_rejects_duplicate_missing_and_cyclic_dependencies() -> None:
    duplicate = _node("data.same", Layer.DATA_SOURCE)
    with pytest.raises(ValueError, match="duplicate node ids"):
        DependencyRegistry((duplicate, duplicate), {"demo": ("data.same",)})

    with pytest.raises(ValueError, match="references missing"):
        DependencyRegistry(
            (
                _node(
                    "product.missing",
                    Layer.PRODUCT,
                    inputs=(DependencyEdge("data.unknown"),),
                ),
            ),
            {"demo": ("product.missing",)},
        )

    with pytest.raises(ValueError, match="contains a cycle"):
        DependencyRegistry(
            (
                _node(
                    "product.a",
                    Layer.PRODUCT,
                    inputs=(DependencyEdge("product.b"),),
                ),
                _node(
                    "product.b",
                    Layer.PRODUCT,
                    inputs=(DependencyEdge("product.a"),),
                ),
            ),
            {"demo": ("product.a",)},
        )


def test_default_scope_closure_contains_only_required_production_branches() -> None:
    short = set(DEFAULT_DAILY_DEPENDENCY_REGISTRY.required_node_ids("short"))
    assert {
        "data.market_daily",
        "data.daily_basic",
        "feature.left_side_unified",
        "score.left_side_unified",
        "score.right_side_unified",
        "score.selector",
        "product.short_snapshot",
    } <= short
    assert "score.b1" not in short
    assert "score.z_skill" not in short
    assert "product.long_pools" not in short
    # The active selector buy/hold artifact consumes the exact-date long
    # factor snapshot, so its PIT inputs are now part of the short closure.
    assert "feature.long_snapshot" in short
    assert "data.financial_pit" in short
    assert "data.analyst_pit" in short
    assert "data.tradability" not in short

    similar = set(DEFAULT_DAILY_DEPENDENCY_REGISTRY.required_node_ids("similar"))
    assert {
        "data.market_daily",
        "data.csi300_daily",
        "data.stock_basic",
        "data.similar_watchlist",
        "feature.similar_reference",
        "feature.similar_target",
        "score.similar",
        "product.similar",
    } <= similar
    assert "feature.selector_live" not in similar


def test_right_side_shadow_is_a_separate_four_layer_research_closure() -> None:
    registry = DEFAULT_DAILY_DEPENDENCY_REGISTRY
    shadow = registry.required_node_ids("rightSideShadow")

    assert shadow == (
        "data.market_daily",
        "feature.strategy_signals",
        "feature.right_side_unified_shadow",
        "score.right_side_unified_shadow",
        "product.right_side_unified_shadow",
    )
    assert registry.nodes["feature.right_side_unified_shadow"].lifecycle == (
        Lifecycle.RESEARCH_ONLY
    )
    assert registry.nodes["score.right_side_unified_shadow"].lifecycle == (
        Lifecycle.RESEARCH_ONLY
    )
    assert registry.nodes["product.right_side_unified_shadow"].lifecycle == (
        Lifecycle.RESEARCH_ONLY
    )
    assert {
        registry.nodes[node_id].layer
        for node_id in shadow
    } == {
        Layer.DATA_SOURCE,
        Layer.FEATURE,
        Layer.MODEL_SCORE,
        Layer.PRODUCT,
    }
    assert "score.selector" not in shadow
    assert "score.right_side_unified_shadow" not in set(
        registry.required_node_ids("short")
    )
    assert "score.right_side_unified_shadow" not in set(
        registry.required_node_ids("all")
    )


def test_right_side_shadow_model_change_dirties_only_shadow_score_and_product() -> None:
    registry = DEFAULT_DAILY_DEPENDENCY_REGISTRY
    target = date(2026, 8, 12)
    active = registry.required_node_ids("rightSideShadow")
    states = {
        node_id: NodeState(
            node_id,
            watermark=target,
            output_fingerprint=f"stable-{node_id}",
        )
        for node_id in active
    }

    entries = _entries_by_id(
        build_dependency_plan(
            registry,
            "rightSideShadow",
            target,
            states,
            changed_nodes=("score.right_side_unified_shadow",),
            include_unused=False,
        )
    )

    assert entries["data.market_daily"].action == "reuse"
    assert entries["feature.strategy_signals"].action == "reuse"
    assert entries["feature.right_side_unified_shadow"].action == "reuse"
    assert entries["score.right_side_unified_shadow"].action == "refresh"
    assert entries["product.right_side_unified_shadow"].action == "refresh"


def test_all_production_strategy_configs_and_calculators_are_registered() -> None:
    project_root = Path(__file__).resolve().parents[1]
    registry = DEFAULT_DAILY_DEPENDENCY_REGISTRY
    registered_sources = {
        source
        for node in registry.nodes.values()
        for source in node.contract_sources
    }
    missing_files = sorted(
        source
        for source in registered_sources
        if not (project_root / source).is_file()
    )
    assert missing_files == []

    production_configs: set[str] = set()
    unregistered_research_configs: set[str] = set()
    for path in sorted((project_root / "configs/strategies").glob("*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        release = payload.get("release") or {}
        lifecycle = str(release.get("lifecycle") or "")
        relative = path.relative_to(project_root).as_posix()
        assert lifecycle in {"production", "research_only"}, (
            f"strategy config must declare release.lifecycle: {relative}"
        )
        dependency_nodes = release.get("dependency_nodes")
        assert isinstance(dependency_nodes, list), (
            f"strategy config must declare release.dependency_nodes: {relative}"
        )
        if lifecycle == "production":
            production_configs.add(relative)
            assert dependency_nodes, (
                f"production strategy has no dependency node: {relative}"
            )
            for node_id in dependency_nodes:
                assert node_id in registry.nodes, (
                    f"strategy references unknown dependency node: {relative} -> {node_id}"
                )
                node = registry.nodes[node_id]
                assert node.lifecycle == Lifecycle.PRODUCTION
                assert relative in node.contract_sources, (
                    f"strategy is not registered on its declared consumer: "
                    f"{relative} -> {node_id}"
                )
        elif lifecycle == "research_only":
            if not dependency_nodes:
                unregistered_research_configs.add(relative)
                continue
            for node_id in dependency_nodes:
                assert node_id in registry.nodes, (
                    f"research strategy references unknown dependency node: "
                    f"{relative} -> {node_id}"
                )
                node = registry.nodes[node_id]
                assert node.lifecycle == Lifecycle.RESEARCH_ONLY
                assert relative in node.contract_sources

    assert production_configs <= registered_sources
    assert unregistered_research_configs.isdisjoint(registered_sources)

    unversioned_live_nodes = sorted(
        node.node_id
        for node in registry.nodes.values()
        if node.lifecycle == Lifecycle.PRODUCTION
        and not node.contract_sources
        and node.artifact is None
    )
    assert unversioned_live_nodes == []

    reachable = {
        node_id
        for scope in registry.scope_roots
        for node_id in registry.required_node_ids(scope)
    }
    orphaned_production = sorted(
        node_id
        for node_id, node in registry.nodes.items()
        if node.lifecycle == Lifecycle.PRODUCTION and node_id not in reachable
    )
    assert orphaned_production == []
    assert all(
        registry.nodes[root].final_gate
        for roots in registry.scope_roots.values()
        for root in roots
    )


def test_production_data_sources_register_their_primary_implementations() -> None:
    registry = DEFAULT_DAILY_DEPENDENCY_REGISTRY
    primary_sources = {
        "data.market_daily": "src/quant/routine/data_refresh.py",
        "data.daily_basic": "src/quant/routine/daily_basic_refresh.py",
        "data.csi300_daily": "src/quant/routine/reference_data_refresh.py",
        "data.stock_basic": "src/quant/routine/reference_data_refresh.py",
        "data.financial_pit": "src/quant/routine/reference_data_refresh.py",
        "data.analyst_pit": "scripts/research/refresh_analyst_forecasts.py",
        "data.top_list": "src/quant/data/long_factor_backfill.py",
        "data.cb_daily": "src/quant/routine/convertible_bond_grid_plan.py",
        "data.cb_reference": "src/quant/routine/convertible_bond_grid_plan.py",
        "data.cb_allotment_events": "src/quant/routine/convertible_bond_allotment.py",
        "data.byd_intraday_training": "src/quant/application/workspaces/byd.py",
        "data.similar_watchlist": "src/quant/webapp/services.py",
    }
    production_source_ids = {
        node.node_id
        for node in registry.nodes.values()
        if node.lifecycle == Lifecycle.PRODUCTION
        and node.layer == Layer.DATA_SOURCE
    }

    assert set(primary_sources) == production_source_ids
    for node_id, primary_source in primary_sources.items():
        assert primary_source in registry.nodes[node_id].contract_sources


def test_legal_empty_features_use_completion_manifests_as_date_truth() -> None:
    registry = DEFAULT_DAILY_DEPENDENCY_REGISTRY
    strategy = registry.nodes["feature.strategy_signals"]
    project = registry.nodes["feature.project_daily"]
    strategy_sidecars = {
        "data/features/b1/b1_gate_candidates.parquet",
        "data/features/b1/b1_family_rule_candidates.parquet",
        "data/features/z_skill_daily_candidates.parquet",
    }

    assert [
        (item.adapter, item.locator, item.date_field)
        for item in strategy.freshness.evidence
        if item.date_field is not None
    ] == [
        (
            "json",
            "data/features/b1/b1_gate_manifest.json",
            "processed_through_date",
        )
    ]
    assert {
        item.locator
        for item in strategy.freshness.evidence
        if item.adapter == "file_fingerprint"
    } == strategy_sidecars
    assert strategy_sidecars | {
        "data/features/b1/b1_gate_manifest.json"
    } == set(strategy.outputs)

    project_evidence = {
        (item.adapter, item.locator, item.date_field)
        for item in project.freshness.evidence
    }
    assert (
        "file_fingerprint",
        "data/features/b1/active_candidate_project_features.parquet",
        None,
    ) in project_evidence
    assert all(item.adapter != "parquet_max" for item in project.freshness.evidence)
    assert all(
        item.locator != "data/features/b1/training_xgb_project_vars.parquet"
        for item in project.freshness.evidence
    )
    assert (
        "data/features/b1/active_candidate_project_features_manifest.json"
        in project.outputs
    )
    assert "data/features/b1/training_xgb_project_vars.parquet" not in project.outputs


@pytest.mark.parametrize(
    "missing_sidecar",
    (
        "data/features/b1/b1_gate_candidates.parquet",
        "data/features/b1/b1_family_rule_candidates.parquet",
        "data/features/z_skill_daily_candidates.parquet",
    ),
    ids=("b1_gate", "b1_family", "z_skill"),
)
def test_strategy_signal_freshness_fails_when_required_sidecar_is_missing(
    tmp_path: Path,
    missing_sidecar: str,
) -> None:
    sidecars = (
        "data/features/b1/b1_gate_candidates.parquet",
        "data/features/b1/b1_family_rule_candidates.parquet",
        "data/features/z_skill_daily_candidates.parquet",
    )
    for locator in sidecars:
        if locator == missing_sidecar:
            continue
        path = tmp_path / locator
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    manifest = tmp_path / "data/features/b1/b1_gate_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        '{"status":"success","processed_through_date":"2026-08-12"}',
        encoding="utf-8",
    )

    states = collect_node_states(DEFAULT_DAILY_DEPENDENCY_REGISTRY, tmp_path)

    assert "feature.strategy_signals" not in states


def test_event_sources_use_independent_successful_poll_watermarks() -> None:
    registry = DEFAULT_DAILY_DEPENDENCY_REGISTRY
    expected = {
        "data.stock_basic": (
            "refresh_reference_inputs.steps.stock_basic",
            "polled_through",
        ),
        "data.financial_pit": (
            "refresh_reference_inputs.steps.financials",
            "polled_through",
        ),
        "data.analyst_pit": (
            "refresh_reference_inputs.steps.analyst_forecast_snapshot",
            "polled_through",
        ),
        "data.top_list": (
            "refresh_reference_inputs.steps.long_factor_sources.datasets.top_list",
            "polled_through",
        ),
        "data.cb_reference": (
            "convertible_bond_plan.data_refresh.reference_poll",
            "polled_through",
        ),
        "data.cb_allotment_events": (
            "convertible_bond_allotment",
            "event_polled_through",
        ),
    }

    for node_id, locator in expected.items():
        evidence = registry.nodes[node_id].freshness.evidence
        assert len(evidence) == 1
        assert (evidence[0].locator, evidence[0].date_field) == locator


def test_postflight_result_backed_nodes_do_not_request_second_refresh() -> None:
    from datetime import date

    from quant.application.daily_dependencies import NodeState
    from quant.routine.daily_dependency_runtime import _postflight_result_backed_nodes

    target = date(2026, 8, 13)
    states = {
        node_id: NodeState(node_id=node_id, watermark=target)
        for node_id in ("data.cb_daily", "feature.cb_grid", "product.cb_grid")
    }

    completed = _postflight_result_backed_nodes(
        DEFAULT_DAILY_DEPENDENCY_REGISTRY,
        states,
        states,
        target,
    )

    assert completed == {"data.cb_daily", "feature.cb_grid", "product.cb_grid"}


def test_chan_effective_artifact_features_prune_unused_top_list_source() -> None:
    registry = DEFAULT_DAILY_DEPENDENCY_REGISTRY
    full_contract = ModelContract(
        node_id="score.chan",
        artifact_hashes=(("chan.joblib", "abc"),),
        features_by_artifact={
            "chan.joblib": ("close", "top_list_count", "top_net_rate"),
        },
        effective_features_by_artifact={"chan.joblib": ("close",)},
        required_feature_union=("close", "top_list_count", "top_net_rate"),
        effective_feature_union=("close",),
        combined_hash="abc",
    )
    catalogs = {
        "feature.chan_live": (
            "close",
            "top_list_count",
            "top_net_amount_ratio",
            "top_net_rate",
        ),
    }

    usage = {
        item.feature_node_id: item
        for item in classify_feature_usage(
            registry,
            "chan",
            {"score.chan": full_contract},
            catalogs,
        )
    }["feature.chan_live"]
    assert usage.effective == ("close",)
    assert usage.contract_only_zero_importance == (
        "top_list_count",
        "top_net_rate",
    )
    assert usage.projection_safe is True

    unpruned = set(registry.required_node_ids("chan"))
    pruned = set(
        registry.required_node_ids(
            "chan",
            effective_feature_requirements={"feature.chan_live": ("close",)},
        )
    )
    assert "data.top_list" in unpruned
    assert "data.top_list" not in pruned
    assert {"feature.chan_live", "score.chan", "product.chan"} <= pruned

    required_top_list = set(
        registry.required_node_ids(
            "chan",
            effective_feature_requirements={
                "feature.chan_live": ("close", "top_list_count"),
            },
        )
    )
    assert "data.top_list" in required_top_list


def test_plan_reuses_exact_and_ttl_but_polls_event_source_conditionally() -> None:
    registry = _planning_registry()
    target = date(2026, 8, 12)
    now = datetime(2026, 8, 13, 9, 0)
    states = {
        "data.exact": NodeState("data.exact", watermark=target),
        "data.ttl": NodeState("data.ttl", checked_at=now - timedelta(days=2)),
        "feature.live": NodeState("feature.live", watermark=target),
        "product.live": NodeState("product.live", watermark=target),
    }

    plan = _entries_by_id(
        build_dependency_plan(registry, "demo", target, states, now=now)
    )

    assert plan["data.exact"].action == "reuse"
    assert plan["data.ttl"].action == "reuse"
    assert plan["data.poll"].action == "poll"
    assert plan["data.poll"].dirty.partitions == ()
    assert plan["feature.live"].action == "refresh_if_changed"
    assert plan["feature.live"].dirty.partitions == ("2026-08-12",)
    assert plan["product.live"].action == "refresh_if_changed"


def test_short_feature_contracts_cover_their_runtime_implementation_closure() -> None:
    registry = DEFAULT_DAILY_DEPENDENCY_REGISTRY
    signal_sources = set(
        registry.nodes["feature.strategy_signals"].contract_sources
    )
    project_sources = set(
        registry.nodes["feature.project_daily"].contract_sources
    )

    assert {
        "scripts/research/analyze_b1_family_rule_backtest.py",
        "scripts/research/analyze_z_skill_entry_exit_backtest.py",
        "src/quant/data/factors/data_adapter.py",
        "src/quant/features/daily_factor_layer.py",
    } <= signal_sources
    assert {
        "scripts/research/refresh_b1_feature_cache.py",
        "scripts/research/train_b1_tushare_models.py",
        "src/quant/data/factors/data_adapter.py",
        "src/quant/features/daily_factor_layer.py",
        "src/quant/features/project_factor_layer.py",
    } <= project_sources


def test_plan_propagates_stale_exact_partition_and_expired_ttl() -> None:
    registry = _planning_registry()
    target = date(2026, 8, 12)
    now = datetime(2026, 8, 13, 9, 0)
    base_states = {
        "data.exact": NodeState("data.exact", watermark=target),
        "data.poll": NodeState("data.poll", polled_through=target),
        "data.ttl": NodeState("data.ttl", checked_at=now),
        "feature.live": NodeState("feature.live", watermark=target),
        "product.live": NodeState("product.live", watermark=target),
    }

    stale_exact = dict(base_states)
    stale_exact["data.exact"] = NodeState(
        "data.exact",
        watermark=target - timedelta(days=1),
    )
    exact_plan = _entries_by_id(
        build_dependency_plan(registry, "demo", target, stale_exact, now=now)
    )
    assert exact_plan["data.exact"].action == "refresh"
    assert exact_plan["data.exact"].dirty.partitions == ("2026-08-12",)
    assert exact_plan["feature.live"].action == "refresh"
    assert exact_plan["product.live"].action == "refresh"

    expired_ttl = dict(base_states)
    expired_ttl["data.ttl"] = NodeState(
        "data.ttl",
        checked_at=now - timedelta(days=8),
    )
    ttl_plan = _entries_by_id(
        build_dependency_plan(registry, "demo", target, expired_ttl, now=now)
    )
    assert ttl_plan["data.ttl"].action == "refresh"
    assert ttl_plan["feature.live"].action == "refresh"
    assert ttl_plan["product.live"].action == "refresh"
