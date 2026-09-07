from __future__ import annotations

import pytest

from quant.routine.checkpoint_store import CheckpointStore
from quant.routine.default_operations import CORE_OPERATION_INPUTS, DEFAULT_DAILY_OPERATION_REGISTRY
from quant.routine.operation_contracts import CacheMode, CachePolicy, InputSnapshot, NodeChanges, OperationContext
from quant.routine.operation_identity import core_input_snapshots, snapshot_from_artifacts, snapshot_from_revision
from tests.test_daily_dag_executor import _operation


@pytest.mark.parametrize("mutation", ["model", "environment", "delete"])
def test_runtime_contract_change_cannot_save_a_successful_checkpoint(tmp_path, monkeypatch, mutation):
    from quant.application.daily_dependencies import DependencyRegistry, Layer
    from quant.routine.dag_executor import DailyDagExecutor
    from quant.routine.operation_registry import OperationRegistry
    from tests.test_daily_dag_executor import _node, _success

    model = tmp_path / "model.bin"
    model.write_bytes(b"original")
    node = _node("product.test", "test", Layer.PRODUCT)
    definition = _operation("test", (node.node_id,), cache=CachePolicy(
        CacheMode.EXACT_DATE, "v1", ("output",), contract_paths=("model.bin",),
        environment_keys=("TEST_RUNTIME_MODEL_MODE",),
    ))

    def handler(context):
        if mutation == "model":
            model.write_bytes(b"changed")
        elif mutation == "delete":
            model.unlink()
        else:
            monkeypatch.setenv("TEST_RUNTIME_MODEL_MODE", "changed")
        (tmp_path / "output").write_bytes(b"unverified")
        return _success(node.node_id)

    store = CheckpointStore(tmp_path, tmp_path / "checkpoints")
    executor = DailyDagExecutor(
        DependencyRegistry((node,), {"demo": (node.node_id,)}),
        OperationRegistry((definition,)), project_root=tmp_path,
        checkpoint_store=store, handlers={"test": handler},
    )
    result = executor.execute(target_trade_date="2026-09-04", scope="demo")
    assert result["status"] == "failed"
    assert not list((tmp_path / "checkpoints").glob("*.json"))


def test_all_five_core_operations_have_complete_declared_inputs_and_model_contracts():
    assert len(CORE_OPERATION_INPUTS) == 5
    for operation_id, inputs in CORE_OPERATION_INPUTS.items():
        definition = DEFAULT_DAILY_OPERATION_REGISTRY.definitions[operation_id]
        assert definition.input_ids == inputs
        assert definition.cache.track_python_imports
        assert definition.cache.output_paths
        assert "src/quant/routine/operation_adapters.py" in definition.cache.contract_paths
    right = DEFAULT_DAILY_OPERATION_REGISTRY.definitions["run_right_side_unified"]
    assert "feature.project_daily" in right.input_ids
    assert "source.market_daily_parquet" in right.input_ids
    assert any(path.endswith("ranking.joblib") for path in right.cache.contract_paths)
    chan = DEFAULT_DAILY_OPERATION_REGISTRY.definitions["refresh_chan_model_scores"]
    assert "data.top_list" in chan.input_ids
    assert "reports/chan_daily/model_filter/live_refresh_manifest.json" in chan.cache.output_paths
    assert "reports/chan_daily/model_filter/chan_model_dataset.parquet" in chan.cache.contract_paths
    assert "score.chan" in chan.produces


def test_chan_training_reference_changes_checkpoint_identity(tmp_path):
    from dataclasses import replace

    relative = "reports/chan_daily/model_filter/chan_model_dataset.parquet"
    definition = DEFAULT_DAILY_OPERATION_REGISTRY.definitions["refresh_chan_model_scores"]
    assert relative in definition.cache.contract_paths
    definition = replace(definition, cache=replace(
        definition.cache, contract_paths=(relative,), optional_contract_paths=(),
        environment_keys=(), track_python_imports=False,
    ))
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(b"frozen-reference-v1")
    store = CheckpointStore(tmp_path, tmp_path / "checkpoints")
    context = OperationContext(
        "2026-09-07", "all", 1, {},
        upstream_fingerprints={node: "source-v1" for node in definition.input_ids},
    )
    first, payload = store.build_identity(definition, context)
    assert relative in payload["contract_hashes"]
    path.write_bytes(b"frozen-reference-v2")
    second, _ = store.build_identity(definition, context)
    assert first != second


def test_wrapper_import_changes_actual_implementation_identity(tmp_path, monkeypatch):
    paths = {
        "scripts/wrapper.py": "from quant.research.actual import run\n",
        "src/quant/research/actual.py": "def run(): return 1\n",
    }
    for relative, contents in paths.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
    definition = _operation("test", ("product.test",), cache=CachePolicy(
        CacheMode.EXACT_DATE, "v1", ("output",), contract_paths=("scripts/wrapper.py",),
        track_python_imports=True, optional_contract_paths=("shadow.joblib",),
        environment_keys=("TEST_FACTOR_SCHEMA",),
    ))
    context = OperationContext("2026-09-04", "demo", 1, {})
    store = CheckpointStore(tmp_path, tmp_path / "checkpoints")
    first, payload = store.build_identity(definition, context)
    assert "src/quant/research/actual.py" in payload["contract_hashes"]
    (tmp_path / "src/quant/research/actual.py").write_text("def run(): return 2\n")
    second = store.build_identity(definition, context)[0]
    assert first != second
    (tmp_path / "shadow.joblib").write_bytes(b"new-optional-model")
    third = store.build_identity(definition, context)[0]
    assert third != second
    monkeypatch.setenv("TEST_FACTOR_SCHEMA", "schema-v2")
    assert store.build_identity(definition, context)[0] != third


def test_core_snapshots_include_historical_partition_corrections_and_source_namespace(tmp_path):
    for relative in ("data/raw/daily_partitioned/old.parquet", "data/raw/daily_partitioned/current.parquet",
                     "data/raw/daily_basic/old.parquet", "data/raw/top_list/old.parquet"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"1")
    canonical = snapshot_from_revision(source_namespace="canonical-db-epoch-1", dataset="daily", revision=3)
    assert canonical != snapshot_from_revision(source_namespace="other-db", dataset="daily", revision=3)
    ids = ("data.market_daily", "source.market_daily_parquet", "data.daily_basic", "data.top_list")
    first = core_input_snapshots(tmp_path, ids, canonical_market=canonical)
    (tmp_path / "data/raw/daily_partitioned/old.parquet").write_bytes(b"2")
    second = core_input_snapshots(tmp_path, ids, canonical_market=canonical)
    assert first["source.market_daily_parquet"].fingerprint != second["source.market_daily_parquet"].fingerprint
    assert first["data.market_daily"] == second["data.market_daily"] == canonical
    assert first["data.daily_basic"] == second["data.daily_basic"]
    with pytest.raises(ValueError, match="canonical snapshot"):
        core_input_snapshots(tmp_path, ("data.market_daily",))
    with pytest.raises(ValueError, match="valid revision"):
        snapshot_from_revision(source_namespace="db", dataset="daily", revision=None)
    with pytest.raises(ValueError, match="materialized"):
        snapshot_from_artifacts(tmp_path, ())


def test_production_stage_requires_pin_and_excludes_selected_outputs(tmp_path):
    from quant.routine.production_dag import pinned_market_inputs, production_stage_inputs, require_pinned_market

    basic = tmp_path / "data/raw/daily_basic"
    basic.mkdir(parents=True)
    (basic / "history.parquet").write_bytes(b"history")
    daily = tmp_path / "data/raw/daily"
    daily.mkdir()
    (daily / "history.parquet").write_bytes(b"prices")
    nodes = ("feature.strategy_signals", "feature.project_daily")
    with pytest.raises(ValueError, match="pinned_market_inputs"):
        production_stage_inputs(tmp_path, nodes)
    canonical = snapshot_from_revision(source_namespace="fixture-db", dataset="daily", revision=1)
    with pinned_market_inputs(canonical):
        snapshots = production_stage_inputs(tmp_path, nodes)
        assert snapshots["data.market_daily"] == canonical
        assert set(snapshots) == {"data.market_daily", "source.market_daily_parquet", "data.daily_basic"}
        signal = InputSnapshot("verified-signal", changes=NodeChanges(keys=("A",), partitions=("20260101",)))
        next_stage = production_stage_inputs(
            tmp_path, ("feature.project_daily",),
            completed_snapshots={"feature.strategy_signals": signal},
        )
        assert next_stage["feature.strategy_signals"] is signal
        with pytest.raises(ValueError, match="unaudited"):
            production_stage_inputs(tmp_path, ("data.market_daily",))
    with pytest.raises(ValueError, match="pinned_market_inputs"):
        require_pinned_market()
