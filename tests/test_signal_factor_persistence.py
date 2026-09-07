from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine

from quant.data.market_data_store import MarketDataStore, MarketDataStoreConfig
from quant.data.market_snapshot import export_market_snapshot, pinned_market_snapshot
from quant.features import daily_factor_layer as layer
from quant.routine.operation_adapters import _strict_python
from quant.routine.operation_contracts import OperationContext
from tests.test_daily_factor_layer import _assert_signal_factors_equal, _daily


SYMBOL = "000001.SZ"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    for key in (
        "DAILY_FACTOR_ROOT", "SIGNAL_FACTOR_ROOT", "MARKET_DATA_SQL_URL",
        "QUANT_PINNED_MARKET_MANIFEST", "QUANT_PINNED_MARKET_FINGERPRINT",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("MARKET_DATA_BACKEND", "parquet")
    layer._signal_calculation_fingerprint.cache_clear()
    yield
    layer._signal_calculation_fingerprint.cache_clear()


@pytest.mark.parametrize("root_env", [None, "DAILY_FACTOR_ROOT", "SIGNAL_FACTOR_ROOT"])
def test_signal_root_is_backward_compatible(tmp_path, monkeypatch, root_env):
    explicit = tmp_path / "explicit"
    expected = explicit
    if root_env:
        expected = tmp_path / root_env
        monkeypatch.setenv(root_env, str(expected))
    if root_env == "SIGNAL_FACTOR_ROOT":
        monkeypatch.setenv("DAILY_FACTOR_ROOT", str(tmp_path / "base"))
    layer.attach_daily_signal_factors(_daily(), SYMBOL, factor_root=explicit)
    assert (layer.signal_factor_symbol_dir(expected, SYMBOL) / "state.json").is_file()
    if root_env:
        assert not explicit.exists()


def test_distinct_sealed_runs_reuse_only_verified_signal_state(tmp_path, monkeypatch):
    persistent = tmp_path / "existing-factors"
    monkeypatch.setenv("DAILY_FACTOR_ROOT", str(persistent))
    daily = _daily()
    # A date-covered but stale base cache must never cross a sealed-run boundary.
    layer.attach_daily_base_factors(daily, SYMBOL)
    for path in layer.factor_symbol_dir(persistent, SYMBOL).glob("*.parquet"):
        cached = pd.read_parquet(path)
        cached["kdj_d_j"] = -999.0
        cached.to_parquet(path, index=False)
    url = f"sqlite:///{tmp_path / 'canonical.sqlite'}"
    engine = create_engine(url)
    store = MarketDataStore(MarketDataStoreConfig(backend="sql", sql_url=url))
    code = """
import json, os
import pandas as pd
from quant.data.market_data_store import MarketDataStore
from quant.features import daily_factor_layer as layer
daily = MarketDataStore().read_frame('daily', '000001.SZ')
actual = layer.attach_daily_signal_factors(daily, '000001.SZ')
expected = layer.attach_daily_signal_factors(daily, '000001.SZ', persist_missing=False)
pd.testing.assert_frame_equal(actual[layer.SIGNAL_FACTOR_COLUMNS],
    expected[layer.SIGNAL_FACTOR_COLUMNS], check_dtype=False, rtol=1e-12, atol=1e-12)
base = layer.attach_daily_base_factors(daily, '000001.SZ')
fresh = layer.attach_daily_base_factors(daily, '000001.SZ', persist_missing=False)
pd.testing.assert_series_equal(base['kdj_d_j'], fresh['kdj_d_j'])
print(json.dumps({'status': 'success', 'end': pd.Timestamp(daily['date'].max()).strftime('%Y%m%d'),
    'mode': actual.attrs['signal_factor_cache_mode'],
    'signal_root': os.environ['SIGNAL_FACTOR_ROOT'],
    'base_root': os.environ['DAILY_FACTOR_ROOT']}))
"""
    revised = daily.copy()
    revised.loc[30, ["vol", "volume"]] *= 2
    results = []
    try:
        for number, frame in enumerate((daily.iloc[:-1], daily, revised), start=1):
            with engine.begin() as connection:
                frame.to_sql("market_daily", connection, if_exists="replace", index=False)
            snapshot = export_market_snapshot(store, tmp_path / f"run-{number}" / "sealed")
            context = OperationContext(
                target_trade_date=frame["date"].max().strftime("%Y%m%d"), scope="all",
                granted_workers=1, upstream_results={}, identity_required=True,
                project_root=PROJECT_ROOT,
            )
            with pinned_market_snapshot(snapshot.manifest_path):
                result = _strict_python(context, ["-c", code], ("probe",), date_field="end")
            assert result.status == "success", result
            results.append(result.node_results["probe"])
    finally:
        engine.dispose()
    assert [item["mode"] for item in results] == ["bootstrap", "incremental", "invalidated_rebuild"]
    assert {item["signal_root"] for item in results} == {str(persistent)}
    assert len({item["base_root"] for item in results}) == 3
    for item in results:
        root = Path(item["base_root"])
        assert layer.factor_symbol_dir(root, SYMBOL).is_dir()
        assert not (root / layer.SIGNAL_FACTOR_LAYER_VERSION).exists()


def test_formula_change_invalidates_state_even_after_legacy_full_run(tmp_path, monkeypatch):
    monkeypatch.setenv("SIGNAL_FACTOR_ROOT", str(tmp_path / "signal"))
    monkeypatch.setenv("DAILY_FACTOR_ROOT", str(tmp_path / "base"))
    daily = _daily()
    layer.attach_daily_signal_factors(daily, SYMBOL)
    state_path = layer.signal_factor_symbol_dir(tmp_path / "signal", SYMBOL) / "state.json"
    old = json.loads(state_path.read_text())
    original = layer.calc_bbi

    def changed_bbi(*args, **kwargs):
        return original(*args, **kwargs) + 1.0

    changed_bbi.__module__ = layer.__name__
    monkeypatch.setattr(layer, "calc_bbi", changed_bbi)
    layer._signal_calculation_fingerprint.cache_clear()
    # The strict legacy full-run path bypasses signal state, leaving it unchanged.
    layer.attach_daily_base_factors(daily, SYMBOL, persist_missing=False)
    assert json.loads(state_path.read_text()) == old
    actual = layer.attach_daily_signal_factors(daily, SYMBOL)
    assert actual.attrs["signal_factor_cache_mode"] == "invalidated_rebuild"
    _assert_signal_factors_equal(
        actual, layer.attach_daily_signal_factors(daily, SYMBOL, persist_missing=False),
    )
    current = json.loads(state_path.read_text())
    assert current["schema_version"] == old["schema_version"]
    assert current["factor_version"] == old["factor_version"]
    assert current["calculation_fingerprint"] != old["calculation_fingerprint"]
    assert layer.attach_daily_signal_factors(daily, SYMBOL).attrs["signal_factor_cache_mode"] == "cache_hit"


def test_fingerprint_ignores_roots_and_unrelated_code_and_skips_third_party(monkeypatch):
    original = layer.inspect.getsource
    inspected = []

    def local_source(value):
        assert value.__module__.startswith("quant.")
        inspected.append(value)
        return original(value)

    monkeypatch.setattr(layer.inspect, "getsource", local_source)
    before = layer._signal_calculation_fingerprint()
    assert layer.calc_bbi in inspected
    assert layer.build_continuous_ohlc in inspected
    assert layer._signal_source_hashes in inspected
    monkeypatch.setenv("SIGNAL_FACTOR_ROOT", "/different/root")
    monkeypatch.setenv("QUANT_PINNED_MARKET_MANIFEST", "/different/run/manifest.json")
    monkeypatch.setattr(layer, "calculate_daily_base_factors", lambda *args: None)
    layer._signal_calculation_fingerprint.cache_clear()
    assert layer._signal_calculation_fingerprint() == before


@pytest.mark.parametrize("change", ["missing_fingerprint", "truncated_prefix"])
def test_unverifiable_signal_state_is_rebuilt(tmp_path, monkeypatch, change):
    monkeypatch.setenv("SIGNAL_FACTOR_ROOT", str(tmp_path))
    daily = _daily()
    layer.attach_daily_signal_factors(daily, SYMBOL)
    if change == "missing_fingerprint":
        path = layer.signal_factor_symbol_dir(tmp_path, SYMBOL) / "state.json"
        state = json.loads(path.read_text())
        del state["calculation_fingerprint"]
        path.write_text(json.dumps(state))
    else:
        daily = daily.iloc[10:]
    actual = layer.attach_daily_signal_factors(daily, SYMBOL)
    assert actual.attrs["signal_factor_cache_mode"] == "invalidated_rebuild"
    _assert_signal_factors_equal(
        actual, layer.attach_daily_signal_factors(daily, SYMBOL, persist_missing=False),
    )
