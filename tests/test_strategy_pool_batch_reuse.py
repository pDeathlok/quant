from copy import deepcopy

import numpy as np
import pytest

from quant.webapp import services


class Model:
    def predict(self, frame):
        return (frame["base"] + frame["matched_count"] * 2
                + frame["group__B1"] * 5 + frame["group__B2"] * 13).to_numpy()


@pytest.fixture
def scenario(monkeypatch):
    date = "2026-09-08"
    signals = [
        {"strategy_key": key, "strategy_family": key, "strategy_group": key,
         "strategy_name": key, "metrics": {"profit_factor": 2.0, "avg_return_pct": 3.0}}
        for key in ("B1", "B2")
    ]
    payload = {
        "signal_date": date, "generated_at": date + "T18:00:00",
        "available_strategies": [{"key": key} for key in ("B1", "B2", "EMPTY")],
        "stocks": [{"symbol": symbol, "date": date, "market_data_date": date,
                    "signals": deepcopy(signals), "selector_score": 99.0,
                    "score_date": date, "model_score_available": True,
                    "feature_quality": {"status": "complete", "date": date}}
                   for symbol in ("000001.SZ", "000002.SZ")],
    }
    source = {row["symbol"]: {"base": index + 1, "date": date,
                             "_score_feature_source": "model_history", "_score_feature_date": date}
              for index, row in enumerate(payload["stocks"])}
    calls = []

    def load(rows):
        calls.append(tuple(row["symbol"] for row in rows))
        return deepcopy(source)

    artifact = {"features": ["base", "matched_count", "group__B1", "group__B2"],
                "model": Model(), "score_reference": np.arange(30)}
    monkeypatch.setattr(services, "_selector_feature_rows_for_score_rows", load)
    monkeypatch.setattr(services, "_selector_buy_hold_models", lambda: {"buy": artifact, "hold": artifact})
    monkeypatch.setattr(services, "_fill_stock_profile", lambda stock, *args: stock)
    monkeypatch.setattr(services, "_calibrated_signal_score", lambda *a, **k: 1.0)
    monkeypatch.setattr(services, "_signal_raw_score", lambda *a, **k: 1.0)
    monkeypatch.setattr(services, "_score_mode_resonance_weight", lambda *a: 0.0)
    monkeypatch.setattr(services, "apply_selector_ranking_source", lambda rows, *a, **k: rows)
    return payload, source, calls


def stable(payload):
    return {key: value for key, value in payload.items() if key != "generated_at"}


def test_shared_features_preserve_per_strategy_results_and_input(scenario):
    payload, source, calls = scenario
    original = deepcopy(payload)
    original_source = deepcopy(source)
    expected = {key: services._filtered_selector_payload(payload, [key])
                for key in services._strategy_keys_from_payload(payload)}
    assert len(calls) == 2  # Empty strategies never request features.
    calls.clear()
    metrics = {}
    snapshots, written = services._build_strategy_pool_snapshots(payload, True, metrics=metrics)
    assert calls == [("000001.SZ", "000002.SZ")]
    assert written == {"ALL": 2, "B1": 2, "B2": 2, "EMPTY": 0}
    for actual, strategies, _ in snapshots[1:]:
        assert stable(actual) == stable(expected[strategies[0]])
    assert snapshots[1][0]["stocks"][0]["historical_buy_score"] != snapshots[2][0]["stocks"][0]["historical_buy_score"]
    assert payload == original and source == original_source
    assert metrics["shared_feature_count"] == 2
    assert set(metrics["strategy_seconds"]) == {"B1", "B2", "EMPTY"}


def test_next_batch_reloads_corrected_same_day_features(scenario):
    payload, source, calls = scenario
    first, _ = services._build_strategy_pool_snapshots(payload, True, metrics={})
    source["000001.SZ"]["base"] = 20
    second, _ = services._build_strategy_pool_snapshots(payload, True, metrics={})
    assert len(calls) == 2
    first_scores = {row["symbol"]: row["historical_buy_score"] for row in first[1][0]["stocks"]}
    second_scores = {row["symbol"]: row["historical_buy_score"] for row in second[1][0]["stocks"]}
    assert first_scores["000001.SZ"] != second_scores["000001.SZ"]


@pytest.mark.parametrize("invalid", ["missing", "old_date", "source_error", "null"])
def test_invalid_shared_feature_never_publishes(scenario, monkeypatch, invalid):
    payload, source, _ = scenario
    row = source["000001.SZ"]
    if invalid == "missing":
        del source["000001.SZ"]
    elif invalid == "old_date":
        row["_score_feature_date"] = "2026-09-07"
    elif invalid == "source_error":
        row["_score_feature_error"] = "incomplete inputs"
    else:
        row["base"] = np.nan
    monkeypatch.setattr(services, "MarketDataStore", lambda *a, **k: pytest.fail("No write before validation"))
    with pytest.raises(RuntimeError, match="incomplete model scores"):
        services._write_strategy_pool_snapshots(payload, True)


def test_stale_candidate_rejected_before_shared_reads(scenario):
    payload, _, calls = scenario
    payload["stocks"][0]["date"] = "2026-09-07"
    with pytest.raises(RuntimeError):
        services._build_strategy_pool_snapshots(payload, True, metrics={})
    assert calls == []


def test_empty_batch_does_not_load_features(scenario):
    payload, _, calls = scenario
    payload["stocks"] = []
    _, written = services._build_strategy_pool_snapshots(payload, True, metrics={})
    assert calls == []
    assert written == {"ALL": 0, "B1": 0, "B2": 0, "EMPTY": 0}


def test_writer_preserves_count_contract_and_reports_timings(scenario, monkeypatch):
    payload, _, _ = scenario
    writes = []
    monkeypatch.setattr(services, "_write_selector_snapshot_batch", lambda batch: writes.append(batch))
    metrics = {}
    result = services._write_strategy_pool_snapshots(payload, True, metrics=metrics)
    assert result == {"ALL": 2, "B1": 2, "B2": 2, "EMPTY": 0}
    assert len(writes) == 1 and len(writes[0]) == 4
    assert metrics["total_seconds"] >= metrics["batch_write_seconds"] >= 0


def test_measurements_do_not_invalidate_dependency_content_identity():
    from quant.routine.daily_dependency_runtime import _semantic_result_payload

    content = {"snapshot": {"strategy_pools": {"ALL": 2}}}
    measured = deepcopy(content)
    measured["snapshot"]["timings"] = {"total_seconds": 1.5}
    measured["finalization_timings"] = {"postflight_seconds": 2.5}
    measured["capture_metrics"] = {"canonical_export_seconds": 3.5}
    assert _semantic_result_payload(measured) == content


def test_all_pools_share_one_ranking_batch_with_injected_context(scenario, monkeypatch):
    payload, _, _ = scenario
    batches = []

    def rank(rows, date, *, ranking_batch):
        batches.append(ranking_batch)
        assert ranking_batch._context_provider is services.current_publication
        return rows

    monkeypatch.setattr(services, "apply_selector_ranking_source", rank)
    services._build_strategy_pool_snapshots(payload, True, metrics={})
    assert len(batches) == 3
    assert all(batch is batches[0] for batch in batches)
