import json

import pandas as pd
import pytest

from quant.strategies.custom.chan_model import CHAN_MODEL_LOGIC_VERSION
from quant.webapp import services
from scripts.research.build_chan_model_strategy import build_strategy_outputs
from scripts.research.refresh_chan_model_live_scores import _write_strategy_outputs


@pytest.fixture
def scored_paths(monkeypatch, tmp_path):
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-01"] * 5 + ["2026-01-02"]),
        "symbol": [f"00000{i}.SZ" for i in range(6)],
        "name": ["测试"] * 6,
        "chan_signal_name": ["三买确认"] * 6,
        "chan_score": [100.] * 6,
        "entry_gap_pct": [1.] * 5 + [float("nan")],
        "pred_target_good": [.1, .2, .3, .4, .5, .95],
        "pred_target_big10": [.1, .2, .3, .4, .5, .8],
        "pred_target_win10": [.1, .2, .3, .4, .5, .8],
        "hold_10d_close": [1., 2., -1., 4., 5., float("nan")],
        "split": ["train"] * 5 + ["live"],
    })
    source = tmp_path / "scores.parquet"
    frame.to_parquet(source)
    output = tmp_path / "strategy"
    output.mkdir()
    monkeypatch.setattr(services, "CHAN_MODEL_SCORED_PATH", source)
    monkeypatch.setattr(services, "CHAN_MODEL_STRATEGY_DIR", output)
    monkeypatch.setattr(services, "CHAN_MODEL_STRATEGY_SCORED_PATH", output / "scored.parquet")
    monkeypatch.setattr(services, "CHAN_MODEL_LATEST_CANDIDATES_PATH", output / "latest.csv")
    monkeypatch.setattr(services, "CHAN_MODEL_SUMMARY_PATH", output / "summary.csv")
    return frame, source, output


def test_payload_exposes_pending_open_candidate(scored_paths):
    payload = services._build_chan_model_strategy_payload()
    assert payload["logic_version"] == CHAN_MODEL_LOGIC_VERSION
    assert payload["signal_date"] == "2026-01-02"
    assert payload["pending_open_count"] == payload["candidate_count"] == 1
    assert payload["executable_count"] == 0
    row = payload["candidates"][0]
    assert row["execution_status"] == "pending_open"
    assert row["execution_label"] == "待开盘核验"
    assert row["entry_gap_pct"] is None


def test_successful_processed_day_with_no_raw_signals_remains_empty(scored_paths):
    frame, source, output = scored_paths
    source.with_name("live_refresh_manifest.json").write_text(json.dumps({
        "status": "success", "processed_through": "2026-01-05",
    }))
    payload = services._build_chan_model_strategy_payload()
    assert payload["signal_date"] == "2026-01-05"
    assert payload["candidates"] == []
    manifest = build_strategy_outputs(source, output, None, 20)
    assert manifest["trade_date"] == "2026-01-05"
    assert manifest["candidate_rows"] == 0
    meta = _write_strategy_outputs(frame, output, 20, "2026-01-05")
    assert meta["latest_signal_date"] == "2026-01-05"
    assert meta["latest_candidates"] == 0


def test_explicit_history_overrides_latest_processed_day(scored_paths):
    _, source, _ = scored_paths
    source.with_name("live_refresh_manifest.json").write_text(json.dumps({
        "status": "success", "processed_through": "2026-01-05",
    }))
    payload = services._build_chan_model_strategy_payload(signal_date="2026-01-01")
    assert payload["signal_date"] == "2026-01-01"
    assert payload["candidate_count"] > 0


def test_obsolete_cached_payload_is_rebuilt(monkeypatch, scored_paths):
    monkeypatch.setattr(services, "_read_workspace_snapshot", lambda *a, **k: {"candidates": []})
    monkeypatch.setattr(services, "_write_workspace_snapshot", lambda *a, **k: None)
    payload = services.get_chan_model_strategy_plan()
    assert payload["logic_version"] == CHAN_MODEL_LOGIC_VERSION
    assert payload["candidate_count"] == 1


def test_old_committed_snapshot_upgrades_without_writing(monkeypatch, scored_paths):
    from types import SimpleNamespace

    monkeypatch.setattr(services, "_read_workspace_snapshot", lambda *a, **k: {"candidates": []})
    monkeypatch.setattr(services, "current_publication", lambda: SimpleNamespace(writable=False))

    def forbidden(*args, **kwargs):
        raise AssertionError("read-only upgrade must not write published artifacts")

    for name in ("atomic_write_csv", "atomic_write_parquet", "_write_workspace_snapshot"):
        monkeypatch.setattr(services, name, forbidden)
    payload = services.get_chan_model_strategy_plan(signal_date="2026-01-01")
    assert payload["signal_date"] == "2026-01-01"
    assert payload["logic_version"] == CHAN_MODEL_LOGIC_VERSION
    assert payload["candidate_count"] > 0
