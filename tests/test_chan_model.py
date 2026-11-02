import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant.strategies.custom.chan_model import (
    add_chan_model_strategy_columns,
    select_chan_model_candidates,
    summarize_chan_model_strategy,
)


def _scored_frame() -> pd.DataFrame:
    rows = []
    for idx, pred in enumerate([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]):
        rows.append(
            {
                "date": pd.Timestamp("2026-01-01"),
                "symbol": f"00000{idx}",
                "name": f"测试{idx}",
                "chan_signal_name": "三买确认" if idx != 9 else "二买确认",
                "chan_score": 100 if idx >= 4 else 92,
                "entry_gap_pct": 2.0 if idx != 8 else 6.5,
                "pred_target_good": pred,
                "pred_target_big10": pred - 0.05,
                "pred_target_win10": pred - 0.02,
                "hold_10d_close": idx - 3,
                "split": "train",
            }
        )
    rows.append(
        {
            "date": pd.Timestamp("2026-01-02"),
            "symbol": "000010",
            "name": "最新",
            "chan_signal_name": "三买确认",
            "chan_score": 100,
            "entry_gap_pct": 1.0,
            "pred_target_good": 0.96,
            "pred_target_big10": 0.80,
            "pred_target_win10": 0.82,
            "hold_10d_close": 5.0,
            "split": "oot",
        }
    )
    return pd.DataFrame(rows)


def test_add_chan_model_strategy_columns_marks_primary_and_expanded():
    out = add_chan_model_strategy_columns(_scored_frame())

    primary = out[out["chan_model_rule_id"].eq("chan_model_primary")]
    expanded = out[out["chan_model_rule_id"].eq("chan_model_expanded")]

    assert not primary.empty
    assert (primary["chan_signal_name"] == "三买确认").all()
    assert (primary.loc[primary["chan_model_signal"].eq(1), "entry_gap_pct"] <= 3).all()
    assert not expanded.empty


def test_select_chan_model_candidates_uses_latest_signal_date():
    selected = select_chan_model_candidates(_scored_frame(), top_n=5)

    assert len(selected) == 1
    assert selected.loc[0, "date"] == pd.Timestamp("2026-01-02")
    assert selected.loc[0, "chan_model_rule_id"] == "chan_model_primary"


def test_summarize_chan_model_strategy_returns_rule_metrics():
    summary = summarize_chan_model_strategy(_scored_frame())

    assert not summary.empty
    assert {"rule_id", "avg_return_10d", "win_rate_10d"} <= set(summary.columns)


def test_close_candidate_survives_unknown_next_open_without_becoming_executable():
    frame = _scored_frame()
    frame.loc[frame.index[-1], "entry_gap_pct"] = float("nan")
    selected = select_chan_model_candidates(frame)
    assert selected["symbol"].tolist() == ["000010"]
    assert selected.iloc[0]["chan_model_execution_status"] == "pending_open"
    assert selected.iloc[0]["chan_model_signal"] == 0
    assert "oot" not in summarize_chan_model_strategy(frame)["split"].tolist()


def test_open_gap_status_boundaries_and_historical_execution_filter():
    frame = _scored_frame()
    latest = frame.iloc[[-1]].copy()
    cases = [(float("nan"), "pending_open", 0), (3.0, "executable", 1),
             (3.01, "reduce_watch", 0), (6.0, "reduce_watch", 0), (6.01, "skip", 0)]
    for gap, status, executable in cases:
        latest["entry_gap_pct"] = gap
        out = add_chan_model_strategy_columns(pd.concat([frame.iloc[:-1], latest]))
        row = out.iloc[-1]
        assert row["chan_model_setup_signal"] == 1
        assert row["chan_model_execution_status"] == status
        assert row["chan_model_signal"] == executable


def test_latest_empty_date_does_not_fall_back_to_old_candidates():
    frame = _scored_frame()
    frame.loc[frame.index[-1], "pred_target_good"] = 0.01
    assert select_chan_model_candidates(frame).empty
    assert not select_chan_model_candidates(frame, trade_date="2026-01-01").empty
