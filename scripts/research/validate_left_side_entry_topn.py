#!/usr/bin/env python
"""Validate the rank-only B1 entry threshold on frozen B/C predictions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quant.data.atomic_io import atomic_write_csv, atomic_write_json


PREDICTIONS = (
    PROJECT_ROOT
    / "reports/research/left_side_unified_v3_group4_input_parity/test_predictions.parquet"
)
LABELS = (
    PROJECT_ROOT
    / "data/research/left_side_unified_v3_group4_input_parity/labels.parquet"
)
REPORT_ROOT = (
    PROJECT_ROOT / "reports/research/left_side_unified_v3_group4_input_parity"
)
TOP_N_CANDIDATES = (5, 10, 20, 30, 50)
FROZEN_PRODUCTION_TOP_N = 20
ROUND_TRIP_COST = 0.002


def main() -> None:
    predictions = pd.read_parquet(PREDICTIONS)
    predictions["date"] = pd.to_datetime(predictions["date"], errors="raise")
    labels = pd.read_parquet(
        LABELS,
        columns=(
            "symbol",
            "date",
            "entry_mode",
            "horizon",
            "terminal_return",
            "mature",
            "locked_limit_up",
        ),
    )
    labels["date"] = pd.to_datetime(labels["date"], errors="raise")
    labels = labels[
        labels["entry_mode"].eq("next_close")
        & labels["horizon"].eq(5)
        & labels["mature"]
        & ~labels["locked_limit_up"]
    ]
    frame = predictions.merge(
        labels[["symbol", "date", "terminal_return"]],
        on=["symbol", "date"],
        how="left",
        validate="many_to_one",
    )
    if frame["terminal_return"].isna().any():
        raise RuntimeError("left-side Top-N validation has incomplete returns")

    rows: list[dict[str, object]] = []
    score_columns = {
        "independent_left_members": "pred_independent_left_members",
        "unified_left_long_task_deep": "pred_unified_left_long_task_deep",
    }
    for fold in ("B", "C"):
        fold_frame = frame[frame["fold"].eq(fold)]
        for top_n in TOP_N_CANDIDATES:
            metrics: dict[str, dict[str, float | int]] = {}
            for model, score_column in score_columns.items():
                selected = (
                    fold_frame.sort_values(
                        ["date", score_column, "symbol"],
                        ascending=[True, False, True],
                        kind="stable",
                    )
                    .groupby("date", sort=False)
                    .head(top_n)
                )
                net = selected["terminal_return"] - ROUND_TRIP_COST
                metrics[model] = {
                    "trades": len(selected),
                    "average_net_return": float(net.mean()),
                    "win_rate": float(net.gt(0.0).mean()),
                }
            baseline = metrics["independent_left_members"]
            candidate = metrics["unified_left_long_task_deep"]
            rows.append(
                {
                    "fold": fold,
                    "top_n": top_n,
                    "baseline_average_net_return": baseline["average_net_return"],
                    "candidate_average_net_return": candidate["average_net_return"],
                    "delta_average_net_return": (
                        candidate["average_net_return"]
                        - baseline["average_net_return"]
                    ),
                    "baseline_win_rate": baseline["win_rate"],
                    "candidate_win_rate": candidate["win_rate"],
                    "trades": candidate["trades"],
                }
            )
    result = pd.DataFrame(rows)
    selected = result[result["top_n"].eq(FROZEN_PRODUCTION_TOP_N)]
    passed = (
        selected["fold"].tolist() == ["B", "C"]
        and selected["delta_average_net_return"].ge(0.0).all()
    )
    report = {
        "status": "success",
        "schema_version": "left-side-rank-entry-topn-validation-v1",
        "entry_mode": "next_close",
        "horizon": 5,
        "round_trip_cost_bps": int(ROUND_TRIP_COST * 10_000),
        "candidate_top_n_values": list(TOP_N_CANDIDATES),
        "production_top_n": FROZEN_PRODUCTION_TOP_N,
        "production_threshold_mode": "none_rank_only",
        "normalization": "daily_cross_section_percentile_v1",
        "gate": "candidate_average_net_return_not_lower_than_independent_on_B_and_C",
        "passed": bool(passed),
        "selected_fold_metrics": selected.to_dict(orient="records"),
        "locked_limit_up_excluded": True,
    }
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(result, REPORT_ROOT / "entry_threshold_topn_metrics.csv", index=False)
    atomic_write_json(report, REPORT_ROOT / "entry_threshold_topn_validation.json")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
