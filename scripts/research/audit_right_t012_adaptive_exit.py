"""Replay frozen adaptive actions, stock selection and every cash scenario."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from audit_overnight_right_provenance import sha256
from run_right_t012_adaptive_exit import load_group, with_action
from run_right_t012_profit_grid import cashbook, portfolio_metrics
from train_right_t012_payoff_v3 import OUTPUT, ROOT, TARGETS, load_protocol, payoff_selection


def main() -> None:
    parent = load_protocol()
    folder = ROOT / "right_t012_adaptive_exit_v1"
    protocol = json.loads((folder / "protocol.json").read_text())
    complete = json.loads((folder / "complete.json").read_text())
    for path, digest in protocol["sources"].items():
        assert sha256(Path(path)) == digest
    assert sha256(folder / "picks.parquet") == complete["picks_sha256"]
    csv_args = dict(dtype={"fold": str}, float_precision="round_trip")
    grid = pd.read_csv(folder / "selection_grid.csv", **csv_args)
    stats = pd.read_csv(folder / "summary.csv", **csv_args)
    curves = pd.read_csv(folder / "daily_curves.csv", parse_dates=["date"], **csv_args)
    picks = pd.read_parquet(folder / "picks.parquet")
    assert len(grid) == 7776
    calendar = pd.DatetimeIndex(pd.read_parquet(OUTPUT / "calendar.parquet").date)
    stages = 0
    for lock in json.loads((folder / "locked_rules.json").read_text()):
        fold, arm, rule = lock["fold"], lock["arm"], lock["rule"]
        eligible = grid.loc[grid.fold.eq(fold) & grid.arm.eq(arm) & grid.dates.ge(20) & grid.unknown.eq(0)]
        if rule is None:
            assert eligible.empty
        else:
            best = eligible.sort_values(["score", "net_return_scenario", "window", "pool", "entry", "top_k", "penalty", "min_mu"],
                ascending=[False, False, True, True, True, True, True, True], kind="stable").iloc[0]
            assert all(best[k] == v for k, v in rule.items())
            assert lock["qualified"] == bool(best.score > 0 and best.first_half_pnl > 0 and best.second_half_pnl > 0)
        frame = load_group(parent, fold, arm, 126 if rule is None else rule["window"])
        boundaries = parent["folds"][fold]
        for stage, start, end in [("selection", boundaries[1], boundaries[2]), ("diagnostic", boundaries[2], boundaries[3])]:
            subset = frame.loc[frame.stage.eq(stage)].reset_index(drop=True)
            action = with_action(subset, 0. if rule is None else rule["penalty"])
            indices = np.array([], dtype=int) if rule is None else payoff_selection(action, rule)
            chosen = action.iloc[indices].reset_index(drop=True)
            actual = picks.loc[picks.fold.eq(fold) & picks.arm.eq(arm) & picks.stage.eq(stage)].reset_index(drop=True)
            pd.testing.assert_frame_equal(chosen[["symbol", "date", "selected_exit"]], actual[["symbol", "date", "selected_exit"]])
            assert np.allclose(chosen.gross, actual.gross, equal_nan=True)
            assert actual.loc[actual.label_status.ne("ok"), "gross"].isna().all()
            if len(actual):
                a = actual[[f"gain_{t}" for t in TARGETS]].to_numpy()-(1+rule["penalty"])*actual[[f"loss_{t}" for t in TARGETS]].to_numpy()
                assert np.array_equal(np.asarray(TARGETS)[a.argmax(axis=1)], actual.selected_exit)
            assert actual.screen_date.lt(actual.date).all() and actual.date.lt(actual.planned_exit_date).all()
            assert actual.planned_exit_date.lt(end).all()
            days = calendar[(calendar >= start) & (calendar < end)]
            k = 1 if rule is None else rule["top_k"]
            selected_stats = stats.loc[stats.fold.eq(fold) & stats.arm.eq(arm) & stats.stage.eq(stage)]
            for row in selected_stats.itertuples():
                metrics, daily = portfolio_metrics(actual.gross.to_numpy(), days.get_indexer(actual.planned_exit_date), len(days), k, row.cost_bps, row.unknown_scenario)
                book, nav = cashbook(actual, days, k, row.cost_bps, row.unknown_scenario)
                assert all(np.isclose(v, getattr(row, key), equal_nan=True) for key, v in (metrics | book).items())
                saved = curves.loc[curves.fold.eq(fold) & curves.arm.eq(arm) & curves.stage.eq(stage) & curves.cost_bps.eq(row.cost_bps) & curves.unknown_scenario.eq(row.unknown_scenario)]
                assert np.allclose(saved.pnl, daily) and np.allclose(saved.cashbook_nav, nav)
            stages += 1
    (folder / "audit_complete.json").write_text(json.dumps(dict(stage_replays=stages, metric_scenarios=len(stats),
        picks_sha256=complete["picks_sha256"], scope="Frozen grid winner, predicted action, source label admission and cashbook replay"), indent=2))
    print(f"Audited {stages} adaptive stage selections and {len(stats)} scenarios")


if __name__ == "__main__":
    main()
