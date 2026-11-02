"""Replay profit grid locks, execution prices, and cash curves from sources."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from audit_overnight_right_provenance import sha256
from run_right_t012_profit_grid import (
    ARMS,
    EXITS,
    cashbook,
    exit_returns,
    portfolio_metrics,
    selected_rows,
)
from train_right_t012 import SIGNALS

from quant.core.paths import PROJECT_ROOT


def main() -> None:
    root = PROJECT_ROOT / "reports/research"
    output = root / "right_t012_profit_grid_v1"
    complete = json.loads((output / "complete.json").read_text())
    protocol = json.loads((output / "protocol.json").read_text())
    for path, digest in protocol["sources"].items():
        assert sha256(Path(path)) == digest, path
    assert sha256(output / "locked_picks.parquet") == complete["picks_sha256"]
    grid = pd.read_csv(output / "selection_grid.csv", dtype={"fold": str}, float_precision="round_trip")
    assert len(grid) == 62208
    picks = pd.read_parquet(output / "locked_picks.parquet")
    summary = pd.read_csv(output / "summary.csv", dtype={"fold": str}, float_precision="round_trip")
    curves = pd.read_csv(output / "daily_curves.csv", parse_dates=["date"], dtype={"fold": str}, float_precision="round_trip")
    columns = ["symbol", "date", "screen_date", "planned_exit_date", "entry_close", "next_open", "next_high", "next_low", "next_close",
        "next_valid_bar", "down_limit", "price_limit_inconsistent", "screen_to_entry_return", *SIGNALS]
    base = pd.read_parquet(root / "right_t012_full_dataset_v4/dataset.parquet", columns=columns)
    signals = pd.read_parquet(root / "right_t012_history_v1/all_daily_signals.parquet", columns=["symbol", "date", *SIGNALS])
    signals["entry_right"] = signals[list(SIGNALS)].eq(True).any(axis=1)
    base = base.merge(signals[["symbol", "date", "entry_right"]], on=["symbol", "date"], how="left", validate="one_to_one")
    calendar = pd.DatetimeIndex(pd.read_parquet(root / "overnight_unfiltered_labels_v1/labels.parquet", columns=["date"]).date.unique()).sort_values()
    count = 0
    for lock in json.loads((output / "locked_rules.json").read_text()):
        fold, arm, rule = lock["fold"], lock["arm"], lock["rule"]
        subgrid = grid.loc[grid.fold.eq(fold) & grid.arm.eq(arm) & grid.dates.ge(20) & grid.unknown.eq(0)]
        if rule is None:
            assert subgrid.empty
        else:
            winner = subgrid.sort_values(["score", "net_return_scenario", "top_k", "exit", "pool", "entry", "rank", "min_expected", "max_risk"],
                ascending=[False, False, True, True, True, True, True, True, True], kind="stable").iloc[0]
            assert all(winner[key] == value for key, value in rule.items())
        frames = {a: pd.read_parquet(root / directory / f"{fold}_{a}_net_close_predictions.parquet",
            columns=["symbol", "date", "stage", "expected_net", "probability", "risk"]) for a, directory in ARMS.items()}
        common = frames["core145"][["symbol", "date"]]
        for f in frames.values():
            common = common.merge(f[["symbol", "date"]], on=["symbol", "date"], validate="one_to_one")
        frame = base.merge(common, on=["symbol", "date"], validate="one_to_one").merge(frames[arm], on=["symbol", "date"], validate="one_to_one")
        boundaries = protocol["folds"][fold]
        for stage, start, end in [("selection", boundaries[1], boundaries[2]), ("diagnostic", boundaries[2], boundaries[3])]:
            subset = frame.loc[frame.stage.eq(stage)].reset_index(drop=True)
            ids = np.array([], dtype=int) if rule is None else selected_rows(subset, rule)
            chosen = subset.iloc[ids]
            actual = picks.loc[picks.fold.eq(fold) & picks.arm.eq(arm) & picks.stage.eq(stage)].reset_index(drop=True)
            pd.testing.assert_frame_equal(chosen[["symbol", "date"]].reset_index(drop=True), actual[["symbol", "date"]])
            assert chosen.screen_date.lt(chosen.date).all() and chosen.date.lt(chosen.planned_exit_date).all()
            days = calendar[(calendar >= start) & (calendar < end)]
            assert chosen.planned_exit_date.lt(end).all()
            k = 1 if rule is None else rule["top_k"]
            gross = np.array([]) if rule is None else exit_returns(chosen)[:, EXITS.index(rule["exit"])]
            assert np.allclose(gross, actual.gross, equal_nan=True)
            if len(actual):
                assert actual.groupby("date").size().max() <= k
                assert np.allclose(actual.weight, .5/k)
            stats = summary.loc[summary.fold.eq(fold) & summary.arm.eq(arm) & summary.stage.eq(stage)]
            for row in stats.itertuples():
                metrics, daily = portfolio_metrics(gross, days.get_indexer(chosen.planned_exit_date), len(days), k, row.cost_bps, row.unknown_scenario)
                book, nav = cashbook(actual, days, k, row.cost_bps, row.unknown_scenario)
                for key, value in (metrics | book).items():
                    assert np.isclose(value, getattr(row, key), equal_nan=True), key
                saved = curves.loc[curves.fold.eq(fold) & curves.arm.eq(arm) & curves.stage.eq(stage) & curves.cost_bps.eq(row.cost_bps) & curves.unknown_scenario.eq(row.unknown_scenario)]
                assert np.allclose(saved.pnl, daily) and np.allclose(saved.cashbook_nav, nav)
            count += 1
    (output / "audit_complete.json").write_text(json.dumps(dict(replayed_stage_selections=count, metric_scenarios=len(summary),
        picks_sha256=complete["picks_sha256"], status="Source and frozen rule replay passed; not independent validation of daily execution assumptions"), indent=2))
    print(f"Audited {count} stage selections and {len(summary)} metric scenarios")


if __name__ == "__main__":
    main()
