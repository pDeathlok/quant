"""Replay forward-stacking selections and persist monthly/badcase diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from audit_overnight_right_provenance import sha256
from train_right_t012 import choose

from quant.core.paths import PROJECT_ROOT


def main() -> None:
    source = PROJECT_ROOT / "reports/research/right_t012_sequence_fusion_v1"
    output = PROJECT_ROOT / "reports/research/right_t012_fusion_audit_v1"
    complete = json.loads((source / "complete.json").read_text())
    if sha256(source / "picks.parquet") != complete["picks_sha256"]:
        raise ValueError("Changed picks")
    folds = json.loads((source / "forward_folds.json").read_text())
    for fold in folds:
        if not pd.Timestamp(fold["fit_max_exit"]) < pd.Timestamp(fold["prediction_min_entry"]):
            raise ValueError("Forward prediction boundary violation")
    picks = pd.read_parquet(source / "picks.parquet")
    seq = pd.read_parquet(source / "sequence_variables.parquet")
    ready = seq.seq_return_close.notna()
    sequence_keys = pd.MultiIndex.from_frame(seq.loc[ready, ["symbol", "date"]])
    grid = pd.read_csv(source / "calibration_grid.csv")
    replays, monthly, diagnostics = [], [], []
    for arm in ["table_matched", "sequence_only", "fusion"]:
        for target in ["net_close", "limit1"]:
            pred = pd.read_parquet(source / f"{arm}_{target}_predictions.parquet")
            selected_grid = grid.loc[grid.arm.eq(arm) & grid.target.eq(target) & grid.dates.ge(10)]
            winner = selected_grid.sort_values(["daily_score", "threshold", "rank"], ascending=[False, True, True]).iloc[0]
            eligible_sequence = pd.MultiIndex.from_frame(pred[["symbol", "date"]]).isin(sequence_keys)
            for split in ["calibration", "q4_diagnostic"]:
                subset = pred.loc[pred.split.eq(split) & eligible_sequence]
                replay = choose(subset, float(winner.threshold), str(winner["rank"]))
                actual = picks.loc[picks.arm.eq(arm) & picks.target.eq(target) & picks.split.eq(split)]
                cols = ["symbol", "date", "net", "event_hit"]
                pd.testing.assert_frame_equal(replay[cols].sort_values("date").reset_index(drop=True),
                                              actual[cols].sort_values("date").reset_index(drop=True))
                if actual.date.duplicated().any():
                    raise ValueError("Multiple positions on a date")
                replays.append(dict(arm=arm, target=target, split=split, rows=len(actual), replay="exact"))
                for month, rows in actual.groupby(actual.date.dt.strftime("%Y-%m")):
                    monthly.append(dict(arm=arm, target=target, split=split, month=month, trades=len(rows),
                                        unknown=int(rows.net.isna().sum()), profitable=int(rows.net.gt(0).sum()),
                                        hits=int(rows.event_hit.eq(True).sum()), net20=float(rows.net.mean()),
                                        net40=float(rows.net.mean() - .002)))
                winner_mean = float(actual.loc[actual.net.gt(0), "net"].mean())
                loser_mean = float(actual.loc[actual.net.le(0), "net"].mean())
                diagnostics.append(dict(arm=arm, target=target, split=split, winner_mean=winner_mean,
                                        loser_mean=loser_mean, worst=float(actual.net.min()),
                                        without_best3=float(actual.net.sort_values(ascending=False).iloc[3:].mean()),
                                        breakeven_win=(-loser_mean / (winner_mean - loser_mean)) if np.isfinite(winner_mean - loser_mean) and winner_mean != loser_mean else None))
    output.mkdir(exist_ok=False)
    pd.DataFrame(replays).to_csv(output / "replays.csv", index=False)
    pd.DataFrame(monthly).to_csv(output / "monthly.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(output / "payoff_diagnostics.csv", index=False)
    (output / "complete.json").write_text(json.dumps(dict(replay_groups=len(replays),
        source_picks_sha256=complete["picks_sha256"], forward_folds=folds,
        script_sha256=sha256(Path(__file__)), status="Exact12 selection/return replays; causal fold boundaries checked; not independent performance confirmation"), indent=2))
    print(pd.DataFrame(diagnostics).to_string(index=False))


if __name__ == "__main__":
    main()
