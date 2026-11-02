"""Replay locked selections and summarize losses without selecting new rules."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from audit_overnight_right_provenance import sha256
from train_right_t012 import choose, summarize

from quant.core.paths import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    args = parser.parse_args()
    root = PROJECT_ROOT / "reports/research"
    source = root / args.directory
    completed = json.loads((source / "complete.json").read_text())
    assert sha256(source / "picks.parquet") == completed["picks_sha256"]
    protocol = json.loads((source / "protocol.json").read_text())
    picks = pd.read_parquet(source / "picks.parquet")
    grid = pd.read_csv(source / "selection_grid.csv")
    summary = pd.read_csv(source / "summary.csv")
    calendar = pd.DatetimeIndex(pd.read_parquet(root / "overnight_unfiltered_labels_v1/labels.parquet", columns=["date"]).date.unique()).sort_values()
    replayed = 0
    for (fold, arm, target, policy, stage), stats in summary.groupby(["fold", "arm", "target", "policy", "stage"]):
        available = grid.loc[grid.fold.eq(fold) & grid.arm.eq(arm) & grid.target.eq(target) & grid.dates.ge(20)]
        if policy == "high_precision":
            available = available.loc[available.event_hit_rate.ge(.8) & available.unknown.eq(0) & available.daily_score.gt(0)]
        sort = ["daily_score", "threshold", "maximum_risk", "rank"]
        asc = [False, True, True, True]
        if policy == "high_precision":
            sort, asc = ["dates", *sort], [False, *asc]
        prediction = pd.read_parquet(source / f"{fold}_{arm}_{target}_predictions.parquet")
        prediction = prediction.loc[prediction.stage.eq(stage)]
        if available.empty:
            expected = prediction.iloc[:0]
        else:
            rule = available.sort_values(sort, ascending=asc).iloc[0]
            expected = choose(prediction.loc[prediction.risk.le(rule.maximum_risk)], rule.threshold, rule["rank"])
        actual = picks.loc[picks.fold.eq(fold) & picks.arm.eq(arm) & picks.target.eq(target) & picks.policy.eq(policy) & picks.stage.eq(stage)]
        def keys(rows: pd.DataFrame) -> set:
            return set(map(tuple, rows[["symbol", "date"]].to_numpy()))
        assert keys(actual) == keys(expected)
        assert not actual.date.duplicated().any()
        boundaries = protocol["folds"][fold]
        start, end = boundaries[1:3] if stage == "selection" else boundaries[2:4]
        days = calendar[(calendar >= start) & (calendar < end)][:-1]
        for row in stats.itertuples():
            rebuilt = summarize(actual.assign(net=actual.net - (row.cost_bps - 20) / 10000), days)
            for name, value in rebuilt.items():
                assert np.isclose(value, getattr(row, name), equal_nan=True), (fold, arm, target, name)
        replayed += 1
    diag = picks.loc[picks.stage.eq("diagnostic")].copy()
    diag["month"] = diag.date.dt.strftime("%Y-%m")
    groups = ["fold", "arm", "target", "policy"]
    monthly = diag.groupby([*groups, "month"]).agg(trades=("net", "size"), known=("net", "count"), mean_net20=("net", "mean"), worst=("net", "min"), event_hits=("event_hit", "sum"))
    monthly.to_csv(source / "monthly_diagnostics.csv")
    tails = []
    for key, rows in diag.groupby(groups):
        known = rows.net.dropna()
        loss, win = known[known <= 0], known[known > 0]
        tails.append(dict(zip(groups, key, strict=True)) | dict(trades=len(rows), unknown=int(rows.net.isna().sum()), mean_win=float(win.mean()), mean_loss=float(loss.mean()), without_best3_mean20=float(known.sort_values().iloc[:-3].mean()), worst5_sum=float(known.nsmallest(5).sum())))
    pd.DataFrame(tails).to_csv(source / "tail_diagnostics.csv", index=False)
    diag.sort_values("net").groupby(groups).head(10).to_csv(source / "worst_trades.csv", index=False)
    reliability = []
    for path in sorted(source.glob("*_predictions.parquet")):
        rows = pd.read_parquet(path)
        rows = rows.loc[rows.stage.eq("diagnostic")].copy()
        rows["probability_bin"] = pd.cut(rows.probability, [-.001, .5, .6, .7, .8, .9, 1.001])
        for key, block in rows.groupby(["fold", "arm", "target", "probability_bin"], observed=True):
            reliability.append(dict(zip(["fold", "arm", "target", "probability_bin"], key, strict=True)) | dict(rows=len(block), dates=block.date.nunique(), unknown=int(block.net.isna().sum()), forecast=float(block.probability.mean()), actual=float(block.event_hit.fillna(False).mean()), mean_net20=float(block.net.mean()), forecast_loss2=float(block.risk.mean()), actual_loss2=float(block.net.lt(-.02).mean())))
    pd.DataFrame(reliability).to_csv(source / "probability_reliability.csv", index=False)
    strata = []
    diag["entry_move"] = pd.cut(diag.screen_to_entry_return, [-np.inf, -.03, 0, .03, .07, np.inf])
    diag["signal_age"] = pd.cut(diag.episode_age, [0, 1, 3, 10, np.inf])
    diag["board"] = np.select([diag.symbol.str.endswith(".BJ"), diag.symbol.str.startswith(("300", "301")), diag.symbol.str.startswith("688")], ["BSE", "ChiNext", "STAR"], default="main")
    for dimension in ["entry_move", "signal_age", "board"]:
        for key, rows in diag.groupby([*groups, dimension], observed=True):
            strata.append(dict(zip([*groups, "bucket"], key, strict=True)) | dict(dimension=dimension, trades=len(rows), unknown=int(rows.net.isna().sum()), mean_net20=float(rows.net.mean()), event_hit_rate=float(rows.event_hit.fillna(False).mean()), worst=float(rows.net.min())))
    pd.DataFrame(strata).to_csv(source / "badcase_strata.csv", index=False)
    (source / "audit_complete.json").write_text(json.dumps(dict(replayed_selections=replayed, picks_sha256=completed["picks_sha256"], scope="Exact locked-rule and metric replay; monthly and tail diagnostics only, no new rule selected"), indent=2))
    print(f"Audited {replayed} locked selections: {source}")


if __name__ == "__main__":
    main()
