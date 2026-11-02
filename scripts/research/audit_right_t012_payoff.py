"""Audit payoff checkpoints, sampled prediction replay, locks and cash curves."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from audit_overnight_right_provenance import sha256
from run_right_t012_profit_grid import cashbook, portfolio_metrics
from threadpoolctl import threadpool_limits
from train_right_t012_payoff import META, OUTPUT, load_protocol, payoff_selection


def main() -> None:
    protocol = load_protocol()
    complete = json.loads((OUTPUT / "complete.json").read_text())
    training = json.loads((OUTPUT / "training_complete.json").read_text())
    assert len(training["models"]) == 4 * len(protocol["groups"])
    prediction_rows = 0
    reliability = []
    for group in protocol["groups"]:
        frame = pd.read_parquet(group["path"], columns=["symbol", "date", "stage", "planned_exit_date", *group["columns"],
            *[f"gross_{t}" for t in protocol["targets"]]])
        boundaries = protocol["folds"][group["fold"]]
        for target in protocol["targets"]:
            name = f"{group['id']}_{target}"
            check = json.loads((OUTPUT / f"{name}_complete.json").read_text())
            assert pd.Timestamp(check["max_fit_exit"]) < pd.Timestamp(boundaries[0])
            assert pd.Timestamp(check["max_cal_exit"]) < pd.Timestamp(boundaries[1])
            assert sha256(OUTPUT / f"{name}.joblib") == check["model_sha256"]
            assert sha256(OUTPUT / f"{name}_predictions.parquet") == check["predictions_sha256"]
            pred = pd.read_parquet(OUTPUT / f"{name}_predictions.parquet")
            expected_keys = frame.loc[frame.stage.isin(["selection", "diagnostic"]), ["symbol", "date", "stage"]].reset_index(drop=True)
            pd.testing.assert_frame_equal(pred[["symbol", "date", "stage"]].reset_index(drop=True), expected_keys)
            outcomes = frame.loc[frame.stage.isin(["selection", "diagnostic"]), f"gross_{target}"].to_numpy()-.004
            descriptive = pred.assign(actual_net=outcomes, mu=pred.gain-pred.loss)
            descriptive = descriptive.loc[descriptive.stage.eq("diagnostic")].copy()
            descriptive["mu_bin"] = pd.cut(descriptive.mu, [-np.inf, 0, .005, .01, .02, np.inf])
            for bucket, block in descriptive.groupby("mu_bin", observed=True):
                reliability.append(dict(model_id=name, bucket=str(bucket), rows=len(block), dates=block.date.nunique(),
                    unknown=int(block.actual_net.isna().sum()), forecast_mu=float(block.mu.mean()),
                    actual_mean_net40=float(block.actual_net.mean()), forecast_loss=float(block.loss.mean()),
                    actual_loss=float((-block.actual_net).clip(lower=0).mean())))
            sample = pred.iloc[np.linspace(0, len(pred)-1, min(128, len(pred)), dtype=int)]
            joined = sample.merge(frame.drop(columns="stage"), on=["symbol", "date"], validate="one_to_one")
            bundle = joblib.load(OUTPUT / f"{name}.joblib")
            x = joined[bundle["columns"]].astype("float32").replace([np.inf, -np.inf], np.nan)
            with threadpool_limits(limits=2):
                for head in ["gain", "loss"]:
                    reproduced = np.maximum(bundle["models"][head].predict(x), 0) * bundle["scales"][head]
                    assert np.allclose(reproduced, joined[head], rtol=1e-10, atol=1e-12)
            prediction_rows += len(sample)
    assert sha256(OUTPUT / "picks.parquet") == complete["picks_sha256"]
    picks = pd.read_parquet(OUTPUT / "picks.parquet")
    csv_args = dict(dtype={"fold": str}, float_precision="round_trip")
    grid = pd.read_csv(OUTPUT / "selection_grid.csv", **csv_args)
    stats = pd.read_csv(OUTPUT / "summary.csv", **csv_args)
    curves = pd.read_csv(OUTPUT / "daily_curves.csv", parse_dates=["date"], **csv_args)
    assert len(grid) == 31968
    days_all = pd.DatetimeIndex(pd.read_parquet(OUTPUT / "calendar.parquet").date)
    stages = 0
    for lock in json.loads((OUTPUT / "locked_rules.json").read_text()):
        fold, arm, rule = lock["fold"], lock["arm"], lock["rule"]
        eligible = grid.loc[grid.fold.eq(fold) & grid.arm.eq(arm) & grid.dates.ge(20) & grid.unknown.eq(0)]
        if rule is None:
            assert eligible.empty
        else:
            best = eligible.sort_values(["score", "net_return_scenario", "window", "exit", "pool", "entry", "top_k", "penalty", "min_mu"],
                ascending=[False, False, True, True, True, True, True, True, True], kind="stable").iloc[0]
            assert all(best[key] == value for key, value in rule.items())
            assert lock["qualified"] == bool(best.score > 0 and best.first_half_pnl > 0 and best.second_half_pnl > 0)
        targets = [f"gross_{t}" for t in protocol["targets"]]
        frame = pd.read_parquet(OUTPUT / f"{fold}_prepared.parquet", columns=[*META, "stage", *targets])
        if rule is not None and arm != "no_model":
            pred = pd.read_parquet(OUTPUT / f"{rule['model_id']}_predictions.parquet")
            frame = frame.merge(pred.drop(columns="stage"), on=["symbol", "date"], validate="one_to_one")
        boundaries = protocol["folds"][fold]
        for stage, start, end in [("selection", boundaries[1], boundaries[2]), ("diagnostic", boundaries[2], boundaries[3])]:
            subset = frame.loc[frame.stage.eq(stage)].reset_index(drop=True)
            ids = np.array([], dtype=int) if rule is None else payoff_selection(subset, rule)
            chosen = subset.iloc[ids]
            actual = picks.loc[picks.fold.eq(fold) & picks.arm.eq(arm) & picks.stage.eq(stage)].reset_index(drop=True)
            pd.testing.assert_frame_equal(chosen[["symbol", "date"]].reset_index(drop=True), actual[["symbol", "date"]])
            if rule is not None:
                assert np.allclose(chosen[f"gross_{rule['exit']}"], actual.gross, equal_nan=True)
                assert actual.groupby("date").size().max() <= rule["top_k"] if len(actual) else True
            assert chosen.screen_date.lt(chosen.date).all() and chosen.date.lt(chosen.planned_exit_date).all()
            assert chosen.planned_exit_date.lt(end).all()
            days = days_all[(days_all >= start) & (days_all < end)]
            k = 1 if rule is None else rule["top_k"]
            records = stats.loc[stats.fold.eq(fold) & stats.arm.eq(arm) & stats.stage.eq(stage)]
            for row in records.itertuples():
                metrics, daily = portfolio_metrics(actual.gross.to_numpy(), days.get_indexer(actual.planned_exit_date), len(days), k, row.cost_bps, row.unknown_scenario)
                book, nav = cashbook(actual, days, k, row.cost_bps, row.unknown_scenario)
                for key, value in (metrics | book).items():
                    assert np.isclose(value, getattr(row, key), equal_nan=True), (fold, arm, key)
                saved = curves.loc[curves.fold.eq(fold) & curves.arm.eq(arm) & curves.stage.eq(stage) & curves.cost_bps.eq(row.cost_bps) & curves.unknown_scenario.eq(row.unknown_scenario)]
                assert np.allclose(saved.pnl, daily) and np.allclose(saved.cashbook_nav, nav)
            stages += 1
    # Monthly cashbook changes include reserved cash and locked-position scenarios.
    monthly = []
    group_names = ["fold", "arm", "stage", "cost_bps", "unknown_scenario"]
    for key, frame in curves.groupby(group_names):
        ordered = frame.sort_values("date").copy()
        ordered["change"] = ordered.cashbook_nav.diff().fillna(ordered.cashbook_nav.iloc[0]-1)
        ordered["month"] = ordered.date.dt.to_period("M").astype(str)
        for month, value in ordered.groupby("month").change.sum().items():
            monthly.append(dict(zip(group_names, key, strict=True)) | dict(month=month, initial_capital_pnl=value))
    pd.DataFrame(monthly).to_csv(OUTPUT / "monthly_cashbook.csv", index=False)
    diagnostic = picks.loc[picks.stage.eq("diagnostic")].copy()
    diagnostic["net40"] = diagnostic.gross-.004
    diagnostic["entry_move"] = pd.cut(diagnostic.screen_to_entry_return, [-np.inf, -.03, 0, .03, .07, np.inf])
    diagnostic["signal_age"] = pd.cut(diagnostic.episode_age, [0, 1, 3, 10, np.inf])
    pieces = []
    for dimension in ["entry_move", "signal_age"]:
        piece = diagnostic.groupby(["fold", "arm", dimension], observed=True).agg(trades=("net40", "size"), known=("net40", "count"), mean_net40=("net40", "mean"), worst=("net40", "min")).reset_index().rename(columns={dimension: "bucket"})
        pieces.append(piece.assign(dimension=dimension))
    pd.concat(pieces).to_csv(OUTPUT / "badcase_strata.csv", index=False)
    diagnostic.sort_values("net40").groupby(["fold", "arm"]).head(10).to_csv(OUTPUT / "worst_trades.csv", index=False)
    pd.DataFrame(reliability).to_csv(OUTPUT / "payoff_reliability.csv", index=False)
    (OUTPUT / "audit_complete.json").write_text(json.dumps(dict(model_pairs=training["dual_models"], sampled_prediction_rows=prediction_rows,
        replayed_stage_selections=stages, metric_scenarios=len(stats), picks_sha256=complete["picks_sha256"],
        scope="Prediction/key/chronology/locked-rule/cashbook replay; daily execution and financial vintages remain assumptions"), indent=2))
    print(f"Audited {training['dual_models']} model pairs, {prediction_rows} prediction rows, {stages} stage selections")


if __name__ == "__main__":
    main()
