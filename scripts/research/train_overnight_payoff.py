"""Freeze win probability; model conditional gain and non-win loss magnitudes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from train_overnight_direct import select
from train_overnight_minutes import evaluate

from quant.research.overnight_payoff import expected_payoff
from quant.research.overnight_uncertainty import date_block_audit

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    source = ROOT / "reports/research/overnight_context_training_v1"
    dataset = ROOT / "reports/research/overnight_context_dataset_v1/dataset.parquet"
    output = ROOT / "reports/research/overnight_payoff_training_v1"
    protocol = json.loads((source / "protocol.json").read_text())
    if hashlib.sha256(dataset.read_bytes()).hexdigest() != protocol["dataset_sha256"]:
        raise ValueError("Base dataset changed")
    output.mkdir(exist_ok=False)
    protocol.update(payoff="p(win)*E(net|win) - (1-p(win))*E(-net|nonwin); 20bps already in labels",
        calibration="each magnitude scaled by weighted observed/predicted mean on its 2025Q3 class",
        probability="frozen source forecasts; no rescore or refit", independent_confirmation=False)
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    all_rows = pd.read_parquet(dataset)
    frame = all_rows[all_rows.decision_eligible].copy().reset_index(drop=True)
    fit = frame.date.lt("2025-07-01") & frame.target_date.lt("2025-07-01")
    cal = frame.date.ge("2025-07-01") & frame.target_date.lt("2025-10-01")
    forecast = frame.date.ge("2025-10-01")
    days = pd.DatetimeIndex(all_rows.loc[all_rows.date.ge("2025-10-01")
        & all_rows.target_date.lt("2026-01-01"), "date"].unique()).sort_values()
    summaries = []
    for family in protocol["features"]:
        for target in protocol["targets"]:
            name = f"{family}_{target}"
            columns = joblib.load(source / f"{name}.joblib")["columns"]
            x = frame[columns].astype("float32").replace([np.inf, -np.inf], np.nan)
            net = frame[f"target_net_{target}"]
            fm, cm = fit & net.notna(), cal & net.notna()
            weights = 1 / frame.loc[fm].groupby("date").date.transform("size")
            weights /= weights.mean()
            cal_weights = 1 / frame.loc[cm].groupby("date").date.transform("size")
            scores = frame.loc[forecast, ["symbol", "date"]].copy()
            bundle = {"columns": columns}
            support = {}
            with threadpool_limits(limits=4):
                for label, condition in (("gain", net.gt(0)), ("loss", net.le(0))):
                    fit_class, cal_class = fm & condition, cm & condition
                    if fit_class.sum() < 100 or cal_class.sum() < 20:
                        raise ValueError("Insufficient conditional class support")
                    model = HistGradientBoostingRegressor(max_iter=120, max_leaf_nodes=15, min_samples_leaf=100,
                        l2_regularization=30, learning_rate=.06, early_stopping=False, random_state=20260912)
                    model.fit(x.loc[fit_class], net[fit_class].abs(), sample_weight=weights.loc[fit_class[fit_class].index])
                    predicted = np.maximum(model.predict(x.loc[cal_class]), 1e-8)
                    cw = cal_weights.loc[cal_class[cal_class].index]
                    scale = float(np.average(net[cal_class].abs(), weights=cw) / np.average(predicted, weights=cw))
                    scores[label] = np.maximum(model.predict(x.loc[forecast]), 0) * scale
                    bundle[label] = {"model": model, "scale": scale}
                    support[label] = {"fit": int(fit_class.sum()), "calibration": int(cal_class.sum()), "scale": scale}
            base = pd.read_parquet(source / f"{name}_predictions.parquet")
            pred = base.merge(scores, on=["symbol", "date"], how="left", validate="one_to_one", sort=False)
            pd.testing.assert_frame_equal(base.reset_index(drop=True), pred[base.columns].reset_index(drop=True))
            pred["base_expected_net"] = pred.expected_net
            pred["expected_net"] = expected_payoff(pred.probability.to_numpy(), pred.gain.to_numpy(), pred.loss.to_numpy())
            pred.to_parquet(output / f"{name}_predictions.parquet", index=False)
            joblib.dump(bundle, output / f"{name}.joblib")
            validation = pred[pred.date.lt("2026-01-01") & pred.target_date.lt("2026-01-01")]
            rows = []
            for probability in protocol["probabilities"]:
                for minimum_net in protocol["minimum_net"]:
                    for top_k in protocol["top_k"]:
                        picks = select(validation, probability, minimum_net, top_k)
                        rows.append({"probability": probability, "minimum_net": minimum_net, "top_k": top_k,
                                     **evaluate(picks, days, top_k)})
            table = pd.DataFrame(rows)
            table.to_csv(output / f"{name}_validation.csv", index=False)
            eligible = table[table.signals.ge(50) & table.dates.ge(20) & table.unknown.eq(0)
                & table.known_win.ge(.8) & table.cost40_daily_score.gt(0) & table.entry_final_at_limit.eq(0)]
            summary = {"name": name, "eligible_rules": len(eligible), "support": support, "probability_unchanged": True}
            if not eligible.empty:
                chosen = eligible.sort_values(["allocation", "cost40_daily_score"], ascending=False).iloc[0]
                picks = select(pred[pred.date.ge("2026-01-01")], chosen.probability, chosen.minimum_net, int(chosen.top_k))
                picks.to_parquet(output / f"{name}_locked_picks.parquet", index=False)
                later_days = pd.DatetimeIndex(all_rows.loc[all_rows.date.ge("2026-01-01"), "date"].unique()).sort_values()
                summary.update(chosen=chosen.to_dict(), exploratory_2026=evaluate(picks, later_days, int(chosen.top_k)),
                               date_blocks=[date_block_audit(picks, later_days, block_days=b) for b in (5, 20)])
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
    (output / "complete.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
