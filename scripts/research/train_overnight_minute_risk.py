"""Add a downside head without changing frozen minute-model profit forecasts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from threadpoolctl import threadpool_limits
from train_overnight_direct import select
from train_overnight_minutes import evaluate

from quant.research.overnight_uncertainty import date_block_audit

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    source = ROOT / "reports/research/overnight_minute_training_v1"
    dataset = ROOT / "reports/research/overnight_minute_dataset_v1/dataset.parquet"
    output = ROOT / "reports/research/overnight_minute_risk_v1"
    protocol = json.loads((source / "protocol.json").read_text())
    if hashlib.sha256(dataset.read_bytes()).hexdigest() != protocol["dataset_sha256"]:
        raise ValueError("Base dataset hash changed")
    if not (source / "complete.json").exists() or protocol["smoke_only"]:
        raise ValueError("Base full experiment must be complete")
    output.mkdir(exist_ok=False)
    protocol.update(risk_event="fixed exit net<-2%", maximum_risk=[1., .2, .1, .05],
                    base_predictions="loaded verbatim; never refit or rescore profit heads",
                    added_search="exploratory validation search; no independent confirmation claim")
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    all_rows = pd.read_parquet(dataset)
    frame = all_rows[all_rows.decision_eligible].copy().reset_index(drop=True)
    fit = frame.date.lt("2025-07-01") & frame.target_date.lt("2025-07-01")
    cal = frame.date.ge("2025-07-01") & frame.target_date.lt("2025-10-01")
    days = pd.DatetimeIndex(all_rows.loc[all_rows.date.ge("2025-10-01")
        & all_rows.target_date.lt("2026-01-01"), "date"].unique()).sort_values()
    summaries = []
    for family in protocol["features"]:
        for target in protocol["targets"]:
            name = f"{family}_{target}"
            bundle = joblib.load(source / f"{name}.joblib")
            columns = bundle["columns"]
            if not all(c.startswith("x_") for c in columns):
                raise ValueError("Non-input feature in frozen bundle")
            x = frame[columns].astype("float32").replace([np.inf, -np.inf], np.nan)
            net = frame[f"target_net_{target}"]
            fm, cm = fit & net.notna(), cal & net.notna()
            if net[fm].lt(-.02).nunique() < 2 or net[cm].lt(-.02).nunique() < 2:
                raise ValueError("Insufficient downside classes")
            weights = 1 / frame.loc[fm].groupby("date").date.transform("size")
            weights /= weights.mean()
            risk = HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=15, min_samples_leaf=100,
                l2_regularization=30, learning_rate=.06, early_stopping=False, random_state=20260912)
            with threadpool_limits(limits=4):
                risk.fit(x.loc[fm], net[fm].lt(-.02), sample_weight=weights)
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(risk.predict_proba(x.loc[cm])[:, 1], net[cm].lt(-.02),
                        sample_weight=1 / frame.loc[cm].groupby("date").date.transform("size"))
                scores = frame.loc[frame.date.ge("2025-10-01"), ["symbol", "date"]].copy()
                scores["risk"] = iso.predict(risk.predict_proba(x.loc[scores.index])[:, 1])
            base = pd.read_parquet(source / f"{name}_predictions.parquet")
            pred = base.merge(scores, on=["symbol", "date"], how="left", validate="one_to_one", sort=False)
            pd.testing.assert_frame_equal(base.reset_index(drop=True), pred[base.columns].reset_index(drop=True))
            if pred.risk.isna().any():
                raise ValueError("Missing risk forecast")
            pred.to_parquet(output / f"{name}_predictions.parquet", index=False)
            joblib.dump({"risk_model": risk, "calibrator": iso, "columns": columns}, output / f"{name}.joblib")
            validation = pred[pred.date.lt("2026-01-01") & pred.target_date.lt("2026-01-01")]
            rows = []
            for maximum_risk in protocol["maximum_risk"]:
                for probability in protocol["probabilities"]:
                    for minimum_net in protocol["minimum_net"]:
                        for top_k in protocol["top_k"]:
                            picks = select(validation, probability, minimum_net, top_k, maximum_risk)
                            rows.append({"maximum_risk": maximum_risk, "probability": probability,
                                "minimum_net": minimum_net, "top_k": top_k, **evaluate(picks, days, top_k)})
            table = pd.DataFrame(rows)
            table.to_csv(output / f"{name}_validation.csv", index=False)
            eligible = table[table.signals.ge(50) & table.dates.ge(20) & table.unknown.eq(0)
                & table.known_win.ge(.8) & table.cost40_daily_score.gt(0) & table.entry_final_at_limit.eq(0)]
            summary = {"name": name, "eligible_rules": len(eligible), "fit_rows": int(fm.sum()),
                "fit_downside_events": int(net[fm].lt(-.02).sum()), "base_unchanged": True}
            if not eligible.empty:
                chosen = eligible.sort_values(["allocation", "cost40_daily_score"], ascending=False).iloc[0]
                picks = select(pred[pred.date.ge("2026-01-01")], chosen.probability, chosen.minimum_net,
                               int(chosen.top_k), chosen.maximum_risk)
                picks.to_parquet(output / f"{name}_locked_picks.parquet", index=False)
                later_days = pd.DatetimeIndex(all_rows.loc[all_rows.date.ge("2026-01-01"), "date"].unique()).sort_values()
                summary.update(chosen=chosen.to_dict(), exploratory_2026=evaluate(picks, later_days, int(chosen.top_k)),
                               date_blocks=[date_block_audit(picks, later_days, block_days=b) for b in (5, 20)])
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
    (output / "complete.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
