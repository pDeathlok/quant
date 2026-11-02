"""Validation-locked selection on causal minute inputs; exploratory price proxies."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from threadpoolctl import threadpool_limits
from train_overnight_direct import metrics, select

from quant.research.overnight_uncertainty import date_block_audit


def evaluate(picks: pd.DataFrame, days: pd.DatetimeIndex, top_k: int) -> dict:
    """Unknowns remain; extra 20bps is stress, not an execution guarantee."""
    result = metrics(picks, days, top_k)
    stressed = picks.copy()
    stressed["net"] -= .002
    result.update({f"cost40_{k}": v for k, v in metrics(stressed, days, top_k).items()})
    result["entry_final_at_limit"] = int(picks.entry_final_at_limit.sum())
    result["source_conflicts"] = int(picks.future_quality.eq("source_price_conflict").sum())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--context", action="store_true")
    parser.add_argument("--targets", nargs="+", choices=["tp05", "tp1", "open", "close", "tp05_nostop", "tp1_nostop",
                        "tp05_limit", "tp1_limit"], default=["tp05", "tp1"])
    args = parser.parse_args()
    manifest = json.loads((args.dataset / "manifest.json").read_text())
    if (args.dataset / "INVALIDATED.md").exists() or not (args.dataset / "complete.json").exists():
        raise ValueError("Dataset missing completion or explicitly invalidated")
    if manifest["smoke_only"] and not args.smoke:
        raise ValueError("Partial-pool dataset cannot support formal experiment")
    path = args.dataset / "dataset.parquet"
    args.output.mkdir(exist_ok=False)
    protocol = {"smoke_only": args.smoke, "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "fit": "2024 through 2025H1, label date before 2025-07-01",
                "calibration": "2025Q3, label date before 2025-10-01",
                "selection": "2025Q4, label date before 2026-01-01",
                "confirmation": "2026 exploratory; previously examined dates, NOT independent proof",
                "targets": args.targets, "features": ["minutes", "minutes_and_lagged_factors"],
                "probabilities": [.5, .6, .7, .8, .85, .9, .95], "minimum_net": [0., .002, .005],
                "top_k": [1, 3, 5], "selection_rule": ">=50 signals,>=20 dates,unknown=0,win>=80%,"
                "positive 40bps daily score,zero observed upper-limit final entries; maximize allocation",
                "execution": "price proxies only; closing auction and next-bar fills remain unproven",
                "postdecision_flags": "never filter individual predictions or trades by future metadata"}
    protocol["watchlist_context"] = args.context
    protocol["target_definition"] = {name: (
        "pre-set upward-rounded sell limit, high penetration by 10bps of entry; no open improvement or intraday stop; close fallback"
        if name.endswith("_limit") else
        "official next-session open / entry official close - 1 - 20bps" if name == "open" else
        "official next-session close / entry official close - 1 - 20bps" if name == "close" else
        "completed minute-close profit trigger, next active bar open proxy; no intraday stop; precommitted close fallback"
        if name.endswith("_nostop") else
        "completed minute-close trigger, next active bar open proxy; 1% stop trigger; precommitted close fallback")
        for name in args.targets}
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    all_rows = pd.read_parquet(path)
    if all_rows.duplicated(["symbol", "date"]).any():
        raise ValueError("Duplicate sample keys")
    frame = all_rows[all_rows.decision_eligible].copy().reset_index(drop=True)
    if not pd.to_datetime(frame.factor_date).lt(frame.date).all():
        raise ValueError("Non-lagged daily input detected")
    fit = frame.date.lt("2025-07-01") & frame.target_date.lt("2025-07-01")
    cal = frame.date.ge("2025-07-01") & frame.target_date.lt("2025-10-01")
    forecast = frame.date.ge("2025-10-01")
    summaries = []
    for family in protocol["features"]:
        columns = [c for c in frame if c.startswith("x_min_")
            or (args.context and c.startswith("x_ctx_")) or (
            family == "minutes_and_lagged_factors" and c.startswith(("x_lag_", "x_factor_")))]
        columns = [c for c in columns if frame.loc[fit, c].nunique() > 1
                   and frame.loc[fit, c].notna().mean() >= .05]
        x = frame[columns].astype("float32").replace([np.inf, -np.inf], np.nan)
        for target in protocol["targets"]:
            name = f"{family}_{target}"
            net = frame[f"target_net_{target}"]
            fm, cm = fit & net.notna(), cal & net.notna()
            if net[fm].gt(0).nunique() < 2 or net[cm].gt(0).nunique() < 2:
                raise ValueError("Insufficient class support")
            weights = 1 / frame.loc[fm].groupby("date").date.transform("size")
            weights /= weights.mean()
            params = dict(max_iter=120, max_leaf_nodes=15, min_samples_leaf=100,
                          l2_regularization=30, learning_rate=.06, early_stopping=False, random_state=20260912)
            clf, reg = HistGradientBoostingClassifier(**params), HistGradientBoostingRegressor(**params)
            with threadpool_limits(limits=4):
                clf.fit(x.loc[fm], net[fm].gt(0), sample_weight=weights)
                reg.fit(x.loc[fm], net[fm], sample_weight=weights)
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(clf.predict_proba(x.loc[cm])[:, 1], net[cm].gt(0),
                        sample_weight=1 / frame.loc[cm].groupby("date").date.transform("size"))
                pred = frame.loc[forecast, ["symbol", "date", "target_date", "entry_final_at_limit", "future_quality"]].copy()
                pred["net"] = net[forecast]
                if f"label_quality_{target}" in frame:
                    pred["future_quality"] = frame.loc[forecast, f"label_quality_{target}"]
                pred["probability"] = iso.predict(clf.predict_proba(x.loc[forecast])[:, 1])
                pred["expected_net"] = reg.predict(x.loc[forecast])
            pred.to_parquet(args.output / f"{name}_predictions.parquet", index=False)
            joblib.dump({"classifier": clf, "regressor": reg, "calibrator": iso, "columns": columns},
                        args.output / f"{name}.joblib")
            validation = pred[pred.date.lt("2026-01-01") & pred.target_date.lt("2026-01-01")]
            days = pd.DatetimeIndex(all_rows.loc[all_rows.date.ge("2025-10-01")
                & all_rows.target_date.lt("2026-01-01"), "date"].unique()).sort_values()
            rows = []
            for probability in protocol["probabilities"]:
                for minimum_net in protocol["minimum_net"]:
                    for top_k in protocol["top_k"]:
                        picks = select(validation, probability, minimum_net, top_k)
                        rows.append({"probability": probability, "minimum_net": minimum_net, "top_k": top_k,
                                     **evaluate(picks, days, top_k)})
            table = pd.DataFrame(rows)
            table.to_csv(args.output / f"{name}_validation.csv", index=False)
            eligible = table[table.signals.ge(50) & table.dates.ge(20) & table.unknown.eq(0)
                             & table.known_win.ge(.8) & table.cost40_daily_score.gt(0)
                             & table.entry_final_at_limit.eq(0)]
            summary = {"name": name, "fit_rows": int(fm.sum()), "features": len(columns), "eligible_rules": len(eligible)}
            if not eligible.empty:
                chosen = eligible.sort_values(["allocation", "cost40_daily_score"], ascending=False).iloc[0]
                later = pred[pred.date.ge("2026-01-01")]
                picks = select(later, chosen.probability, chosen.minimum_net, int(chosen.top_k))
                picks.to_parquet(args.output / f"{name}_locked_picks.parquet", index=False)
                later_days = pd.DatetimeIndex(all_rows.loc[all_rows.date.ge("2026-01-01"), "date"].unique()).sort_values()
                summary.update(chosen=chosen.to_dict(), exploratory_2026=evaluate(picks, later_days, int(chosen.top_k)))
                summary["exploratory_date_block_audit"] = [date_block_audit(picks, later_days, block_days=b)
                    for b in (5, 20) if len(later_days) >= b]
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
    (args.output / "complete.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
