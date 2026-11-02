"""Downside classifier added to frozen broad-pool lag-only profit forecasts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from threadpoolctl import threadpool_limits
from train_overnight_direct import metrics, select

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuilt-current", action="store_true")
    parser.add_argument("--market-parent", action="store_true")
    parser.add_argument("--moneyflow", action="store_true")
    parser.add_argument("--moneyflow-rolling", action="store_true")
    parser.add_argument("--holder-events", action="store_true")
    parser.add_argument("--loss-cutoff", type=float, choices=[.01, .02], default=.02)
    args = parser.parse_args()
    if args.moneyflow_rolling and not args.moneyflow:
        parser.error("--moneyflow-rolling requires --moneyflow")
    if args.holder_events and (not args.market_parent or args.moneyflow):
        parser.error("--holder-events requires market parent and excludes moneyflow")
    parent = ROOT / "reports/research/overnight_lag_expanded_limit_v1"
    dataset = ROOT / "reports/research/tail_overnight/dataset.parquet"
    output = ROOT / "reports/research/overnight_lag_expanded_risk_v1"
    if args.rebuilt_current:
        parent = ROOT / "reports/research/overnight_same_day_rebuilt_limit_gated_v1"
        output = ROOT / "reports/research/overnight_same_day_rebuilt_risk_v1"
        if args.loss_cutoff == .01:
            output = ROOT / "reports/research/overnight_same_day_rebuilt_risk1pct_v1"
    elif args.loss_cutoff != .02:
        raise ValueError("Alternative loss event is scoped to rebuilt current-day experiment")
    if args.market_parent:
        if not args.rebuilt_current or args.loss_cutoff != .02:
            raise ValueError("Market-parent experiment uses rebuilt inputs and fixed2% loss event")
        parent = ROOT / "reports/research/overnight_same_day_market_limit_gated_v2"
        output = ROOT / "reports/research/overnight_same_day_market_risk_v1"
    if args.moneyflow:
        if not args.market_parent:
            parser.error("--moneyflow requires --market-parent --rebuilt-current")
        output = ROOT / "reports/research/overnight_moneyflow_risk_v1"
    if args.moneyflow_rolling:
        output = ROOT / "reports/research/overnight_moneyflow_rolling_risk_v1"
    if args.holder_events:
        output = ROOT / "reports/research/overnight_holder_risk_v1"
    protocol = json.loads((parent / "protocol.json").read_text())
    if not (parent / "complete.json").exists() or hashlib.sha256(dataset.read_bytes()).hexdigest() != protocol["source_sha256"]:
        raise ValueError("Incomplete parent or changed source")
    output.mkdir(exist_ok=False)
    protocol.update(risk_event=f"fixed 20bps exit net < -{args.loss_cutoff}", risk_thresholds=[1., .2, .1, .05],
                    frozen_profit="load parent predictions verbatim; never refit",
                    risk_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if args.moneyflow:
        protocol.update(probability_thresholds=[.7], minimum_net=[None], top_k=[1], risk_thresholds=[.1],
                        ablation="Add six exact T-1 flow ratios only to 2% loss classifier; profit models frozen",
                        confirmation="Reused Q4 diagnostic, not independent confirmation; no threshold search")
    if args.moneyflow_rolling:
        protocol["ablation"] = "Add 24 exact-calendar T-1 and 3/5-day flow features only to 2% loss classifier; profit frozen"
    if args.holder_events:
        protocol.update(probability_thresholds=[.7], minimum_net=[None], top_k=[1], risk_thresholds=[.1],
                        ablation="Only add eight pre-decision observed holder event features to loss classifier",
                        confirmation="Reused Q4 diagnostic; no new threshold search or independent claim")
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    cols = protocol["features"]
    if not all(c.startswith("lag1_") for c in cols) and not args.rebuilt_current:
        raise ValueError("Non-lagged feature")
    lag_cols = [c for c in cols if c.startswith("lag1_")]
    f = pd.read_parquet(dataset, columns=["symbol", "date", "feature_date", "exit_date", *lag_cols])
    input_hashes = []
    if args.rebuilt_current:
        from quant.features.variable_library import PROJECT_FACTOR_COLUMNS

        for record in json.loads((parent / "same_day_sources.json").read_text()):
            path = Path(record["path"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError("Changed parent factor source")
            input_hashes.append(record)
        factors_path = ROOT / "reports/research/overnight_current_full_factors_v1/dataset.parquet"
        factors = pd.read_parquet(factors_path, columns=["symbol", "date", *PROJECT_FACTOR_COLUMNS])
        factors = factors.rename(columns={c: f"today_factor_{c}" for c in PROJECT_FACTOR_COLUMNS})
        f = f.merge(factors, on=["symbol", "date"], how="left", validate="one_to_one")
        raw_path = parent / "same_day_raw_snapshot.parquet"
        f = f.merge(pd.read_parquet(raw_path), on=["symbol", "date"], how="left", validate="one_to_one")
        input_hashes.append({"path": str(raw_path), "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest()})
        if args.market_parent:
            market_path = parent / "market_context.parquet"
            context = pd.read_parquet(market_path)
            market_cols = protocol["market_context_features"]
            f = f.merge(context[["date", *market_cols]], on="date", how="left", validate="many_to_one")
            if f[market_cols].isna().any().any():
                raise ValueError("Missing frozen market context")
            input_hashes.append({"path": str(market_path), "sha256": hashlib.sha256(market_path.read_bytes()).hexdigest()})
    label_path = parent / "limit_label_source.parquet"
    labels = pd.read_parquet(label_path, columns=["symbol", "date", "exit_date", *protocol["targets"]])
    pd.testing.assert_frame_equal(f[["symbol", "date", "exit_date"]], labels[["symbol", "date", "exit_date"]])
    if not f.feature_date.lt(f.date).all():
        raise ValueError("Non-lagged feature date")
    calendar = f[["date", "exit_date"]].drop_duplicates()
    if args.rebuilt_current:
        gate_path = parent / "exact_entry_gate.parquet"
        gate = pd.read_parquet(gate_path)
        pd.testing.assert_frame_equal(f[["symbol", "date"]], gate[["symbol", "date"]])
        eligible = gate.eligible.to_numpy()
        f = f.loc[eligible].reset_index(drop=True)
        labels = labels.loc[eligible].reset_index(drop=True)
        input_hashes.append({"path": str(gate_path), "sha256": hashlib.sha256(gate_path.read_bytes()).hexdigest()})
    base_cols = list(cols)
    if args.moneyflow:
        flow_root = ROOT / "reports/research/overnight_moneyflow_lag_v1"
        if args.moneyflow_rolling:
            flow_root = ROOT / "reports/research/overnight_moneyflow_rolling_v1"
        flow_path = flow_root / "dataset.parquet"
        manifest = json.loads((flow_root / "complete.json").read_text())
        if hashlib.sha256(flow_path.read_bytes()).hexdigest() != manifest["dataset_sha256"]:
            raise ValueError("Moneyflow artifact changed")
        flow_cols = manifest["features"]
        flow = pd.read_parquet(flow_path, columns=["symbol", "date", "flow_date", *flow_cols])
        if not flow.flow_date.lt(flow.date).all():
            raise ValueError("Invalid flow timing")
        old_keys = f[["symbol", "date"]].copy()
        f = f.merge(flow, on=["symbol", "date"], how="left", validate="one_to_one")
        pd.testing.assert_frame_equal(old_keys, f[["symbol", "date"]])
        if f.flow_date.isna().any():
            raise ValueError("Moneyflow candidate keys missing")
        cols = [*base_cols, *flow_cols]
        protocol.update(features=cols, baseline_features=base_cols)
        (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
        input_hashes.append({"path": str(flow_path), "sha256": manifest["dataset_sha256"]})
    if args.holder_events:
        event_root = ROOT / "reports/research/overnight_holder_events_v1"
        event_path = event_root / "dataset.parquet"
        event_manifest = json.loads((event_root / "complete.json").read_text())
        if hashlib.sha256(event_path.read_bytes()).hexdigest() != event_manifest["dataset_sha256"]:
            raise ValueError("Holder features changed")
        event_cols = event_manifest["features"]
        event = pd.read_parquet(event_path)
        old_keys = f[["symbol", "date"]].copy()
        f = f.merge(event[["symbol", "date", "source_stale", *event_cols]], on=["symbol", "date"], how="left", validate="one_to_one")
        pd.testing.assert_frame_equal(old_keys, f[["symbol", "date"]])
        if f.source_stale.isna().any():
            raise ValueError("Missing event candidate keys")
        cols = [*base_cols, *event_cols]
        protocol.update(features=cols, baseline_features=base_cols)
        (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
        input_hashes.append({"path": str(event_path), "sha256": event_manifest["dataset_sha256"]})
    x = f[cols].astype("float32").replace([np.inf, -np.inf], np.nan)
    fit = f.date.lt("2025-07-01") & f.exit_date.lt("2025-07-01")
    cal = f.date.ge("2025-07-01") & f.exit_date.lt("2025-10-01")
    val = f.date.ge("2025-10-01") & f.date.lt("2026-01-01") & f.exit_date.lt("2026-01-01")
    days = pd.DatetimeIndex(calendar.loc[calendar.date.ge("2025-10-01") & calendar.date.lt("2026-01-01")
        & calendar.exit_date.lt("2026-01-01"), "date"].unique()).sort_values()
    summaries, provenance = [], [*input_hashes,
        {"path": str(label_path), "sha256": hashlib.sha256(label_path.read_bytes()).hexdigest()}]
    for target in protocol["targets"]:
        net = labels[target]
        fm, cm = fit & net.notna(), cal & net.notna()
        if net[fm].lt(-args.loss_cutoff).nunique() != 2 or net[cm].lt(-args.loss_cutoff).nunique() != 2:
            raise ValueError("Insufficient risk classes")
        weights = 1 / f.loc[fm].groupby("date").date.transform("size")
        weights /= weights.mean()
        path = parent / f"{target}_predictions.parquet"
        provenance.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        base = pd.read_parquet(path)
        pd.testing.assert_frame_equal(base[["symbol", "date"]].reset_index(drop=True), f.loc[val, ["symbol", "date"]].reset_index(drop=True))
        pred = base.copy()
        with threadpool_limits(limits=4):
            if args.rebuilt_current:
                parent_model = joblib.load(parent / f"{target}.joblib")
                assert parent_model["columns"] == base_cols
                np.testing.assert_allclose(parent_model["calibrator"].predict(
                    parent_model["classifier"].predict_proba(x.loc[val, base_cols])[:, 1]), base.probability, rtol=1e-10, atol=1e-12)
                np.testing.assert_allclose(parent_model["regressor"].predict(x.loc[val, base_cols]), base.expected_net,
                                           rtol=1e-10, atol=1e-12)
            if args.moneyflow or args.holder_events:
                control_root = ROOT / "reports/research/overnight_same_day_market_risk_v1"
                control_path = control_root / f"{target}_risk.joblib"
                control_pred_path = control_root / f"{target}_predictions.parquet"
                for source in (control_path, control_pred_path):
                    provenance.append({"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
                control_model = joblib.load(control_path)
                control = pd.read_parquet(control_pred_path)
                pd.testing.assert_frame_equal(base, control[base.columns])
                if control_model["columns"] != base_cols:
                    raise ValueError("Baseline risk column mismatch")
                np.testing.assert_allclose(control_model["calibrator"].predict(
                    control_model["model"].predict_proba(x.loc[val, base_cols])[:, 1]), control.risk,
                    rtol=1e-10, atol=1e-12)
                pred["baseline_risk"] = control.risk
                if args.moneyflow:
                    pred["moneyflow_available"] = f.loc[val, flow_cols].notna().all(axis=1).to_numpy()
                if args.holder_events:
                    pred["holder_source_stale"] = f.loc[val, "source_stale"].to_numpy()
                    pred["holder_observed_20d"] = f.loc[val, ["holder_observed_IN_20d", "holder_observed_DE_20d"]].sum(axis=1, min_count=2).to_numpy()
                print(f"{target}: baseline profit/risk replay passed; training {len(cols)} risk features", flush=True)
            model = HistGradientBoostingClassifier(**protocol["params"]).fit(x.loc[fm], net[fm].lt(-args.loss_cutoff), sample_weight=weights)
            iso = IsotonicRegression(out_of_bounds="clip").fit(model.predict_proba(x.loc[cm])[:, 1], net[cm].lt(-args.loss_cutoff),
                sample_weight=1 / f.loc[cm].groupby("date").date.transform("size"))
            pred["risk"] = iso.predict(model.predict_proba(x.loc[val])[:, 1])
        pd.testing.assert_frame_equal(base, pred[base.columns])
        pred.to_parquet(output / f"{target}_predictions.parquet", index=False)
        joblib.dump({"model": model, "calibrator": iso, "columns": cols}, output / f"{target}_risk.joblib")
        rows = []
        for probability in protocol["probability_thresholds"]:
            for minimum_net in protocol["minimum_net"]:
                for top_k in protocol["top_k"]:
                    for maximum_risk in protocol["risk_thresholds"]:
                        picks = select(pred, probability, -np.inf if minimum_net is None else minimum_net, top_k, maximum_risk)
                        record = {"probability": probability, "minimum_net": minimum_net, "top_k": top_k,
                                  "maximum_risk": maximum_risk, **metrics(picks, days, top_k),
                                  "postdecision_old_entry_failures": int((~picks.entry_policy_eligible).sum())}
                        stress = picks.copy()
                        stress["net"] -= .002
                        record.update({f"cost40_{k}": v for k, v in metrics(stress, days, top_k).items()})
                        rows.append(record)
        table = pd.DataFrame(rows)
        baseline = pd.read_csv(parent / f"{target}_diagnostics.csv")
        replay = table[table.maximum_risk.eq(1.)].drop(columns="maximum_risk").reset_index(drop=True)
        if not (args.moneyflow or args.holder_events):
            pd.testing.assert_frame_equal(replay, baseline, check_dtype=False, rtol=1e-10, atol=1e-12)
        table.to_csv(output / f"{target}_diagnostics.csv", index=False)
        enough = table[table.signals.ge(50) & table.dates.ge(20)]
        eligible = enough[enough.known_win.ge(.8) & enough.unknown.eq(0) & enough.cost40_daily_score.gt(0)]
        if not args.rebuilt_current:
            eligible = eligible[eligible.postdecision_old_entry_failures.eq(0)]
        summary = {"target": target, "fit_known": int(fm.sum()), "fit_loss_events": int(net[fm].lt(-args.loss_cutoff).sum()),
                   "cal_loss_events": int(net[cm].lt(-args.loss_cutoff).sum()), "baseline_replayed": True,
                   "eligible": len(eligible), "supported_rules": len(enough), "best_supported_diagnostic":
                   None if enough.empty else enough.sort_values("win_missing_as_failure", ascending=False).iloc[0].to_dict()}
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    for record in provenance:
        if hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("Parent input changed during risk experiment")
    (output / "parent_hashes.json").write_text(json.dumps(provenance, indent=2))
    (output / "complete.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
