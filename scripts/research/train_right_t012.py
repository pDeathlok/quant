"""Pure-right T screening, T+1 close entry, T+2 exit research baseline."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from audit_overnight_right_provenance import sha256
from overnight_right_signal_source import load_events
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from threadpoolctl import threadpool_limits
from train_overnight_direct import metrics
from train_overnight_unfiltered_comparison import date_weights, direct_exit_flags

from quant.core.paths import PROJECT_ROOT
from quant.features.variable_library import PROJECT_FACTOR_COLUMNS
from quant.research.short_side_groups import RIGHT_GROUP_MEMBERS

SIGNALS = tuple(dict.fromkeys(s for members in RIGHT_GROUP_MEMBERS.values() for s in members))
KEYS = ["symbol", "date"]


def map_entries(events: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Use global market dates, never per-stock next observed bar."""
    if events.duplicated(KEYS).any() or events[KEYS].isna().any().any():
        raise ValueError("Invalid event keys")
    if calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise ValueError("Invalid calendar")
    links = pd.DataFrame({"screen_date": calendar, "date": pd.Series(calendar).shift(-1),
                          "planned_exit_date": pd.Series(calendar).shift(-2)})
    selected = events.loc[events[list(SIGNALS)].eq(True).any(axis=1)].copy()
    selected = selected.rename(columns={"date": "screen_date"})
    return selected.merge(links, on="screen_date", how="left", validate="many_to_one")


def choose(frame: pd.DataFrame, probability: float, rank: str) -> pd.DataFrame:
    """Selection must not consult outcome columns."""
    accepted = frame.probability.ge(probability) & frame.expected_net.gt(0)
    return (frame.loc[accepted].sort_values(
        ["date", rank, "symbol"], ascending=[True, False, True]
    ).groupby("date", sort=False).head(1).copy())


def summarize(frame: pd.DataFrame, days: pd.DatetimeIndex) -> dict:
    result = metrics(frame, days, 1)
    result.update(worst_net=float(frame.net.min()),
                  event_hits=int(frame.event_hit.eq(True).sum()),
                  event_unknown=int(frame.event_hit.isna().sum()),
                  event_hit_rate=float(frame.event_hit.eq(True).fillna(False).mean()) if len(frame) else float("nan"))
    return result


def main() -> None:
    root = PROJECT_ROOT / "reports/research"
    output = root / "right_t012_baseline_v2"
    if output.exists():
        raise FileExistsError(output)
    signal_path = root / "overnight_current_right_signals_v2/all_daily_signals.parquet"
    events, availability = load_events(signal_path)
    paths = {
        "labels": root / "overnight_unfiltered_labels_v1/labels.parquet",
        "gate": root / "overnight_unfiltered_entry_gate_v1/entry_gate.parquet",
        "factors": root / "overnight_unfiltered_full_factors_v1/dataset.parquet",
        "auxiliary": root / "overnight_unfiltered_auxiliary_v1/dataset.parquet",
        "audit": root / "overnight_unfiltered_exit_audit_v1/exit_audit.parquet",
    }
    hashes = {str(signal_path): sha256(signal_path)}
    for path in paths.values():
        digest = sha256(path)
        if digest != json.loads(path.with_name("complete.json").read_text())["dataset_sha256"]:
            raise ValueError(f"Source hash mismatch: {path}")
        hashes[str(path)] = digest
    protocol_path = root / "overnight_same_day_market_limit_gated_v2/protocol.json"
    parent = json.loads(protocol_path.read_text())
    hashes[str(protocol_path)] = sha256(protocol_path)
    labels = pd.read_parquet(paths["labels"])
    calendar = pd.DatetimeIndex(labels.date.unique()).sort_values()
    frame = map_entries(events, calendar)
    signal_close = labels[KEYS + ["entry_close"]].rename(
        columns={"date": "screen_date", "entry_close": "screen_close"})
    frame = frame.merge(signal_close, on=["symbol", "screen_date"], how="left", validate="one_to_one")
    frame = frame.merge(labels, on=KEYS, how="left", validate="one_to_one")
    del labels
    frame = frame.merge(pd.read_parquet(paths["gate"], columns=KEYS + ["eligible", "unknown_entry", "at_upper"]),
                        on=KEYS, how="left", validate="one_to_one")
    frame["eligible"] = frame.eligible.eq(True)
    known = frame.exit_date.notna()
    if not frame.loc[known, "exit_date"].eq(frame.loc[known, "planned_exit_date"]).all():
        raise ValueError("T+2 labels do not match global calendar")
    if not (frame.screen_date.lt(frame.date) & frame.date.lt(frame.planned_exit_date)).all():
        raise ValueError("Invalid T/T+1/T+2 timing")
    output.mkdir()
    frame.to_parquet(output / "all_entry_candidates.parquet", index=False)
    coverage = frame.groupby("date").agg(candidates=("symbol", "size"), eligible=("eligible", "sum"))
    coverage.to_csv(output / "coverage.csv")
    frame = frame.loc[frame.eligible & frame.planned_exit_date.lt("2026-01-01")].copy()
    filters = [("date", ">=", pd.Timestamp("2025-07-01")), ("date", "<", pd.Timestamp("2026-01-01"))]
    auxiliary = pd.read_parquet(paths["auxiliary"], filters=filters)
    frame = frame.merge(auxiliary, on=KEYS, how="left", validate="one_to_one", indicator=True)
    if not frame._merge.eq("both").all():
        raise ValueError("Missing entry feature keys")
    frame = frame.drop(columns="_merge")
    factors = pd.read_parquet(paths["factors"], columns=KEYS + list(PROJECT_FACTOR_COLUMNS), filters=filters)
    factors = factors.rename(columns={c: f"today_factor_{c}" for c in PROJECT_FACTOR_COLUMNS})
    frame = frame.merge(factors, on=KEYS, how="left", validate="one_to_one", indicator=True)
    if not frame._merge.eq("both").all():
        raise ValueError("Missing market factor keys")
    frame = frame.drop(columns="_merge")
    del auxiliary, factors
    frame["screen_to_entry_return"] = frame.entry_close / frame.screen_close - 1
    columns = [*parent["features"], *SIGNALS, "screen_to_entry_return"]
    if len(parent["features"]) != 187 or not frame.feature_date.lt(frame.date).all():
        raise ValueError("Unexpected feature contract")
    audit = pd.read_parquet(paths["audit"], filters=filters)
    audit_columns = KEYS + ["down_limit", "price_limit_inconsistent", "limit1_exit_uncertain",
                           "limit1_net_after_exit_audit"]
    frame = frame.merge(audit[audit_columns], on=KEYS, how="left", validate="one_to_one")
    close_audit = direct_exit_flags(frame, "net_close")
    frame = pd.concat([frame.reset_index(drop=True), close_audit.reset_index(drop=True)], axis=1)
    masks = {
        "train": frame.date.lt("2025-09-01") & frame.planned_exit_date.lt("2025-09-01"),
        "calibration": frame.date.ge("2025-09-01") & frame.date.lt("2025-10-01") & frame.planned_exit_date.lt("2025-10-01"),
        "q4_diagnostic": frame.date.ge("2025-10-01"),
    }
    params = dict(max_iter=120, max_leaf_nodes=15, min_samples_leaf=200, l2_regularization=30,
                  learning_rate=.06, early_stopping=False, random_state=20260913)
    protocol = dict(timing="T pure-right screen; T+1 close entry; T+2 precommitted exit", sources=hashes,
                    features=columns, params=params, signal_availability=availability,
                    signals=SIGNALS, right_groups=RIGHT_GROUP_MEMBERS,
                    targets=["net_close", "limit1"], costs_bps=[20, 40],
                    partitions={k: int(v.sum()) for k, v in masks.items()},
                    selection="September only: among rules with >=10 dates, maximize 40bps daily score; if none abstain. No pass/precision claim implied.",
                    grid={"probability": [0., .5, .6, .7, .8, .9], "rank": ["probability", "expected_net"]},
                    limitations=["Historical reused diagnostics; no independent confirmation", "Daily close proxy, not14:30 fills",
                                 "Only audited187 inputs plus screen flags/return, not700+", "No sequence model yet",
                                 "Unknown exit kept; -10% score penalty is not actual loss or maximum loss",
                                 "Nominal screen-to-entry return includes corporate actions; price-gap badcase needs review"],
                    script_sha256=sha256(Path(__file__)))
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    x = frame[columns].astype("float32").replace([np.inf, -np.inf], np.nan)
    summaries, predictions, selections, badcases, grids = [], [], [], [], []
    for target in ["net_close", "limit1"]:
        net = frame[f"{target}_net_after_exit_audit"]
        event = frame.limit1_penetrated.astype("boolean") if target == "limit1" else net.gt(0).astype("boolean").mask(net.isna())
        fm = masks["train"] & net.notna()
        cm = masks["calibration"] & net.notna()
        print(f"{target}: fit={fm.sum()} calibration={cm.sum()} features={len(columns)}", flush=True)
        if event[fm].nunique() != 2 or event[cm].nunique() != 2:
            raise ValueError("Insufficient classes")
        with threadpool_limits(limits=4):
            classifier = HistGradientBoostingClassifier(**params).fit(x[fm], event[fm].astype(int), sample_weight=date_weights(frame[fm]))
            regressor = HistGradientBoostingRegressor(**params).fit(x[fm], net[fm], sample_weight=date_weights(frame[fm]))
            calibrator = IsotonicRegression(out_of_bounds="clip").fit(
                classifier.predict_proba(x[cm])[:, 1], event[cm].astype(int), sample_weight=date_weights(frame[cm]))
            probability = calibrator.predict(classifier.predict_proba(x)[:, 1])
            expected_net = regressor.predict(x)
        joblib.dump(dict(classifier=classifier, regressor=regressor, calibrator=calibrator, features=columns), output / f"{target}.joblib")
        fields = KEYS + ["screen_date", "planned_exit_date", "entry_close", "next_open", "next_high", "next_low", "next_close",
                         "label_status", "screen_to_entry_return", *SIGNALS]
        pred = frame[fields].assign(target=target, probability=probability, expected_net=expected_net, net=net,
                                    event_hit=event.mask(net.isna()), split="purged")
        for split, mask in masks.items():
            pred.loc[mask, "split"] = split
        predictions.append(pred)
        cal = pred.loc[masks["calibration"]]
        days_cal = calendar[(calendar >= "2025-09-01") & (calendar < "2025-09-30")]
        grid = []
        for p in protocol["grid"]["probability"]:
            for rank in protocol["grid"]["rank"]:
                picked = choose(cal, p, rank)
                score = summarize(picked.assign(net=picked.net - .002), days_cal)
                grid.append(dict(target=target, threshold=p, rank=rank, **score))
        grid_frame = pd.DataFrame(grid)
        grids.append(grid_frame)
        acceptable = grid_frame.loc[grid_frame.dates.ge(10)]
        winner = None if acceptable.empty else acceptable.sort_values(
            ["daily_score", "threshold", "rank"], ascending=[False, True, True]).iloc[0]
        for split in ["calibration", "q4_diagnostic"]:
            subset = pred.loc[masks[split]]
            start, end = ("2025-09-01", "2025-09-30") if split == "calibration" else ("2025-10-01", "2025-12-31")
            days = calendar[(calendar >= start) & (calendar < end)]
            picked = subset.iloc[:0].copy() if winner is None else choose(subset, float(winner.threshold), str(winner["rank"]))
            picked["selection"] = "calibration_locked"
            selections.append(picked)
            for policy, rows in [("all_candidates", subset), ("calibration_locked", picked)]:
                for cost in [20, 40]:
                    adjusted = rows.assign(net=rows.net - (cost - 20) / 10000)
                    # All-candidate baseline uses equal-weight daily returns, not a leveraged sum.
                    result = summarize(adjusted, days)
                    if policy == "all_candidates":
                        daily = adjusted.net.fillna(-.10).groupby(adjusted.date).mean().reindex(days, fill_value=0)
                        result.update(daily_penalized_mean=float(daily.mean()),
                                      daily_score=float(daily.mean() - 2 * daily.std() / np.sqrt(len(days))),
                                      allocation=float(rows.date.nunique() / len(days)))
                    summaries.append(dict(target=target, split=split, policy=policy, cost_bps=cost,
                                          threshold=None if winner is None else float(winner.threshold), **result))
            badcases.append(picked.loc[picked.net.isna() | picked.net.le(0) | ~picked.event_hit.fillna(False)])
    pd.concat(predictions).to_parquet(output / "predictions.parquet", index=False)
    pd.concat(selections).to_parquet(output / "picks.parquet", index=False)
    pd.concat(badcases).to_parquet(output / "badcases.parquet", index=False)
    pd.concat(grids).to_csv(output / "calibration_grid.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
    for name, digest in hashes.items():
        if sha256(Path(name)) != digest:
            raise ValueError(f"Source changed during run: {name}")
    completion = dict(rows=len(frame), features=len(columns), partitions=protocol["partitions"],
                      picks_sha256=sha256(output / "picks.parquet"), status="Baseline completed; sequence/fusion/badcase iteration pending")
    (output / "complete.json").write_text(json.dumps(completion, indent=2))
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
