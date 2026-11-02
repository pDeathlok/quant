"""Describe Q3 support behind frozen direct-exit calibration; no rule search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from audit_overnight_right_provenance import sha256
from sklearn.isotonic import IsotonicRegression
from threadpoolctl import threadpool_limits
from train_overnight_direct import metrics, select
from train_overnight_unfiltered_comparison import date_weights, partition_masks

from quant.core.paths import PROJECT_ROOT


def support(frame: pd.DataFrame) -> dict[str, float | int]:
    """Concentration is descriptive, not an effective independent sample size."""
    counts = frame.groupby("date").size()
    return {
        "rows": len(frame), "dates": len(counts),
        "largest_date_share": float(counts.max() / len(frame)) if len(frame) else np.nan,
        "wins": int(frame.net.gt(0).sum()),
        "win_rate": float(frame.net.gt(0).mean()),
        "mean_net": float(frame.net.mean()),
    }


def chronological_calibrators(
    scored: pd.DataFrame, boundary: str,
) -> tuple[IsotonicRegression, IsotonicRegression]:
    """Fit only outcomes resolved strictly before the validation boundary."""
    fit = scored.loc[scored.date.ge("2025-07-01") & scored.date.lt(boundary)
                     & scored.exit_date.lt(boundary) & np.isfinite(scored.net)]
    if fit.empty or not fit.exit_date.gt(fit.date).all():
        raise ValueError("Invalid calibration timing or empty calibration data")
    weights = date_weights(fit)
    calibrated = []
    if "event_hit" in fit and fit.event_hit.isna().any():
        raise ValueError("Known return with unknown event")
    event = fit.event_hit.astype(bool) if "event_hit" in fit else fit.net.gt(0)
    for column, outcome in [("raw_probability", event), ("raw_risk", fit.net.lt(-.02))]:
        if not np.isfinite(fit[column]).all() or outcome.nunique() != 2:
            raise ValueError("Invalid calibration scores or insufficient classes")
        calibrated.append(IsotonicRegression(out_of_bounds="clip").fit(
            fit[column], outcome, sample_weight=weights))
    return calibrated[0], calibrated[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chronological-validation", action="store_true")
    parser.add_argument("--event-candidate", action="store_true")
    args = parser.parse_args()
    root = PROJECT_ROOT / "reports/research"
    output = root / ("overnight_chronological_calibration_v1" if args.chronological_validation
                     else "overnight_probability_support_v1")
    if args.event_candidate:
        output = root / ("overnight_event_candidate_chronological_v1" if args.chronological_validation
                         else "overnight_event_candidate_support_v1")
    output.mkdir(exist_ok=False)
    if args.chronological_validation:
        (output / "protocol.json").write_text(json.dumps({
            "folds": [["2025-08-01", "2025-09-01"], ["2025-09-01", "2025-10-01"]],
            "fit_start": "2025-07-01", "fit_exit": "strictly before fold start",
            "validation_exit": "strictly before fold end; unknown stock outcomes retained",
            "rule": "P>=0.7, risk<=0.1, top1; no expected-return gate; no threshold search",
            "models": ("Frozen old-pool limit1 event-head HGB" if args.event_candidate
                       else "Frozen expanded-pool HGB; next-open/close, with/without calendar"),
            "costs_bps": [20, 40], "status": "Reused-history diagnostic; NOT independent confirmation",
        }, indent=2))
    filters = [("date", ">=", pd.Timestamp("2025-07-01")),
               ("date", "<", pd.Timestamp("2025-10-01"))]
    keys = ["symbol", "date"]
    paths = {
        "auxiliary": root / "overnight_unfiltered_auxiliary_v1/dataset.parquet",
        "factor": root / "overnight_unfiltered_full_factors_v1/dataset.parquet",
        "labels": root / "overnight_unfiltered_labels_v1/labels.parquet",
        "gate": root / "overnight_unfiltered_entry_gate_v1/entry_gate.parquet",
        "calendar": root / "overnight_target_calendar_v1/dataset.parquet",
    }
    hashes = {str(path): sha256(path) for path in paths.values()}
    frame = pd.read_parquet(paths["auxiliary"], filters=filters)
    factors = pd.read_parquet(paths["factor"], filters=filters)
    factors = factors.rename(columns={c: f"today_factor_{c}" for c in factors if c not in keys})
    frame = frame.merge(factors, on=keys, validate="one_to_one")
    target_columns = ["limit1", "limit1_penetrated"] if args.event_candidate else ["net_open", "net_close"]
    for kind, columns in [("labels", [*keys, "exit_date", *target_columns]),
                          ("gate", [*keys, "eligible"])]:
        frame = frame.merge(pd.read_parquet(paths[kind], columns=columns, filters=filters),
                            on=keys, validate="one_to_one")
    calendar = pd.read_parquet(paths["calendar"], filters=filters)
    frame = frame.merge(calendar[[*keys, *[c for c in calendar if c.startswith("calendar_target_")]]],
                        on=keys, validate="one_to_one")
    _, mask, _ = partition_masks(frame)
    frame = frame.loc[mask & frame.eligible].reset_index(drop=True)
    if args.event_candidate:
        old_path = root / "tail_overnight/dataset.parquet"
        paths["old_pool"] = old_path
        hashes[str(old_path)] = sha256(old_path)
        frame = frame.merge(pd.read_parquet(old_path, columns=keys, filters=filters),
                            on=keys, validate="one_to_one")
    if not frame.feature_date.lt(frame.date).all():
        raise ValueError("Lag timing mismatch")
    summaries = []
    for target in (["limit1"] if args.event_candidate else ["net_open", "net_close"]):
        for calendar_enabled in ([False] if args.event_candidate else [False, True]):
            suffix = "hgb_calendar" if calendar_enabled else "comparison"
            if args.event_candidate:
                suffix = "hgb_event_head"
            model_root = root / f"overnight_unfiltered_{target}_{suffix}_v1"
            protocol_path = model_root / "protocol.json"
            protocol = json.loads(protocol_path.read_text())
            for path in paths.values():
                if str(path) in protocol["sources"] and hashes[str(path)] != protocol["sources"][str(path)]:
                    raise ValueError(f"Changed source: {path}")
            model_path = model_root / ("old.joblib" if args.event_candidate else "expanded.joblib")
            hashes[str(model_path)] = sha256(model_path)
            hashes[str(protocol_path)] = sha256(protocol_path)
            model = joblib.load(model_path)
            columns = protocol["features"]
            if model["columns"] != columns:
                raise ValueError("Model feature order mismatch")
            sample = frame.copy() if args.chronological_validation else frame.loc[frame[target].notna()].copy()
            x = sample[columns].astype("float32").replace([np.inf, -np.inf], np.nan)
            with threadpool_limits(limits=4):
                raw = model["classifier"].predict_proba(x)[:, 1]
                raw_risk = model["risk"].predict_proba(x)[:, 1]
                expected = model["regressor"].predict(x)
            known = sample[target].notna()
            weights = date_weights(sample.loc[known])
            outcome = sample.limit1_penetrated if args.event_candidate else sample[target].gt(0)
            # Verify reconstruction against both saved Q3 calibrators before counting support.
            for scores, outcome, name in [(raw, outcome, "calibrator"),
                                           (raw_risk, sample[target].lt(-.02), "risk_calibrator")]:
                replay = IsotonicRegression(out_of_bounds="clip").fit(scores[known], outcome[known].astype(bool), sample_weight=weights)
                np.testing.assert_allclose(replay.X_thresholds_, model[name].X_thresholds_, rtol=1e-10)
                np.testing.assert_allclose(replay.y_thresholds_, model[name].y_thresholds_, rtol=1e-10)
            scored = sample[[*keys, "exit_date", target]].rename(columns={target: "net"})
            scored["probability"] = model["calibrator"].predict(raw)
            scored["risk"] = model["risk_calibrator"].predict(raw_risk)
            scored["expected_net"] = expected
            if args.event_candidate:
                scored["event_hit"] = sample.limit1_penetrated
                scored["raw_probability"] = raw
                scored["raw_risk"] = raw_risk
            if args.chronological_validation:
                scored["raw_probability"] = raw
                scored["raw_risk"] = raw_risk
                scored.to_parquet(output / f"{target}_{suffix}_q3.parquet", index=False)
                for lower, upper in [("2025-08-01", "2025-09-01"), ("2025-09-01", "2025-10-01")]:
                    iso, risk_iso = chronological_calibrators(scored, lower)
                    validation = scored.loc[scored.date.ge(lower) & scored.date.lt(upper)
                                            & scored.exit_date.lt(upper)].copy()
                    validation["probability"] = iso.predict(validation.raw_probability)
                    validation["risk"] = risk_iso.predict(validation.raw_risk)
                    days = pd.DatetimeIndex(validation.date.unique()).sort_values()
                    picks = select(validation, .7, -np.inf, 1, .1)
                    picks.to_parquet(output / f"{target}_{suffix}_{lower}_picks.parquet", index=False)
                    joblib.dump({"probability": iso, "risk": risk_iso},
                                output / f"{target}_{suffix}_{lower}_calibrators.joblib")
                    for cost in [20, 40]:
                        adjusted = picks.copy()
                        adjusted["net"] -= (cost - 20) / 10000
                        event_stats = ({"event_hits": int(picks.event_hit.fillna(False).sum()),
                                        "event_unknown": int(picks.event_hit.isna().sum()),
                                        "event_rate_missing_failure": float(picks.event_hit.fillna(False).mean()) if len(picks) else np.nan}
                                       if args.event_candidate else {})
                        summaries.append({"target": target, "calendar": calendar_enabled,
                                          "validation_start": lower, "cost_bps": cost,
                                          "maximum_probability": float(validation.probability.max()),
                                          **metrics(adjusted, days, 1), **event_stats})
                pd.DataFrame(summaries).to_csv(output / "validation.csv", index=False)
                print(f"{target} calendar={calendar_enabled}: chronological validation saved", flush=True)
                continue
            scored.to_parquet(output / f"{target}_{suffix}_q3.parquet", index=False)
            for threshold in [.7, .8, .9, 1.]:
                tail = scored.loc[scored.probability.ge(threshold)]
                filtered = tail.loc[tail.risk.le(.1)]
                top = filtered.sort_values(["date", "probability", "expected_net", "symbol"],
                                           ascending=[True, False, False, True]).drop_duplicates("date")
                for stage, part in [("probability_tail", tail), ("risk_filtered", filtered), ("daily_top1", top)]:
                    event_stats = ({"event_hits": int(part.event_hit.sum()),
                                    "event_rate": float(part.event_hit.mean()) if len(part) else np.nan}
                                   if args.event_candidate else {})
                    summaries.append({"target": target, "calendar": calendar_enabled,
                                      "threshold": threshold, "stage": stage, **support(part), **event_stats})
            pd.DataFrame(summaries).to_csv(output / "support.csv", index=False)
            print(f"{target} calendar={calendar_enabled}: Q3 replay verified, {len(scored)} rows", flush=True)
    for path, digest in hashes.items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"Source changed during audit: {path}")
    (output / "complete.json").write_text(json.dumps({"sources": hashes, "script_sha256": sha256(Path(__file__)),
        "rows": len(summaries), "status": "Descriptive in-calibration support only; no independent success claim"}, indent=2))


if __name__ == "__main__":
    main()
