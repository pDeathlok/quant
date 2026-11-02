"""Independently replay fixed selections and cost metrics for the earlier-fit experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from audit_overnight_right_provenance import sha256

from quant.core.paths import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--event-calendar", action="store_true")
    group.add_argument("--event-xgboost", action="store_true")
    args = parser.parse_args()
    original_windows = args.event_calendar or args.event_xgboost
    root = PROJECT_ROOT / "reports/research/overnight_unfiltered_limit1_hgb_early_calibration_v1"
    if args.event_calendar:
        root = PROJECT_ROOT / "reports/research/overnight_unfiltered_limit1_hgb_event_calendar_v1"
    if args.event_xgboost:
        root = PROJECT_ROOT / "reports/research/overnight_unfiltered_limit1_xgboost_event_head_v1"
    if not (root / "complete.json").exists():
        raise ValueError("Training is still incomplete")
    output = PROJECT_ROOT / "reports/research/overnight_early_calibration_audit_v1"
    if args.event_calendar:
        output = PROJECT_ROOT / "reports/research/overnight_event_calendar_audit_v1"
    if args.event_xgboost:
        output = PROJECT_ROOT / "reports/research/overnight_event_xgboost_audit_v1"
    paths = [root / "protocol.json", root / "complete.json", root / "diagnostics.csv"]
    paths += [root / f"{arm}_{name}.parquet" for arm in ["old", "expanded"] for name in ["predictions", "calibration_support"]]
    paths += [root / f"train_{arm}_select_{pool}_picks.parquet" for arm in ["old", "expanded"] for pool in ["old", "expanded"]]
    hashes = {str(p): sha256(p) for p in paths}
    protocol = json.loads((root / "protocol.json").read_text())
    if original_windows:
        baseline = PROJECT_ROOT / "reports/research/overnight_unfiltered_limit1_hgb_event_head_v1"
        baseline_protocol_path = baseline / "protocol.json"
        hashes[str(baseline_protocol_path)] = sha256(baseline_protocol_path)
        original = json.loads(baseline_protocol_path.read_text())
        fields = ["target", "rule", "fit", "calibration", "evaluations", "costs_bps", "unknown", "exit_policy"]
        if args.event_calendar:
            fields.append("params")
        for field in fields:
            if protocol[field] != original[field]:
                raise ValueError(f"Ablation changed {field}")
        if args.event_calendar and (protocol["features"] != [*original["features"], *protocol["calendar_features"]] or len(protocol["calendar_features"]) != 9):
            raise ValueError("Unexpected calendar schema")
        if args.event_xgboost:
            fixed_path = PROJECT_ROOT / "reports/research/overnight_unfiltered_net_open_xgboost_v1/protocol.json"
            hashes[str(fixed_path)] = sha256(fixed_path)
            fixed = json.loads(fixed_path.read_text())
            if protocol["features"] != original["features"] or protocol["params"] != fixed["params"] or protocol["backend"] != "xgboost":
                raise ValueError("XGBoost configuration/schema changed")
    for path, digest in protocol["sources"].items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"Training source changed: {path}")
    diagnostics = pd.read_csv(root / "diagnostics.csv")
    periods = [("2025Q3", "2025-07-01", "2025-10-01"), ("2025Q4", "2025-10-01", "2026-01-01"),
               ("2026", "2026-01-01", "2026-09-04")]
    calibration_start, calibration_end = "2025-01-01", "2025-07-01"
    if original_windows:
        periods = periods[1:]
        calibration_start, calibration_end = "2025-07-01", "2025-10-01"
    summaries, support_rows = [], []
    checks = 0
    for arm in ["old", "expanded"]:
        support = pd.read_parquet(root / f"{arm}_calibration_support.parquet")
        if not support.date.ge(calibration_start).all() or not support.date.lt(calibration_end).all():
            raise ValueError("Unexpected calibration dates")
        supported = support.loc[support.probability.ge(.7) & support.risk.le(.1)]
        support_rows.append({"arm": arm, "rows": len(supported), "dates": int(supported.date.nunique()),
                             "event_hit_rate": float(supported.event_hit.mean()),
                             "largest_date_share": float(supported.groupby("date").size().max() / len(supported)) if len(supported) else np.nan})
        pred = pd.read_parquet(root / f"{arm}_predictions.parquet")
        if original_windows:
            source = baseline / f"{arm}_predictions.parquet"
            hashes[str(source)] = sha256(source)
            label_columns = ["symbol", "date", "exit_date", "old_pool", "net", "event_hit", "limit1_exit_uncertain", "limit1_net_after_exit_audit"]
            old_labels = pd.read_parquet(source, columns=label_columns)
            pd.testing.assert_frame_equal(pred[label_columns].reset_index(drop=True), old_labels.reset_index(drop=True))
        for pool in ["old", "expanded"]:
            candidates = pred.loc[pred.old_pool] if pool == "old" else pred
            wanted = candidates.loc[candidates.probability.ge(.7) & candidates.risk.le(.1) & candidates.expected_net.ge(-np.inf)]
            wanted = wanted.sort_values(["date", "probability", "expected_net", "symbol"], ascending=[True, False, False, True])
            wanted = wanted.drop_duplicates("date", keep="first")
            saved = pd.read_parquet(root / f"train_{arm}_select_{pool}_picks.parquet")
            pd.testing.assert_frame_equal(wanted.reset_index(drop=True), saved.reset_index(drop=True))
            for period, lower, upper in periods:
                part = saved.loc[saved.date.ge(lower) & saved.date.lt(upper)]
                summaries.append({"train_pool": arm, "selection_pool": pool, "period": period,
                                  "signals": len(part), "event_hits": int(part.event_hit.fillna(False).sum()),
                                  "event_rate_missing_failure": float(part.event_hit.fillna(False).mean()),
                                  "unknown_events": int(part.event_hit.isna().sum()),
                                  "exit_uncertain": int(part.limit1_exit_uncertain.sum()),
                                  "mean_net20": float(part.net.mean()), "mean_net40": float((part.net - .002).mean()),
                                  "worst_net20": float(part.net.min())})
                for view, column in [("raw_mark", "net"), ("exit_audited", "limit1_net_after_exit_audit")]:
                    for cost in [20, 40]:
                        values = part[column] - (cost - 20) / 10000
                        row = diagnostics.loc[diagnostics.train_pool.eq(arm) & diagnostics.selection_pool.eq(pool)
                            & diagnostics.period.eq(period) & diagnostics.view.eq(view) & diagnostics.cost_bps.eq(cost)]
                        if len(row) != 1:
                            raise ValueError("Missing or duplicate metric row")
                        row = row.iloc[0]
                        assert row.signals == len(part)
                        assert row.unknown == values.isna().sum()
                        np.testing.assert_allclose(row.known_mean_net, values.mean(), rtol=1e-12, atol=1e-12)
                        np.testing.assert_allclose(row.win_missing_as_failure, values.gt(0).mean(), rtol=1e-12, atol=1e-12)
                        checks += 1
    if checks != len(diagnostics):
        raise ValueError("Unaudited metric rows")
    output.mkdir(exist_ok=False)
    pd.DataFrame(summaries).to_csv(output / "event_summary.csv", index=False)
    pd.DataFrame(support_rows).to_csv(output / "calibration_support.csv", index=False)
    for path, digest in hashes.items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"Changed audit source: {path}")
    (output / "sources.json").write_text(json.dumps(hashes, indent=2))
    summary = {"selection_files_replayed": 4, "cost_metric_rows_replayed": checks,
               "script_sha256": sha256(Path(__file__)),
               "status": "No robust80%/90% opportunity established; historical periods reused, independent confirmation remains absent"}
    (output / "complete.json").write_text(json.dumps(summary, indent=2))
    report = ["# Earlier training / half-year calibration diagnostic", "",
              "2024 training,2025H1 calibration, fixed probability>=0.7 / risk<=0.1 / dailyTop1. No threshold search.", "",
              "| Training pool | Selection pool | Period | Signals | Event hits | Mean net20 | Mean net40 |",
              "|---|---|---|---:|---:|---:|---:|"]
    if args.event_calendar:
        report[0] = "# Event model with nine target-calendar inputs"
        report[2] = "Original training/calibration dates,187+9 features, fixed probability>=0.7 / risk<=0.1 / dailyTop1. Labels and evaluation keys match baseline exactly."
    if args.event_xgboost:
        report[0] = "# Fixed XGBoost event-model comparison"
        report[2] = "Original187 features and training/calibration dates; fixed existing XGBoost configuration. Labels and evaluation keys match HGB baseline exactly. No parameter search."
    for row in summaries:
        report.append(f"| {row['train_pool']} | {row['selection_pool']} | {row['period']} | {row['signals']} | "
                      f"{row['event_hits']} | {row['mean_net20']:.4%} | {row['mean_net40']:.4%} |")
    report += ["", "Calibration support is descriptive, not out-of-time precision. More supported dates did not establish robust returns and event precision. "
               "Sparse earlier-period signals, historical reuse, and daily-close rather than14:30 fills prevent a success claim. "
               "Event penetration and positive net returns are different metrics. Unknown outcomes remain in denominators.", ""]
    (output / "REPORT.md").write_text("\n".join(report))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
