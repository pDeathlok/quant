"""Fixed two-training-window agreement diagnostic; never tune on outcomes."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from audit_overnight_right_provenance import sha256

from quant.core.paths import PROJECT_ROOT


def consensus(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    columns = ["symbol", "date", "old_pool", "probability", "expected_net", "risk"]
    joined = left[columns].merge(right[columns], on=["symbol", "date"], how="left",
                                 validate="one_to_one", suffixes=("_a", "_b"), indicator=True)
    if not joined["_merge"].eq("both").all() or not joined.old_pool_a.eq(joined.old_pool_b).all():
        raise ValueError("Different candidate keys or old pool membership")
    mask = joined.old_pool_a & joined.probability_a.ge(.7) & joined.probability_b.ge(.7)
    mask &= joined.risk_a.le(.1) & joined.risk_b.le(.1)
    picks = joined.loc[mask].copy()
    picks["agreement_score"] = picks[["probability_a", "probability_b"]].min(axis=1)
    picks["mean_expected_net"] = picks[["expected_net_a", "expected_net_b"]].mean(axis=1)
    picks["maximum_risk"] = picks[["risk_a", "risk_b"]].max(axis=1)
    return picks.sort_values(["date", "agreement_score", "mean_expected_net", "symbol"],
                             ascending=[True, False, False, True]).drop_duplicates("date").drop(columns="_merge")


def main() -> None:
    root = PROJECT_ROOT / "reports/research"
    dirs = [root / "overnight_unfiltered_limit1_hgb_event_head_v1",
            root / "overnight_unfiltered_limit1_hgb_early_calibration_v1"]
    paths = [d / name for d in dirs for name in ["old_predictions.parquet", "protocol.json", "complete.json"]]
    hashes = {str(p): sha256(p) for p in paths}
    for directory in dirs:
        protocol = json.loads((directory / "protocol.json").read_text())
        for path, digest in protocol["sources"].items():
            if sha256(Path(path)) != digest:
                raise ValueError(f"Model input changed: {path}")
    output = root / "overnight_model_agreement_v1"
    output.mkdir(exist_ok=False)
    lock = {"locked_at_utc": datetime.now(timezone.utc).isoformat(), "sources": hashes,
            "rule": "Both old-pool-trained models: P>=0.7 and risk<=0.1; old selection pool; dailyTop1 by minimumP then mean expected net then symbol",
            "score": "Minimum of two calibrated model probabilities is a ranking score, NOT independently calibrated confidence",
            "period": "Common evaluation2025Q4 and2026 throughSeptember3; no2025Q3 which calibrates modelA",
            "exit": "Unchanged1% nominal limit with10bps penetration, otherwise next close;20/40bps costs",
            "status": "Reused-history diagnostic; no independent confirmation and no parameter sweep"}
    (output / "protocol.json").write_text(json.dumps(lock, indent=2))
    columns = ["symbol", "date", "old_pool", "probability", "expected_net", "risk"]
    left, right = [pd.read_parquet(d / "old_predictions.parquet", columns=columns) for d in dirs]
    picks = consensus(left, right)
    picks.to_parquet(output / "frozen_picks.parquet", index=False)
    frozen_sha = sha256(output / "frozen_picks.parquet")
    labels = ["symbol", "date", "net", "event_hit", "limit1_exit_uncertain", "limit1_net_after_exit_audit"]
    a, b = [pd.read_parquet(d / "old_predictions.parquet", columns=labels) for d in dirs]
    common = a.merge(b, on=["symbol", "date"], how="left", validate="one_to_one", suffixes=("", "_b"))
    for column in labels[2:]:
        pd.testing.assert_series_equal(common[column], common[f"{column}_b"], check_names=False)
    results = picks.merge(a, on=["symbol", "date"], how="left", validate="one_to_one")
    results.to_parquet(output / "outcomes.parquet", index=False)
    summaries = []
    for period, lower, upper in [("2025Q4", "2025-10-01", "2026-01-01"), ("2026", "2026-01-01", "2026-09-04")]:
        part = results.loc[results.date.ge(lower) & results.date.lt(upper)]
        summaries.append({"period": period, "signals": len(part), "event_hits": int(part.event_hit.fillna(False).sum()),
                          "event_rate_missing_failure": float(part.event_hit.fillna(False).mean()),
                          "unknown": int(part.net.isna().sum()), "exit_uncertain": int(part.limit1_exit_uncertain.sum()),
                          "profitable20": int(part.net.gt(0).sum()), "mean_net20": float(part.net.mean()),
                          "mean_net40": float((part.net - .002).mean()), "worst_net20": float(part.net.min())})
    pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
    for path, digest in hashes.items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"Source changed: {path}")
    if sha256(output / "frozen_picks.parquet") != frozen_sha:
        raise ValueError("Selection changed after label access")
    report = ["# Two-model agreement diagnostic", "", lock["rule"], "", lock["score"], ""]
    for row in summaries:
        report.append(f"{row['period']}: {row['signals']} signals, {row['event_hits']} event hits, "
                      f"mean net20={row['mean_net20']:.4%}, net40={row['mean_net40']:.4%}; "
                      f"unknown={row['unknown']}, exit uncertainty={row['exit_uncertain']}.")
    report += ["", "These models share features and many training samples; agreement is not independent evidence. "
               "Small samples or reused periods cannot establish80%/90% opportunity precision. No14:30 or actual-fill claim."]
    (output / "REPORT.md").write_text("\n".join(report))
    (output / "complete.json").write_text(json.dumps({"summary": summaries, "frozen_picks_sha256": frozen_sha,
        "outcomes_sha256": sha256(output / "outcomes.parquet"), "script_sha256": sha256(Path(__file__))}, indent=2))
    print(json.dumps(summaries), flush=True)


if __name__ == "__main__":
    main()
