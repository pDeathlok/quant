"""Frozen candidate stability diagnostic; no selection changes or promotion."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
from audit_overnight_right_provenance import sha256

from quant.core.paths import PROJECT_ROOT


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    """Nominal IID 95% interval, NOT dependence/search-adjusted confirmation."""
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("Invalid binomial counts")
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0., centre - half), min(1., centre + half)


def main() -> None:
    root = PROJECT_ROOT / "reports/research"
    source = root / "overnight_unfiltered_limit1_hgb_event_head_v1"
    path = source / "train_old_select_old_picks.parquet"
    label_path = root / "overnight_unfiltered_labels_v1/labels.parquet"
    output = root / "overnight_event_candidate_stability_v1"
    output.mkdir(exist_ok=False)
    hashes = {str(p): sha256(p) for p in [path, label_path, source / "protocol.json", source / "REPORT.md"]}
    original = pd.read_parquet(path)
    frame = original.loc[original.date.ge("2026-01-01")].copy()
    if len(frame) != 30 or frame.date.duplicated().any():
        raise ValueError("Frozen candidate changed")
    match = frame.symbol.eq("002773.SZ") & frame.date.eq("2026-06-08")
    if match.sum() != 1 or frame.net.isna().sum() != 1 or not frame.loc[match, "net"].isna().all():
        raise ValueError("Corporate-action correction target changed")
    labels = pd.read_parquet(label_path, filters=[("symbol", "==", "002773.SZ"),
                                                 ("date", "==", pd.Timestamp("2026-06-08"))])
    row = labels.iloc[0]
    if len(labels) != 1 or (row.entry_close, row.next_pre_close, row.next_high, row.next_close) != (20.63, 19.93, 20.37, 20.12):
        raise ValueError("Corporate-action price evidence changed")
    frame["original_net"] = frame.net
    frame["original_event_hit"] = frame.event_hit
    frame.loc[match, "net"] = (row.next_close + .70 - .14) / row.entry_close - 1 - .002
    frame.loc[match, "event_hit"] = False
    if frame.net.isna().any() or frame.event_hit.isna().any():
        raise ValueError("Other unresolved observations")
    frame.to_parquet(output / "frozen_cash_overlay.parquet", index=False)
    monthly = []
    for month in pd.period_range("2026-01", "2026-09", freq="M"):
        part = frame.loc[frame.date.dt.to_period("M").eq(month)]
        monthly.append({"month": str(month), "signals": len(part), "hits": int(part.event_hit.sum()),
                        "hit_rate": float(part.event_hit.mean()) if len(part) else float("nan"), "mean_net20": float(part.net.mean()),
                        "mean_net40": float((part.net - .002).mean())})
    pd.DataFrame(monthly).to_csv(output / "monthly.csv", index=False)
    results = {"signals": len(frame), "hits": int(frame.event_hit.sum()),
               "nominal_iid_hit_interval95": wilson_interval(int(frame.event_hit.sum()), len(frame)),
               "wins20": int(frame.net.gt(0).sum()),
               "nominal_iid_win20_interval95": wilson_interval(int(frame.net.gt(0).sum()), len(frame)),
               "mean_net20": float(frame.net.mean()), "mean_net40": float((frame.net - .002).mean()),
               "worst_net20": float(frame.net.min()), "without_best3_mean_net20": float(frame.net.sort_values().iloc[:-3].mean()),
               "symbols": int(frame.symbol.nunique()), "sources": hashes,
               "correction_source": "https://disc.static.szse.cn/download/disc/disk03/finalpage/2026-06-01/a0b530ab-874f-4028-b98b-081b7d9076ae.PDF",
               "correction_assumption": "New ordinary individual holding, no prior inventory; cash0.70 minus short-hold tax0.14; nominal sell limit unchanged",
               "status": "Post-selection reused-history diagnostic; no independent confirmation; original artifacts untouched",
               "script_sha256": sha256(Path(__file__))}
    for filename, digest in hashes.items():
        if sha256(Path(filename)) != digest:
            raise ValueError("Source changed during audit")
    (output / "complete.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: v for k, v in results.items() if k not in {"sources", "correction_source"}}, indent=2))


if __name__ == "__main__":
    main()
