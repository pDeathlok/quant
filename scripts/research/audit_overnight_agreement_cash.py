"""Separate cash-dividend overlay for the frozen agreement signal; no reselection."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from audit_overnight_right_provenance import sha256

from quant.core.paths import PROJECT_ROOT

SOURCE_URL = "https://disc.static.szse.cn/disc/disk03/finalpage/2026-07-15/f63ee9bc-992f-489e-a6af-b4de440185e0.PDF"


def main() -> None:
    root = PROJECT_ROOT / "reports/research"
    original = root / "overnight_model_agreement_v1/outcomes.parquet"
    proof_path = original.parent / "complete.json"
    if sha256(original) != json.loads(proof_path.read_text())["outcomes_sha256"]:
        raise ValueError("Frozen outcome file changed")
    label_path = root / "overnight_unfiltered_labels_v1/labels.parquet"
    limit_path = root / "overnight_unfiltered_limits_v1/20260721.parquet"
    paths = [original, proof_path, label_path, limit_path]
    hashes = {str(p): sha256(p) for p in paths}
    result = pd.read_parquet(original)
    match = result.symbol.eq("001979.SZ") & result.date.eq("2026-07-20")
    if match.sum() != 1 or result.net.isna().sum() != 1 or not result.loc[match, "net"].isna().all():
        raise ValueError("Unexpected unresolved signal")
    labels = pd.read_parquet(label_path, filters=[("symbol", "==", "001979.SZ"), ("date", "==", pd.Timestamp("2026-07-20"))])
    if len(labels) != 1:
        raise ValueError("Missing source label")
    row = labels.iloc[0]
    np.testing.assert_allclose([row.entry_close, row.next_open, row.next_high, row.next_low, row.next_close, row.next_pre_close],
                               [6.97, 6.90, 7.09, 6.85, 6.92, 6.92], rtol=0, atol=1e-12)
    if row.exit_date != pd.Timestamp("2026-07-21") or row.label_status != "corporate_action_unresolved":
        raise ValueError("Different corporate action boundary")
    # Official implementation notice: 0.5110/10 gross,0.1022/10 tax
    # for personal holdings <=1month. Assume new position, no FIFO old lots.
    cash = (.5110 - .1022) / 10
    target = np.ceil(row.entry_close * 1.01 * 100 - 1e-9) / 100
    hit = row.next_high >= target + row.entry_close * .001
    limits = pd.read_parquet(limit_path)
    limit = limits.loc[limits.ts_code.eq("001979.SZ")]
    if len(limit) != 1 or not limit.trade_date.astype(str).eq("20260721").all():
        raise ValueError("Missing exact exit limits")
    limit = limit.iloc[0]
    if not (limit.down_limit > 0 and limit.up_limit > 0 and row.next_low >= limit.down_limit - .011
            and row.next_high <= limit.up_limit + .011 and hit):
        raise ValueError("Unresolved exit path or bounds")
    net = (target + cash) / row.entry_close - 1 - .002
    result["original_net"] = result.net
    result["original_event_hit"] = result.event_hit
    result.loc[match, "net"] = net
    result.loc[match, "event_hit"] = bool(hit)
    # Original exit audit stays untouched; this overlay is explicitly separate.
    output = root / "overnight_model_agreement_cash_overlay_v1"
    output.mkdir(exist_ok=False)
    result.to_parquet(output / "outcomes.parquet", index=False)
    period = result.loc[result.date.ge("2026-01-01")]
    summary = {"signals_2026": len(period), "hits_2026": int(period.event_hit.sum()),
               "hit_rate_2026": float(period.event_hit.mean()), "mean_net20": float(period.net.mean()),
               "mean_net40": float((period.net - .002).mean()), "cash_net_per_share": cash,
               "sell_limit": target, "corrected_signal_net20": net, "source_url": SOURCE_URL,
               "assumption": "New personal holding<=1month, no prior FIFO inventory; no nominal target adjustment",
               "status": "Descriptive cash overlay only; no independent confidence or fill guarantee"}
    for path, digest in hashes.items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"Source changed: {path}")
    (output / "sources.json").write_text(json.dumps(hashes, indent=2))
    (output / "complete.json").write_text(json.dumps({**summary, "dataset_sha256": sha256(output / "outcomes.parquet"),
                                                     "script_sha256": sha256(Path(__file__))}, indent=2))
    (output / "REPORT.md").write_text(
        "# Separate dividend correction for frozen agreement signal\n\n"
        f"[Official implementation notice]({SOURCE_URL}): record2026-07-20, ex/pay2026-07-21; "
        "gross0.0511/share, personal<=1month tax0.01022/share. New holding/no prior FIFO inventory assumed.\n\n"
        f"Nominal sell limit remains{target:.2f}; actual high7.09 exceeds required penetration. "
        f"Net cash{cash:.5f}/share is added to sale proceeds, not to the high or trigger.\n\n"
        f"2026: {len(period)} signals, {int(period.event_hit.sum())} event hits, "
        f"mean net20={period.net.mean():.4%}, net40={(period.net - .002).mean():.4%}. "
        "Still below80%, tiny sample and reused history. Original labels/picks/audit flags remain unchanged.\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
