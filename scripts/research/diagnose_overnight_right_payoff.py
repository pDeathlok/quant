"""Q3-only payoff diagnostics; no new selection, no deletion of losing trades."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from audit_overnight_right_provenance import sha256
from audit_overnight_same_day_left import quarter_mask

from quant.core.paths import PROJECT_ROOT
from quant.research.right_side_unified_signals import CANONICAL_RIGHT_SIDE_SIGNALS


def breakdown(frame: pd.DataFrame, cost_bps: int) -> dict:
    net = frame.net - (cost_bps - 20) / 10000
    wins, losses = net.gt(0), net.lt(0)
    gain, loss = net[wins].mean(), -net[losses].mean()
    known = net.notna()
    open_known = known & frame.next_open.gt(0)
    open_return = frame.next_open / frame.entry_close - 1
    closing_drift = (frame.next_close - frame.next_open) / frame.entry_close
    negative_at_open = losses & open_known & open_return.lt(0)
    negative_after_open = losses & open_known & open_return.ge(0)
    contribution = net.fillna(0)
    if not np.isclose(contribution.sum(), net[wins].sum() + net[losses].sum()):
        raise ValueError("Payoff decomposition failed")
    return {
        "cost_bps": cost_bps, "trades": len(frame), "dates": int(frame.date.nunique()),
        "unknown": int((~known).sum()), "wins": int(wins.sum()), "losses": int(losses.sum()),
        "known_win_rate": float(wins.sum() / known.sum()) if known.any() else None,
        "mean_gain": float(gain) if wins.any() else None,
        "mean_loss_magnitude": float(loss) if losses.any() else None,
        "break_even_win_rate_nonzero": float(loss / (gain + loss)) if wins.any() and losses.any() else None,
        "known_mean_net": float(net.mean()), "worst_net": float(net.min()),
        "loss_over_2pct_count": int(net.lt(-.02).sum()),
        "loss_over_2pct_share_of_loss_magnitude": float(-net[net.lt(-.02)].sum() / -net[losses].sum()) if losses.any() else None,
        "losers_already_below_entry_at_open": int(negative_at_open.sum()),
        "losers_not_below_entry_at_open": int(negative_after_open.sum()),
        "losers_unknown_open": int((losses & ~open_known).sum()),
        "losers_mean_open_return": float(open_return[losses & open_known].mean()),
        "losers_mean_open_to_close_change_per_entry": float(closing_drift[losses & open_known].mean()),
        "same_members_open_net_mean": float((open_return[open_known] - cost_bps / 10000).mean()),
        "same_members_open_win_rate": float(open_return[open_known].gt(cost_bps / 10000).mean()),
        "exit_ambiguous": int(frame.exit_ambiguous.sum()),
    }


def main() -> None:
    parent = PROJECT_ROOT / "reports/research/overnight_same_day_right_v1"
    output = PROJECT_ROOT / "reports/research/overnight_right_payoff_diagnostic_v1"
    if not (parent / "complete.json").exists():
        raise ValueError("Parent unfinished")
    output.mkdir(exist_ok=False)
    paths = sorted((PROJECT_ROOT / "data/raw/daily_partitioned").glob("year_month=20250[789]/data.parquet"))
    hashes = {str(p): sha256(p) for p in paths}
    protocol = {"scope": "All 14 signals, both preset targets, Q3 only, fixed parent selections",
                "purpose": "Diagnostic, not a selectable or independently validated strategy",
                "opening_exit": "Counterfactual daily mark on same members; not a verified auction fill",
                "unknown": "Kept in counts; decomposition of known marks is not whole-population realized PnL",
                "raw_sha256": hashes, "script_sha256": sha256(Path(__file__))}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    bars = ds.dataset([str(p) for p in paths], format="parquet").to_table(
        columns=["ts_code", "trade_date", "open"]).to_pandas()
    bars = bars.rename(columns={"ts_code": "symbol", "open": "next_open"})
    bars["exit_date"] = pd.to_datetime(bars.trade_date)
    original = json.loads((parent / "q3_rules.json").read_text())
    records = []
    for target in (50, 100):
        source = parent / f"tp{target}_outcomes.parquet"
        hashes[str(source)] = sha256(source)
        frame = pd.read_parquet(source, filters=[("date", ">=", pd.Timestamp("2025-07-01")),
                                                 ("date", "<", pd.Timestamp("2025-10-01"))])
        frame = frame.loc[quarter_mask(frame, "2025Q3")].copy()
        frame = frame.merge(bars[["symbol", "exit_date", "next_open"]],
                            on=["symbol", "exit_date"], how="left", validate="many_to_one")
        for signal in CANONICAL_RIGHT_SIDE_SIGNALS:
            selected = frame.loc[frame[signal].eq(True)]
            for cost in (20, 40):
                result = breakdown(selected, cost)
                baseline = next(r for r in original if r["signal"] == signal and r["take_profit"] == target / 10000)[f"cost{cost}"]
                if (result["trades"] != baseline["signals"] or result["unknown"] != baseline["unknown"]
                        or not np.isclose(result["known_mean_net"], baseline["known_mean_net"], equal_nan=True)):
                    raise ValueError("Frozen Q3 selection/returns mismatch")
                records.append({"signal": signal, "target_bps": target, **result})
    for path, digest in hashes.items():
        if sha256(Path(path)) != digest:
            raise ValueError("Input changed")
    (output / "sources.json").write_text(json.dumps(hashes, indent=2))
    pd.DataFrame(records).to_csv(output / "breakdown.csv", index=False)
    (output / "complete.json").write_text(json.dumps({"rows": len(records), "parent_replay": "passed", "Q4_read": False}))
    print(pd.DataFrame(records).loc[lambda f: f.signal.eq("KENGQI")].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
