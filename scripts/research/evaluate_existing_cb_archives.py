"""Review every archived CB equity curve without reselecting production rules."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_existing_strategy_regimes import OUT, ROOT, dates, daily_sources
from run_strategy_lab import metrics


def main() -> None:
    market = pd.read_csv(OUT / "market_context.csv", parse_dates=["date"])
    states = market.set_index("date")[["market_fast", "market_slow"]].shift(1)
    paths = sorted((ROOT / "reports/convertible_bond").rglob("equity.csv"))
    curves, inventory = {}, []
    common = None
    for path in paths:
        frame = pd.read_csv(path)
        frame["date"] = dates(frame.trade_date)
        frame = frame.sort_values("date").set_index("date")
        assert not frame.index.duplicated().any()
        summary_path = path.with_name("summary.json")
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        initial = summary.get("config", {}).get("initial_cash", 1_000_000)
        values = frame.equity.pct_change(fill_method=None)
        values.iloc[0] = frame.equity.iloc[0] / initial - 1
        assert np.allclose(initial * (1 + values).cumprod(), frame.equity, rtol=1e-10, atol=1e-8)
        name = str(path.parent.relative_to(ROOT / "reports/convertible_bond"))
        exposure = frame.exposure if "exposure" in frame else pd.Series(np.nan, index=frame.index)
        curves[name] = pd.DataFrame({"daily_return": values, "exposure": exposure})
        available = frame.index[frame.index >= pd.Timestamp("2024-01-03")]
        common = available if common is None else common.intersection(available)
        inventory.append({"scenario": name, "source": str(path.relative_to(ROOT)),
                          "family": name.split("/")[0], "start": frame.index.min(), "end": frame.index.max(),
                          "days": len(frame), "initial_capital": initial,
                          "fee_reconciliation_gap": float((1 + frame.daily_return).prod() - frame.equity.iloc[-1] / initial),
                          "source_caveats": json.dumps(summary.get("caveats", []), ensure_ascii=False)})
    common = common.sort_values()
    assert len(common) >= 230
    rows, strata = [], []
    for name, frame in curves.items():
        for period, sample in (("native", frame), ("cb_common_2024_on", frame.reindex(common)),
                               ("2026", frame.loc["2026-01-01":])):
            rows.append({"scenario": name, "family": name.split("/")[0], "period": period,
                         "start": sample.index.min(), "end": sample.index.max(), "days": len(sample),
                         **metrics(sample.daily_return, sample.exposure)})
        work = frame.join(states)
        for horizon in states.columns:
            for state, group in work.groupby(horizon):
                strata.append({"scenario": name, "horizon": horizon, "state": state, "days": len(group),
                               "mean_daily_return": group.daily_return.mean(),
                               "mean_exposure": group.exposure.mean(),
                               "positive_day_rate": group.daily_return.gt(0).mean()})
    pd.DataFrame(inventory).to_csv(OUT / "cb_archive_inventory.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "cb_archive_comparison.csv", index=False)
    pd.DataFrame(strata).to_csv(OUT / "cb_archive_market_strata.csv", index=False)
    benchmarks = []
    for name, frame in daily_sources().items():
        sample = frame.reindex(common)
        assert sample.daily_return.notna().all()
        benchmarks.append({"strategy": name, "start": common.min(), "end": common.max(), "days": len(common),
                           **metrics(sample.daily_return, sample.exposure)})
    pd.DataFrame(benchmarks).to_csv(OUT / "cb_common_other_benchmarks.csv", index=False)
    info = {"curves": len(curves), "common_days": len(common), "common_start": str(common.min().date()),
            "common_end": str(common.max().date()), "status": "archive diagnostics; execution and premium audit pending"}
    (OUT / "cb_archive_manifest.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
    print(json.dumps(info), flush=True)
    print(pd.DataFrame(rows).query("period=='native'").nlargest(4, "annual_return")
          [["scenario", "annual_return", "max_drawdown"]].to_string(index=False))


if __name__ == "__main__":
    main()
