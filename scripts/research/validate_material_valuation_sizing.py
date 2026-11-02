"""Monthly沪深300 ETF allocation from point-in-time index PE history."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/public_strategy_validation_20260929/material_backtests"
spec = importlib.util.spec_from_file_location("public_validation", ROOT / "scripts/research/validate_public_strategies.py")
validation = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(validation)


def simulate(returns: pd.Series, targets: dict[pd.Timestamp, float]) -> tuple[pd.Series, float]:
    """Close signal, execute at next close; cash earns zero; 10bp per traded notional."""
    etf_weight, wealth, pending, turnover_total = 0.0, 1.0, None, 0.0
    output = []
    for day, asset_return in returns.items():
        previous = wealth
        portfolio_return = etf_weight * asset_return
        wealth *= 1 + portfolio_return
        etf_weight = etf_weight * (1 + asset_return) / (1 + portfolio_return)
        if pending is not None:
            turnover = abs(pending - etf_weight)
            wealth *= 1 - 0.001 * turnover
            turnover_total += turnover
            etf_weight = pending
            pending = None
        if day in targets:
            pending = targets[day]
        output.append(wealth / previous - 1)
    return pd.Series(output, index=returns.index), turnover_total


def run() -> pd.DataFrame:
    index = pd.read_parquet(OUT / "source/index_dailybasic_000300_SH.parquet")
    index["date"] = pd.to_datetime(index.trade_date)
    pe = index.set_index("date").sort_index().pe_ttm.astype(float)
    etf = pd.read_parquet(ROOT / "reports/public_strategy_validation_20260929/source/510300_SH_2018_2026.parquet")
    etf["date"] = pd.to_datetime(etf.trade_date)
    adjusted = etf.set_index("date").sort_index().eval("close * adj_factor")
    asset_returns = adjusted.pct_change(fill_method=None).loc["2019-01-01":"2026-09-24"].dropna()

    # Rolling rank uses only observations on or before the signal close.
    percentile = pe.rolling(756, min_periods=252).apply(
        lambda x: np.count_nonzero(x <= x[-1]) / len(x), raw=True
    )
    observation = pd.DataFrame({"pe_ttm": pe, "percentile_3y": percentile})
    observation = observation.reindex(asset_returns.index).ffill()
    month_end = observation.groupby(observation.index.to_period("M")).tail(1)
    month_end = month_end.loc["2019-01-01":].dropna()
    rank_exposure = pd.Series(np.select(
        [month_end.percentile_3y <= 0.2, month_end.percentile_3y >= 0.6],
        [0.8, 0.3], default=0.6,
    ), index=month_end.index)
    absolute_exposure = pd.Series(np.select(
        [month_end.pe_ttm < 12, month_end.pe_ttm > 50],
        [0.8, 0.3], default=0.6,
    ), index=month_end.index)
    signals = month_end.copy()
    signals["rank_exposure"] = rank_exposure
    signals["absolute_exposure"] = absolute_exposure
    signals.to_csv(OUT / "valuation_position_signals.csv", index_label="date")

    targets = {
        "fixed_60pct": {day: 0.6 for day in month_end.index},
        "full_100pct": {day: 1.0 for day in month_end.index},
        "rolling_pe_percentile": rank_exposure.to_dict(),
        "absolute_pe_12_50": absolute_exposure.to_dict(),
    }
    returns, rows = {}, []
    for name, scenario in targets.items():
        series, turnover = simulate(asset_returns, scenario)
        series = series.loc["2019-02-01":]
        returns[name] = series
        for period, sample in (("full", series), ("2023_onward", series.loc["2023-01-01":])):
            rows.append({"scenario": name, "period": period, **validation.metrics(sample),
                         "turnover_notional_full": turnover})
    pd.DataFrame(returns).to_csv(OUT / "valuation_position_daily_returns.csv", index_label="date")
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "valuation_position_comparison.csv", index=False)
    (OUT / "valuation_position_notes.json").write_text(json.dumps({
        "data": "Tushare index_dailybasic 000300.SH pe_ttm and fund_daily/fund_adj 510300.SH",
        "signal": "Last ETF trading close each month: rolling 756-day PE percentile, min 252 observations; <=20% 80% ETF, >=60% 30% ETF, otherwise 60% ETF",
        "absolute_comparator": "PE<12: 80% ETF; PE>50: 30% ETF; otherwise 60% ETF",
        "execution": "Signal at month-end close, execute at next close, earn new position from following session; 0.10% per traded notional; zero cash return",
        "limitations": ["One index and one ETF only", "No interest on cash", "PE data begins 2018, so initial percentile window has fewer than 756 observations", "Parameters are exploratory and were chosen after seeing this history"],
        "signal_counts": {"rolling": rank_exposure.value_counts().sort_index().to_dict(),
                          "absolute": absolute_exposure.value_counts().sort_index().to_dict()},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(run()[["scenario", "period", "annual_return", "max_drawdown", "turnover_notional_full"]].to_string(index=False))
