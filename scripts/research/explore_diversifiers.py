"""Extend the frozen six-sleeve study with accessible non-equity diversifiers."""
from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import explore_six_sleeves as base
import fetch_six_sleeve_exploration as fetcher

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/research/diversifiers_20261111"
NEW_CODES = ("511260.SH", "511090.SH", "159985.SZ", "159981.SZ", "159980.SZ")
LABELS = {"511010.SH": "5年国债ETF", "511260.SH": "10年国债ETF", "511090.SH": "30年国债ETF",
          "159985.SZ": "豆粕期货ETF", "159981.SZ": "能源化工期货ETF", "159980.SZ": "有色金属期货ETF"}


def fetch() -> None:
    from dotenv import dotenv_values
    token = dotenv_values(ROOT / ".env").get("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("Configured Tushare token missing")
    fetcher.OUT = OUT / "source"
    fetcher.OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        manifest = list(pool.map(lambda code: fetcher.fetch_one(code, token), NEW_CODES))
    (fetcher.OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))


def read_panel() -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    p, v, audit = base.load_data()
    for code in NEW_CODES:
        f = pd.read_parquet(OUT / "source" / f"{code.replace('.', '_')}.parquet")
        f.index = pd.to_datetime(f.trade_date)
        if f.index.has_duplicates:
            raise ValueError("Duplicate data")
        f = f.reindex(p.index)
        raw = f.close*f.adj_factor
        p[code] = raw.ffill()
        v[code] = raw.notna() & f.amount.gt(0)
        returns = p[code].pct_change(fill_method=None)
        if returns.abs().max() > .35:
            raise ValueError(f"Inspect adjustment outlier: {code}")
        audit.append({"code": code, "first_quote": str(raw.first_valid_index().date()),
                      "missing_dates_after_listing": p.index[(p.index >= raw.first_valid_index()) & raw.isna()].strftime("%Y-%m-%d").tolist(),
                      "max_abs_daily_return": float(returns.abs().max())})
    return p, v, audit


def frozen_core_targets(prices: pd.DataFrame) -> dict[int, np.ndarray]:
    signals = base.sleeve_targets(prices[list(base.CODES)])
    allocations = pd.read_csv(base.OUT / "allocation_signals.csv")
    allocations = allocations[allocations.portfolio.eq("six_erc")]
    targets = {}
    for _, row in allocations.iterrows():
        t = prices.index.get_loc(pd.Timestamp(row.date))
        target = np.zeros(len(prices.columns))
        target[:len(base.CODES)] = row[list(base.SLEEVES)].to_numpy(dtype=float) @ signals[t]
        targets[t] = target
    return targets


def mix_targets(core: dict[int, np.ndarray], columns: pd.Index,
                candidates: dict[str, float]) -> dict[int, np.ndarray]:
    share = sum(candidates.values())
    if not 0 <= share <= 1:
        raise ValueError("Invalid added allocation")
    static = np.zeros(len(columns))
    for code, weight in candidates.items():
        if code != "cash":
            static[columns.get_loc(code)] = weight
    return {t: target*(1-share)+static for t, target in core.items()}


def candidate_stats(prices: pd.DataFrame, valid: pd.DataFrame, core: base.Result,
                    code: str, start: pd.Timestamp) -> dict:
    mask = prices.index >= start
    ret = prices[code].pct_change(fill_method=None).loc[mask]
    baseline = core.returns.loc[mask]
    weekly = pd.DataFrame({"asset": ret, "core": baseline})
    weekly = ((1+weekly).resample("W-FRI").prod()-1)
    threshold = baseline.quantile(.05)
    worst = baseline <= threshold
    down = baseline < 0
    rolling = ret.rolling(120).corr(baseline).dropna()
    weights = np.zeros(len(prices.columns)); weights[prices.columns.get_loc(code)] = 1
    i = int(prices.index.searchsorted(start))
    standalone = base.simulate(prices, valid, {i-1: weights}, start=i)
    return {"code": code, "name": LABELS[code], "start": str(start.date()), "observations": int(mask.sum()),
            "correlation_daily": float(ret.corr(baseline)), "correlation_weekly": float(weekly.asset.corr(weekly.core)),
            "corr_on_core_down_days": float(ret[down].corr(baseline[down])),
            "mean_return_on_core_worst5pct_days": float(ret[worst].mean()),
            "positive_fraction_on_core_worst5pct_days": float((ret[worst] > 0).mean()),
            "rolling120_corr_min": float(rolling.min()), "rolling120_corr_max": float(rolling.max()),
            **base.metrics(standalone.returns.loc[mask])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true")
    if parser.parse_args().fetch:
        fetch(); return
    OUT.mkdir(parents=True, exist_ok=True)
    p, v, audit = read_panel()
    core_targets = frozen_core_targets(p)
    i = int(p.index.searchsorted(base.START)); mask = p.index >= base.START
    core = base.simulate(p, v, core_targets, start=i)
    old = pd.read_csv(base.OUT / "daily_returns.csv", index_col=0, parse_dates=True).six_erc
    np.testing.assert_allclose(core.returns.loc[mask], old, atol=1e-12)
    specs = {"original_six_erc": {}, "core80_cash20": {"cash": .2},
             "core80_gov5_20": {"511010.SH": .2}, "core80_gov10_20": {"511260.SH": .2},
             "core80_soymeal20": {"159985.SZ": .2}, "core80_energychem20": {"159981.SZ": .2},
             "core80_metals20": {"159980.SZ": .2},
             "core80_commodity_basket20": {"159985.SZ": .2/3, "159981.SZ": .2/3, "159980.SZ": .2/3},
             "core60_cash40": {"cash": .4},
             "core60_gov5_20_gov10_20": {"511010.SH": .2, "511260.SH": .2},
             "core60_gov5_20_commodities20": {"511010.SH": .2, "159985.SZ": .2/3,
                                              "159981.SZ": .2/3, "159980.SZ": .2/3}}
    # Fixed shares are comparison probes, not recommended portfolio weights.
    results = {name: base.simulate(p, v, mix_targets(core_targets, p.columns, targets), start=i)
               for name, targets in specs.items()}
    returns = pd.DataFrame({name: result.returns.loc[mask] for name, result in results.items()})
    returns.to_csv(OUT / "daily_returns.csv", index_label="date")
    comparison = pd.DataFrame({name: {**base.metrics(result.returns.loc[mask], result.turnover.loc[mask]),
                                     "average_cash": float(1-result.weights.loc[mask].sum(axis=1).mean())}
                               for name, result in results.items()}).T
    comparison.to_csv(OUT / "comparison.csv", index_label="portfolio")
    statistics = [candidate_stats(p, v, core, code, base.START)
                  for code in ("511010.SH", "511260.SH", "159985.SZ", "159981.SZ", "159980.SZ")]
    pd.DataFrame(statistics).to_csv(OUT / "candidate_stats.csv", index=False)
    annual = (1+returns).groupby(returns.index.year).prod()-1
    annual.to_csv(OUT / "annual_returns.csv", index_label="year")
    sections = []
    for label, start, end in [("202106_2023", "2021-06-01", "2023-12-31"),
                              ("2024_onward", "2024-01-01", "2026-11-10")]:
        for name in returns:
            sections.append({"segment": label, "portfolio": name, **base.metrics(returns.loc[start:end, name])})
    pd.DataFrame(sections).to_csv(OUT / "segments.csv", index=False)
    stress = []
    for window, start, end in [("2022", "2022-01-01", "2022-12-31"),
                               ("2024_reversal", "2024-09-01", "2024-10-31"),
                               ("2025_april", "2025-04-01", "2025-04-30")]:
        for name in returns:
            met = base.metrics(returns.loc[start:end, name])
            stress.append({"window": window, "portfolio": name,
                           **{k: met[k] for k in ("total_return", "max_drawdown", "worst_day")}})
    pd.DataFrame(stress).to_csv(OUT / "historical_stress.csv", index=False)
    # 30-year bond history is shorter: all comparison portfolios restart from cash.
    short_start = pd.Timestamp("2023-07-03")
    si = int(p.index.searchsorted(short_start)); smask = p.index >= short_start
    short_specs = {"original_six_erc": {}, "core80_cash20": {"cash": .2},
                   "core80_gov5_20": {"511010.SH": .2}, "core80_gov10_20": {"511260.SH": .2},
                   "core80_gov30_20": {"511090.SH": .2}}
    short_results = {name: base.simulate(p, v, mix_targets(core_targets, p.columns, targets), start=si)
                     for name, targets in short_specs.items()}
    pd.DataFrame({name: base.metrics(result.returns.loc[smask], result.turnover.loc[smask])
                  for name, result in short_results.items()}).T.to_csv(OUT / "gov30_short_window.csv", index_label="portfolio")
    short_returns = pd.DataFrame({name: result.returns.loc[smask] for name, result in short_results.items()})
    short_returns.to_csv(OUT / "gov30_short_daily_returns.csv")
    pd.DataFrame([candidate_stats(p, v, short_results["original_six_erc"], "511090.SH", short_start)]).to_csv(OUT / "gov30_stats.csv", index=False)
    covars = p[["511010.SH", "511260.SH", "511090.SH", "159985.SZ", "159981.SZ", "159980.SZ"]].pct_change(fill_method=None)
    covars.loc[smask].corr().to_csv(OUT / "candidate_correlations_short.csv")
    manifest = {"start": str(base.START.date()), "end": str(p.index[-1].date()), "observations": int(mask.sum()),
                "short_window_start": str(short_start.date()), "fixed_additions": specs,
                "independent_out_of_sample": False, "original_reproduction_max_error": float(abs(core.returns.loc[mask]-old).max()),
                "source_audit": audit, "skipped_rebalances": {n: r.skipped for n, r in results.items()},
                "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT / "summary.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(comparison.round(5).to_string())
    print(pd.DataFrame(statistics).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
