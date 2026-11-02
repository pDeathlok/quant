"""Short-sample THS industry net-flow and matched ETF diagnostic."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tushare as ts
from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/public_strategy_validation_20260929/industry_netflow"
SECTOR_SOURCE = ROOT / "reports/public_strategy_validation_20260929/sector_turnover_surge/source"
MAPPING = {
    "512800.SH": ("881155.TI", "银行"),
    "512880.SH": ("881157.TI", "证券"),
    "512480.SH": ("881121.TI", "半导体"),
    "512200.SH": ("881153.TI", "房地产"),
    "512690.SH": ("881273.TI", "白酒"),
    "515220.SH": ("881105.TI", "煤炭开采加工"),
    "515790.SH": ("881279.TI", "光伏设备"),
}
START = pd.Timestamp("2024-12-16")

spec = importlib.util.spec_from_file_location("public_validation", ROOT / "scripts/research/validate_public_strategies.py")
validation = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(validation)


def load_flows(token: str) -> pd.DataFrame:
    cache = OUT / "source"
    cache.mkdir(parents=True, exist_ok=True)
    pro = ts.pro_api(token)
    frames = []
    for _, (code, expected_name) in MAPPING.items():
        path = cache / f"{code.replace('.', '_')}.parquet"
        if path.exists():
            frame = pd.read_parquet(path)
        else:
            frame = pro.moneyflow_ind_ths(ts_code=code, start_date="20240901", end_date="20260924")
            if frame.empty or set(frame.industry.dropna()) != {expected_name}:
                raise RuntimeError(f"Industry mapping or history invalid: {code}")
            frame.to_parquet(path, index=False)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def run(token: str) -> dict:
    flows = load_flows(token)
    flows["date"] = pd.to_datetime(flows.trade_date)
    flow = flows.pivot(index="date", columns="ts_code", values="net_amount").sort_index()
    flow = flow.rename(columns={v[0]: k for k, v in MAPPING.items()})
    parts = []
    for code in MAPPING:
        part = pd.read_parquet(SECTOR_SOURCE / f"{code.replace('.', '_')}.parquet")
        part["date"] = pd.to_datetime(part.trade_date)
        part["adjusted_close"] = part.close * part.adj_factor
        parts.append(part[["date", "ts_code", "adjusted_close"]])
    price = pd.concat(parts).pivot(index="date", columns="ts_code", values="adjusted_close")
    dates = price.index.intersection(flow.index)
    price, flow = price.reindex(dates), flow.reindex(dates)
    five_day_net = flow.rolling(5, min_periods=5).sum()
    prior_scale = five_day_net.shift(5).rolling(60, min_periods=60).std()
    flow_strength = five_day_net / prior_scale.replace(0, np.nan)
    return20 = price.pct_change(20, fill_method=None)
    ma60 = price.rolling(60, min_periods=60).mean()
    week_ends = price.groupby(price.index.to_period("W-FRI")).tail(1).index
    targets = {"price_momentum": {}, "netflow_plus_price": {}, "netflow_plus_price_60pct": {}}
    signals = []
    for day in week_ends:
        eligible = (return20.loc[day] > 0) & (price.loc[day] > ma60.loc[day])
        price_picks = return20.loc[day][eligible].sort_values(ascending=False).head(2).index.tolist()
        flow_eligible = eligible & (five_day_net.loc[day] > 0) & (flow_strength.loc[day] > 0)
        flow_picks = flow_strength.loc[day][flow_eligible].sort_values(ascending=False).head(2).index.tolist()
        targets["price_momentum"][day] = {code: 0.5 for code in price_picks}
        targets["netflow_plus_price"][day] = {code: 0.5 for code in flow_picks}
        targets["netflow_plus_price_60pct"][day] = {code: 0.3 for code in flow_picks}
        signals.append({"date": day, "price_picks": ",".join(price_picks),
                        "flow_picks": ",".join(flow_picks)})
    results, daily = [], {}
    for mode, target in targets.items():
        series, turnover, rebalances = validation.simulate(price, target)
        series = series.loc[START:]
        daily[mode] = series
        results.append({"strategy": mode, **validation.metrics(series),
                        "turnover_notional": turnover, "rebalances": rebalances,
                        "active_week_fraction": sum(bool(x) for x in
                            (targets[mode][day] for day in week_ends if day >= START))
                            / sum(day >= START for day in week_ends)})
    benchmark_frame = pd.read_parquet(
        ROOT / "reports/public_strategy_validation_20260929/source/510300_SH_2018_2026.parquet"
    )
    benchmark_frame["date"] = pd.to_datetime(benchmark_frame.trade_date)
    benchmark = (benchmark_frame.set_index("date").sort_index().close
                 * benchmark_frame.set_index("date").sort_index().adj_factor)
    baseline = benchmark.pct_change(fill_method=None).fillna(0).reindex(dates).loc[START:]
    baseline.iloc[0] -= validation.FEE_ONE_WAY
    results.append({"strategy": "buy_hold_510300", **validation.metrics(baseline)})
    daily["buy_hold_510300"] = baseline
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(daily).to_csv(OUT / "daily_returns.csv", index_label="date")
    pd.DataFrame(signals).to_csv(OUT / "weekly_signals.csv", index=False)
    pd.DataFrame(results).to_csv(OUT / "summary.csv", index=False)
    payload = {"sample_start": str(START.date()), "sample_end": str(dates.max().date()),
               "mapping": MAPPING, "flow_first_date": str(flow.index.min().date()),
               "results": results,
               "cautions": ["Only about two years of THS industry flow history are available.",
                            "THS net amount is a vendor estimate, not audited ETF net subscriptions.",
                            "Each THS industry is an approximate mapping to its sector ETF.",
                            "Rules were explored after seeing turnover-surge failure; not a blind test."]}
    (OUT / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    token = os.getenv("TUSHARE_TOKEN") or dotenv_values(ROOT / ".env").get("TUSHARE_TOKEN")
    if not token:
        raise SystemExit("TUSHARE_TOKEN missing")
    result = run(token)
    print(pd.DataFrame(result["results"])[["strategy", "annual_return", "max_drawdown"]].to_string(index=False))
