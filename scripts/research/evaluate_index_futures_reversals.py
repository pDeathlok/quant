"""Audit rolling short trades and test additional futures factors and reversal rules."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from backtest_index_futures import MULTIPLIERS, ROOT, metrics, simulate
from evaluate_index_futures_directions import CAPITAL, directional_panel, load_bars
from evaluate_index_futures_labels import build_labels
from evaluate_index_futures_lr import ETFS, FUTURES_FACTORS, etf_features

FEE = .0003


def short_episodes(ledger, bars, fee=FEE, direction=-1):
    trades = []
    active = None
    for row in ledger.itertuples():
        code = row.positions.split(":")[0] if row.positions else ""
        product = code[:2] if row.positions.endswith(f":{direction:+d}") else ""
        if active and active["product"] != product:
            active["pnl"] -= bars.loc[(row.entry_date, active["code"]), "open"] * MULTIPLIERS[active["product"]] * fee
            trades.append(dict(active, exit=row.entry_date))
            active = None
        if not product:
            continue
        if active is None:
            active = dict(product=product, entry=row.entry_date, code="", pnl=0., days=0)
        multiplier = MULTIPLIERS[product]
        opening = bars.loc[(row.entry_date, code), "open"]
        ending = bars.loc[(row.exit_date, code), "open"]
        if active["code"] != code:
            active["pnl"] -= opening * multiplier * fee
            if active["code"]:
                active["pnl"] -= bars.loc[(row.entry_date, active["code"]), "open"] * multiplier * fee
        active["code"] = code
        active["days"] += 1
        active["pnl"] += direction * (ending - opening) * multiplier
    if active:
        active["pnl"] -= ending * multiplier * fee
        trades.append(dict(active, exit=row.exit_date))
    return pd.DataFrame(trades)


def futures_features(bars):
    d = bars.reset_index().sort_values(["ts_code", "trade_date"])
    g = d.groupby("ts_code")
    d["future_range"] = (d.high - d.low) / d.pre_close
    d["future_close_location"] = (d.close - d.low) / (d.high - d.low).replace(0, np.nan)
    d["future_gap"] = d.open / d.pre_close - 1
    d["future_ret5"] = g.close.pct_change(5)
    d["future_volume_ratio"] = d.vol / g.vol.transform(lambda x: x.shift().rolling(20, min_periods=5).mean())
    d["future_oi5"] = g.oi.pct_change(5)
    d["future_settle_gap"] = d.close / d.settle - 1
    return d[["trade_date", "ts_code", *d.filter(regex="^future_").columns]]


def main():
    out = ROOT / "reversal_research"
    out.mkdir(exist_ok=True)
    bars = load_bars()
    roll = pd.read_csv(ROOT / "rolling_selection/6_24_stable_0.15.csv",
                       dtype={"entry_date": str, "exit_date": str}, keep_default_na=False)
    shorts = short_episodes(roll, bars)
    shorts.to_csv(out / "short_episodes.csv", index=False)
    panel = directional_panel()
    panel = pd.concat(build_labels(group) for _, group in panel.groupby("symbol"))
    etf = pd.concat(etf_features(symbol, ticker) for symbol, ticker in ETFS.items())
    panel = panel.merge(etf, on=["trade_date", "symbol"], how="left", validate="one_to_one")
    future = futures_features(bars)
    panel = panel.merge(future, on=["trade_date", "ts_code"], validate="one_to_one")
    panel = panel.replace([np.inf, -np.inf], np.nan)
    etf_columns = etf.filter(regex="^etf_").columns.tolist()
    futures_columns = future.filter(regex="^future_").columns.tolist()
    base = FUTURES_FACTORS + etf_columns
    # This ablation starts in 2023 because the exported walk-forward panel starts in 2020.
    variants = {"base": base, "extended": base + futures_columns,
                "etf_only": etf_columns, "futures_only": FUTURES_FACTORS[:4] + futures_columns}
    score_parts = []
    for year in (2023, 2024, 2025, 2026):
        test = panel[panel.entry_date.str.startswith(str(year))].copy()
        train = panel[panel.direction_20_end.lt(test.trade_date.min())].dropna(subset=["direction_20"])
        for name, columns in variants.items():
            # Impute from the training period only, including contract warm-up gaps.
            fill = train[columns].median()
            model = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=1000))
            model.fit(train[columns].fillna(fill), train.direction_20)
            prob = model.predict_proba(test[columns].fillna(fill))
            test[name] = prob[:, 2] - prob[:, 0]
        score_parts.append(test)
    panel = pd.concat(score_parts)
    low = panel.etf_drawdown_20.le(-.05)
    reversal = low & panel.etf_ret_1.gt(0) & panel.etf_vol_ratio.gt(1.2)
    confirmed = reversal & panel.future_close_location.ge(.65) & panel.future_volume_ratio.ge(1)
    panel["buy_dip"] = low.astype(float)
    panel["reversal"] = reversal.astype(float)
    panel["confirmed"] = confirmed.astype(float)
    records = []
    for period, data in {
        "2023–2024": panel[panel.exit_date.lt("20250101")],
        "2025–2026": panel[panel.entry_date.ge("20250101")],
    }.items():
        groups = [list(g.itertuples(index=False)) for _, g in data.groupby("entry_date")]
        for name in ("carry", *variants, "buy_dip", "reversal", "confirmed"):
            threshold, cadence = (0.0005, 10) if name == "carry" else (0.20, 10) if name in variants else (0.5, 1)
            ledger = simulate(groups, name, threshold, 1, "both", cadence, FEE,
                              capital=CAPITAL, enforce_margin=True, lot_cap=1)
            assert ledger.gross_lots.le(1).all()
            assert np.allclose(ledger.equity, CAPITAL + ledger.pnl.cumsum(), atol=1e-6, rtol=0)
            events = data[data.direction_20_end.le(ledger.exit_date.max())].dropna(subset=["direction_20"])
            selected = events[events[name].gt(threshold)]
            records.append(dict(period=period, policy=name, forced=int(ledger.forced_exit.sum()),
                                signals=len(selected), long_hit_rate=selected.direction_20.eq(1).mean(),
                                **metrics(ledger, CAPITAL)))
    result = pd.DataFrame(records)
    result.to_csv(out / "summary.csv", index=False)
    print(shorts.groupby(shorts.entry.str[:4]).pnl.agg(["size", "sum", lambda x: x.gt(0).sum()]))
    print(result[["period", "policy", "total_return", "max_drawdown", "active_days",
                  "short_days", "signals", "long_hit_rate", "forced"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
