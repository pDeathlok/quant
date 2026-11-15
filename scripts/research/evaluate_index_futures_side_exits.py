"""Compare separate long/short entry models and opposing-signal exits."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from backtest_index_futures import ROOT, build_panel, metrics, simulate
from evaluate_index_futures_directions import CAPITAL
from evaluate_index_futures_labels import build_labels
from evaluate_index_futures_lr import ETFS, FUTURES_FACTORS, etf_features
from evaluate_index_futures_reversals import futures_features
from evaluate_index_futures_directions import load_bars

ENTRY = .45
EXIT = .35


def forecasts():
    source, _ = build_panel()
    panel = pd.concat(build_labels(group) for _, group in source.groupby("symbol"))
    etf = pd.concat(etf_features(symbol, ticker) for symbol, ticker in ETFS.items())
    future = futures_features(load_bars())
    columns = FUTURES_FACTORS + etf.filter(regex="^etf_").columns.tolist()
    columns += future.filter(regex="^future_").columns.tolist()
    panel = panel.merge(etf, on=["symbol", "trade_date"], how="left", validate="one_to_one")
    panel = panel.merge(future, on=["ts_code", "trade_date"], validate="one_to_one")
    panel = panel.replace([np.inf, -np.inf], np.nan)
    end = etf.groupby("symbol").trade_date.max().min()
    for year in range(2020, int(end[:4]) + 1):
        test = panel[panel.entry_date.str.startswith(str(year)) & panel.exit_date.le(end)].copy()
        for horizon in (5, 20, 60):
            label = f"direction_{horizon}"
            cutoff = test.trade_date.min()
            train = panel[panel[f"{label}_end"].lt(cutoff)].dropna(subset=[label])
            train = train[train.etf_ret_1.notna()]
            fill = train[columns].median()
            for side, direction in (("long", 1), ("short", -1)):
                model = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=1000))
                model.fit(train[columns].fillna(fill), train[label].eq(direction))
                test[f"{side}_{horizon}"] = model.predict_proba(test[columns].fillna(fill))[:, 1]
        yield test


def replay(groups, horizon, exit_horizon, fee, margin, confirm):
    by_date = {group[0].entry_date: group for group in groups}
    elapsed = -1
    entered = 0
    events = []
    opposing_days = 0

    def select(day, positions):
        nonlocal elapsed, entered, opposing_days
        elapsed += 1
        rows = by_date[day]
        if positions:
            code, (lots, _) = next(iter(positions.items()))
            held = next((row for row in rows if row.symbol == code[:2]), None)
            opposing = "short" if lots > 0 else "long"
            probability = (getattr(held, f"{opposing}_{exit_horizon}")
                           if exit_horizon and held and pd.notna(held.etf_ret_1) else 0.)
            own_side = "long" if lots > 0 else "short"
            own = getattr(held, f"{own_side}_{exit_horizon}") if exit_horizon and held else 0.
            qualifies = probability >= EXIT and (not confirm or probability - own >= .10)
            opposing_days = opposing_days + 1 if qualifies else 0
            early = opposing_days >= (2 if confirm else 1)
            if elapsed - entered >= 20 or early:
                reason = "opposing_signal" if early and elapsed - entered < 20 else "holding_limit"
                events.append(dict(date=day, reason=reason, direction=lots, probability=probability))
                return ("flat", .5, 1, "both", None)
            return None
        opposing_days = 0
        if elapsed % 5:
            return ("flat", .5, 1, "both", None)
        entered = elapsed
        return (f"entry_{horizon}", ENTRY, 10_000, "both", None)

    ledger = simulate(groups, "flat", .5, 1, "both", 1, fee, margin_rate=margin,
                      capital=CAPITAL, enforce_margin=True, lot_cap=1, policy_selector=select)
    assert ledger.gross_lots.le(1).all()
    assert np.allclose(ledger.equity, CAPITAL + ledger.pnl.cumsum(), atol=1e-6, rtol=0)
    return ledger, events


def main():
    out = ROOT / "side_exits"
    out.mkdir(exist_ok=True)
    panel = pd.concat(forecasts(), ignore_index=True)
    panel["flat"] = 0.
    for horizon in (20, 60):
        long = panel[f"long_{horizon}"]
        short = panel[f"short_{horizon}"]
        # If both heads are confident, require a clear separation to trade.
        panel[f"entry_{horizon}"] = np.where(
            (long - short).abs().ge(.10) & panel.etf_ret_1.notna(),
            np.where(long > short, long, -short), 0.)
    summaries, yearly, exits = [], [], []
    for start in ("20200101", "20220101", "20250101"):
        groups = [list(g.itertuples(index=False))
                  for _, g in panel[panel.entry_date.ge(start)].groupby("entry_date")]
        for horizon in (20, 60):
            for exit_horizon, confirm in ((0, False), (5, False), (20, False), (5, True), (20, True)):
                for fee, margin in ((.0003, .15), (.0006, .25)):
                    ledger, events = replay(groups, horizon, exit_horizon, fee, margin, confirm)
                    fields = dict(start=start, entry_horizon=horizon, exit_horizon=exit_horizon, confirmed=confirm, margin=margin)
                    tag = f"{start}_{horizon}_{exit_horizon}_{int(confirm)}_{margin:.2f}"
                    ledger.to_csv(out / f"{tag}.csv", index=False)
                    exits.extend(dict(**fields, **event) for event in events)
                    early = [event for event in events if event["reason"] == "opposing_signal"]
                    summaries.append(dict(**fields, **metrics(ledger, CAPITAL),
                                          close_long=sum(event["direction"] > 0 for event in early),
                                          close_short=sum(event["direction"] < 0 for event in early),
                                          forced=int(ledger.forced_exit.sum())))
                    equity = CAPITAL
                    for year, part in ledger.groupby(ledger.exit_date.str[:4]):
                        path = np.r_[equity, part.equity]
                        yearly.append(dict(**fields, year=year, total_return=path[-1] / equity - 1,
                                           max_drawdown=(path / np.maximum.accumulate(path) - 1).min()))
                        equity = path[-1]
    pd.DataFrame(summaries).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(yearly).to_csv(out / "annual.csv", index=False)
    pd.DataFrame(exits).to_csv(out / "exits.csv", index=False)


if __name__ == "__main__":
    main()
