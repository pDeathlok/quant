"""Historical walk-forward replay with policy parameters frozen across windows."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from backtest_index_futures import ROOT, build_panel, metrics, simulate
from evaluate_index_futures_labels import CAPITAL, build_labels
from evaluate_index_futures_lr import ETFS, FUTURES_FACTORS, etf_features

POLICIES = {"carry": ("carry", 0.0005, 10),
            "lr5": ("score_direction_5", 0.20, 10),
            "lr60": ("score_direction_60", 0.30, 20)}


def main():
    out = ROOT / "long_oos"
    out.mkdir(exist_ok=True)
    source, _ = build_panel()
    panel = pd.concat(build_labels(group) for _, group in source.groupby("symbol"))
    etfs = pd.concat(etf_features(symbol, ticker) for symbol, ticker in ETFS.items())
    features = FUTURES_FACTORS + etfs.filter(regex="^etf_").columns.tolist()
    panel = pd.merge_asof(
        panel.sort_values("trade_date").assign(signal_time=lambda d: pd.to_datetime(d.trade_date)),
        etfs.rename(columns={"trade_date": "etf_date"}).sort_values("etf_date").assign(
            etf_time=lambda d: pd.to_datetime(d.etf_date)),
        left_on="signal_time", right_on="etf_time", by="symbol",
        tolerance=pd.Timedelta(days=3),
    )
    panel = panel.replace([np.inf, -np.inf], np.nan)
    end = etfs.groupby("symbol").trade_date.max().min()
    forecasts = []
    audit = []
    for year in range(2020, int(end[:4]) + 1):
        test = panel[panel.entry_date.str.startswith(str(year)) & panel.exit_date.le(end)].copy()
        if test[features].isna().any().any():
            raise ValueError(f"Missing evaluation features in {year}")
        for label in ("direction_5", "direction_60"):
            last_date = test.trade_date.min()
            train = panel[panel[f"{label}_end"].lt(last_date)].dropna(subset=[*features, label])
            model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000))
            model.fit(train[features], train[label])
            assert list(model.classes_) == [-1, 0, 1]
            probabilities = model.predict_proba(test[features])
            test[f"score_{label}"] = probabilities[:, 2] - probabilities[:, 0]
            audit.append(dict(year=year, label=label, rows=len(train),
                              training_label_end=train[f"{label}_end"].max(),
                              first_signal=last_date, products=",".join(sorted(train.symbol.unique()))))
        test["carry"] = -test.f_basis_annual / 244
        forecasts.append(test)
    forecasts = pd.concat(forecasts)
    forecasts.loc[forecasts.etf_date.ne(forecasts.trade_date),
                  ["symbol", "trade_date", "etf_date"]].to_csv(out / "stale_etf.csv", index=False)
    forecasts[["symbol", "trade_date", "entry_date", "etf_date",
               "score_direction_5", "score_direction_60", "carry"]].to_parquet(
                   out / "predictions.parquet", index=False)

    pd.DataFrame(audit).to_csv(out / "training_audit.csv", index=False)
    results = []
    annual = []
    for start in ("20200101", "20220101", "20250101"):
        sample = forecasts[forecasts.entry_date.ge(start)]
        groups = [list(group.itertuples(index=False))
                  for _, group in sample.groupby("entry_date")]
        for name, (score, threshold, cadence) in POLICIES.items():
            for fee, margin in ((0.0003, 0.15), (0.0006, 0.25)):
                ledger = simulate(groups, score, threshold, 1, "both", cadence, fee,
                                  margin_rate=margin, capital=CAPITAL, enforce_margin=True, lot_cap=1)
                assert ledger.gross_lots.le(1).all()
                assert (ledger.equity - CAPITAL - ledger.pnl.cumsum()).abs().lt(1e-6).all()
                ledger.to_csv(out / f"{start}_{name}_{margin:.2f}.csv", index=False)
                results.append(dict(start=ledger.entry_date.min(), end=ledger.exit_date.max(),
                                    policy=name, margin=margin, fee=fee,
                                    forced_exits=int(ledger.forced_exit.sum()),
                                    **metrics(ledger, capital=CAPITAL)))
                starting_equity = CAPITAL
                for year, rows in ledger.groupby(ledger.exit_date.str[:4]):
                    ending_equity = rows.equity.iloc[-1]
                    annual.append(dict(window=start, year=year, policy=name, margin=margin,
                                       annual_period_return=ending_equity / starting_equity - 1,
                                       forced_exits=int(rows.forced_exit.sum())))
                    starting_equity = ending_equity
    pd.DataFrame(results).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(annual).to_csv(out / "annual.csv", index=False)
    print(pd.DataFrame(results)[["start", "end", "policy", "margin", "total_return",
                                 "max_drawdown", "short_days", "forced_exits"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
