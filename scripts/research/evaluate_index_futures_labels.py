"""Compare long, short and flat logistic policies on tradable rolled futures labels."""
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from backtest_index_futures import ROOT, metrics, simulate
from evaluate_index_futures_lr import ETFS, FUTURES_FACTORS, etf_features

CAPITAL = 500_000
LABELS = {"direction_1": (1, 0.002), "direction_5": (5, 0.008),
          "direction_20": (20, 0.02), "direction_60": (60, 0.04),
          "path_20": (20, 0.03)}


def build_labels(group):
    group = group.sort_values("entry_date").copy()
    # One constant lot, marked open to open. Contract changes add each leg's
    # P&L separately and incur both closing and opening costs.
    start = group.entry_open.to_numpy()
    finish = group.exit_open.to_numpy()
    codes = group.ts_code.to_numpy()
    dates = group.entry_date.to_numpy()
    ends = group.exit_date.to_numpy()
    for name, (days, hurdle) in LABELS.items():
        labels = np.full(len(group), np.nan)
        maturity = np.full(len(group), None, dtype=object)
        for first in range(len(group) - days + 1):
            last = first + days
            if not np.array_equal(ends[first:last-1], dates[first+1:last]):
                continue
            moves = np.cumsum(finish[first:last] - start[first:last])
            rolls = codes[first+1:last] != codes[first:last-1]
            roll_cost = np.cumsum((start[first+1:last] + finish[first:last-1]) * rolls)
            costs = 0.0003 * (start[first] + finish[first:last] + np.r_[0, roll_cost])
            long = (moves - costs) / start[first]
            short = (-moves - costs) / start[first]
            direction = int(long[-1] >= hurdle) - int(short[-1] >= hurdle)
            if name == "path_20" and (
                (direction == 1 and long.min() <= -hurdle)
                or (direction == -1 and short.min() <= -hurdle)
            ):
                direction = 0
            labels[first] = direction
            maturity[first] = ends[last-1]
        group[name] = labels
        group[f"{name}_end"] = maturity
    return group


def main():
    out = ROOT / "lr_labels"
    out.mkdir(exist_ok=True)
    source = pd.read_parquet(ROOT / "predictions.parquet")
    panel = pd.concat(build_labels(g) for _, g in source.groupby("symbol"))
    signals = pd.concat(etf_features(symbol, ticker) for symbol, ticker in ETFS.items())
    features = FUTURES_FACTORS + signals.filter(regex="^etf_").columns.tolist()
    panel = panel.merge(signals, on=["symbol", "trade_date"], validate="one_to_one")
    panel = panel.replace([np.inf, -np.inf], np.nan)
    end = signals.groupby("symbol").trade_date.max().min()
    forecasts = []
    for year in (2023, 2024, 2025, 2026):
        test = panel[panel.entry_date.str.startswith(str(year)) & panel.exit_date.le(end)].copy()
        assert not test[features].isna().any().any()
        for name in LABELS:
            train = panel[panel[f"{name}_end"].lt(test.trade_date.min())].dropna(subset=features+[name])
            model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000))
            model.fit(train[features], train[name])
            assert list(model.classes_) == [-1, 0, 1]
            prob = model.predict_proba(test[features])
            test[f"score_{name}"] = prob[:, 2] - prob[:, 0]
            joblib.dump(dict(model=model, features=features, label=name, definition=LABELS[name]),
                        out / f"{name}_{year}.joblib")
        forecasts.append(test)
    predictions = pd.concat(forecasts)
    predictions.to_parquet(out / "predictions.parquet", index=False)
    periods = {
        "validation": predictions[predictions.exit_date.lt("20250101")],
        "holdout": predictions[predictions.entry_date.ge("20250101")],
    }
    groups = {key: [list(g.itertuples(index=False)) for _, g in frame.groupby("entry_date")]
              for key, frame in periods.items()}

    def replay(period, score, threshold, cadence, friction=0.0003, margin=0.15):
        ledger = simulate(groups[period], score, threshold, 1, "both", cadence, friction,
                          margin_rate=margin, capital=CAPITAL, enforce_margin=True, lot_cap=1)
        assert ledger.gross_lots.le(1).all()
        assert (ledger.equity - CAPITAL - ledger.pnl.cumsum()).abs().lt(1e-6).all()
        return ledger, dict(**metrics(ledger, capital=CAPITAL),
                            forced_exits=int(ledger.forced_exit.sum()))

    candidates = []
    for label in LABELS:
        for cadence in (1, 5, 10, 20):
            for threshold in (0.05, 0.10, 0.15, 0.20, 0.30):
                _, stats = replay("validation", f"score_{label}", threshold, cadence)
                candidates.append(dict(label=label, threshold=threshold, cadence=cadence, **stats))
    grid = pd.DataFrame(candidates)
    grid.to_csv(out / "validation_grid.csv", index=False)
    results = []
    picks = [("carry", "carry", 0.0005, 10)]
    for label, rows in grid.groupby("label"):
        eligible = rows.query("short_days >= 20 and long_days >= 20 and flat_days >= 20 and forced_exits == 0")
        if eligible.empty:
            continue
        for objective, keys in (("return", ["total_return", "sharpe"]),
                                ("stable", ["sharpe", "max_drawdown"])):
            best = eligible.sort_values(keys, ascending=False).iloc[0]
            picks.append((f"{label}_{objective}", f"score_{label}", best.threshold, int(best.cadence)))
    for policy, score, threshold, cadence in picks:
        for period in groups:
            for friction, margin in ((0.0003, 0.15), (0.0006, 0.25)):
                ledger, stats = replay(period, score, threshold, cadence, friction, margin)
                ledger.to_csv(out / f"{policy}_{period}_{margin:.2f}.csv", index=False)
                results.append(dict(policy=policy, period=period, threshold=threshold,
                                    cadence=cadence, friction=friction, margin=margin, **stats))
    summary = pd.DataFrame(results)
    summary.to_csv(out / "summary.csv", index=False)
    print(summary.query("margin == 0.15")[["policy", "period", "threshold", "cadence",
          "total_return", "max_drawdown", "short_days", "forced_exits"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
