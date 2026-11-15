"""Walk-forward logistic regression for downside filtering of one-lot carry."""
import joblib
import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from backtest_index_futures import HOLDOUT, ROOT, VALIDATION, metrics, price_factors, simulate
from test_index_futures_etf_filter import CACHE, CAPITAL, ETFS

DOWNSIDE = -0.03
THRESHOLDS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40)
STOCK_FACTORS = (
    "jq_VOSC", "jq_WVAD", "jq_MFI14", "jq_arron_up_25", "jq_arron_down_25",
    "jq_Skewness20", "jq_Kurtosis20",
)
FUTURES_FACTORS = [
    "f_basis_annual", "f_basis_change", "f_oi_change", "f_days",
    "f_momentum_5", "f_momentum_20", "f_volatility_20",
]


def etf_features(symbol, ticker, *, bars=None, adj=None):
    code = f"{ticker}.SH"
    if bars is None or adj is None:
        bars, adj = (
            pd.concat(pd.read_parquet(path) for path in sorted((CACHE / name).glob(f"{code}_*.parquet")))
            for name in ("fund_daily", "fund_adj")
        )
    bars = bars.merge(adj[["trade_date", "adj_factor"]], on="trade_date",
                      validate="one_to_one").sort_values("trade_date")
    for field in ("open", "high", "low", "close"):
        bars[field] *= bars.adj_factor
    bars = bars.rename(columns={"trade_date": "date", "vol": "volume"})
    bars["symbol"] = symbol
    bars["ret"] = bars.close.pct_change()
    bars["turnover_rate"] = np.nan
    factors = price_factors(pl.from_pandas(bars)).to_pandas()[["date", *STOCK_FACTORS]]
    factors = factors.rename(columns={name: f"etf_{name}" for name in STOCK_FACTORS})
    volume = bars.volume.rolling(20).mean().to_numpy()
    factors["etf_jq_WVAD"] /= volume
    for window in (5, 20, 60):
        factors[f"etf_ma_gap_{window}"] = (bars.close / bars.close.rolling(window).mean() - 1).to_numpy()
    factors["etf_vol_ratio"] = (bars.volume / bars.volume.shift().rolling(20).mean()).to_numpy()
    factors["etf_ret_1"] = bars.ret.to_numpy()
    factors["etf_drawdown_20"] = (bars.close / bars.close.rolling(20).max() - 1).to_numpy()
    return factors.rename(columns={"date": "trade_date"}).assign(symbol=symbol)


def main():
    out = ROOT / "lr_risk"
    out.mkdir(exist_ok=True)
    signals = pd.concat(etf_features(symbol, ticker) for symbol, ticker in ETFS.items())
    features = FUTURES_FACTORS + signals.filter(regex="^etf_").columns.tolist()
    source = pd.read_parquet(ROOT / "predictions.parquet")
    end = signals.groupby("symbol").trade_date.max().min()
    panel = source.merge(signals, on=["trade_date", "symbol"], how="left", validate="one_to_one")
    panel = panel.replace([np.inf, -np.inf], np.nan)
    forecasts = []
    coefficients = []
    for year in (2023, 2024, 2025, 2026):
        cutoff = f"{year}0101"
        train = panel[panel.exit_5d_date.lt(cutoff)].dropna(subset=features + ["target_5d"])
        test = panel[panel.entry_date.str.startswith(str(year)) & panel.exit_date.le(end)].copy()
        if test[features].isna().any().any():
            raise ValueError("Missing ETF or futures features on evaluation sessions")
        assert train.exit_5d_date.max() < test.entry_date.min()
        label = train.target_5d.mul(5).le(DOWNSIDE)
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000))
        model.fit(train[features], label)
        test["risk_probability"] = model.predict_proba(test[features])[:, 1]
        forecasts.append(test)
        coefficients.extend(dict(year=year, feature=feature, coefficient=value)
                            for feature, value in zip(features, model[-1].coef_[0]))
        joblib.dump(dict(model=model, features=features, trained_before=cutoff,
                         downside=DOWNSIDE, train_rows=len(train)), out / f"model_{year}.joblib")
    forecasts = pd.concat(forecasts, ignore_index=True)
    forecasts.to_parquet(out / "predictions.parquet", index=False)
    pd.DataFrame(coefficients).to_csv(out / "coefficients.csv", index=False)
    periods = {
        "validation": forecasts[forecasts.entry_date.ge(VALIDATION) & forecasts.exit_date.lt(HOLDOUT)],
        "holdout": forecasts[forecasts.entry_date.ge(HOLDOUT)],
    }
    results = []
    classification = []

    def replay(sample, period, threshold, friction):
        sample = sample.assign(lr_exit=False if threshold is None else sample.risk_probability.ge(threshold))
        groups = [list(g.itertuples(index=False)) for _, g in sample.groupby("entry_date")]
        ledger = simulate(groups, "carry", 0.0005, 1, "both", 10, friction,
                          capital=CAPITAL, enforce_margin=True, lot_cap=1, risk_filter="lr_exit")
        assert ledger.gross_lots.le(1).all()
        assert (ledger.equity - CAPITAL - ledger.pnl.cumsum()).abs().lt(1e-6).all()
        name = "baseline" if threshold is None else str(threshold)
        ledger.to_csv(out / f"{period}_{name}_{friction:.4f}.csv", index=False)
        results.append(dict(period=period, rule=name, friction=friction,
                            forced_exits=int(ledger.forced_exit.sum()),
                            **metrics(ledger, capital=CAPITAL)))

    for period, sample in periods.items():
        observed = sample[sample.exit_5d_date.le(sample.exit_date.max())].dropna(subset=["target_5d"])
        actual = observed.target_5d.mul(5).le(DOWNSIDE)
        prob = observed.risk_probability
        classification.append(dict(period=period, observations=len(actual), event_rate=actual.mean(),
                                   auc=roc_auc_score(actual, prob),
                                   average_precision=average_precision_score(actual, prob),
                                   brier=brier_score_loss(actual, prob)))
    for threshold in (None, *THRESHOLDS):
        replay(periods["validation"], "validation", threshold, 0.0003)
    validation = pd.DataFrame(results)
    baseline = validation.iloc[0]
    eligible = validation.iloc[1:].query("active_days >= 30 and total_return >= 0 and forced_exits == 0")
    if not eligible.empty:
        winner = eligible.sort_values(["max_drawdown", "total_return"], ascending=False).iloc[0]
        chosen = float(winner.rule) if winner.max_drawdown > baseline.max_drawdown else None
    else:
        chosen = None
    for friction in (0.0003, 0.0006):
        replay(periods["holdout"], "holdout", None, friction)
        if chosen is not None:
            replay(periods["holdout"], "holdout", chosen, friction)
    # A fixed 20% diagnostic shows model behavior even when validation rejects deployment.
    if chosen != 0.20:
        replay(periods["holdout"], "holdout", 0.20, 0.0003)
    pd.DataFrame(results).assign(selected_threshold=chosen).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(classification).to_csv(out / "classification.csv", index=False)
    print("Selected:", chosen)
    print(pd.DataFrame(classification).round(4).to_string(index=False))
    print(pd.DataFrame(results)[["period", "rule", "friction", "total_return", "max_drawdown",
                                "active_days", "forced_exits"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
