"""Replay ETF risk filters against the one-lot carry strategy."""
import pandas as pd

from backtest_index_futures import ROOT, metrics, simulate

ETFS = {"IF": "510300", "IH": "510050", "IC": "510500", "IM": "512100"}
CACHE = ROOT.parent / "short_side_ownership"
RULES = ("baseline", "ma20", "ma60", "ma20_and_volume", "volume_shock", "drawdown_8")
CAPITAL = 500_000


def etf_signals(ticker):
    code = f"{ticker}.SH"
    bars, adjustment = (
        pd.concat(pd.read_parquet(path) for path in sorted((CACHE / name).glob(f"{code}_*.parquet")))
        for name in ("fund_daily", "fund_adj")
    )
    d = bars.merge(adjustment[["trade_date", "adj_factor"]], on="trade_date",
                   validate="one_to_one").sort_values("trade_date")
    close = d.close * d.adj_factor
    volume_ratio = d.vol / d.vol.shift().rolling(20).mean()
    d["ma20"] = close < close.rolling(20).mean()
    d["ma60"] = close < close.rolling(60).mean()
    d["ma20_and_volume"] = d.ma20 & (close.pct_change() < 0) & (volume_ratio > 1.5)
    d["volume_shock"] = (close.pct_change() <= -0.03) & (volume_ratio > 1.5)
    d["drawdown_8"] = close / close.rolling(20).max() - 1 <= -0.08
    d["etf_close"] = close
    d["etf_return"] = close.pct_change()
    d["volume_ratio"] = volume_ratio
    return d[["trade_date", "etf_close", "etf_return", "volume_ratio", *RULES[1:]]]


def main():
    out = ROOT / "etf_risk"
    out.mkdir(exist_ok=True)
    signals = pd.concat(etf_signals(ticker).assign(symbol=symbol)
                        for symbol, ticker in ETFS.items())
    predictions = pd.read_parquet(ROOT / "predictions.parquet")
    end = signals.groupby("symbol").trade_date.max().min()
    data = predictions.query("entry_date >= '20230101' and exit_date <= @end").merge(
        signals, on=["trade_date", "symbol"], how="left", validate="many_to_one")
    assert data.etf_close.notna().all()
    stats = []
    for period, sample in {
        "validation": data[data.exit_date.lt("20250101")],
        "holdout": data[data.entry_date.ge("20250101")],
    }.items():
        groups = [list(g.itertuples(index=False)) for _, g in sample.groupby("entry_date")]
        for rule in RULES:
            ledger = simulate(groups, "carry", 0.0005, 1, "both", 10, 0.0003,
                              capital=CAPITAL, enforce_margin=True, lot_cap=1,
                              risk_filter=None if rule == "baseline" else rule)
            assert ledger.gross_lots.le(1).all()
            assert (ledger.equity - CAPITAL - ledger.pnl.cumsum()).abs().lt(1e-6).all()
            ledger.to_csv(out / f"{period}_{rule}.csv", index=False)
            stress = ledger[ledger.exit_date.between("20250317", "20250409")]
            stats.append(dict(period=period, rule=rule, end=ledger.exit_date.max(),
                              stress_pnl=stress.pnl.iloc[1:].sum(),
                              **metrics(ledger, capital=CAPITAL)))
    pd.DataFrame(stats).to_csv(out / "summary.csv", index=False)
    print(pd.DataFrame(stats)[["period", "rule", "end", "total_return", "max_drawdown",
                               "sharpe", "stress_pnl"]].round(4).to_string(index=False))
    print(signals.query("symbol == 'IM' and '20250402' <= trade_date <= '20250408'").to_string(index=False))


if __name__ == "__main__":
    main()
