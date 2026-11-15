"""Compare fixed directional and spread policies on the historical daily panel."""

import numpy as np
import pandas as pd

from backtest_index_futures import INDICES, MULTIPLIERS, ROOT, build_panel, metrics, simulate

CAPITAL = 500_000
FEE = 0.0003


def load_bars():
    return pd.concat(pd.read_parquet(ROOT / f"{symbol}_daily.parquet")
                     for symbol in INDICES).set_index(["trade_date", "ts_code"]).sort_index()


def cash_risk(row, lots, equity, margin_rate):
    return row.vol_percentile <= 0.80


def margin_reserve(row, lots, equity, margin_rate):
    adverse_move = max(0.05, 3 * row.f_volatility_20)
    stressed_price = row.entry_open * (1 - np.sign(lots) * adverse_move)
    loss = abs(lots) * row.entry_open * adverse_move * row.multiplier
    margin = abs(lots) * stressed_price * margin_rate * row.multiplier
    return equity - loss - margin >= 0.20 * equity


def paired_panel(panel, bars):
    """Select the two nearest liquid IC expiries with at least 10 days remaining."""
    days = panel[panel.symbol.eq("IC")][["trade_date", "entry_date", "exit_date", "spot_close"]]
    expiry = pd.read_parquet(ROOT / "IC_contracts.parquet").set_index("ts_code").delist_date
    ic = bars.reset_index().loc[lambda d: d.ts_code.str.startswith("IC")].merge(days, on="trade_date")
    ic["expiry"] = ic.ts_code.map(expiry)
    ic["days_left"] = (pd.to_datetime(ic.expiry) - pd.to_datetime(ic.trade_date)).dt.days
    ic = ic.query("days_left >= 10 and oi >= 1000 and vol >= 100").sort_values(["trade_date", "expiry"])
    ic["leg"] = ic.groupby("trade_date").cumcount()
    ic = ic[ic.leg.lt(2)].copy()
    quotes = ic.pivot(index="trade_date", columns="leg", values="close")
    spot = days.set_index("trade_date").spot_close
    spread = ((quotes[1] - quotes[0]) / spot).reindex(spot.index)
    history = spread.shift()
    z = (spread - history.rolling(60, min_periods=40).mean()) / history.rolling(60, min_periods=40).std()
    # Daily signal with a neutral zone. Long the cheaper leg when the spread is unusually wide.
    direction = np.sign(z).where(z.abs().ge(1.5), 0)
    ic["pair"] = ic.trade_date.map(direction).fillna(0) * (1 - 2 * ic.leg)
    ic["symbol"] = ic.leg.map({0: "IC_near", 1: "IC_far"})
    ic["multiplier"] = MULTIPLIERS["IC"]
    for day, field in (("entry", "open"), ("entry", "settle"), ("exit", "open")):
        ic[f"{day}_{field}"] = bars[field].reindex(
            pd.MultiIndex.from_arrays([ic[f"{day}_date"], ic.ts_code])).to_numpy()
    available = ic[ic[["entry_open", "entry_settle", "exit_open"]].gt(0).all(axis=1)]
    complete = available.groupby("trade_date").size().loc[lambda count: count.eq(2)].index
    fallback = panel[panel.symbol.eq("IC") & ~panel.trade_date.isin(complete)].assign(pair=0)
    return pd.concat([available[available.trade_date.isin(complete)], fallback], ignore_index=True)


def stress_ledger(ledger, quotes, fee, margin):
    rows = []
    equity = CAPITAL
    previous = {}
    for day in ledger.itertuples():
        current = {code: int(lots) for code, lots in
                   (leg.split(":") for leg in day.positions.split(";") if leg)}
        gap_buffer = equity - sum(abs(lots) * quotes[(day.entry_date, code)][0]
                                  * MULTIPLIERS[code[:2]] * margin for code, lots in previous.items())
        fees = sum(abs(current.get(code, 0) - previous.get(code, 0))
                   * quotes[(day.entry_date, code)][0] * MULTIPLIERS[code[:2]] * fee
                   for code in current.keys() | previous.keys())
        buffer = equity - fees
        for code, lots in current.items():
            opening, low, high = quotes[(day.entry_date, code)]
            adverse = low if lots > 0 else high
            multiplier = MULTIPLIERS[code[:2]]
            buffer += (lots * (adverse - opening) - abs(lots) * adverse * margin) * multiplier
        rows.append((gap_buffer, buffer))
        equity = day.equity
        previous = current
    return pd.DataFrame(rows, columns=["gap_buffer", "intraday_buffer"])


def directional_panel():
    panel, _ = build_panel()
    oos = pd.read_parquet(ROOT / "long_oos/predictions.parquet")
    scores = ["score_direction_5", "score_direction_60", "carry"]
    panel = panel.merge(oos[["trade_date", "symbol", *scores]], on=["trade_date", "symbol"],
                        validate="one_to_one")
    for window in (20, 60, 120):
        panel[f"trend{window}"] = panel[f"f_momentum_{window}"] / panel.f_volatility_20 / np.sqrt(window)
    panel["trend_carry"] = panel.trend60.where(panel.trend60 * panel.carry > 0, 0)
    panel["lr60_carry"] = panel.score_direction_60.where(panel.score_direction_60 * panel.carry > 0, 0)
    panel["vol_percentile"] = panel.groupby("symbol").f_volatility_20.transform(
        lambda vol: vol.rolling(120, min_periods=60).rank(pct=True)).fillna(1)
    return panel


def main():
    out = ROOT / "strategy_directions"
    out.mkdir(exist_ok=True)
    panel = directional_panel()
    scores = ["score_direction_5", "score_direction_60", "carry"]
    bars = load_bars()
    quotes = dict(zip(bars.index, bars[["open", "low", "high"]].itertuples(index=False, name=None)))
    cross = panel[panel.symbol.isin(["IF", "IM"])].copy()
    trend = cross.pivot(index="entry_date", columns="symbol", values="trend60")
    spread = trend.IM - trend.IF
    direction = np.sign(spread).where(spread.abs().ge(0.5), 0)
    cross = cross[cross.entry_date.isin(spread.dropna().index)].copy()
    cross["pair"] = cross.entry_date.map(direction) * cross.symbol.map({"IM": 1, "IF": -1})
    calendar = paired_panel(panel, bars)
    policies = {
        "carry": (panel, "carry", 0.0005, 10, "both", None),
        "lr5": (panel, scores[0], 0.20, 10, "both", None),
        "lr5_long_only": (panel, scores[0], 0.20, 10, "long", None),
        "lr60": (panel, scores[1], 0.30, 20, "both", None),
        "trend20": (panel, "trend20", 0.25, 5, "both", None),
        "trend60": (panel, "trend60", 0.25, 10, "both", None),
        "trend120": (panel, "trend120", 0.25, 20, "both", None),
        "trend60_long_only": (panel, "trend60", 0.25, 10, "long", None),
        "trend_carry": (panel, "trend_carry", 0.25, 10, "both", None),
        "trend_carry_vol": (panel, "trend_carry", 0.25, 10, "both", cash_risk),
        "carry_vol": (panel, "carry", 0.0005, 10, "both", cash_risk),
        "carry_reserve": (panel, "carry", 0.0005, 10, "both", margin_reserve),
        "lr5_reserve": (panel, scores[0], 0.20, 10, "both", margin_reserve),
        "lr60_carry_reserve": (panel, "lr60_carry", 0.30, 20, "both", margin_reserve),
        "lr60_carry": (panel, "lr60_carry", 0.30, 20, "both", None),
        "calendar_IC": (calendar, "pair", 0, 1, "both", None),
        "relative_IM_IF": (cross, "pair", 0, 1, "both", None),
    }
    results = []
    for start in ("20200101", "20220101", "20250101"):
        for name, (data, score, threshold, cadence, side, risk_filter) in policies.items():
            pair = score == "pair"
            sample = data[data.entry_date.ge(start)].copy()
            if pair:
                sample[score] = sample[score].where(sample.groupby("entry_date")[score].transform("size").eq(2), 0).fillna(0)
            groups = [list(g.itertuples(index=False)) for _, g in sample.groupby("entry_date")]
            for fee, margin in ((FEE, 0.15), (FEE * 2, 0.25)):
                ledger = simulate(groups, score, threshold, 1, side, cadence, fee,
                                  margin_rate=margin, capital=CAPITAL, enforce_margin=True,
                                  lot_cap=2 if pair else 1, atomic=pair, position_filter=risk_filter)
                if pair:
                    assert ledger.gross_lots.isin([0, 2]).all()
                assert (ledger.equity - CAPITAL - ledger.pnl.cumsum()).abs().lt(1e-6).all()
                ledger = ledger.join(stress_ledger(ledger, quotes, fee, margin))
                ledger.to_csv(out / f"{start}_{name}_{margin:.2f}.csv", index=False)
                results.append(dict(window=start, policy=name, margin=margin, fee=fee,
                                    start=ledger.entry_date.min(), end=ledger.exit_date.max(),
                                    forced=int(ledger.forced_exit.sum()),
                                    gap_breaches=int(ledger.gap_buffer.lt(0).sum()),
                                    intraday_breaches=int(ledger.intraday_buffer.lt(0).sum()),
                                    **metrics(ledger, capital=CAPITAL)))
    pd.DataFrame(results).to_csv(out / "summary.csv", index=False)


if __name__ == "__main__":
    main()
