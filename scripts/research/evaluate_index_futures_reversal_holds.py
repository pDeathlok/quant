"""Evaluate reversal entries with explicit holding and exit rules by calendar year."""
import numpy as np
import pandas as pd

from backtest_index_futures import ROOT, metrics, simulate
from evaluate_index_futures_directions import CAPITAL, directional_panel, load_bars
from evaluate_index_futures_lr import ETFS, etf_features
from evaluate_index_futures_reversals import futures_features, short_episodes

STARTS = ("20200101", "20220101", "20250101")


def main():
    out = ROOT / "reversal_holds"
    out.mkdir(exist_ok=True)
    bars = load_bars()
    panel = directional_panel()
    etf = pd.concat(etf_features(symbol, ticker) for symbol, ticker in ETFS.items())
    panel = panel.merge(etf, on=["trade_date", "symbol"], how="left", validate="one_to_one")
    panel = panel.merge(futures_features(bars), on=["trade_date", "ts_code"], validate="one_to_one")
    panel["reversal"] = (panel.etf_drawdown_20.le(-.05) & panel.etf_ret_1.gt(0)
                         & panel.etf_vol_ratio.gt(1.2))
    panel["confirmed"] = (panel.reversal & panel.future_close_location.ge(.65)
                          & panel.future_volume_ratio.ge(1))
    panel["carry_entry"] = panel.reversal & panel.carry.gt(.0005)
    # Test a separate bearish reversal rather than assuming a long signal's inverse.
    panel["bearish"] = (panel.etf_drawdown_20.ge(-.01) & panel.etf_ret_1.lt(0)
                        & panel.etf_vol_ratio.gt(1.2) & panel.future_close_location.le(.35)
                        & panel.future_volume_ratio.ge(1))
    panel["flat"] = 0.
    for symbol in ETFS:
        panel[f"long_{symbol}"] = panel.symbol.eq(symbol).astype(float)
        panel[f"short_{symbol}"] = -panel.symbol.eq(symbol).astype(float)
    closes = bars.close.to_dict()
    opens = bars.open.to_dict()
    configurations = [(signal, days, barrier) for signal in
                      ("reversal", "confirmed", "carry_entry", "bearish")
                      for days in (10, 20) for barrier in (False, True)]
    summaries, annual, episodes = [], [], []
    for start in STARTS:
        groups = [list(g.itertuples(index=False))
                  for _, g in panel[panel.entry_date.ge(start)].groupby("entry_date")]
        by_day = {rows[0].entry_date: rows for rows in groups}
        for fee, margin in ((.0003, .15), (.0006, .25)):
            for signal, days, barrier in [("carry", 0, False), *configurations]:
                held_since = 0
                first_price = 0.
                rolled_pnl = 0.
                last_code = ""
                last_open = 0.
                step = -1
                side = -1 if signal == "bearish" else 1

                def select(day, positions):
                    nonlocal held_since, first_price, rolled_pnl, last_code, last_open, step
                    step += 1
                    rows = by_day[day]
                    if positions:
                        code = next(iter(positions))
                        row = next(r for r in rows if r.symbol == code[:2])
                        mark = closes[(row.trade_date, code)]
                        if last_code != code:
                            raise ValueError("Unexpected contract after roll")
                        price_pnl = side * (rolled_pnl + mark - last_open)
                        # Closing signal uses the previous close of the actually held contract.
                        if step - held_since >= days or (
                            barrier and (price_pnl >= .04 * first_price or price_pnl <= -.025 * first_price)
                        ):
                            return ("flat", .5, 1, "both", None)
                        if row.ts_code != code:
                            rolled_pnl += opens[(day, code)] - last_open
                            last_code, last_open = row.ts_code, row.entry_open
                        return None
                    candidates = [row for row in rows if getattr(row, signal)]
                    if not candidates:
                        return ("flat", .5, 1, "both", None)
                    row = max(candidates, key=lambda r: r.etf_vol_ratio)
                    held_since, first_price, rolled_pnl = step, row.entry_open, 0.
                    last_code, last_open = row.ts_code, row.entry_open
                    return (f"{'long' if side > 0 else 'short'}_{row.symbol}", .5, 10_000, "both", None)

                ledger = simulate(groups, "carry", .0005, 1, "both", 10, fee,
                                  margin_rate=margin, capital=CAPITAL, enforce_margin=True,
                                  lot_cap=1, policy_selector=None if signal == "carry" else select)
                assert ledger.gross_lots.le(1).all()
                assert np.allclose(ledger.equity, CAPITAL + ledger.pnl.cumsum(), atol=1e-6, rtol=0)
                name = "carry" if not days else f"{signal}_{days}_{'barrier' if barrier else 'fixed'}"
                fields = dict(start=start, policy=name, margin=margin, fee=fee)
                if signal != "carry":
                    trades = short_episodes(ledger, bars, fee=fee, direction=side)
                    if not trades.empty:
                        assert np.isclose(trades.pnl.sum(), ledger.pnl.sum())
                        episodes.append(trades.assign(**fields))
                summaries.append(dict(**fields, **metrics(ledger, CAPITAL),
                                      forced=int(ledger.forced_exit.sum())))
                equity = CAPITAL
                for year, part in ledger.groupby(ledger.exit_date.str[:4]):
                    values = np.r_[equity, part.equity]
                    annual.append(dict(**fields, year=year, total_return=values[-1] / equity - 1,
                                       max_drawdown=(values / np.maximum.accumulate(values) - 1).min(),
                                       forced=int(part.forced_exit.sum())))
                    equity = values[-1]
    pd.DataFrame(summaries).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(annual).to_csv(out / "annual.csv", index=False)
    pd.concat(episodes).to_csv(out / "episodes.csv", index=False)


if __name__ == "__main__":
    main()
