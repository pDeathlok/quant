"""Re-evaluate frozen and validation-selected policies at RMB 500,000."""
from pathlib import Path

import pandas as pd

from backtest_index_futures import HOLDOUT, ROOT, VALIDATION, metrics, simulate

CAPITAL = 500_000
PARAMETERS = ["score", "threshold", "lots_per_product", "side", "cadence"]


def main():
    output = Path(ROOT / "capital_500k")
    output.mkdir(exist_ok=True)
    predictions = pd.read_parquet(ROOT / "predictions.parquet")
    periods = {
        "validation": predictions[predictions.entry_date.ge(VALIDATION)
                                  & predictions.exit_date.lt(HOLDOUT)],
        "holdout": predictions[predictions.entry_date.ge(HOLDOUT)],
    }
    groups = {name: [list(rows.itertuples(index=False))
                     for _, rows in panel.groupby("entry_date", sort=True)]
              for name, panel in periods.items()}

    def replay(params, period, cap, friction=0.0003, margin=0.15, symbol=None):
        rows = groups[period]
        if symbol:
            rows = [[row for row in group if row.symbol == symbol] for group in rows]
        ledger = simulate(rows, *params, friction=friction, margin_rate=margin,
                          capital=CAPITAL, enforce_margin=True, lot_cap=cap)
        assert ledger.gross_lots.le(cap).all()
        assert (ledger.equity - CAPITAL - ledger.pnl.cumsum()).abs().lt(1e-6).all()
        stats = metrics(ledger, capital=CAPITAL)
        stats.update(max_lots=int(ledger.gross_lots.max()),
                     reduced_days=int(ledger.gross_lots.lt(ledger.requested_lots).sum()),
                     forced_exits=int(ledger.forced_exit.sum()),
                     settlement_breaches=int(ledger.settlement_breach.sum()))
        return ledger, stats

    policies = pd.read_csv(ROOT / "validation.csv")[PARAMETERS]
    frozen = pd.read_csv(ROOT / "selected.csv").query("period == 'validation' and friction == 0.0003")
    selected = []
    for cap in (1, 2):
        evaluations = []
        for params in policies.itertuples(index=False, name=None):
            _, stats = replay(params, "validation", cap)
            evaluations.append(dict(zip(PARAMETERS, params), **stats))
        candidates = pd.DataFrame(evaluations)
        candidates.to_csv(output / f"validation_{cap}.csv", index=False)
        eligible = candidates.query("active_days >= 60 and forced_exits == 0 and settlement_breaches == 0")
        picks = []
        for name, sort in {
            "best_return": ["total_return", "sharpe"],
            "best_sharpe": ["sharpe", "max_drawdown"],
            "low_drawdown": ["max_drawdown", "sharpe"],
        }.items():
            pool = eligible.query("total_return > 0") if name == "low_drawdown" else eligible
            winner = pool.sort_values(sort, ascending=False).iloc[0]
            picks.append((name, tuple(winner[PARAMETERS]), None))
        for row in frozen.itertuples(index=False):
            if row.objective in ("return", "drawdown", "two_way") or row.objective.startswith("long_"):
                symbol = row.objective[5:] if row.objective.startswith("long_") else None
                picks.append((f"original_{row.objective}",
                              (row.score, row.threshold, row.lots, row.side, row.cadence), symbol))
        for name, params, symbol in picks:
            for period in groups:
                for friction, margin in ((0.0003, 0.15), (0.0006, 0.25)):
                    ledger, stats = replay(params, period, cap, friction, margin, symbol)
                    ledger.to_csv(output / f"{name}_{cap}_{period}_{margin:.2f}.csv", index=False)
                    selected.append(dict(policy=name, cap=cap, period=period, margin=margin,
                                         friction=friction, **dict(zip(PARAMETERS, params)), **stats))
    pd.DataFrame(selected).to_csv(output / "selected.csv", index=False)


if __name__ == "__main__":
    main()
