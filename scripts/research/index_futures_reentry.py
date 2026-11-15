"""Causal product/side reentry memory; rolls are not new trades."""
import numpy as np


class ReentryGate:
    def __init__(self, rule: str = "none"):
        if rule not in ("none", "cooldown5", "reset"):
            raise ValueError(rule)
        self.rule = rule
        self.previous = None
        self.previous_day = None
        self.blocked = {}

    def observe(self, day, step, positions, rows):
        current = next(((code[:2], 1 if lots > 0 else -1)
                        for code, (lots, _) in positions.items()), None)
        # At this opening we observe the previous session's executed position.
        # A transition to flat occurred at the previous opening, not today.
        if self.previous is not None and current != self.previous:
            self.blocked[self.previous] = (step-1, self.previous_day)
        self.previous, self.previous_day = current, day
        if self.rule != "reset":
            return
        for row in rows:
            if not all(np.isfinite(x) for x in (row.etf_ret_1, row.long_60, row.short_60)):
                continue  # Missing evidence cannot reset an old signal.
            direction = 1 if row.long_60 > row.short_60 else -1
            valid = max(row.long_60, row.short_60) > .45 and abs(row.long_60-row.short_60) >= .10
            for side in (1, -1):
                key = (row.symbol, side)
                if key in self.blocked and row.trade_date >= self.blocked[key][1] and not (valid and side == direction):
                    del self.blocked[key]

    def admits(self, symbol: str, direction: int, step: int) -> bool:
        if self.rule == "none" or (symbol, direction) not in self.blocked:
            return True
        return self.rule == "cooldown5" and step-self.blocked[symbol, direction][0] > 5
