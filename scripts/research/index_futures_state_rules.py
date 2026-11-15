"""Side-symmetric state-dependent evidence and realized-account exit rules."""
import numpy as np


def evidence_admits(probability: float, net: float, risk: float) -> bool:
    if not np.isfinite(probability):
        if np.isfinite(net) or np.isfinite(risk):
            raise ValueError("Partial prediction without probability")
        return True
    return bool(np.isfinite(net) and np.isfinite(risk) and probability>=.55 and net-.5*risk>0)


def holding_limit(state: int, direction: int) -> int:
    return 20 if state==direction else 10 if state==0 else 5


def profit_giveback(state: int, direction: int, peak: float, net: float,
                    entry_equity: float, adaptive: bool = True) -> bool:
    fraction=.5 if not adaptive or state==direction else .25
    return peak>=.05*entry_equity and net<=peak*(1-fraction)
