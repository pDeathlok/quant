import numpy as np
import pandas as pd
from run_right_t012_adaptive_exit import with_action


def frame() -> pd.DataFrame:
    values = {}
    for name, gain, loss, gross in [("open", .02, .001, np.nan), ("close", .01, .002, .5),
                                     ("tp2", .01, .003, -.1), ("tp5", .01, .004, .1)]:
        values[f"gain_{name}"] = [gain]
        values[f"loss_{name}"] = [loss]
        values[f"gross_{name}"] = [gross]
    return pd.DataFrame(values)


def test_action_never_uses_future_return_or_knownness() -> None:
    data = frame()
    chosen = with_action(data, .5)
    assert chosen.selected_exit.iloc[0] == "open"
    assert np.isnan(chosen.gross.iloc[0])
    for c in data:
        if c.startswith("gross_"):
            data[c] = 999.
    assert with_action(data, .5).selected_exit.iloc[0] == "open"
