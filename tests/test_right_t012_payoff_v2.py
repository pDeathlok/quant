import numpy as np
import pandas as pd
import pytest
from train_right_t012_payoff_v3 import exit_returns


def sample() -> pd.DataFrame:
    return pd.DataFrame(dict(entry_close=[10., 10.], next_open=[10., 10.], next_high=[11., 11.],
        next_low=[9.5, 9.5], next_close=[10.5, 10.5], next_valid_bar=[True, True],
        down_limit=[9., 9.], price_limit_inconsistent=[False, False],
        label_status=["ok", "corporate_action_unresolved"], next_pre_close=[10., 9.5]))


def test_corporate_action_is_unknown_for_every_exit_even_if_limit_hit() -> None:
    values = exit_returns(sample())
    assert values[0, 1] == pytest.approx(.05)
    assert np.isnan(values[1]).all()


def test_cannot_silently_omit_label_admission_columns() -> None:
    with pytest.raises(ValueError, match="admission"):
        exit_returns(sample().drop(columns="label_status"))


def test_reference_price_mismatch_cannot_be_overridden_by_ok_status() -> None:
    frame = sample()
    frame["label_status"] = "ok"
    assert np.isnan(exit_returns(frame)[1]).all()
