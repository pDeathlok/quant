import numpy as np
import pandas as pd
import pytest

from quant.research.overnight_uncertainty import date_block_audit


def test_same_day_replicates_do_not_create_independent_evidence():
    dates = pd.bdate_range("2025-01-01", periods=30)
    picks = pd.DataFrame({"date": dates, "net": [.01] * 25 + [-.03] * 5})
    one = date_block_audit(picks, dates)
    many = date_block_audit(pd.concat([picks] * 100), dates)
    assert one["win_missing_as_failure_percentile95"] == many["win_missing_as_failure_percentile95"]
    assert one["effective_signal_dates"] == pytest.approx(many["effective_signal_dates"])
    assert one["known_mean_net_percentile95"] == pytest.approx(many["known_mean_net_percentile95"])


def test_unknown_returns_are_not_cash_and_dates_must_be_complete():
    dates = pd.bdate_range("2025-01-01", periods=30)
    picks = pd.DataFrame({"date": dates[:2], "net": [.01, np.nan]})
    result = date_block_audit(picks, dates)
    assert result["unknown"] == 1
    assert result["win_missing_as_failure"] == .5
    assert result["known_mean_net"] == .01
    assert result["effective_signal_dates"] == 2.
    with pytest.raises(ValueError, match="outside"):
        date_block_audit(picks, dates[1:])


def test_empty_and_degenerate_samples_do_not_claim_certainty():
    dates = pd.bdate_range("2025-01-01", periods=30)
    empty = date_block_audit(pd.DataFrame({"date": [], "net": []}), dates)
    assert empty["win_missing_as_failure_percentile95"] is None
    assert empty["nonempty_draws"] == 0
    perfect = date_block_audit(pd.DataFrame({"date": dates, "net": .01}), dates)
    assert perfect["degenerate_win_sample"]
