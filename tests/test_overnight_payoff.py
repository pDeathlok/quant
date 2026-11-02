import numpy as np
import pytest

from quant.research.overnight_payoff import expected_payoff


def test_high_win_rate_does_not_imply_positive_payoff():
    result = expected_payoff(np.array([.8, .95]), np.array([.003, .003]), np.array([.025, .025]))
    assert result == pytest.approx([-.0026, .0016])
    with pytest.raises(ValueError):
        expected_payoff(np.array([1.1]), np.array([.003]), np.array([.025]))
    with pytest.raises(ValueError):
        expected_payoff(np.array([.8]), np.array([.003]), np.array([-1.]))
