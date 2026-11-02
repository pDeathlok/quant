"""Conditional payoff mixture, with costs already included in each outcome."""

import numpy as np


def expected_payoff(probability: np.ndarray, gain: np.ndarray, loss: np.ndarray) -> np.ndarray:
    probability, gain, loss = np.broadcast_arrays(probability, gain, loss)
    if (not np.isfinite([probability, gain, loss]).all() or (probability < 0).any()
            or (probability > 1).any() or (gain < 0).any() or (loss < 0).any()):
        raise ValueError("Invalid conditional payoff inputs")
    return probability * gain - (1 - probability) * loss
