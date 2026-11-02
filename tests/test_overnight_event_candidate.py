import pytest
from audit_overnight_event_candidate import wilson_interval


def test_small_sample_does_not_prove_high_probability() -> None:
    lower, upper = wilson_interval(25, 30)
    assert lower < .8 < 25 / 30 < upper
    assert wilson_interval(3, 3)[0] < .8
    assert wilson_interval(0, 30)[0] == 0
    with pytest.raises(ValueError):
        wilson_interval(1, 0)
