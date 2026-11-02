import pandas as pd
import pytest
from audit_overnight_model_agreement import consensus


def scores() -> pd.DataFrame:
    return pd.DataFrame({"symbol": ["a", "b", "c"], "date": pd.to_datetime(["2026-01-05"] * 3),
                         "old_pool": [True] * 3, "probability": [.95, .8, .99],
                         "expected_net": [.01] * 3, "risk": [.05, .05, .2]})


def test_consensus_requires_both_models_and_does_not_use_outcomes() -> None:
    a = scores()
    b = scores()
    b["probability"] = [.69, .81, .99]
    wanted = consensus(a, b)
    assert wanted.symbol.tolist() == ["b"]
    assert wanted.agreement_score.tolist() == [.8]
    pd.testing.assert_frame_equal(wanted, consensus(a.assign(net=[1., -1., 1.]), b.assign(event_hit=[True, False, True])))


def test_consensus_rejects_missing_prediction_and_membership_disagreement() -> None:
    a = scores()
    with pytest.raises(ValueError, match="Different candidate"):
        consensus(a, a.iloc[1:])
    b = a.copy()
    b.loc[0, "old_pool"] = False
    with pytest.raises(ValueError, match="Different candidate"):
        consensus(a, b)
