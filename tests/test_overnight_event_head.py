import pandas as pd
import pytest
from train_overnight_unfiltered_comparison import event_classification_labels


def test_event_target_does_not_confuse_profitable_fallback_or_unknown() -> None:
    frame = pd.DataFrame({"limit1": [.008, .003, -.03, float("nan")],
                          "limit1_penetrated": pd.array([True, False, False, None], dtype="boolean")})
    result = event_classification_labels(frame, "limit1")
    assert result.iloc[:3].tolist() == [True, False, False]
    assert pd.isna(result.iloc[3])
    frame.loc[3, "limit1_penetrated"] = False
    with pytest.raises(ValueError, match="knownness"):
        event_classification_labels(frame, "limit1")
