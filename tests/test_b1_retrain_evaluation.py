from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


RESEARCH_SCRIPTS = Path(__file__).parents[1] / "scripts" / "research"
if str(RESEARCH_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(RESEARCH_SCRIPTS))

from evaluate_b1_retrained_models import (  # noqa: E402
    EntryRule,
    _optional_threshold,
    apply_entry_rule,
    build_test_calibrated_entry_rules,
)


def test_apply_entry_rule_uses_direct_probability_thresholds() -> None:
    data = pd.DataFrame(
        {
            "pred_up5_es": [0.7, 0.4],
            "pred_up8_es": [0.6, 0.8],
            "pred_up10_es": [0.5, 0.5],
            "pred_down2_es": [0.2, 0.2],
            "pred_down3_es": [0.3, 0.6],
        }
    )
    rule = EntryRule("rule", min_up5=0.6, max_down3=0.4)

    assert apply_entry_rule(data, rule).tolist() == [True, False]


def test_adaptive_thresholds_are_calibrated_from_test_only() -> None:
    data = pd.DataFrame(
        {
            "split": ["test", "test", "oot"],
            "pred_up5_es": [0.1, 0.3, 0.99],
            "pred_up8_es": [0.2, 0.4, 0.99],
            "pred_up10_es": [0.3, 0.5, 0.99],
            "pred_down2_es": [0.4, 0.6, 0.01],
            "pred_down3_es": [0.5, 0.7, 0.01],
        }
    )

    rules = build_test_calibrated_entry_rules(data)
    rule = next(item for item in rules if item.name == "testq__up10_0.50__down3_0.10")

    assert rule.min_up10 == 0.4
    assert rule.max_down3 == 0.52


def test_nan_threshold_from_dataframe_is_restored_to_none() -> None:
    assert _optional_threshold(float("nan")) is None
    assert _optional_threshold(0.4) == 0.4
