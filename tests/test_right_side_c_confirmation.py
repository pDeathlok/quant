from __future__ import annotations

import pandas as pd

from quant.research.right_side_c_confirmation import (
    POLICY_VERSION,
    c_confirmation_payload,
    evaluate_c_confirmation,
)


def _inputs(delta: float = 0.02):
    frozen = {
        "decision": "freeze_for_c_confirmation",
        "selected_candidate": "unified_long_task_deep",
        "scope": {"entry_mode": "next_close", "horizon": 5, "label": "good_path5"},
    }
    base = {
        "entry_mode": "next_close", "horizon": 5, "label": "good_path5", "fold": "C",
        "candidate": "unified_long_task_deep", "status": "ok", "paired_rows": 1000,
        "month_blocks": 7, "confidence_level": .95, "delta_pr_auc": delta,
        "delta_pr_auc_ci_low": -.001, "delta_pr_auc_ci_high": .04,
        "delta_pr_auc_bootstrap_valid": 500, "delta_top_lift": .03,
        "delta_daily_top_k_avg_terminal_return": .001,
    }
    paired = pd.DataFrame([
        {**base, "comparison_scope": "all_events"},
        {**base, "comparison_scope": "independent_model_rows", "paired_rows": 990},
    ])
    metrics = pd.DataFrame([{
        "entry_mode": "next_close", "horizon": 5, "label": "good_path5", "fold": "C",
        "experiment": "independent", "signal": "ALL", "rows": 1000, "fallback_rows": 10,
    }])
    signals = []
    for index in range(6):
        for experiment, ap in (("independent", .30), ("unified_long_task_deep", .31)):
            signals.append({
                "entry_mode": "next_close", "horizon": 5, "label": "good_path5", "fold": "C",
                "experiment": experiment, "signal": f"S{index}", "rows": 300,
                "positives": 100, "average_precision": ap,
            })
    return frozen, paired, metrics, pd.DataFrame(signals)


def test_c_confirmation_passes_only_to_shadow() -> None:
    result = evaluate_c_confirmation(*_inputs())
    assert result.decision == "confirmed_for_shadow"
    assert result.candidate == "unified_long_task_deep"
    assert result.checks["passed"].all()


def test_c_confirmation_rejects_negative_pr_auc() -> None:
    result = evaluate_c_confirmation(*_inputs(delta=-.01))
    assert result.decision == "rejected_after_c"
    failed = result.checks.loc[~result.checks["passed"], "check"].tolist()
    assert "delta_pr_auc_positive" in failed


def test_c_confirmation_cli_payload_contract() -> None:
    result = evaluate_c_confirmation(*_inputs())
    payload = c_confirmation_payload(result)

    assert payload["policy_version"] == POLICY_VERSION
    assert payload["decision"] == "confirmed_for_shadow"
    assert payload["candidate"] == "unified_long_task_deep"
    assert payload["checks"]
    assert payload["limitations"]
