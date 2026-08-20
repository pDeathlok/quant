from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.research.right_side_playbook_dataset import (
    EVENT_FACTOR_COLUMNS,
    FirstLayerPredictionContract,
    PLAYBOOK_OUTCOME_COLUMNS,
    audit_narrow_playbook_tables,
    audit_reusable_playbook_events,
    join_playbook_event_outcomes,
    narrow_playbook_outcomes,
    prepare_playbook_events,
)
from quant.research.right_side_playbook_model import (
    FIRST_LAYER_FOLD_COLUMN,
    FIRST_LAYER_PROVENANCE_COLUMN,
    FIRST_LAYER_SCORE_COLUMN,
)
from quant.research.right_side_playbook_policy import (
    DEFAULT_PLAYBOOK_CATALOG,
    NO_TRADE_PLAYBOOK_ID,
)


def _factors() -> pd.DataFrame:
    values: dict[str, object] = {
        "symbol": ["000001.SZ", "000002.SZ"],
        "date": [pd.Timestamp("2024-01-02"), pd.Timestamp("2025-01-02")],
    }
    for position, column in enumerate(EVENT_FACTOR_COLUMNS):
        values[column] = [float(position), float(position + 1)]
    for column in ("B2", "has_right_signal"):
        values[column] = [True, True]
    return pd.DataFrame(values)


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fold": ["A", "B"],
            "symbol": ["000001.SZ", "000002.SZ"],
            "date": [pd.Timestamp("2024-01-02"), pd.Timestamp("2025-01-02")],
            FIRST_LAYER_SCORE_COLUMN: [0.4, 0.7],
            FIRST_LAYER_PROVENANCE_COLUMN: ["oof", "test"],
            FIRST_LAYER_FOLD_COLUMN: ["A", "B"],
        }
    )


def _outcomes(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for event in events[["fold", "event_id", "symbol", "date"]].to_dict("records"):
        for spec in DEFAULT_PLAYBOOK_CATALOG:
            no_trade = spec.playbook_id == NO_TRADE_PLAYBOOK_ID
            rows.append(
                {
                    **event,
                    "playbook_id": spec.playbook_id,
                    "playbook_version": spec.version,
                    "entry_mode": "no_trade" if no_trade else spec.entry.mode,
                    "exit_policy_id": spec.exit.policy_id,
                    "eligible": True,
                    "eligibility_reason": "eligible",
                    "mature": True,
                    "maturity_reason": "not_applicable" if no_trade else "mature",
                    "entry_date": pd.NaT if no_trade else event["date"] + pd.Timedelta(days=1),
                    "entry_price": np.nan if no_trade else 10.0,
                    "entry_raw_price": np.nan if no_trade else 10.0,
                    "exit_date": pd.NaT if no_trade else event["date"] + pd.Timedelta(days=3),
                    "exit_price": np.nan if no_trade else 10.1,
                    "exit_raw_price": np.nan if no_trade else 10.1,
                    "exit_reason": "no_trade" if no_trade else "expiry",
                    "gross_return": 0.0 if no_trade else 0.01,
                    "net_return": 0.0 if no_trade else 0.0085,
                    "mae": 0.0 if no_trade else -0.01,
                    "holding_sessions": 0 if no_trade else 2,
                    "ambiguous_bar": False,
                    "round_trip_cost": 0.0 if no_trade else 0.0015,
                    "round_trip_cost_bps": 0.0 if no_trade else 15.0,
                    "locked_limit_up": False,
                    "locked_limit_source": "not_applicable" if no_trade else "ohlc_4p8_proxy",
                    "open_at_up_limit": False,
                    "close_at_up_limit": False,
                    "open_gap": np.nan if no_trade else 0.01,
                    # This simulates an accidental wide policy-engine payload;
                    # the normalized outcome table must remove it.
                    "rs_accidental_factor_copy": 99.0,
                }
            )
    return pd.DataFrame(rows)


def test_narrow_tables_keep_wide_factors_once_and_cover_every_action() -> None:
    events = prepare_playbook_events(_factors(), _predictions())
    outcomes = narrow_playbook_outcomes(_outcomes(events))
    audit = audit_narrow_playbook_tables(events, outcomes)

    assert len(events) == 2
    assert len(outcomes) == 2 * len(DEFAULT_PLAYBOOK_CATALOG)
    assert "rs_accidental_factor_copy" not in outcomes
    assert set(EVENT_FACTOR_COLUMNS).isdisjoint(outcomes.columns)
    assert audit["rule_feature_count"] == 118
    assert audit["actions_per_event"] == 9
    assert audit["known_utility_regular_coverage"] == pytest.approx(1.0)
    assert audit["eligible_immature_null_utility_rows"] == 0
    joined = join_playbook_event_outcomes(
        events,
        outcomes,
        event_feature_columns=["B2", EVENT_FACTOR_COLUMNS[0]],
    )
    assert len(joined) == len(outcomes)
    assert joined.groupby(["fold", "event_id"])[FIRST_LAYER_SCORE_COLUMN].nunique().eq(1).all()


def test_event_builder_rejects_missing_factor_rows_and_fold_c() -> None:
    with pytest.raises(ValueError, match="lack factor rows"):
        prepare_playbook_events(_factors().head(1), _predictions())

    contaminated = _predictions().copy()
    contaminated.loc[1, "fold"] = "C"
    contaminated.loc[1, FIRST_LAYER_FOLD_COLUMN] = "C"
    with pytest.raises(ValueError, match="fold C is forbidden"):
        prepare_playbook_events(_factors(), contaminated)


def test_first_layer_candidate_and_score_column_must_match() -> None:
    with pytest.raises(ValueError, match="does not match score column"):
        FirstLayerPredictionContract(
            score_column="pred_unified_long_task_deep",
            selected_candidate="unified_long_task_deep_beam_residual_v3",
        )


def test_reusable_events_require_exact_selected_score_and_provenance(tmp_path) -> None:
    events = prepare_playbook_events(_factors(), _predictions())
    events_path = tmp_path / "events.parquet"
    predictions_path = tmp_path / "predictions.parquet"
    factor_path = tmp_path / "factors.parquet"
    events.to_parquet(events_path, index=False)
    _predictions().to_parquet(predictions_path, index=False)
    _factors().to_parquet(factor_path, index=False)

    audit = audit_reusable_playbook_events(
        events,
        _predictions(),
        events_path=events_path,
        predictions_path=predictions_path,
        factor_dataset_path=factor_path,
    )

    assert audit["rows"] == 2
    assert audit["fold_rows"] == {"A": 1, "B": 1}
    assert audit["reuse_validation"]["keys"] == "exact_A_B_one_to_one"
    assert len(audit["reuse_validation"]["event_sha256"]) == 64

    corrupted = events.copy()
    corrupted.loc[0, FIRST_LAYER_SCORE_COLUMN] += 0.01
    with pytest.raises(ValueError, match="scores differ"):
        audit_reusable_playbook_events(
            corrupted,
            _predictions(),
            events_path=events_path,
            predictions_path=predictions_path,
            factor_dataset_path=factor_path,
        )


def test_audit_rejects_action_gaps_and_factor_copies() -> None:
    events = prepare_playbook_events(_factors(), _predictions())
    outcomes = narrow_playbook_outcomes(_outcomes(events))
    with pytest.raises(ValueError, match="every event"):
        audit_narrow_playbook_tables(events, outcomes.iloc[:-1].copy())

    copied = outcomes.copy()
    copied[EVENT_FACTOR_COLUMNS[0]] = 0.0
    with pytest.raises(ValueError, match="must not be repeated"):
        audit_narrow_playbook_tables(events, copied)


def test_narrow_outcome_schema_is_explicit_and_stable() -> None:
    events = prepare_playbook_events(_factors(), _predictions())
    outcomes = narrow_playbook_outcomes(_outcomes(events))

    assert tuple(outcomes.columns) == PLAYBOOK_OUTCOME_COLUMNS
    assert str(outcomes["holding_sessions"].dtype) == "Int64"
    assert pd.api.types.is_datetime64_any_dtype(outcomes["entry_date"])
