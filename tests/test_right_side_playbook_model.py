from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.features.variable_library import PROJECT_FACTOR_COLUMNS
from quant.research.right_side_playbook_model import (
    FIRST_LAYER_FOLD_COLUMN,
    FIRST_LAYER_PROVENANCE_COLUMN,
    FIRST_LAYER_SCORE_COLUMN,
    NO_TRADE_PLAYBOOK_ID,
    SharedPlaybookModel,
    admit_default_event_features,
    apply_execution_gate,
    attach_playbook_utility,
    audit_playbook_model_dataset,
    evaluate_playbook_selections,
    fit_static_global_playbook,
    fit_static_per_signal_playbooks,
    fold_by_name,
    score_static_global_playbook,
    score_static_per_signal_playbooks,
    select_no_trade_baseline,
    select_oracle_playbook,
    select_planned_playbook,
    split_playbook_time_fold,
    validate_first_layer_score_contract,
    validate_playbook_feature_columns,
)
from quant.research.right_side_unified import RIGHT_SIDE_SIGNALS
from quant.research.right_side_unified_features import RULE_FEATURE_COLUMNS


def _event_actions(
    event_id: str,
    *,
    date: str,
    fold: str,
    context: float,
    p1_utility: float,
    p2_utility: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for playbook_id, entry_mode, exit_policy_id, utility in (
        ("OPEN_EXPIRY_T5", "next_open", "expiry_T5", p1_utility),
        ("CLOSE_TP4_SL2", "next_close", "tp4_sl2_T5", p2_utility),
        (NO_TRADE_PLAYBOOK_ID, "none", "none", 0.0),
    ):
        mae = -0.04 if playbook_id != NO_TRADE_PLAYBOOK_ID else 0.0
        net_return = utility + 0.25 * abs(min(mae, 0.0))
        rows.append(
            {
                "event_id": event_id,
                "symbol": "000001.SZ",
                "date": pd.Timestamp(date),
                "fold": fold,
                "playbook_id": playbook_id,
                "entry_mode": entry_mode,
                "exit_policy_id": exit_policy_id,
                "eligible": True,
                "mature": True,
                "net_return": net_return,
                "mae": mae,
                "round_trip_cost": (
                    0.0 if playbook_id == NO_TRADE_PLAYBOOK_ID else 0.0015
                ),
                "context": context,
                FIRST_LAYER_SCORE_COLUMN: 0.65,
                FIRST_LAYER_PROVENANCE_COLUMN: "test",
                FIRST_LAYER_FOLD_COLUMN: fold,
            }
        )
    return rows


def _synthetic() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold, year, size in (("A", 2024, 300), ("B", 2025, 120)):
        for position in range(size):
            context = -1.0 if position % 2 == 0 else 1.0
            rows.extend(
                _event_actions(
                    f"{fold}-{position}",
                    date=f"{year}-{position % 12 + 1:02d}-{position % 20 + 1:02d}",
                    fold=fold,
                    context=context,
                    p1_utility=0.10 if context > 0 else -0.02,
                    p2_utility=0.06 if context < 0 else -0.08,
                )
            )
    return pd.DataFrame(rows)


def test_utility_formula_and_no_trade_fixed_zero() -> None:
    frame = pd.DataFrame(
        {
            "playbook_id": ["A", "B", NO_TRADE_PLAYBOOK_ID],
            "mature": [True, True, True],
            "net_return": [0.05, -0.01, 99.0],
            "mae": [-0.04, 0.02, -1.0],
        }
    )
    actual = attach_playbook_utility(frame)

    assert actual["utility"].tolist() == pytest.approx([0.04, -0.01, 0.0])


def test_future_columns_and_in_sample_first_layer_scores_fail_closed() -> None:
    with pytest.raises(ValueError, match="forbidden model features"):
        validate_playbook_feature_columns(["context", "net_return", "open_gap"])

    frame = _synthetic().head(3).copy()
    frame[FIRST_LAYER_PROVENANCE_COLUMN] = "in_sample"
    with pytest.raises(ValueError, match="provenance"):
        validate_first_layer_score_contract(frame)


def test_audit_checks_unique_actions_and_outcome_masks() -> None:
    frame = _synthetic().head(6).copy()
    audit = audit_playbook_model_dataset(frame, event_feature_columns=["context"])

    assert audit["events"] == 2
    assert audit["planned_action_training_rows"] == 4
    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        audit_playbook_model_dataset(duplicate, event_feature_columns=["context"])
    nonfinite = frame.copy()
    nonfinite.loc[0, "net_return"] = np.inf
    with pytest.raises(ValueError, match="finite net_return"):
        audit_playbook_model_dataset(nonfinite, event_feature_columns=["context"])
    bad_no_trade = frame.copy()
    bad_no_trade.loc[
        bad_no_trade["playbook_id"].eq(NO_TRADE_PLAYBOOK_ID), "round_trip_cost"
    ] = 0.0015
    with pytest.raises(ValueError, match="NO_TRADE net_return"):
        audit_playbook_model_dataset(bad_no_trade, event_feature_columns=["context"])


def test_default_feature_admission_drops_all_null_project_but_keeps_all_rules() -> None:
    values = {
        column: [float(position), float(position + 1), float(position + 2)]
        for position, column in enumerate(RULE_FEATURE_COLUMNS)
    }
    values.update({column: [False, True, False] for column in RIGHT_SIDE_SIGNALS})
    values[PROJECT_FACTOR_COLUMNS[0]] = [1.0, 2.0, 3.0]
    values[PROJECT_FACTOR_COLUMNS[1]] = [np.nan, np.nan, np.nan]
    frame = pd.DataFrame(values)

    admitted = admit_default_event_features(frame)

    assert PROJECT_FACTOR_COLUMNS[0] in admitted
    assert PROJECT_FACTOR_COLUMNS[1] not in admitted
    assert set(RULE_FEATURE_COLUMNS) <= set(admitted)
    assert set(RIGHT_SIDE_SIGNALS) <= set(admitted)


def test_ineligible_high_score_plan_is_cancelled_not_replaced_ex_post() -> None:
    frame = pd.DataFrame(
        {
            "fold": ["B", "B", "B", "B", "B", "B"],
            "event_id": ["one"] * 3 + ["two"] * 3,
            "playbook_id": ["A", "B", NO_TRADE_PLAYBOOK_ID] * 2,
            "eligible": [False, True, True, True, True, True],
            "predicted_utility": [10.0, 0.02, 0.0, -0.01, 0.0, 0.0],
        }
    )
    planned = select_planned_playbook(frame)
    assert planned.set_index("event_id").loc["one", "playbook_id"] == "A"
    selected = apply_execution_gate(planned, frame).set_index("event_id")

    assert selected.loc["one", "playbook_id"] == NO_TRADE_PLAYBOOK_ID
    assert selected.loc["one", "planned_playbook_id"] == "A"
    assert selected.loc["one", "execution_status"] == "cancelled_ineligible"
    assert selected.loc["two", "playbook_id"] == NO_TRADE_PLAYBOOK_ID
    assert bool(selected.loc["two", "abstained"])


def test_planned_action_tie_break_is_stable_and_does_not_choose_worst() -> None:
    frame = pd.DataFrame(
        {
            "fold": ["B", "B", "B"],
            "event_id": ["tie"] * 3,
            "playbook_id": ["B", "A", NO_TRADE_PLAYBOOK_ID],
            "eligible": [True, True, True],
            "predicted_utility": [0.03, 0.03, 0.0],
        }
    )

    selected = select_planned_playbook(frame).iloc[0]
    assert selected["playbook_id"] == "B"
    assert selected["predicted_utility"] == pytest.approx(0.03)


def test_shared_model_learns_context_specific_actions_from_prior_oos_fold() -> None:
    frame = _synthetic()
    split = split_playbook_time_fold(frame, fold_by_name("B"))
    model = SharedPlaybookModel.fit(
        split.train,
        event_feature_columns=["context"],
        fold="B",
    )
    scored = model.score_actions(split.test)
    planned = select_planned_playbook(scored)
    selected = apply_execution_gate(planned, scored)
    expected = np.where(
        selected["context"].to_numpy() > 0,
        "OPEN_EXPIRY_T5",
        "CLOSE_TP4_SL2",
    )

    assert (selected["playbook_id"].to_numpy() == expected).mean() >= 0.95
    assert not selected["abstained"].any()


def test_normalized_full_fit_matches_context_policy_without_wide_action_copy(
    tmp_path,
) -> None:
    frame = _synthetic()
    train = frame.loc[frame["fold"].eq("A")].copy()
    test = frame.loc[frame["fold"].eq("B")].copy()
    event_columns = [
        "fold",
        "event_id",
        "symbol",
        "date",
        FIRST_LAYER_SCORE_COLUMN,
        FIRST_LAYER_PROVENANCE_COLUMN,
        FIRST_LAYER_FOLD_COLUMN,
        "context",
    ]
    events = train[event_columns].drop_duplicates(["fold", "event_id"])
    outcomes = train.drop(
        columns=[
            "context",
            FIRST_LAYER_SCORE_COLUMN,
            FIRST_LAYER_PROVENANCE_COLUMN,
            FIRST_LAYER_FOLD_COLUMN,
        ]
    )

    model = SharedPlaybookModel.fit_normalized(
        events,
        outcomes,
        event_feature_columns=["context"],
        fold="B",
        scratch_dir=tmp_path,
    )
    scored = model.score_actions(test)
    selected = apply_execution_gate(select_planned_playbook(scored), scored)
    expected = np.where(
        selected["context"].to_numpy() > 0,
        "OPEN_EXPIRY_T5",
        "CLOSE_TP4_SL2",
    )

    assert (selected["playbook_id"].to_numpy() == expected).mean() >= 0.95
    assert model.training_rows == 600
    assert model.eligible_mature_training_rows == 600
    assert model.ineligible_cancel_training_rows == 0


def test_time_split_keeps_all_event_actions_together() -> None:
    frame = _synthetic()
    split = split_playbook_time_fold(frame, fold_by_name("B"))

    assert set(split.train["fold"]) == {"A"}
    assert set(split.test["fold"]) == {"B"}
    assert split.train["date"].max() < split.test["date"].min()
    frame.loc[frame["event_id"].eq("B-0"), "event_id"] = "A-0"
    with pytest.raises(ValueError, match="cross train/test"):
        split_playbook_time_fold(frame, fold_by_name("B"))


def test_manifest_freezes_features_catalog_cost_utility_and_cutoff() -> None:
    frame = _synthetic()
    split = split_playbook_time_fold(frame, fold_by_name("B"))
    model = SharedPlaybookModel.fit(
        split.train,
        event_feature_columns=["context"],
        fold="B",
    )
    manifest = model.manifest(
        playbook_catalog_version="right-side-playbook-v1",
        data_cutoff="2025-12-31",
    )

    assert manifest["event_features"] == ["context"]
    assert manifest["playbook_catalog_version"] == "right-side-playbook-v1"
    assert len(manifest["playbook_catalog_sha256"]) == 64
    assert {row["playbook_id"] for row in manifest["playbook_catalog"]} >= {
        NO_TRADE_PLAYBOOK_ID,
        "next_open__expiry_t5_close",
    }
    assert manifest["round_trip_cost_source"] == "pre_registered_playbook_parameter"
    assert "T+1 ineligible" in manifest["utility_formula"]
    assert manifest["evaluation_fold"] == "B"
    assert manifest["training_first_layer_score_folds"] == ["A"]
    assert manifest["data_cutoff"] == "2025-12-31"
    assert manifest["release_status"] == "research_only_not_production"


def test_static_baselines_are_fitted_on_a_and_precommitted_on_b() -> None:
    frame = _synthetic()
    frame["positive_context"] = frame["context"].gt(0)
    frame["negative_context"] = frame["context"].lt(0)
    train = frame[frame["fold"].eq("A")].copy()
    test = frame[frame["fold"].eq("B")].copy()

    global_choice = fit_static_global_playbook(train)
    global_selected = apply_execution_gate(
        select_planned_playbook(score_static_global_playbook(test, global_choice)),
        test,
    )
    policy = fit_static_per_signal_playbooks(
        train,
        signal_columns=["positive_context", "negative_context"],
    )
    signal_scored = score_static_per_signal_playbooks(test, policy)
    signal_planned = select_planned_playbook(signal_scored)
    signal_selected = apply_execution_gate(signal_planned, signal_scored)

    assert global_choice.playbook_id == "OPEN_EXPIRY_T5"
    assert set(signal_selected.loc[signal_selected["context"].gt(0), "playbook_id"]) == {
        "OPEN_EXPIRY_T5"
    }
    assert set(signal_selected.loc[signal_selected["context"].lt(0), "playbook_id"]) == {
        "CLOSE_TP4_SL2"
    }
    global_metrics = evaluate_playbook_selections(global_selected, arm="static_global")
    signal_metrics = evaluate_playbook_selections(signal_selected, arm="static_per_signal")
    assert signal_metrics["average_event_net_return"] > global_metrics["average_event_net_return"]
    assert signal_metrics["event_coverage"] == pytest.approx(1.0)
    assert signal_metrics["unevaluated_events"] == 0
    assert signal_metrics["unevaluated_trades"] == 0
    assert sum(signal_metrics["execution_status_counts"].values()) == signal_metrics["events"]


def test_no_trade_and_oracle_bound_event_level_metrics() -> None:
    test = _synthetic().loc[lambda frame: frame["fold"].eq("B")].copy()
    no_trade = select_no_trade_baseline(test)
    oracle = select_oracle_playbook(test)
    no_trade_metrics = evaluate_playbook_selections(no_trade, arm="no_trade")
    oracle_metrics = evaluate_playbook_selections(oracle, arm="oracle")

    assert no_trade_metrics["average_event_net_return"] == pytest.approx(0.0)
    assert no_trade_metrics["executed_trade_rate"] == pytest.approx(0.0)
    assert oracle_metrics["average_event_net_return"] > 0.0
    assert oracle_metrics["event_coverage"] == pytest.approx(1.0)
    assert "not a capital curve" in oracle_metrics["warning"]


def test_ineligible_planned_action_is_a_zero_utility_training_target() -> None:
    frame = _synthetic().head(3).copy()
    frame.loc[frame["playbook_id"].eq("OPEN_EXPIRY_T5"), ["eligible", "mature"]] = False
    frame.loc[
        frame["playbook_id"].eq("OPEN_EXPIRY_T5"), ["net_return", "mae"]
    ] = np.nan

    enriched = attach_playbook_utility(frame)
    cancelled = enriched["playbook_id"].eq("OPEN_EXPIRY_T5")
    audit = audit_playbook_model_dataset(frame, event_feature_columns=["context"])

    assert enriched.loc[cancelled, "utility"].eq(0.0).all()
    assert audit["ineligible_cancel_training_rows"] == 1
    assert audit["planned_action_training_rows"] == 2
