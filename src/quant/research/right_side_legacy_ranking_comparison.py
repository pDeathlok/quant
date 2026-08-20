"""Read-only ranking comparison against the selector-consumed legacy artifacts."""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from quant.research.right_side_factor_increment_comparison import (
    compare_rule_feature_versions,
)


KEY_COLUMNS: tuple[str, ...] = ("symbol", "date", "fold")


def compare_candidate_with_legacy_artifact(
    candidate_predictions: pd.DataFrame,
    legacy_predictions: pd.DataFrame,
    *,
    candidate_column: str = "pred_unified_long_task_deep",
    legacy_column: str = "legacy_quality_score",
    legacy_label_column: str = "good_path5",
    folds: Sequence[str] = ("A", "B"),
    top_fraction: float = 0.10,
    daily_top_k: int = 10,
    bootstrap_iterations: int = 500,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Compare identical symbol/date rows without extrapolating legacy overlap.

    The target is explicitly the legacy table's next-open good_path5 label,
    because that is the executable contract under which the old artifact was
    consumed.  Candidate training entry mode is retained as provenance.
    """

    selected_folds = tuple(dict.fromkeys(str(value) for value in folds))
    if not selected_folds or set(selected_folds) - {"A", "B"}:
        raise ValueError("legacy artifact promotion comparison may use A/B only")
    candidate_required = {
        *KEY_COLUMNS,
        "entry_mode",
        "horizon",
        "label",
        legacy_label_column,
        candidate_column,
    }
    legacy_required = {
        *KEY_COLUMNS,
        "entry_mode",
        "horizon",
        legacy_label_column,
        legacy_column,
        "legacy_signal_timing_match_any",
        "legacy_temporal_status",
    }
    missing_candidate = candidate_required - set(candidate_predictions.columns)
    missing_legacy = legacy_required - set(legacy_predictions.columns)
    if missing_candidate or missing_legacy:
        raise ValueError(
            "legacy comparison columns missing; "
            f"candidate={sorted(missing_candidate)} legacy={sorted(missing_legacy)}"
        )
    candidate = candidate_predictions[
        candidate_predictions["fold"].astype(str).isin(selected_folds)
    ].copy()
    legacy = legacy_predictions[
        legacy_predictions["fold"].astype(str).isin(selected_folds)
    ].copy()
    if candidate.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("candidate predictions contain duplicate symbol/date/fold rows")
    if legacy.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("legacy predictions contain duplicate symbol/date/fold rows")
    candidate["date"] = pd.to_datetime(candidate["date"], errors="raise")
    legacy["date"] = pd.to_datetime(legacy["date"], errors="raise")
    candidate_counts = candidate.groupby("fold", sort=True).size().to_dict()
    candidate_entry_modes = (
        candidate.groupby("fold", sort=True)["entry_mode"].agg(
            lambda values: "|".join(sorted(set(values.astype(str))))
        ).to_dict()
    )
    candidate = candidate.rename(
        columns={legacy_label_column: "candidate_contract_good_path5"}
    )
    legacy = legacy.rename(
        columns={
            "entry_mode": "legacy_entry_mode",
            "horizon": "legacy_horizon",
            legacy_label_column: "legacy_good_path5",
        }
    )
    merged = candidate.merge(
        legacy,
        on=list(KEY_COLUMNS),
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("candidate and legacy artifact have no A/B symbol/date overlap")
    merged["candidate_score"] = merged[candidate_column]
    merged["legacy_score"] = merged[legacy_column]
    merged["legacy_next_open_good_path5"] = merged["legacy_good_path5"]

    frames: list[pd.DataFrame] = []
    label_contracts = (
        (
            "candidate_training_contract_good_path5",
            "candidate_contract_good_path5",
            "horizon",
        ),
        (
            "legacy_next_open_good_path5",
            "legacy_next_open_good_path5",
            "legacy_horizon",
        ),
    )
    for comparison_label, target_column, horizon_column in label_contracts:
        target_frame = merged.copy()
        target_frame["comparison_label"] = comparison_label
        target_frame["comparison_horizon"] = pd.to_numeric(
            target_frame[horizon_column], errors="raise"
        ).astype(int)
        for scope, scoped in (
            ("all_exact_symbol_date_overlap", target_frame),
            (
                "legacy_signal_timing_match_any",
                target_frame[
                    target_frame["legacy_signal_timing_match_any"]
                    .fillna(False)
                    .astype(bool)
                ].copy(),
            ),
        ):
            result = compare_rule_feature_versions(
                scoped,
                candidate_column="candidate_score",
                baseline_column="legacy_score",
                label_column=target_column,
                group_columns=("comparison_label", "comparison_horizon", "fold"),
                top_fraction=top_fraction,
                bootstrap_iterations=bootstrap_iterations,
                random_seed=random_seed,
                daily_top_k=daily_top_k,
            )
            result.insert(0, "comparison_scope", scope)
            result["candidate_training_entry_mode"] = result["fold"].map(
                candidate_entry_modes
            )
            result["legacy_entry_mode"] = "next_open"
            temporal_status = scoped.groupby("fold", sort=True)[
                "legacy_temporal_status"
            ].agg(lambda values: "|".join(sorted(set(values.astype(str)))))
            result["legacy_temporal_status"] = result["fold"].map(temporal_status)
            result["legacy_coverage_denominator"] = result["fold"].map(candidate_counts)
            result["legacy_overlap_coverage"] = (
                result["paired_rows"] / result["legacy_coverage_denominator"]
            )
            agreement = scoped.groupby("fold", sort=True).apply(
                lambda group: float(
                    (
                        pd.to_numeric(group["candidate_contract_good_path5"])
                        == pd.to_numeric(group["legacy_next_open_good_path5"])
                    ).mean()
                ),
                include_groups=False,
            )
            result["candidate_legacy_label_agreement"] = result["fold"].map(agreement)
            result["extrapolation_allowed"] = False
            result["interpretation_limit"] = (
                "exact legacy-factor overlap only; historical predicates/timing differ; "
                "coverage result must not be extrapolated to all canonical events"
            )
            frames.append(result)
    return pd.concat(frames, ignore_index=True, sort=False)


__all__ = ["compare_candidate_with_legacy_artifact"]
