"""Post-hoc audits for the A->B right-side playbook experiment.

This module never fits or scores a model.  It validates the immutable artifact
chain and decomposes already-selected B-fold outcomes by arm, action, signal,
and month.  Returns remain overlapping event outcomes, not a capital curve.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from math import comb
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from quant.research.right_side_playbook_dataset import file_sha256
from quant.research.right_side_playbook_model import FORBIDDEN_MODEL_FEATURE_COLUMNS
from quant.research.right_side_playbook_policy import (
    DEFAULT_PLAYBOOK_CATALOG,
    NO_TRADE_PLAYBOOK_ID,
    playbook_catalog_hash,
)
from quant.research.right_side_unified import RIGHT_SIDE_SIGNALS
from quant.research.right_side_unified_features import (
    LEGACY_RULE_FEATURE_COLUMNS_SHA256_V1,
    LEGACY_RULE_FEATURE_COLUMNS_V1,
    LEGACY_RULE_FEATURE_SCHEMA_VERSION_V1,
    RULE_FEATURE_COLUMNS,
    RULE_FEATURE_COLUMNS_SHA256,
    RULE_FEATURE_SCHEMA_VERSION,
    rule_feature_columns_sha256,
)


EXPECTED_FIRST_LAYER_CANDIDATE = "unified_long_task_deep_rule105"
EXPECTED_FIRST_LAYER_SCORE_COLUMN = f"pred_{EXPECTED_FIRST_LAYER_CANDIDATE}"
EXPECTED_FOLDS = ("A", "B")
EXPECTED_REGULAR_COST_BPS = 15.0


@dataclass(frozen=True)
class MonthBlockBootstrapSpec:
    iterations: int = 10_000
    confidence_level: float = 0.95
    random_seed: int = 42

    def __post_init__(self) -> None:
        if self.iterations <= 0:
            raise ValueError("bootstrap iterations must be positive")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be in (0, 1)")


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], *, name: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} missing required columns: {sorted(missing)}")


def _finite(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.notna() & np.isfinite(numeric.to_numpy(dtype=float))


def _safe_mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric.loc[np.isfinite(numeric.to_numpy(dtype=float))]
    return float(numeric.mean()) if len(numeric) else np.nan


def summarize_outcomes_by_fold_action(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Summarize all preregistered counterfactual actions in A and B."""

    required = {
        "fold",
        "event_id",
        "playbook_id",
        "entry_mode",
        "exit_policy_id",
        "eligible",
        "mature",
        "net_return",
        "mae",
        "round_trip_cost_bps",
    }
    _require_columns(outcomes, required, name="playbook outcomes")
    if set(outcomes["fold"].astype(str)) != set(EXPECTED_FOLDS):
        raise ValueError("outcome summary requires exactly A/B and never C")
    rows: list[dict[str, Any]] = []
    keys = ["fold", "playbook_id", "entry_mode", "exit_policy_id"]
    for key, group in outcomes.groupby(keys, sort=True, dropna=False):
        eligible = group["eligible"].fillna(False).astype(bool)
        mature = group["mature"].fillna(False).astype(bool)
        known = _finite(group["net_return"])
        net = pd.to_numeric(group.loc[known, "net_return"], errors="coerce")
        mae = pd.to_numeric(group.loc[known, "mae"], errors="coerce")
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "events": int(len(group)),
                "eligible_events": int(eligible.sum()),
                "mature_events": int((eligible & mature).sum()),
                "known_return_events": int(known.sum()),
                "known_return_coverage": float(known.mean()),
                "average_net_return": float(net.mean()) if len(net) else np.nan,
                "win_rate": float(net.gt(0).mean()) if len(net) else np.nan,
                "average_mae": float(mae.mean()) if len(mae) else np.nan,
                "round_trip_cost_bps": _safe_mean(group["round_trip_cost_bps"]),
            }
        )
    return pd.DataFrame(rows).sort_values(keys, kind="stable").reset_index(drop=True)


def summarize_selections_by_arm_action(selections: pd.DataFrame) -> pd.DataFrame:
    """Summarize planned and executed B actions for every policy arm."""

    required = {
        "arm",
        "fold",
        "event_id",
        "playbook_id",
        "planned_playbook_id",
        "net_return",
        "mae",
    }
    _require_columns(selections, required, name="playbook selections")
    if set(selections["fold"].astype(str)) != {"B"}:
        raise ValueError("selection summary requires B only")
    rows: list[dict[str, Any]] = []
    for stage, action_column in (
        ("planned", "planned_playbook_id"),
        ("executed", "playbook_id"),
    ):
        for (arm, action), group in selections.groupby(
            ["arm", action_column], sort=True, dropna=False
        ):
            arm_total = int((selections["arm"].astype(str) == str(arm)).sum())
            known = _finite(group["net_return"])
            net = pd.to_numeric(group.loc[known, "net_return"], errors="coerce")
            trade = group["playbook_id"].astype(str).ne(NO_TRADE_PLAYBOOK_ID)
            trade_known = trade & known
            trade_net = pd.to_numeric(
                group.loc[trade_known, "net_return"], errors="coerce"
            )
            rows.append(
                {
                    "fold": "B",
                    "arm": str(arm),
                    "stage": stage,
                    "playbook_id": str(action),
                    "events": int(len(group)),
                    "selection_rate": float(len(group) / arm_total),
                    "covered_events": int(known.sum()),
                    "coverage": float(known.mean()),
                    "executed_trades": int(trade.sum()),
                    "average_event_net_return": (
                        float(net.mean()) if len(net) else np.nan
                    ),
                    "average_trade_net_return": (
                        float(trade_net.mean()) if len(trade_net) else np.nan
                    ),
                    "trade_win_rate": (
                        float(trade_net.gt(0).mean()) if len(trade_net) else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["arm", "stage", "selection_rate", "playbook_id"],
        ascending=[True, True, False, True],
        kind="stable",
    ).reset_index(drop=True)


def attach_signal_identities(
    selections: pd.DataFrame,
    events: pd.DataFrame,
    *,
    signal_columns: Sequence[str] = RIGHT_SIDE_SIGNALS,
) -> pd.DataFrame:
    """Join the one-row event strategy identities to each selected arm row."""

    signals = tuple(dict.fromkeys(signal_columns))
    _require_columns(
        events,
        {"fold", "event_id", *signals},
        name="playbook events",
    )
    if events.duplicated(["fold", "event_id"]).any():
        raise ValueError("event signal table contains duplicate keys")
    joined = selections.merge(
        events[["fold", "event_id", *signals]],
        on=["fold", "event_id"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if joined["_merge"].ne("both").any():
        raise ValueError("selected policies lack event signal identities")
    return joined.drop(columns="_merge")


def summarize_selections_by_signal(
    selections_with_signals: pd.DataFrame,
    *,
    signal_columns: Sequence[str] = RIGHT_SIDE_SIGNALS,
) -> pd.DataFrame:
    """Compare arm outcomes inside each overlapping strategy identity."""

    signals = tuple(dict.fromkeys(signal_columns))
    _require_columns(
        selections_with_signals,
        {"arm", "fold", "playbook_id", "net_return", *signals},
        name="signal selections",
    )
    rows: list[dict[str, Any]] = []
    for arm, arm_rows in selections_with_signals.groupby("arm", sort=True):
        for signal in signals:
            group = arm_rows.loc[arm_rows[signal].fillna(False).astype(bool)]
            known = _finite(group["net_return"])
            trade = group["playbook_id"].astype(str).ne(NO_TRADE_PLAYBOOK_ID)
            trade_known = trade & known
            net = pd.to_numeric(group.loc[known, "net_return"], errors="coerce")
            trade_net = pd.to_numeric(
                group.loc[trade_known, "net_return"], errors="coerce"
            )
            rows.append(
                {
                    "fold": "B",
                    "arm": str(arm),
                    "signal": signal,
                    "events": int(len(group)),
                    "covered_events": int(known.sum()),
                    "coverage": float(known.mean()) if len(group) else np.nan,
                    "executed_trades": int(trade.sum()),
                    "trade_rate": float(trade.mean()) if len(group) else np.nan,
                    "average_event_net_return": (
                        float(net.mean()) if len(net) else np.nan
                    ),
                    "average_trade_net_return": (
                        float(trade_net.mean()) if len(trade_net) else np.nan
                    ),
                    "win_rate": (
                        float(trade_net.gt(0).mean()) if len(trade_net) else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["signal", "arm"], kind="stable"
    ).reset_index(drop=True)


def compare_shared_to_static_by_signal(
    signal_summary: pd.DataFrame,
    *,
    shared_arm: str = "shared_playbook_model",
    static_arm: str = "static_per_signal",
) -> pd.DataFrame:
    """Return compact shared-minus-static signal deltas."""

    shared = signal_summary.loc[
        signal_summary["arm"].astype(str).eq(shared_arm)
    ].set_index("signal")
    static = signal_summary.loc[
        signal_summary["arm"].astype(str).eq(static_arm)
    ].set_index("signal")
    if set(shared.index) != set(RIGHT_SIDE_SIGNALS) or set(static.index) != set(
        RIGHT_SIDE_SIGNALS
    ):
        raise ValueError("shared/static signal summary does not cover all 14 signals")
    output = pd.DataFrame(index=list(RIGHT_SIDE_SIGNALS))
    output.index.name = "signal"
    for column in (
        "events",
        "coverage",
        "trade_rate",
        "average_event_net_return",
        "average_trade_net_return",
        "win_rate",
    ):
        output[f"shared_{column}"] = shared[column]
        output[f"static_{column}"] = static[column]
    output["delta_average_event_net_return"] = (
        output["shared_average_event_net_return"]
        - output["static_average_event_net_return"]
    )
    output["delta_trade_rate"] = (
        output["shared_trade_rate"] - output["static_trade_rate"]
    )
    output["delta_win_rate"] = output["shared_win_rate"] - output["static_win_rate"]
    return output.reset_index()


def paired_monthly_stability(
    selections: pd.DataFrame,
    *,
    shared_arm: str = "shared_playbook_model",
    static_arm: str = "static_per_signal",
    bootstrap_spec: MonthBlockBootstrapSpec = MonthBlockBootstrapSpec(),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Pair shared/static event returns and bootstrap whole calendar months."""

    required = {"arm", "fold", "event_id", "date", "net_return", "playbook_id"}
    _require_columns(selections, required, name="playbook selections")
    subset = selections.loc[
        selections["arm"].astype(str).isin([shared_arm, static_arm]),
        ["arm", "event_id", "date", "net_return", "playbook_id"],
    ].copy()
    if subset.duplicated(["arm", "event_id"]).any():
        raise ValueError("paired monthly selections contain duplicate arm/event keys")
    wide_return = subset.pivot(index="event_id", columns="arm", values="net_return")
    wide_date = subset.pivot(index="event_id", columns="arm", values="date")
    wide_action = subset.pivot(index="event_id", columns="arm", values="playbook_id")
    if not pd.to_datetime(wide_date[shared_arm]).eq(
        pd.to_datetime(wide_date[static_arm])
    ).all():
        raise ValueError("shared/static event dates differ")
    shared_return = pd.to_numeric(wide_return[shared_arm], errors="coerce")
    static_return = pd.to_numeric(wide_return[static_arm], errors="coerce")
    common = _finite(shared_return) & _finite(static_return)
    paired = pd.DataFrame(
        {
            "date": pd.to_datetime(wide_date.loc[common, shared_arm]),
            "shared_net_return": shared_return.loc[common],
            "static_net_return": static_return.loc[common],
            "shared_trade": wide_action.loc[common, shared_arm]
            .astype(str)
            .ne(NO_TRADE_PLAYBOOK_ID),
            "static_trade": wide_action.loc[common, static_arm]
            .astype(str)
            .ne(NO_TRADE_PLAYBOOK_ID),
        },
        index=wide_return.index[common],
    )
    paired["delta_net_return"] = (
        paired["shared_net_return"] - paired["static_net_return"]
    )
    paired["month"] = paired["date"].dt.to_period("M").astype(str)
    monthly = (
        paired.groupby("month", sort=True)
        .agg(
            paired_events=("delta_net_return", "size"),
            shared_average_event_net_return=("shared_net_return", "mean"),
            static_average_event_net_return=("static_net_return", "mean"),
            delta_average_event_net_return=("delta_net_return", "mean"),
            shared_trade_rate=("shared_trade", "mean"),
            static_trade_rate=("static_trade", "mean"),
        )
        .reset_index()
    )
    if len(monthly) < 2:
        raise ValueError("monthly stability requires at least two calendar months")

    month_sums = (
        paired.groupby("month", sort=True)["delta_net_return"].sum().to_numpy(float)
    )
    month_counts = (
        paired.groupby("month", sort=True)["delta_net_return"].size().to_numpy(float)
    )
    generator = np.random.default_rng(bootstrap_spec.random_seed)
    draws = generator.integers(
        0,
        len(month_sums),
        size=(bootstrap_spec.iterations, len(month_sums)),
    )
    sampled_sums = month_sums[draws].sum(axis=1)
    sampled_counts = month_counts[draws].sum(axis=1)
    bootstrapped = sampled_sums / sampled_counts
    alpha = 1.0 - bootstrap_spec.confidence_level
    delta = paired["delta_net_return"]
    observed_sum = float(month_sums.sum())
    if len(month_sums) <= 20:
        signed_sums = np.fromiter(
            (
                float(np.dot(signs, month_sums))
                for signs in product((-1.0, 1.0), repeat=len(month_sums))
            ),
            dtype=float,
            count=2 ** len(month_sums),
        )
        sign_flip_method = f"exact_all_2^{len(month_sums)}_month_sign_patterns"
    else:
        signs = generator.choice(
            (-1.0, 1.0),
            size=(bootstrap_spec.iterations, len(month_sums)),
        )
        signed_sums = signs @ month_sums
        sign_flip_method = "monte_carlo_month_sign_flip"
    positive_months = int(monthly["delta_average_event_net_return"].gt(0).sum())
    negative_months = int(monthly["delta_average_event_net_return"].lt(0).sum())
    nonzero_months = positive_months + negative_months
    sign_test_positive_tail = (
        float(
            sum(comb(nonzero_months, k) for k in range(positive_months, nonzero_months + 1))
            / (2**nonzero_months)
        )
        if nonzero_months
        else 1.0
    )
    summary = {
        "shared_arm": shared_arm,
        "static_arm": static_arm,
        "paired_events": int(len(paired)),
        "shared_total_events": int(
            selections["arm"].astype(str).eq(shared_arm).sum()
        ),
        "static_total_events": int(
            selections["arm"].astype(str).eq(static_arm).sum()
        ),
        "paired_coverage": float(
            len(paired)
            / min(
                selections["arm"].astype(str).eq(shared_arm).sum(),
                selections["arm"].astype(str).eq(static_arm).sum(),
            )
        ),
        "months": int(len(monthly)),
        "event_weighted_delta_average_net_return": float(delta.mean()),
        "mean_monthly_delta_average_net_return": float(
            monthly["delta_average_event_net_return"].mean()
        ),
        "median_monthly_delta_average_net_return": float(
            monthly["delta_average_event_net_return"].median()
        ),
        "positive_months": positive_months,
        "negative_months": negative_months,
        "zero_months": int(
            monthly["delta_average_event_net_return"].eq(0).sum()
        ),
        "negative_month_ratio": float(
            monthly["delta_average_event_net_return"].lt(0).mean()
        ),
        "bootstrap": {
            **asdict(bootstrap_spec),
            "sampling_unit": "calendar_month_with_all_paired_events",
            "ci_low": float(np.quantile(bootstrapped, alpha / 2.0)),
            "ci_high": float(np.quantile(bootstrapped, 1.0 - alpha / 2.0)),
            "probability_delta_gt_zero": float(np.mean(bootstrapped > 0.0)),
        },
        "month_cluster_sign_flip": {
            "method": sign_flip_method,
            "null": "calendar-month contribution signs are exchangeable around zero",
            "one_sided_p_value_shared_gt_static": float(
                np.mean(signed_sums >= observed_sum)
            ),
            "two_sided_p_value": float(
                np.mean(np.abs(signed_sums) >= abs(observed_sum))
            ),
        },
        "monthly_direction_sign_test": {
            "nonzero_months": nonzero_months,
            "one_sided_p_value_shared_positive": sign_test_positive_tail,
            "two_sided_p_value": min(1.0, 2.0 * sign_test_positive_tail),
            "warning": "tests direction only and ignores the magnitude of monthly deltas",
        },
        "warning": (
            "12-month development-fold stability only; paired overlapping event "
            "returns are not a capital curve"
        ),
    }
    return monthly, summary


def validate_artifact_chain(
    *,
    dataset_manifest: dict[str, Any],
    model_manifest: dict[str, Any],
    first_layer_manifests: Sequence[dict[str, Any]],
    events_path: Path,
    outcomes_path: Path,
    dataset_manifest_path: Path,
    model_artifact_path: Path,
    serialized_model_event_features: Sequence[str],
    metrics_event_features: Sequence[str],
    observed_artifact_paths: Mapping[str, Path],
    parquet_fold_metadata: Mapping[str, Mapping[str, Any]],
    outcomes: pd.DataFrame,
    selections: pd.DataFrame,
) -> dict[str, Any]:
    """Validate selected 105 score -> 118 layer-2 features -> B selections."""

    prediction_source = dataset_manifest["first_layer_prediction_source"]
    if prediction_source["selected_candidate"] != EXPECTED_FIRST_LAYER_CANDIDATE:
        raise ValueError("playbook dataset did not freeze the selected 105 candidate")
    if prediction_source["score_column"] != EXPECTED_FIRST_LAYER_SCORE_COLUMN:
        raise ValueError("playbook dataset uses the wrong first-layer score column")
    if prediction_source["fold_provenance"].get("C") != "forbidden_not_read":
        raise ValueError("dataset manifest does not explicitly forbid fold C")
    factor_source = dataset_manifest["factor_source"]
    if (
        int(factor_source["rule_feature_count"]) != len(RULE_FEATURE_COLUMNS)
        or factor_source["rule_feature_schema_version"] != RULE_FEATURE_SCHEMA_VERSION
        or factor_source["rule_feature_columns_sha256"] != RULE_FEATURE_COLUMNS_SHA256
    ):
        raise ValueError("layer-2 118-rule-factor contract drifted")

    if len(first_layer_manifests) != 2:
        raise ValueError("artifact chain requires exact first-layer A/B manifests")
    manifest_folds: set[str] = set()
    for manifest in first_layer_manifests:
        fold = manifest["fold"]
        fold_name = fold.get("name") if isinstance(fold, dict) else fold
        manifest_folds.add(str(fold_name))
        if manifest["experiment"] != EXPECTED_FIRST_LAYER_CANDIDATE:
            raise ValueError("first-layer manifest experiment drifted")
        if (
            int(manifest["rule_feature_count"]) != len(LEGACY_RULE_FEATURE_COLUMNS_V1)
            or manifest["rule_feature_schema_version"]
            != LEGACY_RULE_FEATURE_SCHEMA_VERSION_V1
            or manifest["rule_feature_columns_sha256"]
            != LEGACY_RULE_FEATURE_COLUMNS_SHA256_V1
        ):
            raise ValueError("selected first-layer 105-factor contract drifted")
    if manifest_folds != set(EXPECTED_FOLDS):
        raise ValueError("first-layer manifests must be exact A/B")

    if model_manifest["evaluation_fold"] != "B" or model_manifest[
        "training_first_layer_score_folds"
    ] != ["A"]:
        raise ValueError("playbook model fold contract drifted from A -> B")
    if model_manifest["fold_policy"] != "A_train_to_B_test_only_C_forbidden":
        raise ValueError("playbook model manifest does not forbid C")
    expected_fold_bounds = {
        "events": ("A", "B"),
        "outcomes": ("A", "B"),
        "selections": ("B", "B"),
    }
    for name, (expected_min, expected_max) in expected_fold_bounds.items():
        metadata = parquet_fold_metadata[name]
        if metadata["global_min"] != expected_min or metadata["global_max"] != expected_max:
            raise ValueError(f"{name} parquet metadata indicates a fold outside contract")
    features = tuple(model_manifest["event_features"])
    if len(features) != len(set(features)):
        raise ValueError("playbook model feature contract contains duplicates")
    missing_rules = set(RULE_FEATURE_COLUMNS) - set(features)
    missing_signals = set(RIGHT_SIDE_SIGNALS) - set(features)
    forbidden = set(features) & FORBIDDEN_MODEL_FEATURE_COLUMNS
    if missing_rules or missing_signals or forbidden:
        raise ValueError(
            "playbook model feature contract failed: "
            f"missing_rules={sorted(missing_rules)} "
            f"missing_signals={sorted(missing_signals)} forbidden={sorted(forbidden)}"
        )
    serialized_features = tuple(str(value) for value in serialized_model_event_features)
    reported_features = tuple(str(value) for value in metrics_event_features)
    if serialized_features != features:
        raise ValueError("serialized model event features differ from model manifest")
    if reported_features != features:
        raise ValueError("policy metrics event features differ from model manifest")
    manifest_model_path = Path(str(model_manifest["model"])).resolve()
    if Path(model_artifact_path).resolve() != manifest_model_path:
        raise ValueError("audited model path differs from model manifest")
    if model_manifest["playbook_catalog_sha256"] != playbook_catalog_hash():
        raise ValueError("playbook model catalog hash drifted")

    expected_hashes = {
        "model": (Path(model_artifact_path), file_sha256(model_artifact_path)),
        "events": (Path(events_path), model_manifest["events"]["sha256"]),
        "outcomes": (Path(outcomes_path), model_manifest["outcomes"]["sha256"]),
        "dataset_manifest": (
            Path(dataset_manifest_path),
            model_manifest["dataset_manifest"]["sha256"],
        ),
    }
    artifact_hashes: dict[str, str] = {}
    for name, (path, expected_hash) in expected_hashes.items():
        actual_hash = file_sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(f"{name} artifact hash differs from model manifest")
        artifact_hashes[name] = actual_hash
    for name, path in observed_artifact_paths.items():
        artifact_hashes[name] = file_sha256(Path(path))

    if set(outcomes["fold"].astype(str)) != set(EXPECTED_FOLDS):
        raise ValueError("outcomes contain a fold outside A/B")
    if set(selections["fold"].astype(str)) != {"B"}:
        raise ValueError("selections contain a fold outside B")
    no_trade = outcomes["playbook_id"].astype(str).eq(NO_TRADE_PLAYBOOK_ID)
    regular = ~no_trade
    regular_cost = pd.to_numeric(
        outcomes.loc[regular, "round_trip_cost_bps"], errors="coerce"
    )
    no_trade_cost = pd.to_numeric(
        outcomes.loc[no_trade, "round_trip_cost_bps"], errors="coerce"
    )
    if not regular_cost.eq(EXPECTED_REGULAR_COST_BPS).all() or not no_trade_cost.eq(
        0.0
    ).all():
        raise ValueError("outcome costs are not exact 15bps regular / 0bps NO_TRADE")
    eligible = outcomes["eligible"].fillna(False).astype(bool)
    mature = outcomes["mature"].fillna(False).astype(bool)
    if ((regular & (~eligible | ~mature)) & outcomes["net_return"].notna()).any():
        raise ValueError("unknown/ineligible outcome rows leak realized net return")
    selected_mature = selections["mature"].fillna(False).astype(bool)
    selected_trade = selections["playbook_id"].astype(str).ne(NO_TRADE_PLAYBOOK_ID)
    selected_cost = pd.to_numeric(selections["round_trip_cost"], errors="coerce")
    expected_regular_cost = EXPECTED_REGULAR_COST_BPS / 10_000.0
    if not np.isclose(
        selected_cost.loc[selected_trade].to_numpy(dtype=float),
        expected_regular_cost,
        rtol=0.0,
        atol=1e-12,
    ).all() or not np.isclose(
        selected_cost.loc[~selected_trade].to_numpy(dtype=float),
        0.0,
        rtol=0.0,
        atol=1e-12,
    ).all():
        raise ValueError("selected policy costs are not exact 15bps trade / zero no-trade")
    if ((selected_trade & ~selected_mature) & selections["net_return"].notna()).any():
        raise ValueError("immature selected rows leak net return")
    immature_selected = selected_trade & ~selected_mature
    immature_selected_by_arm = {
        str(arm): int(count)
        for arm, count in selections.loc[immature_selected].groupby("arm").size().items()
    }

    a = outcomes["fold"].astype(str).eq("A")
    known_a_regular = regular & a & (
        ~eligible | (eligible & mature & outcomes["net_return"].notna())
    )
    if int(model_manifest["training_rows"]) != int(known_a_regular.sum()):
        raise ValueError("model training row count differs from known A action targets")

    return {
        "status": "pass",
        "fold_contract": "A_train_to_B_test_only_C_forbidden",
        "parquet_fold_metadata": dict(parquet_fold_metadata),
        "first_layer": {
            "selected_candidate": EXPECTED_FIRST_LAYER_CANDIDATE,
            "score_column": EXPECTED_FIRST_LAYER_SCORE_COLUMN,
            "rule_feature_schema_version": LEGACY_RULE_FEATURE_SCHEMA_VERSION_V1,
            "rule_feature_count": len(LEGACY_RULE_FEATURE_COLUMNS_V1),
            "rule_feature_columns_sha256": LEGACY_RULE_FEATURE_COLUMNS_SHA256_V1,
        },
        "second_layer": {
            "rule_feature_schema_version": RULE_FEATURE_SCHEMA_VERSION,
            "rule_feature_count": len(RULE_FEATURE_COLUMNS),
            "rule_feature_columns_sha256": RULE_FEATURE_COLUMNS_SHA256,
            "strategy_identity_count": len(RIGHT_SIDE_SIGNALS),
            "model_event_feature_count": len(features),
            "model_event_feature_columns_sha256": rule_feature_columns_sha256(
                features
            ),
            "serialized_model_features_match_manifest": True,
            "policy_metrics_features_match_manifest": True,
            "model_training_rows": int(model_manifest["training_rows"]),
        },
        "cost_contract": {
            "regular_round_trip_cost_bps": EXPECTED_REGULAR_COST_BPS,
            "no_trade_round_trip_cost_bps": 0.0,
            "selected_policy_regular_round_trip_cost": expected_regular_cost,
            "selected_policy_costs_match_outcome_contract": True,
            "status": "pass",
        },
        "maturity_contract": {
            "immature_regular_rows": int((regular & eligible & ~mature).sum()),
            "immature_regular_rows_with_net_return": int(
                ((regular & eligible & ~mature) & outcomes["net_return"].notna()).sum()
            ),
            "immature_selected_rows": int((selected_trade & ~selected_mature).sum()),
            "immature_selected_rows_by_arm": immature_selected_by_arm,
            "immature_selected_rows_with_net_return": int(
                ((selected_trade & ~selected_mature) & selections["net_return"].notna()).sum()
            ),
            "status": "pass",
        },
        "artifact_sha256": artifact_hashes,
        "artifact_hash_semantics": {
            "events_outcomes_dataset_manifest": "matched pre-existing model manifest",
            "model_and_report_chain": "computed post-hoc identity digests",
        },
    }


def capital_curve_feasibility(selections: pd.DataFrame) -> dict[str, Any]:
    """Audit whether selections can support a future capital-aware simulator."""

    required = {
        "arm",
        "symbol",
        "date",
        "playbook_id",
        "entry_date",
        "exit_date",
        "net_return",
        "round_trip_cost",
    }
    _require_columns(selections, required, name="capital feasibility selections")
    shared = selections.loc[
        selections["arm"].astype(str).eq("shared_playbook_model")
    ].copy()
    trade = shared["playbook_id"].astype(str).ne(NO_TRADE_PLAYBOOK_ID)
    mature_trade = trade & shared["net_return"].notna()
    entry = pd.to_datetime(shared["entry_date"], errors="coerce")
    exit_date = pd.to_datetime(shared["exit_date"], errors="coerce")
    valid_dates = entry.notna() & exit_date.notna() & exit_date.ge(entry)
    valid_return = _finite(shared["net_return"])
    valid_cost = _finite(shared["round_trip_cost"])
    complete_trade_rows = mature_trade & valid_dates & valid_return & valid_cost
    missing_trade_rows = mature_trade & ~complete_trade_rows
    mature = shared.loc[complete_trade_rows].copy()
    signal_dates = pd.to_datetime(shared["date"], errors="coerce").dropna()
    sessions = pd.DatetimeIndex(
        sorted(
            set(signal_dates)
            | set(pd.to_datetime(mature["entry_date"]).dropna())
            | set(pd.to_datetime(mature["exit_date"]).dropna())
        )
    )
    occupancy = np.zeros(len(sessions), dtype=np.int64)
    entries = np.zeros(len(sessions), dtype=np.int64)
    if len(sessions) and len(mature):
        differences = np.zeros(len(sessions) + 1, dtype=np.int64)
        entry_counts = pd.to_datetime(mature["entry_date"]).value_counts()
        exit_counts = pd.to_datetime(mature["exit_date"]).value_counts()
        for entry_date, count in entry_counts.items():
            position = int(sessions.searchsorted(entry_date, side="left"))
            differences[position] += int(count)
            entries[position] += int(count)
        for exit_value, count in exit_counts.items():
            position = int(sessions.searchsorted(exit_value, side="right"))
            differences[position] -= int(count)
        occupancy = np.cumsum(differences[:-1])
    occupancy_summary = {
        "observed_sessions": int(len(sessions)),
        "median_raw_concurrent_candidates": (
            float(np.median(occupancy)) if len(occupancy) else np.nan
        ),
        "p95_raw_concurrent_candidates": (
            float(np.quantile(occupancy, 0.95)) if len(occupancy) else np.nan
        ),
        "maximum_raw_concurrent_candidates": (
            int(occupancy.max()) if len(occupancy) else 0
        ),
        "maximum_raw_new_entries_per_session": (
            int(entries.max()) if len(entries) else 0
        ),
        "warning": (
            "all selected candidates are counted with one slot and no top-K/cash cap; "
            "this is an unconstrained occupancy envelope, not a portfolio"
        ),
    }
    return {
        "shared_executed_trades": int(trade.sum()),
        "shared_mature_trades": int(mature_trade.sum()),
        "trade_rows_with_symbol_entry_exit_net_cost": int(complete_trade_rows.sum()),
        "trade_rows_missing_required_simulation_fields": int(missing_trade_rows.sum()),
        "can_build_assumption_bound_occupancy_backtest": bool(
            not missing_trade_rows.any()
        ),
        "unconstrained_occupancy_envelope": occupancy_summary,
        "is_true_capital_curve_now": False,
        "available_fields": [
            "symbol",
            "signal_date",
            "entry_date",
            "exit_date",
            "realized_net_return_including_fixed_cost",
            "chosen_action",
        ],
        "missing_portfolio_contract": [
            "initial_capital",
            "position_sizing_rule",
            "maximum_concurrent_positions",
            "maximum_new_positions_per_day",
            "cash_reservation_and_same_day_exit_entry_order",
            "candidate_priority_when_capacity_is_full",
            "board_lot_rounding",
            "liquidity_capacity_and_market_impact",
            "daily_mark_to_market_path_for_open_positions",
        ],
        "production_ready": False,
        "production_blockers": [
            "only one 12-month B development evaluation; require untouched future shadow window",
            "monthly block confidence interval and negative-month rate must pass a preregistered strategy gate",
            "event returns overlap and are not constrained by cash or concurrent-position capacity",
            "shared model lowers win rate and worsens the event-level drawdown proxy versus the best static policy",
            "runtime T+1 cancellation path has zero observed cases after upstream exclusion and needs shadow verification",
            "daily mark-to-market capital curve with sizing, capacity, and execution rules has not been run",
        ],
    }


__all__ = [
    "EXPECTED_FIRST_LAYER_CANDIDATE",
    "EXPECTED_FIRST_LAYER_SCORE_COLUMN",
    "MonthBlockBootstrapSpec",
    "attach_signal_identities",
    "capital_curve_feasibility",
    "compare_shared_to_static_by_signal",
    "paired_monthly_stability",
    "summarize_outcomes_by_fold_action",
    "summarize_selections_by_arm_action",
    "summarize_selections_by_signal",
    "validate_artifact_chain",
]
