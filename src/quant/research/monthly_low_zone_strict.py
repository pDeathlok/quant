"""Strict cohort-level validation for monthly blood-chip reversal candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from typing import Any, Mapping

import numpy as np
import pandas as pd


STRICT_PERIODS: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {
    "development_2013_2016": (
        pd.Timestamp("2013-01-01"),
        pd.Timestamp("2016-12-31"),
    ),
    "exposed_validation_2017_2020": (
        pd.Timestamp("2017-01-01"),
        pd.Timestamp("2020-12-31"),
    ),
    "seen_diagnostic_2021_2024": (
        pd.Timestamp("2021-01-01"),
        pd.Timestamp("2024-12-31"),
    ),
    "time_out_2025_to_cutoff": (
        pd.Timestamp("2025-01-01"),
        pd.Timestamp("2026-07-31"),
    ),
}

HISTORICAL_STRICT_PERIODS = tuple(list(STRICT_PERIODS)[:3])
PRIMARY_STRICT_STRUCTURE = "systemic_deep_no_new_low_dd70_survival80"


@dataclass(frozen=True)
class StrictLowZoneConfig:
    """Frozen gates and statistical contract for the strict iteration."""

    minimum_breadth_constituents: int = 500
    maximum_positive_share_20d: float = 0.20
    maximum_median_return_20d: float = -0.08
    primary_drawdown: float = 0.70
    maximum_financial_age_days: int = 550
    minimum_annual_history_years: int = 3
    primary_survival_share: float = 0.80
    minimum_completed_cohorts_each_period: int = 2
    minimum_completed_cohorts_total: int = 12
    minimum_event_win_rate: float = 0.85
    minimum_profit_factor: float = 2.0
    minimum_positive_cohort_share: float = 0.80
    minimum_worst_cohort_return: float = -0.05
    bootstrap_iterations: int = 10_000
    bootstrap_seed: int = 20260826

    def __post_init__(self) -> None:
        if self.minimum_breadth_constituents < 1:
            raise ValueError("minimum_breadth_constituents must be positive")
        if not 0.0 <= self.maximum_positive_share_20d <= 1.0:
            raise ValueError("maximum_positive_share_20d must be in [0, 1]")
        if not -1.0 < self.maximum_median_return_20d < 0.0:
            raise ValueError("maximum_median_return_20d must be in (-1, 0)")
        for name in ("primary_drawdown", "primary_survival_share"):
            value = float(getattr(self, name))
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be in (0, 1)")
        if self.maximum_financial_age_days < 1:
            raise ValueError("maximum_financial_age_days must be positive")
        if self.minimum_annual_history_years < 1:
            raise ValueError("minimum_annual_history_years must be positive")
        if self.minimum_completed_cohorts_each_period < 1:
            raise ValueError("minimum_completed_cohorts_each_period must be positive")
        if self.minimum_completed_cohorts_total < 1:
            raise ValueError("minimum_completed_cohorts_total must be positive")
        if self.bootstrap_iterations < 100:
            raise ValueError("bootstrap_iterations must be at least 100")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StrictStructureSpec:
    """One pre-registered structural ablation, never an optimizer candidate."""

    name: str
    rule: str
    drawdown: float
    survival_share: float | None
    market_gate: str = "systemic_deep"


STRICT_STRUCTURE_SPECS: tuple[StrictStructureSpec, ...] = (
    StrictStructureSpec(
        "old_panic_direct_dd50_no_survival",
        "anchor_direct",
        0.50,
        None,
        "old_panic",
    ),
    StrictStructureSpec(
        "systemic_deep_direct_dd70_no_survival",
        "anchor_direct",
        0.70,
        None,
    ),
    StrictStructureSpec(
        "systemic_deep_direct_dd70_survival60",
        "anchor_direct",
        0.70,
        0.60,
    ),
    StrictStructureSpec(
        "systemic_deep_direct_dd70_survival80",
        "anchor_direct",
        0.70,
        0.80,
    ),
    StrictStructureSpec(
        "systemic_deep_no_new_low_dd60_survival80",
        "no_new_low_20",
        0.60,
        0.80,
    ),
    StrictStructureSpec(
        PRIMARY_STRICT_STRUCTURE,
        "no_new_low_20",
        0.70,
        0.80,
    ),
    StrictStructureSpec(
        "systemic_deep_no_new_low_dd80_survival80",
        "no_new_low_20",
        0.80,
        0.80,
    ),
    StrictStructureSpec(
        "systemic_deep_range_reclaim_dd70_survival80",
        "range_mid_reclaim",
        0.70,
        0.80,
    ),
)


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def add_strict_gate_columns(
    events: pd.DataFrame,
    config: StrictLowZoneConfig,
) -> pd.DataFrame:
    """Attach causal market, drawdown and point-in-time survival gates."""

    required = {
        "breadth_constituents",
        "breadth_positive_share_20d",
        "breadth_median_return_20d",
        "drawdown_from_prior_peak",
        "signal_date",
    }
    _require_columns(events, required, "events")
    out = events.copy()
    signal_date = pd.to_datetime(out["signal_date"], errors="coerce")
    available_at = pd.to_datetime(
        out.get("annual_quality_available_at"), errors="coerce"
    )
    financial_age = pd.to_numeric(out.get("financial_age_days"), errors="coerce")
    annual_history = pd.to_numeric(out.get("annual_history_years"), errors="coerce")
    profit_share = pd.to_numeric(
        out.get("profit_positive_share_5y"), errors="coerce"
    )
    cfo_share = pd.to_numeric(out.get("cfo_positive_share_5y"), errors="coerce")
    latest_profit = pd.to_numeric(out.get("income_n_income_attr_p"), errors="coerce")
    latest_cfo = pd.to_numeric(out.get("cashflow_n_cashflow_act"), errors="coerce")
    out["gate_old_panic"] = (
        pd.to_numeric(out["breadth_positive_share_20d"], errors="coerce").le(0.30)
        & pd.to_numeric(out["breadth_median_return_20d"], errors="coerce").le(-0.05)
        & pd.to_numeric(out["breadth_constituents"], errors="coerce").ge(
            config.minimum_breadth_constituents
        )
    ).fillna(False)
    out["gate_systemic_deep"] = (
        pd.to_numeric(out["breadth_constituents"], errors="coerce").ge(
            config.minimum_breadth_constituents
        )
        & pd.to_numeric(out["breadth_positive_share_20d"], errors="coerce").le(
            config.maximum_positive_share_20d
        )
        & pd.to_numeric(out["breadth_median_return_20d"], errors="coerce").le(
            config.maximum_median_return_20d
        )
    ).fillna(False)
    drawdown = pd.to_numeric(out["drawdown_from_prior_peak"], errors="coerce")
    for threshold in (0.50, 0.60, 0.70, 0.80):
        out[f"gate_dd{int(threshold * 100)}"] = drawdown.le(-threshold).fillna(False)
    out["financial_information_causal"] = available_at.le(signal_date).fillna(False)
    for share in (0.60, 0.80):
        out[f"gate_survival{int(share * 100)}"] = (
            out["financial_information_causal"]
            & financial_age.between(
                0, config.maximum_financial_age_days, inclusive="both"
            )
            & annual_history.ge(config.minimum_annual_history_years)
            & profit_share.ge(share)
            & cfo_share.ge(share)
            & latest_profit.gt(0.0)
            & latest_cfo.gt(0.0)
        ).fillna(False)
    return out


def materialize_strict_structures(
    gated_events: pd.DataFrame,
    specs: tuple[StrictStructureSpec, ...] = STRICT_STRUCTURE_SPECS,
) -> pd.DataFrame:
    """Return a long table for the frozen nested structural ablations."""

    _require_columns(gated_events, {"rule"}, "gated_events")
    parts: list[pd.DataFrame] = []
    for spec in specs:
        market_column = f"gate_{spec.market_gate}"
        drawdown_column = f"gate_dd{int(round(spec.drawdown * 100))}"
        required = {market_column, drawdown_column}
        if spec.survival_share is not None:
            required.add(f"gate_survival{int(round(spec.survival_share * 100))}")
        _require_columns(gated_events, required, "gated_events")
        mask = (
            gated_events["rule"].eq(spec.rule)
            & gated_events[market_column].fillna(False)
            & gated_events[drawdown_column].fillna(False)
        )
        if spec.survival_share is not None:
            mask &= gated_events[
                f"gate_survival{int(round(spec.survival_share * 100))}"
            ].fillna(False)
        selected = gated_events.loc[mask].copy()
        selected["structure"] = spec.name
        parts.append(selected)
    if not parts:
        return gated_events.head(0).assign(structure=pd.Series(dtype="string"))
    return pd.concat(parts, ignore_index=True, sort=False)


def wilson_interval(
    successes: int,
    trials: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    """Wilson score interval without an optional statistics dependency."""

    if trials < 1:
        return np.nan, np.nan
    if successes < 0 or successes > trials:
        raise ValueError("successes must be between zero and trials")
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * sqrt(proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials**2))
        / denominator
    )
    return center - radius, center + radius


def cohort_bootstrap_interval(
    cohort_returns: pd.Series,
    *,
    iterations: int,
    seed: int,
) -> tuple[float, float, float]:
    """Bootstrap the mean by independent cohort, not by correlated stock rows."""

    values = pd.to_numeric(cohort_returns, errors="coerce").dropna().to_numpy(float)
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    if iterations < 100:
        raise ValueError("iterations must be at least 100")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(iterations, len(values)))
    estimates = values[indices].mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )


def leave_one_cohort_out_minimum(cohort_returns: pd.Series) -> float:
    """Return the weakest mean after deleting each cohort once."""

    values = pd.to_numeric(cohort_returns, errors="coerce").dropna().to_numpy(float)
    if len(values) < 2:
        return np.nan
    total = float(values.sum())
    means = (total - values) / (len(values) - 1)
    return float(means.min())


def summarize_strict_events(
    structure_events: pd.DataFrame,
    config: StrictLowZoneConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize stock paths and anchor-month returns for each frozen structure."""

    required = {
        "structure",
        "month_period",
        "horizon",
        "target_return",
        "entry_status",
        "outcome_completed",
        "net_return",
    }
    _require_columns(structure_events, required, "structure_events")
    frame = structure_events.copy()
    frame["anchor_month"] = pd.PeriodIndex(frame["month_period"], freq="M").to_timestamp(
        how="end"
    ).normalize()
    completed = frame.loc[
        frame["entry_status"].eq("accepted")
        & frame["outcome_completed"].fillna(False)
        & frame.get(
            "baseline_outcome_completed", pd.Series(True, index=frame.index)
        ).fillna(False)
    ].copy()
    completed["net_return"] = pd.to_numeric(completed["net_return"], errors="coerce")
    completed = completed.dropna(subset=["net_return"])
    cohort = (
        completed.groupby(
            ["structure", "horizon", "target_return", "anchor_month"],
            observed=True,
            sort=True,
        )
        .agg(
            cohort_return=("net_return", "mean"),
            events=("net_return", "size"),
            winners=("net_return", lambda values: int(values.gt(0.0).sum())),
            worst_event_return=("net_return", "min"),
        )
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    period_scopes: Mapping[str, tuple[pd.Timestamp, pd.Timestamp]] = {
        **STRICT_PERIODS,
        "historical_2013_2024": (
            STRICT_PERIODS[HISTORICAL_STRICT_PERIODS[0]][0],
            STRICT_PERIODS[HISTORICAL_STRICT_PERIODS[-1]][1],
        ),
    }
    grouping = ["structure", "horizon", "target_return"]
    for keys, group in completed.groupby(grouping, observed=True, sort=True):
        matching_cohorts = cohort.loc[
            cohort["structure"].eq(keys[0])
            & cohort["horizon"].eq(keys[1])
            & np.isclose(cohort["target_return"], float(keys[2]))
        ]
        for period, (start, end) in period_scopes.items():
            scoped = group.loc[group["anchor_month"].between(start, end)]
            scoped_cohorts = matching_cohorts.loc[
                matching_cohorts["anchor_month"].between(start, end)
            ]
            returns = scoped["net_return"]
            cohort_returns = scoped_cohorts.set_index("anchor_month")["cohort_return"]
            gains = float(returns.loc[returns > 0.0].sum())
            losses = float(-returns.loc[returns <= 0.0].sum())
            event_low, event_high = wilson_interval(int(returns.gt(0.0).sum()), len(returns))
            cohort_low, cohort_high = wilson_interval(
                int(cohort_returns.gt(0.0).sum()), len(cohort_returns)
            )
            mean, bootstrap_low, bootstrap_high = cohort_bootstrap_interval(
                cohort_returns,
                iterations=config.bootstrap_iterations,
                seed=config.bootstrap_seed,
            )
            rows.append(
                {
                    "period": period,
                    "structure": keys[0],
                    "horizon": int(keys[1]),
                    "target_return": float(keys[2]),
                    "completed_events": int(len(returns)),
                    "completed_cohorts": int(len(cohort_returns)),
                    "event_win_rate": float(returns.gt(0.0).mean()) if len(returns) else np.nan,
                    "event_win_wilson_low": event_low,
                    "event_win_wilson_high": event_high,
                    "median_event_return": float(returns.median()) if len(returns) else np.nan,
                    "mean_event_return": float(returns.mean()) if len(returns) else np.nan,
                    "profit_factor": gains / losses if losses > 0.0 else (np.inf if gains > 0.0 else np.nan),
                    "positive_cohort_share": float(cohort_returns.gt(0.0).mean()) if len(cohort_returns) else np.nan,
                    "positive_cohort_wilson_low": cohort_low,
                    "positive_cohort_wilson_high": cohort_high,
                    "cohort_equal_mean_return": mean,
                    "cohort_bootstrap_ci95_low": bootstrap_low,
                    "cohort_bootstrap_ci95_high": bootstrap_high,
                    "worst_cohort_return": float(cohort_returns.min()) if len(cohort_returns) else np.nan,
                    "leave_one_cohort_out_min_mean": leave_one_cohort_out_minimum(cohort_returns),
                }
            )
    return pd.DataFrame(rows), cohort


def decide_strict_structure(
    metrics: pd.DataFrame,
    config: StrictLowZoneConfig,
    *,
    portfolio_metrics: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Apply frozen historical checks to the primary structure only."""

    primary = metrics.loc[
        metrics["structure"].eq(PRIMARY_STRICT_STRUCTURE)
        & metrics["horizon"].eq(504)
        & np.isclose(metrics["target_return"], 0.15)
    ].set_index("period")
    periods = primary.reindex(HISTORICAL_STRICT_PERIODS)
    overall = primary.reindex(["historical_2013_2024"]).iloc[0]
    periods_present = not periods.isna().all(axis=1).any()
    sample_checks = {
        "historical_periods_present": bool(periods_present),
        "minimum_two_completed_cohorts_each_period": bool(
            periods_present
            and periods["completed_cohorts"].ge(
                config.minimum_completed_cohorts_each_period
            ).all()
        ),
        "minimum_twelve_completed_cohorts_total": bool(
            pd.notna(overall.get("completed_cohorts"))
            and overall["completed_cohorts"] >= config.minimum_completed_cohorts_total
        ),
    }
    quality_checks = {
        "event_win_rate_at_least_85pct_each": bool(
            periods_present
            and periods["event_win_rate"].ge(config.minimum_event_win_rate).all()
        ),
        "positive_event_median_each": bool(
            periods_present and periods["median_event_return"].gt(0.0).all()
        ),
        "profit_factor_at_least_two_each": bool(
            periods_present
            and periods["profit_factor"].ge(config.minimum_profit_factor).all()
        ),
        "positive_cohort_mean_each": bool(
            periods_present and periods["cohort_equal_mean_return"].gt(0.0).all()
        ),
        "positive_cohort_share_at_least_80pct_overall": bool(
            pd.notna(overall.get("positive_cohort_share"))
            and overall["positive_cohort_share"] >= config.minimum_positive_cohort_share
        ),
        "worst_cohort_no_worse_than_minus_5pct": bool(
            pd.notna(overall.get("worst_cohort_return"))
            and overall["worst_cohort_return"] >= config.minimum_worst_cohort_return
        ),
        "cohort_bootstrap_lower_bound_positive": bool(
            pd.notna(overall.get("cohort_bootstrap_ci95_low"))
            and overall["cohort_bootstrap_ci95_low"] > 0.0
        ),
        "leave_one_cohort_out_minimum_positive": bool(
            pd.notna(overall.get("leave_one_cohort_out_min_mean"))
            and overall["leave_one_cohort_out_min_mean"] > 0.0
        ),
    }
    neighborhood = metrics.loc[
        metrics["structure"].isin(
            [
                "systemic_deep_no_new_low_dd60_survival80",
                PRIMARY_STRICT_STRUCTURE,
                "systemic_deep_no_new_low_dd80_survival80",
            ]
        )
        & metrics["period"].eq("historical_2013_2024")
        & metrics["horizon"].eq(504)
        & np.isclose(metrics["target_return"], 0.15)
    ]
    stable = (
        neighborhood["cohort_equal_mean_return"].gt(0.0)
        & neighborhood["cohort_bootstrap_ci95_low"].gt(0.0)
        & neighborhood["leave_one_cohort_out_min_mean"].gt(0.0)
    )
    neighborhood_check = {
        "two_of_three_drawdown_neighbors_robust": bool(stable.sum() >= 2),
        "robust_neighbors": neighborhood.loc[stable, "structure"].tolist(),
    }
    event_passed = all(sample_checks.values()) and all(quality_checks.values()) and bool(
        neighborhood_check["two_of_three_drawdown_neighbors_robust"]
    )
    portfolio_checks: dict[str, Any] = {"portfolio_evaluated": False}
    portfolio_passed = False
    if portfolio_metrics is not None and not portfolio_metrics.empty:
        selected = portfolio_metrics.set_index("period").reindex(HISTORICAL_STRICT_PERIODS)
        complete = not selected.isna().all(axis=1).any()
        portfolio_checks = {
            "portfolio_evaluated": True,
            "portfolio_periods_present": bool(complete),
            "positive_cagr_each": bool(complete and selected["cagr"].gt(0.0).all()),
            "maximum_drawdown_no_worse_than_10pct_each": bool(
                complete and selected["maximum_drawdown"].ge(-0.10).all()
            ),
            "worst_rolling_24m_no_worse_than_10pct_each": bool(
                complete and selected["worst_rolling_24m_return"].ge(-0.10).all()
            ),
        }
        portfolio_passed = all(portfolio_checks.values())
    if event_passed and portfolio_passed:
        status = "historically_robust_research_candidate"
    elif not all(sample_checks.values()) and all(quality_checks.values()):
        status = "promising_but_independent_sample_insufficient"
    else:
        status = "strict_structure_failed_robustness"
    return {
        "primary_structure": PRIMARY_STRICT_STRUCTURE,
        "primary_horizon_sessions": 504,
        "primary_target_return": 0.15,
        "sample_checks": sample_checks,
        "quality_checks": quality_checks,
        "neighborhood_check": neighborhood_check,
        "portfolio_checks": portfolio_checks,
        "event_gate_passed": event_passed,
        "portfolio_gate_passed": portfolio_passed,
        "status": status,
        "deployment_eligible": False,
        "deployment_blocker": (
            "all historical periods were exposed before this rule; require at least "
            "two newly completed non-adjacent systemic-panic anchor months"
        ),
    }
