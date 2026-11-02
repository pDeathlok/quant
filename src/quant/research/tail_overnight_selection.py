"""Close-proxy overnight target selection and cash-aware research scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd


def take_profit_return(frame, threshold, cost_bps=20.0):
    """Next-day target limit or close; prices conditional on fill assumptions."""
    if not 0 < threshold < 1 or not np.isfinite(cost_bps) or cost_bps < 0:
        raise ValueError("invalid threshold or costs")
    observed = frame[["y_open", "y_high", "y_close"]].notna().all(axis=1)
    value = frame.y_close.mask(frame.y_high.ge(threshold), threshold)
    value = value.mask(frame.y_open.ge(threshold), frame.y_open)
    return (value - cost_bps / 10000).where(observed)


def add_selection_features(frame):
    out = frame.copy()
    for col in (
        "tail_return",
        "tail_log_amount",
        "tail_turnover_est_pct",
        "tail_close_position",
        "lag1_daily_gap_mean_20",
        "lag1_volatility_20d",
    ):
        out[f"context_rank_{col}"] = out.groupby("date")[col].rank(pct=True)
    for col in ("tail_return", "tail_range", "tail_log_amount"):
        mean = out.groupby("date")[col].transform("mean")
        out[f"context_relative_{col}"] = out[col] - mean
    positive = out.tail_return.gt(0).where(out.tail_return.notna())
    out["context_candidate_up_fraction"] = positive.groupby(out.date).transform("mean")
    out["context_candidate_return_mean"] = out.groupby("date").tail_return.transform("mean")
    out["context_candidate_return_std"] = out.groupby("date").tail_return.transform("std")
    out["context_factor_coverage"] = (
        out[[c for c in out if c.startswith("lag1_")]].notna().mean(axis=1)
    )
    return out.replace([np.inf, -np.inf], np.nan)


def choose_candidates(frame, *, pool, top_k, minimum_net, maximum_risk):
    if pool not in ("union", "short", "good", "long") or top_k < 1:
        raise ValueError("invalid pool or top_k")
    if not np.isfinite(minimum_net) or not 0 <= maximum_risk <= 1:
        raise ValueError("invalid score/risk thresholds")
    mask = frame.predicted_net.ge(minimum_net) & frame.predicted_risk.le(maximum_risk)
    mask &= frame.predicted_net.notna() & frame.predicted_risk.notna()
    if pool != "union":
        mask &= frame[f"pool_{pool}"].eq(True)
    order = ["date", "predicted_net"]
    ascending = [True, False]
    if "raw_predicted_net" in frame:
        order.append("raw_predicted_net")
        ascending.append(False)
    return (
        frame.loc[mask]
        .sort_values([*order, "symbol"], ascending=[*ascending, True])
        .groupby("date")
        .head(top_k)
        .copy()
    )


def selection_metrics(picks, calendar, *, top_k):
    """Fixed allocation slots; no-investment dates earn zero.

    Missing outcomes are disclosed and assigned a conservative -10% mark for
    search ranking only. This is not a realized portfolio or execution model.
    """
    days = pd.DatetimeIndex(calendar).sort_values().unique()
    if top_k < 1 or len(days) == 0:
        raise ValueError("positive top_k and calendar required")
    if picks.groupby("date").size().gt(top_k).any():
        raise ValueError("more picks than capital slots")
    if not picks.date.isin(days).all():
        raise ValueError("pick date outside evaluation calendar")
    marked = picks.target_net.fillna(-0.10)
    daily = (marked.groupby(picks.date).sum() / top_k).reindex(days, fill_value=0.0)
    mean = float(daily.mean())
    standard_error = float(daily.std(ddof=1) / np.sqrt(len(daily))) if len(daily) > 1 else 0.0
    return {
        "selected_rows": len(picks),
        "active_days": int(picks.date.nunique()),
        "calendar_days": len(days),
        "missing_outcomes": int(picks.target_net.isna().sum()),
        "unresolved_open_exits": int((~picks.exit_open_executable).sum())
        if "exit_open_executable" in picks
        else 0,
        "daily_mark_mean": mean,
        "daily_mark_standard_error": standard_error,
        "selection_score": mean - 2 * standard_error,
        "trade_mark_mean": float(marked.mean()) if len(marked) else 0.0,
        "trade_win_rate": float(marked.gt(0).mean()) if len(marked) else 0.0,
        "daily_mark_p05": float(daily.quantile(0.05)),
        "allocation_fraction": len(picks) / (len(days) * top_k),
    }
