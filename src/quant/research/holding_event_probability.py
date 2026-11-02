"""High-confidence next-session event probabilities for existing holdings."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

EVENT_FAMILIES = {
    "high_up": ("y_high", "up"),
    "open_up": ("y_open", "up"),
    "close_up": ("y_close", "up"),
    "open_down": ("y_open", "down"),
    "close_down": ("y_close", "down"),
}
TARGETS = {
    f"{family}{level}": (column, level / 100 * (1 if direction == "up" else -1), direction)
    for family, (column, direction) in EVENT_FAMILIES.items()
    for level in (1, 2, 3)
}
THRESHOLDS = (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)


def event_labels(frame):
    result = pd.DataFrame(index=frame.index)
    for name, (column, level, direction) in TARGETS.items():
        outcome = frame[column].ge(level) if direction == "up" else frame[column].le(level)
        result[name] = outcome.astype(float).where(frame[column].notna())
    return result


def add_t0_interactions(frame):
    out = frame.copy()
    close = 1 + out.tail_return
    open_ = 1 + out.tail_gap
    high = close / (1 + out.tail_drawdown_from_high)
    low = close / (1 + out.tail_rebound_from_low)
    width = (high - low).replace(0, np.nan)
    body = close - open_
    out["t0x_body_range_signed"] = body / width
    out["t0x_body_range_absolute"] = body.abs() / width
    out["t0x_lower_shadow_share"] = (pd.concat([close, open_], axis=1).min(axis=1) - low) / width
    out["t0x_high_excursion"] = high - 1
    out["t0x_low_excursion"] = low - 1
    out["t0x_gap_reversal_product"] = -out.tail_gap * body
    out["t0x_gap_fill_fraction"] = (-body / out.tail_gap.replace(0, np.nan)).clip(-3, 3)
    out["t0x_vwap_deviation_per_range"] = out.tail_price_to_vwap / out.tail_range.replace(0, np.nan)
    out["t0x_return_per_sqrt_turnover"] = out.tail_return / np.sqrt(
        out.tail_turnover_est_pct.clip(lower=0).replace(0, np.nan)
    )
    out["t0x_turnover_change"] = out.tail_turnover_est_pct - out.lag1_turnover_rate
    out["t0x_relative_volume_acceleration"] = (
        out.tail_volume_vs_lag1_ma5 - out.tail_volume_vs_lag1_ma20
    )
    out["t0x_volume_body_pressure"] = out.t0x_body_range_signed * np.log1p(
        out.tail_volume_vs_lag1_ma20.clip(lower=0)
    )
    out["t0x_range_expansion"] = out.tail_range / out.lag1_daily_range.replace(0, np.nan)
    out["t0x_return_surprise"] = out.tail_return - out.lag1_daily_return_mean_20
    out["t0x_return_surprise_scaled"] = out.t0x_return_surprise / out.lag1_daily_gap_std_20.replace(
        0, np.nan
    )
    out["t0x_close_location_volume"] = (out.tail_close_position - 0.5) * np.log1p(
        out.tail_volume_vs_lag1_ma20.clip(lower=0)
    )
    out["t0x_drawdown_volume"] = out.tail_drawdown_from_high * np.log1p(
        out.tail_volume_vs_lag1_ma20.clip(lower=0)
    )
    out["t0x_turnover_range_pressure"] = out.tail_turnover_est_pct * out.tail_range
    return out.replace([np.inf, -np.inf], np.nan)


def probability_metrics(y, probability, *, bins=10):
    y, probability = np.asarray(y, dtype=float), np.asarray(probability, dtype=float)
    if len(y) == 0 or not np.isfinite(y).all() or not np.isfinite(probability).all():
        raise ValueError("finite observed outcomes and probabilities required")
    if not np.isin(y, [0, 1]).all() or (probability < 0).any() or (probability > 1).any():
        raise ValueError("binary targets and probabilities in [0,1] required")
    bin_id = np.minimum((probability * bins).astype(int), bins - 1)
    rows = []
    ece = 0.0
    for b in range(bins):
        mask = bin_id == b
        if not mask.any():
            continue
        expected = float(probability[mask].mean())
        observed = float(y[mask].mean())
        ece += mask.mean() * abs(expected - observed)
        rows.append(
            {
                "bin": b,
                "count": int(mask.sum()),
                "predicted_mean": expected,
                "observed_rate": observed,
            }
        )
    base = float(y.mean())
    two_classes = len(np.unique(y)) == 2
    return {
        "samples": len(y),
        "base_rate": base,
        "brier": float(brier_score_loss(y, probability)),
        "base_rate_brier": base * (1 - base),
        "ece": float(ece),
        "roc_auc": float(roc_auc_score(y, probability)) if two_classes else None,
        "pr_auc": float(average_precision_score(y, probability)) if two_classes else None,
    }, rows


def threshold_metrics(frame, thresholds=THRESHOLDS, *, bootstrap=400, seed=42):
    """Conditional precision with same-date and local temporal dependence.

    Resample consecutive five-session blocks, including no-signal days. This
    is an exploratory interval, not simultaneous protection over model search.
    """
    observed = frame.dropna(subset=["outcome", "probability"]).copy()
    if observed.empty:
        return pd.DataFrame()
    days = sorted(observed.date.unique())
    day_ids = {d: i for i, d in enumerate(days)}
    block = (observed.date.map(day_ids) // 5).to_numpy(dtype=int)
    block_count = (len(days) + 4) // 5
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(block_count, np.full(block_count, 1 / block_count), size=bootstrap)
    y = observed.outcome.to_numpy(dtype=float)
    p = observed.probability.to_numpy(dtype=float)
    base = y.mean()
    rows = []
    for threshold in thresholds:
        selected = p >= threshold
        count = int(selected.sum())
        hits = float(y[selected].sum())
        nblock = np.bincount(block, weights=selected.astype(float), minlength=block_count)
        hblock = np.bincount(block, weights=selected * y, minlength=block_count)
        denominators = draws @ nblock
        valid = denominators > 0
        boot = (draws @ hblock)[valid] / denominators[valid]
        precision = hits / count if count else np.nan
        rows.append(
            {
                "threshold": threshold,
                "signals": count,
                "signal_dates": int(observed.loc[selected, "date"].nunique()),
                "precision": precision,
                "recall": hits / y.sum() if y.sum() else np.nan,
                "coverage": count / len(y),
                "base_rate": base,
                "lift": precision / base if base else np.nan,
                "mean_probability": float(p[selected].mean()) if count else np.nan,
                "precision_low90": float(np.quantile(boot, 0.05)) if len(boot) else np.nan,
                "precision_high90": float(np.quantile(boot, 0.95)) if len(boot) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def choose_high_confidence(curves):
    eligible = curves[
        (curves.signals >= 50)
        & (curves.signal_dates >= 20)
        & (curves.precision >= 0.8)
        & (curves.precision_low90 >= 0.7)
        & (curves.lift >= 1.2)
    ]
    if eligible.empty:
        return None
    return (
        eligible.sort_values(
            ["coverage", "precision_low90", "feature_group", "threshold"],
            ascending=[False, False, True, True],
        )
        .iloc[0]
        .to_dict()
    )


def coherent_event_probabilities(probabilities):
    """Least-squares decreasing projection for nested 1/2/3% events.

    Applied identically before validation, confirmation and inference.
    Open/close upside implies the corresponding high-price event. Upside
    and downside at the same observation time are mutually exclusive.
    """
    out = probabilities.copy()
    for prefix in EVENT_FAMILIES:
        family = [f"{prefix}{level}" for level in (1, 2, 3)]
        a, b, c = [out[k].to_numpy(dtype=float).copy() for k in family]
        values = np.stack([a, b, c], axis=1)
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ValueError("invalid event probability")
        average = (a + b + c) / 3
        ab = (a + b) / 2
        bc = (b + c) / 2
        merge_ab = a < b
        all_ab = merge_ab & (ab < c)
        all_bc = (~merge_ab) & (b < c) & (a < bc)
        pair_bc = (~merge_ab) & (b < c) & ~all_bc
        a = np.where(merge_ab, ab, a)
        b = np.where(merge_ab, ab, b)
        b = np.where(pair_bc, bc, b)
        c = np.where(pair_bc, bc, c)
        all_three = all_ab | all_bc
        for name, value in zip(family, [a, b, c]):
            out[name] = np.where(all_three, average, value)
    for observation in ("open", "close"):
        for level in (1, 2, 3):
            up = f"{observation}_up{level}"
            out[up] = np.minimum(out[up], out[f"high_up{level}"])
        # One scale for all levels preserves nested probabilities after enforcing
        # mutually exclusive up/down events at this observation time.
        scale = (out[f"{observation}_up1"] + out[f"{observation}_down1"]).clip(lower=1.0)
        for direction in ("up", "down"):
            for level in (1, 2, 3):
                out[f"{observation}_{direction}{level}"] /= scale
    return out
