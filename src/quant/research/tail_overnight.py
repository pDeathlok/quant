"""Tail-entry research contracts; no broker or production-model side effects.

All returns are fractions. Daily execution is an approximation, never an
assertion that a next-session high or a limit-price print was fillable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

POOL_NAMES = ("good", "short", "long", "watchlist")


def _keys(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    out = frame.copy()
    for col in ("symbol", "date"):
        if col not in out:
            raise ValueError(f"{name} missing {col}")
    if out.symbol.isna().any() or out.symbol.astype(str).str.strip().eq("").any():
        raise ValueError(f"{name} missing symbol")
    out["symbol"] = out.symbol.astype(str)
    dates = out.date if pd.api.types.is_datetime64_any_dtype(out.date) else out.date.astype(str)
    out["date"] = pd.to_datetime(dates, errors="raise").dt.normalize()
    if out.date.isna().any() or out.duplicated(["symbol", "date"]).any():
        raise ValueError(f"{name} missing/duplicate date keys")
    return out


def calendar_links(calendar: Sequence) -> pd.DataFrame:
    dates = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(calendar)).astype(str))).normalize()
    if len(dates) < 2 or dates.hasnans or dates.duplicated().any():
        raise ValueError("calendar needs distinct nonmissing market dates")
    dates = dates.sort_values()
    return pd.DataFrame(
        {
            "date": dates,
            "feature_date": pd.Series(dates).shift(1),
            "exit_date": pd.Series(dates).shift(-1),
        }
    )


def pool_union(pools: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    if set(pools) - set(POOL_NAMES):
        raise ValueError("unknown pool")
    parts = []
    for name, frame in pools.items():
        part = _keys(frame[["symbol", "date"]].drop_duplicates(), name)
        part[f"pool_{name}"] = True
        parts.append(part)
    if not parts:
        return pd.DataFrame(columns=["symbol", "date", *[f"pool_{p}" for p in POOL_NAMES]])
    out = pd.concat(parts).groupby(["symbol", "date"], as_index=False).max()
    for name in POOL_NAMES:
        col = f"pool_{name}"
        out[col] = out[col].eq(True) if col in out else False
    return out


def attach_lagged_features(events, features, calendar, feature_columns):
    """Exact global T-1 join; missing yesterday is not an as-of carry forward."""
    out = _keys(events, "events")
    links = calendar_links(calendar)
    if not out.date.isin(links.date).all():
        raise ValueError("event date outside calendar")
    out = out.merge(links[["date", "feature_date"]], on="date", validate="many_to_one")
    cols = list(feature_columns)
    if len(set(cols)) != len(cols) or any(c.startswith("y_") for c in cols):
        raise ValueError("invalid feature allowlist")
    right = _keys(features, "features")[["symbol", "date", *cols]]
    right = right.rename(columns={"date": "feature_date", **{c: f"lag1_{c}" for c in cols}})
    return out.merge(right, on=["symbol", "feature_date"], how="left", validate="many_to_one")


def build_snapshot_features(snapshot: pd.DataFrame, *, mode: str) -> pd.DataFrame:
    """Require normalized shares/yuan/percentage-point turnover at ingestion.

    close_proxy deliberately uses full-day information, so its fitted model
    cannot be presented as a validated 14:30 model.
    """
    if mode not in ("close_proxy", "snapshot_1430"):
        raise ValueError("unknown input mode")
    out = _keys(snapshot, "snapshot")
    required = [
        "price",
        "open",
        "high",
        "low",
        "pre_close",
        "volume_shares",
        "amount_yuan",
        "turnover_pct",
    ]
    missing = set(required) - set(out)
    if missing:
        raise ValueError(f"snapshot missing {sorted(missing)}")
    for col in required:
        out[col] = pd.to_numeric(out[col], errors="raise")
    if (out[["price", "open", "high", "low", "pre_close"]] <= 0).any().any():
        raise ValueError("snapshot prices must be positive")
    if (out[["volume_shares", "amount_yuan", "turnover_pct"]] < 0).any().any():
        raise ValueError("snapshot volume/amount/turnover must be nonnegative")
    factor = 1.0
    if mode == "snapshot_1430":
        live_required = [
            "price",
            "open",
            "high",
            "low",
            "pre_close",
            "volume_shares",
            "amount_yuan",
        ]
        if not np.isfinite(out[live_required].to_numpy(dtype=float)).all():
            raise ValueError("live quote has missing/nonfinite prices or activity")
        if (out.high < out[["price", "open", "low"]].max(axis=1)).any() or (
            out.low > out[["price", "open", "high"]].min(axis=1)
        ).any():
            raise ValueError("live quote has inconsistent OHLC")
        if "quote_time" not in out:
            raise ValueError("quote_time is required")
        quote = pd.to_datetime(out.quote_time, errors="raise")
        if isinstance(quote.dtype, pd.DatetimeTZDtype):
            quote = quote.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
        cutoff = out.date + pd.Timedelta(hours=14, minutes=30)
        age = (cutoff - quote).dt.total_seconds()
        if quote.isna().any() or (~age.between(0, 120)).any():
            raise ValueError("quote must be from 14:28..14:30 on decision date")
        elapsed = 120 + (quote.dt.hour - 13) * 60 + quote.dt.minute + quote.dt.second / 60
        factor = 240 / elapsed

    def safe(value):
        return value.replace(0, np.nan)

    width = safe(out.high - out.low)
    result = out[["symbol", "date"]].copy()
    result["input_mode"] = mode
    result["tail_return"] = out.price / safe(out.pre_close) - 1
    result["tail_gap"] = out.open / safe(out.pre_close) - 1
    result["tail_body"] = out.price / safe(out.open) - 1
    result["tail_range"] = (out.high - out.low) / safe(out.pre_close)
    result["tail_close_position"] = (out.price - out.low) / width
    result["tail_upper_shadow_share"] = (out.high - out[["price", "open"]].max(axis=1)) / width
    result["tail_rebound_from_low"] = out.price / safe(out.low) - 1
    result["tail_drawdown_from_high"] = out.price / safe(out.high) - 1
    result["tail_price_to_vwap"] = out.price / safe(out.amount_yuan / safe(out.volume_shares)) - 1
    result["tail_volume_est_shares"] = out.volume_shares * factor
    result["tail_amount_est_yuan"] = out.amount_yuan * factor
    result["tail_turnover_est_pct"] = out.turnover_pct * factor
    result["tail_log_amount"] = np.log1p(result.tail_amount_est_yuan)
    return result.replace([np.inf, -np.inf], np.nan)


def add_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Project volume averages use Tushare hands; snapshots use shares."""
    out = frame.copy()
    pairs = {
        "tail_volume_vs_lag1_ma5": ("tail_volume_est_shares", "lag1_volume_ma5", 100.0),
        "tail_volume_vs_lag1_ma20": ("tail_volume_est_shares", "lag1_volume_ma20", 100.0),
        "tail_turnover_vs_lag1": ("tail_turnover_est_pct", "lag1_turnover_rate", 1.0),
    }
    for name, (left, right, multiplier) in pairs.items():
        if left in out and right in out:
            out[name] = out[left] / (out[right] * multiplier).replace(0, np.nan)
    if "lag1_volatility_20d" in out:
        out["tail_return_vs_lag1_volatility"] = out.tail_return / out.lag1_volatility_20d.replace(
            0, np.nan
        )
    return out.replace([np.inf, -np.inf], np.nan)


def build_labels(events, daily, calendar, *, cost_bps=20.0):
    """Raw-price labels with corporate-action quarantine and explicit fill flags.

    Absent precise limit data use conservative >=4.8% / <=-4.8% raw-price
    proxies. These are diagnostics, not historical exact-limit evidence.
    Suspended/missing exits remain in the output; y_open_net is a hypothetical
    open mark after costs, not a filled portfolio return.
    """
    if not np.isfinite(cost_bps) or cost_bps < 0:
        raise ValueError("cost_bps must be finite and nonnegative")
    out = _keys(events, "events")
    links = calendar_links(calendar)
    if not out.date.isin(links.date).all():
        raise ValueError("event date outside calendar")
    out = out.merge(links[["date", "exit_date"]], on="date", how="left", validate="many_to_one")
    bars = _keys(daily, "daily")
    required = ["open", "high", "low", "close", "pre_close", "volume_shares"]
    missing = set(required) - set(bars)
    if missing:
        raise ValueError(f"daily missing {sorted(missing)}")
    for col in ["up_limit", "down_limit"]:
        if col not in bars:
            bars[col] = np.nan
    if "is_suspended" not in bars:
        bars["is_suspended"] = False
    bars["is_suspended"] = bars.is_suspended.astype(str).str.lower().isin(["true", "1", "1.0"])
    columns = [*required, "up_limit", "down_limit", "is_suspended"]
    entry = bars[["symbol", "date", *columns]].rename(columns={c: f"entry_{c}" for c in columns})
    nxt = bars[["symbol", "date", *columns]].rename(
        columns={"date": "exit_date", **{c: f"next_{c}" for c in columns}}
    )
    out = out.merge(entry, on=["symbol", "date"], how="left", validate="many_to_one")
    out = out.merge(nxt, on=["symbol", "exit_date"], how="left", validate="many_to_one")
    entry_ok = (
        out.entry_close.gt(0) & out.entry_volume_shares.gt(0) & out.entry_is_suspended.eq(False)
    )
    exact_up = out.entry_up_limit.gt(0)
    entry_at_limit = (out.entry_close >= out.entry_up_limit - 0.0051).where(
        exact_up, out.entry_close / out.entry_pre_close - 1 >= 0.048
    )
    out["entry_limit_source"] = np.where(exact_up, "exact", "conservative_4p8_proxy")
    out["entry_executable"] = (entry_ok & ~entry_at_limit).fillna(False)
    complete = out[[f"next_{c}" for c in ["open", "high", "low", "close"]]].gt(0).all(axis=1)
    complete &= out.next_volume_shares.gt(0) & out.next_is_suspended.eq(False)
    out["corporate_action_gap"] = (out.next_pre_close - out.entry_close).abs().gt(0.011)
    out["outcome_observed"] = (
        entry_ok & complete & out.next_pre_close.gt(0) & ~out.corporate_action_gap
    ).fillna(False)
    exact_down = out.next_down_limit.gt(0)
    exit_at_limit = (out.next_open <= out.next_down_limit + 0.0051).where(
        exact_down, out.next_open / out.next_pre_close - 1 <= -0.048
    )
    out["exit_limit_source"] = np.where(exact_down, "exact", "conservative_4p8_proxy")
    out["exit_open_executable"] = (out.outcome_observed & ~exit_at_limit).fillna(False)
    for col in ("open", "high", "low", "close"):
        out[f"y_{col}"] = (out[f"next_{col}"] / out.entry_close - 1).where(out.outcome_observed)
    out["y_open_net"] = out.y_open - cost_bps / 10000
    out["y_open_positive"] = out.y_open_net.gt(0).astype("boolean").where(out.outcome_observed)
    for bps in (50, 100, 200, 300):
        out[f"y_hit_up_{bps}bps"] = (
            out.y_high.ge(bps / 10000).astype("boolean").where(out.outcome_observed)
        )
    for bps in (100, 200, 300):
        out[f"y_hit_down_{bps}bps"] = (
            out.y_low.le(-bps / 10000).astype("boolean").where(out.outcome_observed)
        )
    out["y_open_bucket"] = pd.cut(
        out.y_open, [-np.inf, -0.02, -0.01, 0, 0.01, 0.02, np.inf], labels=False
    )
    out["round_trip_cost_bps"] = cost_bps
    return out


def purged_split(frame, validation_start, test_start):
    validation_start, test_start = pd.Timestamp(validation_start), pd.Timestamp(test_start)
    if validation_start >= test_start:
        raise ValueError("validation must precede test")
    return (
        frame[(frame.date < validation_start) & (frame.exit_date < validation_start)].copy(),
        frame[
            (frame.date >= validation_start)
            & (frame.date < test_start)
            & (frame.exit_date < test_start)
        ].copy(),
        frame[frame.date >= test_start].copy(),
    )


def daily_exit_scenarios(outcomes, *, take_profit=0.01, stop_loss=0.01, cost_bps=20.0):
    """OHLC exit scenarios conditional on fills, not executable portfolio PnL.

    Opening gaps execute at the open, not the crossed trigger. When both
    barriers occur after the open, report both possible order assumptions.
    Intraday gaps, queue priority and suspended positions require minute/order
    data and are deliberately not inferred from daily highs and lows.
    """
    if not (0 < take_profit < 1 and 0 < stop_loss < 1):
        raise ValueError("barriers must be fractions between 0 and 1")
    if not np.isfinite(cost_bps) or cost_bps < 0:
        raise ValueError("cost_bps must be finite and nonnegative")
    required = ["y_open", "y_high", "y_low", "y_close"]
    if set(required) - set(outcomes):
        raise ValueError("missing outcome returns")
    valid = outcomes[required].notna().all(axis=1)
    opening = outcomes.y_open.ge(take_profit) | outcomes.y_open.le(-stop_loss)
    high_hit = outcomes.y_high.ge(take_profit)
    low_hit = outcomes.y_low.le(-stop_loss)
    both = valid & ~opening & high_hit & low_hit
    stop_first = outcomes.y_close.copy()
    take_first = outcomes.y_close.copy()
    stop_first = stop_first.mask(high_hit, take_profit).mask(low_hit, -stop_loss)
    take_first = take_first.mask(low_hit, -stop_loss).mask(high_hit, take_profit)
    stop_first = stop_first.mask(opening, outcomes.y_open)
    take_first = take_first.mask(opening, outcomes.y_open)
    cost = cost_bps / 10000
    return pd.DataFrame(
        {
            "next_close_net": (outcomes.y_close - cost).where(valid),
            "bracket_stop_first_net": (stop_first - cost).where(valid),
            "bracket_take_first_net": (take_first - cost).where(valid),
            "ambiguous_order": both,
        },
        index=outcomes.index,
    )
