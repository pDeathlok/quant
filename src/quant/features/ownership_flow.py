"""Point-in-time ownership and ETF flow variables used by short-side research."""
from __future__ import annotations

from collections.abc import Mapping
from itertools import groupby
from operator import itemgetter

import numpy as np
import pandas as pd

ETF_BASKETS = {
    "csi300": ("000300.SH", ("510300.SH", "510310.SH", "510330.SH", "159919.SZ")),
    "sse50": ("000016.SH", ("510050.SH", "510100.SH")),
    "csi500": ("000905.SH", ("510500.SH", "159922.SZ")),
    "csi1000": ("000852.SH", ("512100.SH",)),
    "star50": ("000688.SH", ("588000.SH", "588080.SH")),
    "chinext": ("399006.SZ", ("159915.SZ",)),
    "chinext50": ("399673.SZ", ("159949.SZ",)),
    "csi_a500": ("000510.SH", ("563360.SH", "159338.SZ")),
}
ETF_GROUPS = {name: symbols for name, (_, symbols) in ETF_BASKETS.items()}

KEYS = ["symbol", "date"]
HOLDER_COUNT_COLUMNS = (
    "own_holder_log_count", "own_holder_change_count", "own_holder_change_pct",
    "own_holder_change_pct_prior", "own_holder_change_pct_accel", "own_holder_report_age_days",
    "own_holder_change_pct_per_30d",
)
CONCENTRATION_COLUMNS = tuple(
    f"own_{kind}_{measure}" for kind in ("top", "float") for measure in (
        "top1_pct", "top5_pct", "top10_pct", "top10_change_pp", "top10_change_pct",
        "top10_change_shares", "num_ge5", "report_age_days",
    )
)
TRANSACTION_COLUMNS = tuple(
    f"own_events_{measure}_{window}" for window in (5, 20, 60) for measure in (
        "net_pct", "net_shares", "count", "buy_share",
    )
)
ETF_MEASURES = (
    "flow_yuan_1", "flow_yuan_5", "flow_yuan_20", "flow_pct_1",
    "flow_pct_5", "flow_pct_20", "flow_accel", "flow_z60",
    "share_pct_5", "share_pct_20", "positive_fraction_5",
    "turnover_multiple_20",
)


def dates(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values.astype("string"), format="mixed", errors="coerce")


def _dated(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    frame["symbol"] = frame.ts_code.astype("string")
    frame["ann_date"] = dates(frame.ann_date)
    if "end_date" in frame:
        frame["end_date"] = dates(frame.end_date)
        frame = frame[frame.ann_date.ge(frame.end_date)]
    return frame.dropna(subset=["symbol", "ann_date"])


def _history_states(frame: pd.DataFrame, value_columns: list[str], depth: int) -> pd.DataFrame:
    """Keep the latest observable version of each period as releases arrive."""
    states = []
    ordered = frame.sort_values(["symbol", "ann_date", "end_date"])
    records = ordered[["symbol", "ann_date", "end_date", *value_columns]].itertuples(index=False, name=None)
    for symbol, group in groupby(records, itemgetter(0)):
        known: dict = {}
        for ann_date, release in groupby(group, itemgetter(1)):
            for row in release:
                known[row[2]] = row[3:]
            latest = sorted(known, reverse=True)[:depth]
            state = {"symbol": symbol, "ann_date": ann_date}
            for offset, period in enumerate(latest):
                state[f"period_{offset}"] = period
                state.update({f"{column}_{offset}": value
                              for column, value in zip(value_columns, known[period], strict=True)})
            states.append(state)
    columns = ["symbol", "ann_date"]
    for offset in range(depth):
        columns.extend([f"period_{offset}", *(f"{c}_{offset}" for c in value_columns)])
    out = pd.DataFrame(states, columns=columns)
    for offset in range(depth):
        out[f"period_{offset}"] = pd.to_datetime(out[f"period_{offset}"])
    out["ann_date"] = pd.to_datetime(out.ann_date)
    out["symbol"] = out.symbol.astype("string")
    return out


def _attach(keys: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    return pd.merge_asof(
        keys.assign(symbol=keys.symbol.astype("string")).sort_values("date"), states.assign(ann_date=states.ann_date.astype(keys.date.dtype)).sort_values("ann_date"),
        by="symbol", left_on="date", right_on="ann_date", allow_exact_matches=False,
    )


def holder_count_features(keys: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    frame = _dated(raw)
    frame["holder_num"] = pd.to_numeric(frame.holder_num, errors="coerce")
    frame = frame[frame.holder_num.gt(0)]
    # Conflicting copies with identical observation and publication date are ambiguous.
    by_key = ["symbol", "end_date", "ann_date"]
    counts = frame.groupby(by_key).holder_num.agg(["min", "max"]).reset_index()
    counts = counts[counts["min"].eq(counts["max"])].rename(columns={"min": "holder_num"})
    joined = _attach(keys, _history_states(counts, ["holder_num"], 3))
    current, previous, prior = (joined[f"holder_num_{i}"] for i in range(3))
    gap = (joined.period_0 - joined.period_1).dt.days
    joined["own_holder_log_count"] = np.log1p(current)
    joined["own_holder_change_count"] = current - previous
    joined["own_holder_change_pct"] = current / previous - 1
    joined["own_holder_change_pct_prior"] = previous / prior - 1
    joined["own_holder_change_pct_accel"] = (
        joined.own_holder_change_pct - joined.own_holder_change_pct_prior
    )
    joined["own_holder_report_age_days"] = (joined.date - joined.period_0).dt.days
    joined["own_holder_change_pct_per_30d"] = joined.own_holder_change_pct * 30 / gap.where(gap.gt(0))
    return joined[KEYS + list(HOLDER_COUNT_COLUMNS)]


def concentration_features(
    keys: pd.DataFrame, raw: pd.DataFrame, kind: str,
) -> pd.DataFrame:
    frame = _dated(raw)
    frame = frame[frame.holder_name.notna()].copy()
    ratio = "hold_ratio" if kind == "top" else "hold_float_ratio"
    frame["ratio"] = pd.to_numeric(frame[ratio], errors="coerce")
    frame["shares"] = pd.to_numeric(frame.hold_amount, errors="coerce")
    frame = frame[frame.ratio.between(0, 100) & frame.shares.gt(0)]
    entity_key = ["symbol", "end_date", "ann_date", "holder_name"]
    frame = frame.drop_duplicates(entity_key + ["ratio", "shares"])
    frame = frame[~frame.duplicated(entity_key, keep=False)]
    group_key = entity_key[:-1]
    frame = frame.sort_values(group_key + ["shares"], ascending=[True, True, True, False])
    groups = frame.groupby(group_key, sort=False)
    frame["rank"] = groups.cumcount() + 1
    frame["ratio5"] = frame.ratio.where(frame["rank"].le(5))
    frame["above5"] = frame.ratio.ge(5).astype(int)
    states = groups.agg(top1_pct=("ratio", "first"), top5_pct=("ratio5", "sum"),
                        top10_pct=("ratio", "sum"), top10_shares=("shares", "sum"),
                        num_ge5=("above5", "sum"), count=("ratio", "size")).reset_index()
    states = states[states["count"].eq(10) & states.top10_pct.le(100.15)]
    values = ["top1_pct", "top5_pct", "top10_pct", "top10_shares", "num_ge5"]
    joined = _attach(keys, _history_states(states, values, 2))
    prefix = f"own_{kind}_"
    for measure in ("top1_pct", "top5_pct", "top10_pct", "num_ge5"):
        joined[prefix + measure] = joined[f"{measure}_0"]
    joined[prefix + "top10_change_pp"] = joined.top10_pct_0 - joined.top10_pct_1
    joined[prefix + "top10_change_pct"] = joined.top10_pct_0 / joined.top10_pct_1.where(joined.top10_pct_1.gt(0)) - 1
    joined[prefix + "top10_change_shares"] = joined.top10_shares_0 - joined.top10_shares_1
    joined[prefix + "report_age_days"] = (joined.date - joined.period_0).dt.days
    return joined[KEYS + [column for column in CONCENTRATION_COLUMNS if column.startswith(prefix)]]


def transaction_features(
    keys: pd.DataFrame, raw: pd.DataFrame, calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    frame = _dated(raw)
    frame = frame[frame.in_de.isin(["IN", "DE"])].drop_duplicates()
    frame["session"] = calendar.searchsorted(frame.ann_date, side="right")
    frame["sign"] = np.where(frame.in_de.eq("IN"), 1, -1)
    frame["net_pct"] = pd.to_numeric(frame.change_ratio, errors="coerce").abs() * frame.sign
    frame["net_shares"] = pd.to_numeric(frame.change_vol, errors="coerce").abs() * frame.sign
    frame["buy"] = frame.sign.eq(1).astype(float)
    out = []
    groups = {symbol: sub.sort_values("session") for symbol, sub in frame.groupby("symbol", sort=False)}
    columns = ["net_pct", "net_shares", "buy"]
    for symbol, group in keys.groupby("symbol", sort=False):
        group = group.copy()
        events = groups.get(symbol, frame.iloc[:0])
        observed = calendar.get_indexer(group.date)
        event_sessions = events.session.to_numpy()
        totals = np.vstack([np.zeros(3), np.nan_to_num(events[columns].to_numpy(dtype=float)).cumsum(axis=0)])
        missing = np.vstack([np.zeros(2), events[columns[:2]].isna().to_numpy().cumsum(axis=0)])
        right = np.searchsorted(event_sessions, observed, side="right")
        for window in (5, 20, 60):
            left = np.searchsorted(event_sessions, observed - window, side="right")
            total, count = totals[right] - totals[left], right - left
            incomplete = (missing[right] - missing[left]) > 0
            for index, measure in enumerate(columns[:2]):
                group[f"own_events_{measure}_{window}"] = np.where(incomplete[:, index], np.nan, total[:, index])
            group[f"own_events_count_{window}"] = count
            group[f"own_events_buy_share_{window}"] = np.divide(
                total[:, 2], count, out=np.full(len(count), np.nan), where=count > 0,
            )
        out.append(group)
    return pd.concat(out, ignore_index=True)


def etf_flow_features(
    shares: pd.DataFrame, navs: pd.DataFrame, bars: pd.DataFrame, adjustments: pd.DataFrame,
    listings: pd.DataFrame, groups: Mapping[str, tuple[str, ...]], calendar: pd.DatetimeIndex,
    *, include_observation_dates: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build date-level features and an unlagged coverage panel for source audits."""
    launch = listings.drop_duplicates("ts_code").set_index("ts_code").list_date
    launch = dates(launch)
    parts = []
    for code in sorted({code for group in groups.values() for code in group}):
        start = launch.get(code)
        if pd.isna(start):
            raise ValueError(f"Missing ETF listing date for {code}")
        panel = pd.DataFrame(index=calendar)
        panel["active"] = calendar >= start

        for raw, date_col, fields in (
            (shares, "trade_date", ["fd_share"]),
            (navs, "nav_date", ["unit_nav", "ann_date"]),
            (bars, "trade_date", ["amount"]),
            (adjustments, "trade_date", ["adj_factor"]),
        ):
            sub = raw.loc[raw.ts_code.eq(code), [date_col, *fields]].copy()
            sub[date_col] = dates(sub[date_col])
            sub = sub.drop_duplicates()
            sub = sub[~sub[date_col].duplicated(keep=False)]
            panel = panel.join(sub.set_index(date_col))

        share = pd.to_numeric(panel.fd_share, errors="coerce").where(panel.active)
        nav = pd.to_numeric(panel.unit_nav, errors="coerce").where(panel.unit_nav.gt(0))
        adj = pd.to_numeric(panel.adj_factor, errors="coerce")
        adjacent = panel.active & panel.active.shift(fill_value=False) & share.gt(0) & share.shift().gt(0)
        # Exclude adjustment dates conservatively, including cash dividend adjustments.
        unchanged = adj.gt(0) & adj.shift().gt(0) & np.isclose(adj, adj.shift(), rtol=1e-8, atol=0)
        announcement = dates(panel.ann_date)
        observed = (adjacent & unchanged & nav.gt(0) & nav.shift().gt(0)
                    & announcement.notna() & announcement.shift().notna())
        panel["flow"] = ((share - share.shift()) * nav * 10_000).where(observed)
        panel["asset"] = share * nav * 10_000
        panel["prev_asset"] = panel.asset.shift().where(observed)
        panel["share_pct"] = (share / share.shift() - 1).where(observed)
        panel["turnover"] = pd.to_numeric(panel.amount, errors="coerce") * 1000
        panel["adjustment_day"] = (adj.gt(0) & adj.shift().gt(0) & ~unchanged) & adjacent
        panel["code"] = code
        # The account probe showed D shares lagging D close. Use at least D+2 and
        # at least the first session strictly after the NAV announcement.
        ready = np.maximum.reduce([
            np.arange(len(calendar)) + 2,
            calendar.searchsorted(announcement, side="right"),
            np.where(announcement.shift().notna(), calendar.searchsorted(announcement.shift(), side="right"), 0),
        ])
        panel["ready_session"] = np.where(observed, ready, -1)
        parts.append(panel.rename_axis("trade_date").reset_index())
    detailed = pd.concat(parts, ignore_index=True)
    out = {}
    group_map = {"all": tuple(sorted(detailed.code.unique())), **groups}
    for name, codes in group_map.items():
        rows = detailed.loc[detailed.code.isin(codes) & detailed.active].copy()
        grouped = rows.groupby("trade_date")
        totals = grouped.agg(
            expected=("code", "size"), covered=("flow", "count"), flow=("flow", "sum"),
            asset=("prev_asset", "sum"), share_pct=("share_pct", "mean"),
            turnover=("turnover", "sum"), bars=("turnover", "count"),
            ready_session=("ready_session", "max"),
        ).reindex(calendar)
        # A tiny data hole must not look like a complete zero-flow basket.
        valid = totals.covered.eq(totals.expected) & totals.expected.gt(0)
        flow = totals.flow.where(valid)
        asset = totals.asset.where(valid & totals.asset.gt(0))
        avg_share = totals.share_pct.where(valid)
        daily_pct = flow / asset
        features = {
            "flow_yuan_1": (flow, 1), "flow_pct_1": (daily_pct, 1),
            "flow_accel": (daily_pct.rolling(5).mean() - daily_pct.shift(5).rolling(15).mean(), 20),
            "flow_z60": ((daily_pct - daily_pct.shift(1).rolling(60, min_periods=45).mean())
                         / daily_pct.shift(1).rolling(60, min_periods=45).std().replace(0, np.nan), 61),
        }
        for window in (5, 20):
            # One missing product/date may leave a window uncomputable, so keep
            # source coverage separate and use no more than one incomplete day.
            minimum = window - 1
            rolling = flow.rolling(window, min_periods=minimum).sum()
            features[f"flow_yuan_{window}"] = (rolling, window)
            features[f"flow_pct_{window}"] = (rolling / asset.shift(window - 1), window)
            share_growth = (1 + avg_share).rolling(window, min_periods=minimum).apply(np.prod, raw=True) - 1
            features[f"share_pct_{window}"] = (share_growth, window)
        positive = daily_pct.gt(0).where(daily_pct.notna()).astype(float).rolling(5, min_periods=4).mean()
        features["positive_fraction_5"] = (positive, 5)
        source_positions = np.arange(len(calendar))
        for metric, (values, window) in features.items():
            ready = totals.ready_session.where(valid).rolling(window, min_periods=1).max().to_numpy()
            usable = values.notna().to_numpy() & np.isfinite(ready)
            releases = pd.DataFrame({"source": source_positions[usable], "ready": ready[usable]})
            releases = releases.sort_values(["ready", "source"])
            observed_at = np.searchsorted(releases.ready, source_positions, side="right") - 1
            output = np.full(len(calendar), np.nan)
            record_date = include_observation_dates and metric in {"flow_yuan_5", "flow_yuan_20", "flow_pct_5", "flow_pct_20"}
            if record_date:
                selected_dates = np.full(len(calendar), np.datetime64("NaT"), dtype="datetime64[ns]")
            if not releases.empty:
                chosen = releases.source.cummax().to_numpy()[np.maximum(observed_at, 0)]
                fresh = (observed_at >= 0) & (source_positions - chosen <= 5)
                output[fresh] = values.to_numpy()[chosen[fresh]]
                if record_date:
                    selected_dates[fresh] = calendar.to_numpy(dtype="datetime64[ns]")[chosen[fresh]]
            out[f"etf_{name}_{metric}"] = output
            if record_date:
                out[f"etf_{name}_{metric}_through"] = selected_dates
                if metric.startswith("flow_yuan_"):
                    counts = np.full(len(calendar), np.nan)
                    if not releases.empty:
                        available = flow.rolling(window, min_periods=1).count().to_numpy()
                        counts[fresh] = available[chosen[fresh]]
                    out[f"etf_{name}_{metric}_days"] = counts
        # A close-entry order can only use a completed prior session's turnover.
        turnover = totals.turnover.where(totals.bars.eq(totals.expected) & totals.expected.gt(0))
        out[f"etf_{name}_turnover_multiple_20"] = (
            turnover / turnover.shift(1).rolling(20, min_periods=18).mean()
        ).shift(1)
        rows["published_by_d2"] = rows.ready_session.between(
            0, calendar.get_indexer(rows.trade_date) + 2,
        )
        out[f"etf_{name}_coverage"] = (
            rows.groupby("trade_date").published_by_d2.mean().reindex(calendar).shift(2)
        )
    return pd.DataFrame(out, index=calendar).rename_axis("date").reset_index(), detailed
