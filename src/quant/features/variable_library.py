"""Project-wide variable library.

This module is intentionally strategy-neutral. B1 is only the first consumer of
these variables; future strategies should import the project variable list and
feature calculators from here instead of copying strategy-local helpers.
"""

from __future__ import annotations

import re
from time import perf_counter

import numpy as np
import pandas as pd

from quant.data.factors import KDJ


PROJECT_FACTOR_COLUMNS = [
    "alpha003",
    "alpha004",
    "alpha005",
    "alpha006",
    "alpha009",
    "alpha191_01",
    "alpha191_02",
    "alpha191_03",
    "alpha191_06",
    "alpha191_07",
    "alpha191_09",
    "alpha191_11",
    "alpha191_12",
    "alpha191_13",
    "alpha191_15",
    "amplitude_1",
    "amplitude_20",
    "atr_14",
    "bb_lower",
    "bb_middle",
    "bb_upper",
    "bias_12",
    "bias_24",
    "bias_6",
    "cci",
    "change",
    "close",
    "downside_volatility_20d",
    "downside_volatility_60d",
    "ema_10",
    "ema_20",
    "ema_5",
    "high",
    "kdj_d_k",
    "kdj_d_d",
    "kdj_d_j",
    "kdj_w_k",
    "kdj_w_d",
    "kdj_w_j",
    "kdj_m_k",
    "kdj_m_d",
    "kdj_m_j",
    "keltner_lower",
    "keltner_upper",
    "keltner_width",
    "low",
    "bbi",
    "bbi_ma60_diff",
    "bbi_ma60_ratio",
    "ma_10",
    "ma_120",
    "ma_20",
    "ma_5",
    "ma_60",
    "macd_dif",
    "macd_dea",
    "macd_hist",
    "mass_index",
    "momentum_20d",
    "momentum_5d",
    "momentum_60d",
    "obv",
    "open",
    "parabolic_sar",
    "pct_chg",
    "pre_close",
    "price_level",
    "price_log",
    "psy_24",
    "return_10d",
    "return_120d",
    "return_1d",
    "return_5d",
    "return_60d",
    "reversal_5d",
    "rsi_12",
    "volume_ma5",
    "volume_ma10",
    "volume_ma20",
    "volume_ma60",
    "volume_ema5",
    "volume_ema10",
    "volume_ema20",
    "volume_relative_5d",
    "volume_relative_20d",
    "volume_relative_60d",
    "volume_change_1d",
    "volume_change_3d",
    "volume_change_5d",
    "volume_zscore_20d",
    "volume_breakout_20d",
    "volume_breakout_60d",
    "volume_price_strength_5d",
    "weekly_ma55",
    "weekly_ma144",
    "weekly_ma233",
    "weekly_ma55_slope",
    "weekly_ma144_slope",
    "weekly_bull_ma55_144",
    "weekly_bull_ma55_144_233",
    "yidong_20d",
    "strong_yidong_20d",
    "days_since_yidong",
    "post_yidong_shrink",
    "ground_volume_60d",
    "b2_confirm_3d",
    "s1_distribution",
    "sell_score_simple",
    "volatility_20d",
    "volatility_60d",
    "vortex_minus",
    "vortex_plus",
    "turnover_rate",
    "turnover_rate_f",
    "ts_volume_ratio",
    "pe",
    "pe_ttm",
    "pb",
    "ps",
    "ps_ttm",
    "dv_ratio",
    "dv_ttm",
    "total_share",
    "float_share",
    "free_share",
    "total_mv",
    "circ_mv",
    "total_mv_log",
    "circ_mv_log",
    "free_share_ratio",
    "float_share_ratio",
    "float_mv_ratio",
    "free_float_share_ratio",
    "turnover_rate_ma5",
    "turnover_rate_ma20",
    "turnover_rate_rel20",
    "turnover_rate_f_ma5",
    "turnover_rate_f_ma20",
    "turnover_rate_f_rel20",
    "ts_volume_ratio_ma5",
    "ts_volume_ratio_ma20",
    "ts_volume_ratio_rel20",
    "total_mv_change_20d",
    "circ_mv_change_20d",
    "pe_ttm_inv",
    "pb_inv",
    "ps_ttm_inv",
]


def build_continuous_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Return OHLC adjusted to a continuous current-price scale for indicators.

    Tushare daily OHLC can contain ex-right price jumps while pct_chg/pre_close
    remain continuous. Rolling technical indicators should use a continuous
    price series; display and trade simulation can still use raw prices.
    """
    out = df.copy()
    if not {"open", "high", "low", "close"} <= set(out.columns):
        return out
    if "pre_close" not in out.columns:
        return out

    if "date" in out.columns:
        order = out.assign(_order_date=pd.to_datetime(out["date"], errors="coerce")).sort_values("_order_date").index
    elif "trade_date" in out.columns:
        trade_date = pd.to_datetime(out["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
        order = out.assign(_order_date=trade_date).sort_values("_order_date").index
    else:
        order = out.index

    sorted_frame = out.loc[order].copy()
    if len(sorted_frame) < 2:
        return out

    close = pd.to_numeric(
        sorted_frame["close"],
        errors="coerce",
    ).to_numpy(dtype=float)
    pre_close = pd.to_numeric(
        sorted_frame["pre_close"],
        errors="coerce",
    ).to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ratios = pre_close[1:] / close[:-1]
    ratios = np.where(
        np.isfinite(ratios) & (ratios > 0),
        ratios,
        1.0,
    )
    factor_values = np.ones(len(sorted_frame), dtype=float)
    factor_values[:-1] = np.cumprod(ratios[::-1])[::-1]
    factor = pd.Series(
        factor_values,
        index=sorted_frame.index,
        dtype=float,
    )

    for col in ["open", "high", "low", "close"]:
        out[col] = pd.to_numeric(out[col], errors="coerce") * factor.reindex(out.index).fillna(1.0)
    return out

EXTRA_FEATURE_COLUMNS = [
    "kdj_d_k",
    "kdj_d_d",
    "kdj_d_j",
    "kdj_w_k",
    "kdj_w_d",
    "kdj_w_j",
    "kdj_m_k",
    "kdj_m_d",
    "kdj_m_j",
    "bbi",
    "bbi_ma60_diff",
    "bbi_ma60_ratio",
    "macd_dif",
    "macd_dea",
    "macd_hist",
    "volume_ma5",
    "volume_ma10",
    "volume_ma20",
    "volume_ma60",
    "volume_ema5",
    "volume_ema10",
    "volume_ema20",
    "volume_relative_5d",
    "volume_relative_20d",
    "volume_relative_60d",
    "volume_change_1d",
    "volume_change_3d",
    "volume_change_5d",
    "volume_zscore_20d",
    "volume_breakout_20d",
    "volume_breakout_60d",
    "volume_price_strength_5d",
    "weekly_ma55",
    "weekly_ma144",
    "weekly_ma233",
    "weekly_ma55_slope",
    "weekly_ma144_slope",
    "weekly_bull_ma55_144",
    "weekly_bull_ma55_144_233",
    "yidong_20d",
    "strong_yidong_20d",
    "days_since_yidong",
    "post_yidong_shrink",
    "ground_volume_60d",
    "b2_confirm_3d",
    "s1_distribution",
    "sell_score_simple",
]

DAILY_BASIC_SOURCE_COLUMNS = [
    "ts_code",
    "trade_date",
    "turnover_rate",
    "turnover_rate_f",
    "volume_ratio",
    "pe",
    "pe_ttm",
    "pb",
    "ps",
    "ps_ttm",
    "dv_ratio",
    "dv_ttm",
    "total_share",
    "float_share",
    "free_share",
    "total_mv",
    "circ_mv",
]


def calc_bbi(close: pd.Series) -> pd.Series:
    """Calculate BBI from close prices.

    BBI is strategy-neutral as a technical variable, though B1 currently uses it
    as part of its candidate signal.
    """
    return (
        close.rolling(3).mean()
        + close.rolling(6).mean()
        + close.rolling(12).mean()
        + close.rolling(24).mean()
    ) / 4


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    base = df[["date", "open", "high", "low", "close", "volume"]].dropna(subset=["date"]).copy()
    base = base.sort_values("date").set_index("date")
    out = base.resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })
    return out.dropna(subset=["open", "high", "low", "close"]).reset_index()


def _align_resampled_kdj(df: pd.DataFrame, rule: str, prefix: str) -> pd.DataFrame:
    bars = _resample_ohlcv(df, rule)
    result = pd.DataFrame(index=df.index)
    for col in [f"{prefix}_k", f"{prefix}_d", f"{prefix}_j"]:
        result[col] = np.nan
    if len(bars) < 10:
        return result

    kdj = KDJ().compute(bars)
    bars = bars[["date"]].copy()
    bars[f"{prefix}_k"] = kdj["K"]
    bars[f"{prefix}_d"] = kdj["D"]
    bars[f"{prefix}_j"] = kdj["J"]
    left = df[["date"]].dropna().sort_values("date")
    if left.empty:
        return result
    aligned = pd.merge_asof(
        left,
        bars.sort_values("date"),
        on="date",
        direction="backward",
    )
    aligned.index = left.index
    return aligned.reindex(df.index)[[f"{prefix}_k", f"{prefix}_d", f"{prefix}_j"]]


def _align_weekly_ma(df: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)
    for col in [
        "weekly_ma55",
        "weekly_ma144",
        "weekly_ma233",
        "weekly_ma55_slope",
        "weekly_ma144_slope",
        "weekly_bull_ma55_144",
        "weekly_bull_ma55_144_233",
    ]:
        result[col] = np.nan

    bars = _resample_ohlcv(df, "W-FRI")
    if bars.empty:
        return result
    for window in [55, 144, 233]:
        bars[f"weekly_ma{window}"] = bars["close"].rolling(window).mean()
    bars["weekly_ma55_slope"] = bars["weekly_ma55"].diff(4)
    bars["weekly_ma144_slope"] = bars["weekly_ma144"].diff(4)
    bars["weekly_bull_ma55_144"] = (
        (bars["weekly_ma55"] > bars["weekly_ma144"])
        & (bars["weekly_ma55_slope"] > 0)
    ).astype(float)
    bars["weekly_bull_ma55_144_233"] = (
        (bars["weekly_ma55"] > bars["weekly_ma144"])
        & (bars["weekly_ma144"] > bars["weekly_ma233"])
        & (bars["weekly_ma55_slope"] > 0)
        & (bars["weekly_ma144_slope"] > 0)
    ).astype(float)
    keep = [
        "date",
        "weekly_ma55",
        "weekly_ma144",
        "weekly_ma233",
        "weekly_ma55_slope",
        "weekly_ma144_slope",
        "weekly_bull_ma55_144",
        "weekly_bull_ma55_144_233",
    ]
    left = df[["date"]].dropna().sort_values("date")
    if left.empty:
        return result
    aligned = pd.merge_asof(
        left,
        bars[keep].sort_values("date"),
        on="date",
        direction="backward",
    )
    aligned.index = left.index
    return aligned.reindex(df.index)[keep[1:]]


def calculate_project_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate reusable project-level variables not covered by legacy factors."""
    out = pd.DataFrame(index=df.index)
    price_df = build_continuous_ohlc(df)
    close = price_df["close"]
    volume = df["volume"]

    kdj_d = KDJ().compute(price_df)
    out["kdj_d_k"] = kdj_d["K"]
    out["kdj_d_d"] = kdj_d["D"]
    out["kdj_d_j"] = kdj_d["J"]
    out = pd.concat([
        out,
        _align_resampled_kdj(price_df, "W-FRI", "kdj_w"),
        _align_resampled_kdj(price_df, "ME", "kdj_m"),
        _align_weekly_ma(price_df),
    ], axis=1)

    bbi = calc_bbi(close)
    ma60 = close.rolling(60).mean()
    out["bbi"] = bbi
    out["bbi_ma60_diff"] = bbi - ma60
    out["bbi_ma60_ratio"] = bbi / ma60.replace(0, np.nan)

    ema12 = close.ewm(span=12, adjust=True).mean()
    ema26 = close.ewm(span=26, adjust=True).mean()
    out["macd_dif"] = ema12 - ema26
    out["macd_dea"] = out["macd_dif"].ewm(span=9, adjust=True).mean()
    out["macd_hist"] = out["macd_dif"] - out["macd_dea"]

    for window in [5, 10, 20, 60]:
        out[f"volume_ma{window}"] = volume.rolling(window).mean()
    for window in [5, 10, 20]:
        out[f"volume_ema{window}"] = volume.ewm(span=window, adjust=True).mean()
    for window in [5, 20, 60]:
        out[f"volume_relative_{window}d"] = volume / volume.rolling(window).mean().replace(0, np.nan)

    out["volume_change_1d"] = volume.pct_change(1)
    out["volume_change_3d"] = volume.pct_change(3)
    out["volume_change_5d"] = volume.pct_change(5)
    vol_mean_20 = volume.rolling(20).mean()
    vol_std_20 = volume.rolling(20).std()
    out["volume_zscore_20d"] = (volume - vol_mean_20) / vol_std_20.replace(0, np.nan)
    out["volume_breakout_20d"] = volume / volume.shift(1).rolling(20).max().replace(0, np.nan)
    out["volume_breakout_60d"] = volume / volume.shift(1).rolling(60).max().replace(0, np.nan)
    out["volume_price_strength_5d"] = close.pct_change(5) * out["volume_relative_20d"]

    yidong = (out["volume_relative_5d"] >= 2.0) & (df["pct_chg"] > 2)
    strong_yidong = (out["volume_relative_5d"] >= 3.0) & (df["pct_chg"] >= 5)
    out["yidong_20d"] = yidong.shift(1).rolling(20, min_periods=1).max().astype(float)
    out["strong_yidong_20d"] = strong_yidong.shift(1).rolling(20, min_periods=1).max().astype(float)
    last_yidong_pos = pd.Series(np.where(yidong, np.arange(len(df)), np.nan), index=df.index).ffill()
    pos = pd.Series(np.arange(len(df)), index=df.index)
    out["days_since_yidong"] = pos - last_yidong_pos
    out.loc[last_yidong_pos.isna(), "days_since_yidong"] = np.nan
    out["post_yidong_shrink"] = (
        (out["yidong_20d"] > 0)
        & (out["volume_relative_20d"] < 0.8)
        & (df["pct_chg"] <= 2)
    ).astype(float)
    vol_rank_60 = volume.rolling(60, min_periods=20).rank(pct=True)
    out["ground_volume_60d"] = (vol_rank_60 <= 0.2).astype(float)

    amplitude = (df["high"] - df["low"]) / df["close"].shift(1).replace(0, np.nan) * 100
    b2_today = (
        (df["pct_chg"] >= 4)
        & (out["volume_relative_5d"] >= 1.5)
        & (df["high"] <= df["close"] * 1.01)
        & (out["kdj_d_j"] < 55)
    )
    out["b2_confirm_3d"] = b2_today.rolling(3, min_periods=1).max().astype(float)

    close_position = (df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)
    recent_high_20 = df["high"].rolling(20, min_periods=10).max()
    recent_low_20 = df["low"].shift(1).rolling(19, min_periods=10).min()
    up_pct_20 = (recent_high_20 - recent_low_20) / recent_low_20.replace(0, np.nan)
    is_yinxian = df["close"] < df["open"]
    is_jiayin = (df["close"] < df["open"]) & (df["close"] > df["close"].shift(1))
    fangliang = volume > volume.shift(1).rolling(5, min_periods=3).mean() * 1.5
    out["s1_distribution"] = (
        (up_pct_20 > 0.15)
        & (df["close"] >= recent_high_20 * 0.90)
        & ((is_yinxian & fangliang) | (is_jiayin & fangliang))
        & (close_position <= 0.30)
    ).astype(float)

    ma5 = close.rolling(5).mean()
    close_up = df["close"] > df["close"].shift(1)
    bbi_ok = df["close"] >= bbi
    not_vol_yinxian = ~(is_yinxian & fangliang)
    trend_up = ma5 > ma5.shift(1)
    j_not_dead = (out["kdj_d_j"] >= out["kdj_d_d"]) | (out["kdj_d_j"] < 80)
    out["sell_score_simple"] = (
        close_up.astype(int)
        + bbi_ok.astype(int)
        + not_vol_yinxian.astype(int)
        + trend_up.astype(int)
        + j_not_dead.astype(int)
    )
    return out


_DAILY_BASIC_DATE_PATTERN = re.compile(r"(?:^|_)(\d{8})$")
_DAILY_BASIC_ROLLING_LOOKBACK = 20
_DAILY_BASIC_INITIAL_HISTORY_FILES = 32


def _daily_basic_file_date(path) -> pd.Timestamp | None:
    match = _DAILY_BASIC_DATE_PATTERN.search(path.stem)
    if match is None:
        return None
    return pd.to_datetime(match.group(1), format="%Y%m%d", errors="coerce")


def _read_daily_basic_file(path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, columns=DAILY_BASIC_SOURCE_COLUMNS)
    except Exception:
        # Preserve compatibility with older cache files whose schemas may not
        # contain every current source column.
        return pd.read_parquet(path)


def _bounded_daily_basic_frames(
    files,
    *,
    target_keys: pd.DataFrame,
    history_rows: int,
) -> tuple[list[pd.DataFrame], int]:
    """Read the smallest exact history needed for target symbol/date keys.

    Rolling daily_basic variables need at most ``history_rows`` prior symbol
    observations. Files are therefore read backwards until every symbol that
    is present on a target date has enough history. This remains exact for
    suspended stocks because a sparse symbol automatically expands the window.
    """

    keys = target_keys[["ts_code", "trade_date"]].dropna().drop_duplicates().copy()
    if keys.empty:
        return [], 0
    keys["ts_code"] = keys["ts_code"].astype(str)
    keys["trade_date"] = keys["trade_date"].astype(str)
    target_symbols = set(keys["ts_code"])
    target_dates = set(keys["trade_date"])
    start = pd.to_datetime(keys["trade_date"].min(), format="%Y%m%d", errors="coerce")
    end = pd.to_datetime(keys["trade_date"].max(), format="%Y%m%d", errors="coerce")
    if pd.isna(start) or pd.isna(end):
        return [_read_daily_basic_file(path) for path in files], len(files)

    dated_files = [(path, _daily_basic_file_date(path)) for path in files]
    known = [(path, file_date) for path, file_date in dated_files if pd.notna(file_date)]
    unknown = [path for path, file_date in dated_files if pd.isna(file_date)]
    if not known or unknown:
        # An unparseable filename may contain any date range. Falling back is
        # slower but preserves the historical loader's correctness contract.
        return [_read_daily_basic_file(path) for path in files], len(files)

    known.sort(key=lambda item: item[1])
    in_range = [(path, file_date) for path, file_date in known if start <= file_date <= end]
    before = [(path, file_date) for path, file_date in known if file_date < start]

    frames: list[pd.DataFrame] = []
    files_read = 0

    def read_selected(selected) -> None:
        nonlocal files_read
        for path, _ in selected:
            frame = _read_daily_basic_file(path)
            files_read += 1
            if "ts_code" in frame.columns:
                frame = frame[frame["ts_code"].astype(str).isin(target_symbols)]
            if not frame.empty:
                frames.append(frame)

    read_selected(in_range)

    target_present: set[str] = set()
    if frames:
        current = pd.concat(frames, ignore_index=True, sort=False)
        if {"ts_code", "trade_date"} <= set(current.columns):
            current_keys = current["trade_date"].astype(str).isin(target_dates)
            target_present = set(current.loc[current_keys, "ts_code"].astype(str))
    if not target_present:
        return frames, files_read

    remaining = list(reversed(before))
    history_counts = {symbol: 0 for symbol in target_present}
    batch_size = max(history_rows, _DAILY_BASIC_INITIAL_HISTORY_FILES)
    while remaining and any(count < history_rows for count in history_counts.values()):
        batch = remaining[:batch_size]
        remaining = remaining[batch_size:]
        previous_len = len(frames)
        read_selected(batch)
        for frame in frames[previous_len:]:
            if "ts_code" not in frame.columns:
                continue
            counts = frame["ts_code"].astype(str).value_counts()
            for symbol in history_counts:
                history_counts[symbol] += int(counts.get(symbol, 0))

    return frames, files_read


def load_daily_basic_features(
    daily_basic_dir,
    *,
    target_keys: pd.DataFrame | None = None,
    history_rows: int = _DAILY_BASIC_ROLLING_LOOKBACK,
) -> pd.DataFrame:
    """Load and derive project variables from Tushare daily_basic parquet files.

    ``target_keys`` enables an exact incremental path: only requested symbols,
    dates, and the prior observations needed by rolling variables are loaded.
    Omitting it preserves the historical full-load behavior used by research
    and model-training jobs.
    """

    files = sorted(daily_basic_dir.glob("*.parquet"))
    if not files:
        return pd.DataFrame()

    started = perf_counter()
    if target_keys is None:
        source_frames = [_read_daily_basic_file(path) for path in files]
        files_read = len(files)
    else:
        source_frames, files_read = _bounded_daily_basic_frames(
            files,
            target_keys=target_keys,
            history_rows=history_rows,
        )

    frames = []
    for df in source_frames:
        present = [col for col in DAILY_BASIC_SOURCE_COLUMNS if col in df.columns]
        if {"ts_code", "trade_date"} <= set(present):
            frames.append(df[present].copy())
    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True).drop_duplicates(["ts_code", "trade_date"], keep="last")
    out = out.rename(columns={"volume_ratio": "ts_volume_ratio"})
    out["trade_date"] = out["trade_date"].astype(str)
    out = out.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    for col in ["total_mv", "circ_mv"]:
        if col in out.columns:
            out[f"{col}_log"] = np.log(out[col].replace(0, np.nan))
    if {"free_share", "total_share"} <= set(out.columns):
        out["free_share_ratio"] = out["free_share"] / out["total_share"].replace(0, np.nan)
    if {"float_share", "total_share"} <= set(out.columns):
        out["float_share_ratio"] = out["float_share"] / out["total_share"].replace(0, np.nan)
    if {"circ_mv", "total_mv"} <= set(out.columns):
        out["float_mv_ratio"] = out["circ_mv"] / out["total_mv"].replace(0, np.nan)
    if {"free_share", "float_share"} <= set(out.columns):
        out["free_float_share_ratio"] = out["free_share"] / out["float_share"].replace(0, np.nan)

    grouped = out.groupby("ts_code", group_keys=False)
    for source_col, prefix in [
        ("turnover_rate", "turnover_rate"),
        ("turnover_rate_f", "turnover_rate_f"),
        ("ts_volume_ratio", "ts_volume_ratio"),
    ]:
        if source_col not in out.columns:
            continue
        out[f"{prefix}_ma5"] = grouped[source_col].transform(lambda s: s.rolling(5, min_periods=3).mean())
        out[f"{prefix}_ma20"] = grouped[source_col].transform(lambda s: s.rolling(20, min_periods=10).mean())
        out[f"{prefix}_rel20"] = out[source_col] / out[f"{prefix}_ma20"].replace(0, np.nan)

    for col in ["total_mv", "circ_mv"]:
        if col in out.columns:
            out[f"{col}_change_20d"] = grouped[col].pct_change(20)
    for source_col, out_col in [("pe_ttm", "pe_ttm_inv"), ("pb", "pb_inv"), ("ps_ttm", "ps_ttm_inv")]:
        if source_col in out.columns:
            out[out_col] = 1 / out[source_col].replace(0, np.nan)
    result = out.replace([np.inf, -np.inf], np.nan)
    print(
        "loaded daily_basic features: "
        f"files={files_read}/{len(files)} rows={len(result)} elapsed={perf_counter() - started:.1f}s",
        flush=True,
    )
    return result


def merge_daily_basic_features(
    data: pd.DataFrame,
    daily_basic_dir,
    *,
    min_match_rate: float | None = None,
) -> pd.DataFrame:
    """Merge project daily_basic variables onto a daily feature frame."""
    out = data.copy()
    if "trade_date" not in out.columns and "date" in out.columns:
        out["trade_date"] = pd.to_datetime(out["date"]).dt.strftime("%Y%m%d")
    if "trade_date" in out.columns:
        out["trade_date"] = out["trade_date"].astype(str)
    if "ts_code" not in out.columns and "symbol" in out.columns:
        out["ts_code"] = out["symbol"].astype(str)

    target_keys = (
        out[["ts_code", "trade_date"]]
        if {"ts_code", "trade_date"} <= set(out.columns)
        else None
    )
    daily_basic = load_daily_basic_features(daily_basic_dir, target_keys=target_keys)
    if daily_basic.empty:
        print(f"daily_basic not found under {daily_basic_dir}; using daily-only variables", flush=True)
        if min_match_rate is not None and len(out):
            raise RuntimeError(
                "daily_basic feature coverage below required threshold: "
                f"matched=0.00% required={min_match_rate:.2%}"
            )
        return data

    existing_daily_basic_cols = [
        col for col in daily_basic.columns
        if col not in {"ts_code", "trade_date"} and col in out.columns
    ]
    if existing_daily_basic_cols:
        out = out.drop(columns=existing_daily_basic_cols)

    merged = out.merge(daily_basic, on=["ts_code", "trade_date"], how="left")
    matched = merged["turnover_rate"].notna().mean() if "turnover_rate" in merged.columns else 0.0
    print(f"merged daily_basic features: rows={len(daily_basic)} matched_rate={matched:.2%}", flush=True)
    if min_match_rate is not None and len(merged) and matched < min_match_rate:
        raise RuntimeError(
            "daily_basic feature coverage below required threshold: "
            f"matched={matched:.2%} required={min_match_rate:.2%}"
        )
    return merged.replace([np.inf, -np.inf], np.nan)
