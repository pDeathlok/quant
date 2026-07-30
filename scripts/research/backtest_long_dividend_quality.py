"""Backtest the L1 long-term dividend quality strategy.

This research script intentionally lives outside the production selector path.
It uses local Tushare-first parquet data, detects missing daily_basic coverage,
and writes a compact report under reports/long_dividend_quality/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from quant.data import MarketDataStore, MarketDataStoreConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DAILY_DIR = PROJECT_ROOT / "data/raw/daily"
DAILY_BASIC_DIR = PROJECT_ROOT / "data/raw/daily_basic"
DAILY_BASIC_CACHE_DIR = PROJECT_ROOT / "data/cache/source_merge/tushare"
STOCK_BASIC_PATH = PROJECT_ROOT / "data/raw/stock_basic.parquet"
FINA_INDICATOR_PATH = PROJECT_ROOT / "data/raw/fina_indicator.parquet"
ANALYST_FORECAST_PATH = PROJECT_ROOT / "data/raw/analyst_forecasts.parquet"
INDEX_300_PATH = PROJECT_ROOT / "data/raw/index_000300.SH.parquet"
REPORT_DIR = PROJECT_ROOT / "reports/long_dividend_quality"
RESEARCH_CACHE_DIR = PROJECT_ROOT / "data/research/long_dividend_quality"

MARKET_REGIME_VARIANTS = {
    "v4_market_regime",
    "v5_industry_cap",
    "v6_t_overlay",
    "v7_t_overlay_light",
    "v8_growth_sleeve",
    "v9_growth_sleeve_capped",
    "v10_bluechip_growth_sleeve",
    "v11_analyst_growth_sleeve",
    "v12_quality_growth_sleeve",
    "v13_mega_quality_growth_sleeve",
    "v14_selective_mega_growth_sleeve",
    "v15_analyst_quality_rank_sleeve",
    "v16_mega_rank_growth_sleeve",
    "v17_staged_entry_sleeve",
    "v18_style_grid_overlay_sleeve",
    "v19_style_grid_full_entry_sleeve",
    "v20_fast_staged_grid_sleeve",
    "v21_forecast_dual_score_sleeve",
    "v22_forecast_dual_score_grid_sleeve",
    "v23_forecast_rank_grid_sleeve",
    "v24_forecast_tiebreak_grid_sleeve",
    "v25_forecast_guardrail_grid_sleeve",
    "v26_market_trend_compounder_sleeve",
    "v27_cautious_compounder_sleeve",
    "v28_overheat_guarded_compounder_sleeve",
    "v29_overheat_throttle_grid_sleeve",
    "v30_concentrated_trend_sleeve",
    "v31_bull_bear_exposure_sleeve",
    "v32_empty_bear_sleeve",
    "v33_bull_boost_defensive_bear_sleeve",
    "v34_pit_universe_guarded_sleeve",
    "v35_pit_universe_riskon_recovery_sleeve",
}
GROWTH_VARIANTS = {
    "v8_growth_sleeve",
    "v9_growth_sleeve_capped",
    "v10_bluechip_growth_sleeve",
    "v11_analyst_growth_sleeve",
    "v12_quality_growth_sleeve",
    "v13_mega_quality_growth_sleeve",
    "v14_selective_mega_growth_sleeve",
    "v15_analyst_quality_rank_sleeve",
    "v16_mega_rank_growth_sleeve",
    "v17_staged_entry_sleeve",
    "v18_style_grid_overlay_sleeve",
    "v19_style_grid_full_entry_sleeve",
    "v20_fast_staged_grid_sleeve",
    "v21_forecast_dual_score_sleeve",
    "v22_forecast_dual_score_grid_sleeve",
    "v23_forecast_rank_grid_sleeve",
    "v24_forecast_tiebreak_grid_sleeve",
    "v25_forecast_guardrail_grid_sleeve",
    "v26_market_trend_compounder_sleeve",
    "v27_cautious_compounder_sleeve",
    "v28_overheat_guarded_compounder_sleeve",
    "v29_overheat_throttle_grid_sleeve",
    "v30_concentrated_trend_sleeve",
    "v31_bull_bear_exposure_sleeve",
    "v32_empty_bear_sleeve",
    "v33_bull_boost_defensive_bear_sleeve",
    "v34_pit_universe_guarded_sleeve",
    "v35_pit_universe_riskon_recovery_sleeve",
}
CAPPED_GROWTH_VARIANTS = {
    "v9_growth_sleeve_capped",
    "v10_bluechip_growth_sleeve",
    "v11_analyst_growth_sleeve",
    "v12_quality_growth_sleeve",
    "v13_mega_quality_growth_sleeve",
    "v14_selective_mega_growth_sleeve",
    "v15_analyst_quality_rank_sleeve",
    "v16_mega_rank_growth_sleeve",
    "v17_staged_entry_sleeve",
    "v18_style_grid_overlay_sleeve",
    "v19_style_grid_full_entry_sleeve",
    "v20_fast_staged_grid_sleeve",
    "v21_forecast_dual_score_sleeve",
    "v22_forecast_dual_score_grid_sleeve",
    "v23_forecast_rank_grid_sleeve",
    "v24_forecast_tiebreak_grid_sleeve",
    "v25_forecast_guardrail_grid_sleeve",
    "v26_market_trend_compounder_sleeve",
    "v27_cautious_compounder_sleeve",
    "v28_overheat_guarded_compounder_sleeve",
    "v29_overheat_throttle_grid_sleeve",
    "v30_concentrated_trend_sleeve",
    "v31_bull_bear_exposure_sleeve",
    "v32_empty_bear_sleeve",
    "v33_bull_boost_defensive_bear_sleeve",
    "v34_pit_universe_guarded_sleeve",
    "v35_pit_universe_riskon_recovery_sleeve",
}
ANALYST_QUALITY_RANK_VARIANTS = {
    "v15_analyst_quality_rank_sleeve",
    "v16_mega_rank_growth_sleeve",
    "v17_staged_entry_sleeve",
    "v18_style_grid_overlay_sleeve",
    "v19_style_grid_full_entry_sleeve",
    "v20_fast_staged_grid_sleeve",
    "v21_forecast_dual_score_sleeve",
    "v22_forecast_dual_score_grid_sleeve",
    "v23_forecast_rank_grid_sleeve",
    "v24_forecast_tiebreak_grid_sleeve",
    "v25_forecast_guardrail_grid_sleeve",
    "v26_market_trend_compounder_sleeve",
    "v27_cautious_compounder_sleeve",
    "v28_overheat_guarded_compounder_sleeve",
    "v29_overheat_throttle_grid_sleeve",
    "v30_concentrated_trend_sleeve",
    "v31_bull_bear_exposure_sleeve",
    "v32_empty_bear_sleeve",
    "v33_bull_boost_defensive_bear_sleeve",
    "v34_pit_universe_guarded_sleeve",
    "v35_pit_universe_riskon_recovery_sleeve",
}
FORECAST_DUAL_SCORE_VARIANTS = {
    "v21_forecast_dual_score_sleeve",
    "v22_forecast_dual_score_grid_sleeve",
}
FORECAST_RANK_VARIANTS = {
    "v23_forecast_rank_grid_sleeve",
}
FORECAST_LIGHT_RANK_VARIANTS = {
    "v24_forecast_tiebreak_grid_sleeve",
}
FORECAST_GUARDRAIL_VARIANTS = {
    "v25_forecast_guardrail_grid_sleeve",
    "v26_market_trend_compounder_sleeve",
    "v27_cautious_compounder_sleeve",
    "v28_overheat_guarded_compounder_sleeve",
}
COMPOUNDER_VARIANTS = {
    "v26_market_trend_compounder_sleeve",
    "v27_cautious_compounder_sleeve",
    "v28_overheat_guarded_compounder_sleeve",
    "v29_overheat_throttle_grid_sleeve",
    "v30_concentrated_trend_sleeve",
    "v31_bull_bear_exposure_sleeve",
    "v32_empty_bear_sleeve",
    "v33_bull_boost_defensive_bear_sleeve",
    "v34_pit_universe_guarded_sleeve",
    "v35_pit_universe_riskon_recovery_sleeve",
}
MARKET_TIMING_VARIANTS = {
    "v30_concentrated_trend_sleeve",
    "v31_bull_bear_exposure_sleeve",
    "v32_empty_bear_sleeve",
    "v33_bull_boost_defensive_bear_sleeve",
    "v34_pit_universe_guarded_sleeve",
    "v35_pit_universe_riskon_recovery_sleeve",
}
STAGED_ENTRY_VARIANTS = {
    "v17_staged_entry_sleeve",
    "v18_style_grid_overlay_sleeve",
    "v20_fast_staged_grid_sleeve",
}
STYLE_GRID_VARIANTS = {
    "v18_style_grid_overlay_sleeve",
    "v19_style_grid_full_entry_sleeve",
    "v20_fast_staged_grid_sleeve",
    "v22_forecast_dual_score_grid_sleeve",
    "v23_forecast_rank_grid_sleeve",
    "v24_forecast_tiebreak_grid_sleeve",
    "v25_forecast_guardrail_grid_sleeve",
    "v26_market_trend_compounder_sleeve",
    "v27_cautious_compounder_sleeve",
    "v28_overheat_guarded_compounder_sleeve",
    "v29_overheat_throttle_grid_sleeve",
    "v30_concentrated_trend_sleeve",
    "v31_bull_bear_exposure_sleeve",
    "v32_empty_bear_sleeve",
    "v33_bull_boost_defensive_bear_sleeve",
    "v34_pit_universe_guarded_sleeve",
    "v35_pit_universe_riskon_recovery_sleeve",
}
PIT_UNIVERSE_VARIANTS = {
    "v34_pit_universe_guarded_sleeve",
    "v35_pit_universe_riskon_recovery_sleeve",
}
TEA_MASTER_VARIANTS = {
    "v36_tea_master_concentrated",
    "v37_tea_master_core_satellite",
    "v38_tea_master_balanced",
    "v39_tea_master_optimized",
}
TEA_MASTER_RECOMMENDED_VARIANTS = {
    "v40_tea_master_regime_grid",
}
TEA_MASTER_GUARDED_VARIANTS = {
    "v41_tea_master_pit_guarded",
    "v42_tea_master_growth_throttle",
    "v43_tea_master_core_only",
    "v44_tea_master_defensive_neutral",
    "v45_tea_master_riskon_quality",
}
MARKET_REGIME_VARIANTS |= TEA_MASTER_VARIANTS
MARKET_REGIME_VARIANTS |= TEA_MASTER_RECOMMENDED_VARIANTS
MARKET_REGIME_VARIANTS |= TEA_MASTER_GUARDED_VARIANTS
GROWTH_VARIANTS |= TEA_MASTER_VARIANTS
GROWTH_VARIANTS |= TEA_MASTER_RECOMMENDED_VARIANTS
GROWTH_VARIANTS |= TEA_MASTER_GUARDED_VARIANTS
CAPPED_GROWTH_VARIANTS |= TEA_MASTER_VARIANTS
CAPPED_GROWTH_VARIANTS |= TEA_MASTER_RECOMMENDED_VARIANTS
CAPPED_GROWTH_VARIANTS |= TEA_MASTER_GUARDED_VARIANTS
ANALYST_QUALITY_RANK_VARIANTS |= TEA_MASTER_VARIANTS
ANALYST_QUALITY_RANK_VARIANTS |= TEA_MASTER_RECOMMENDED_VARIANTS
ANALYST_QUALITY_RANK_VARIANTS |= TEA_MASTER_GUARDED_VARIANTS
FORECAST_GUARDRAIL_VARIANTS |= TEA_MASTER_VARIANTS
FORECAST_GUARDRAIL_VARIANTS |= TEA_MASTER_RECOMMENDED_VARIANTS
COMPOUNDER_VARIANTS |= TEA_MASTER_VARIANTS
COMPOUNDER_VARIANTS |= TEA_MASTER_RECOMMENDED_VARIANTS
COMPOUNDER_VARIANTS |= TEA_MASTER_GUARDED_VARIANTS
MARKET_TIMING_VARIANTS |= TEA_MASTER_VARIANTS
MARKET_TIMING_VARIANTS |= TEA_MASTER_RECOMMENDED_VARIANTS
MARKET_TIMING_VARIANTS |= TEA_MASTER_GUARDED_VARIANTS
STYLE_GRID_VARIANTS |= TEA_MASTER_VARIANTS
STYLE_GRID_VARIANTS |= TEA_MASTER_RECOMMENDED_VARIANTS
STYLE_GRID_VARIANTS |= TEA_MASTER_GUARDED_VARIANTS
PIT_UNIVERSE_VARIANTS |= TEA_MASTER_VARIANTS
PIT_UNIVERSE_VARIANTS |= TEA_MASTER_RECOMMENDED_VARIANTS
PIT_UNIVERSE_VARIANTS |= TEA_MASTER_GUARDED_VARIANTS


@dataclass(frozen=True)
class BacktestConfig:
    variant: str = "baseline"
    start: str = "20130101"
    end: str | None = None
    max_positions: int = 15
    min_positions: int = 8
    target_total_weight: float = 0.75
    max_symbol_weight: float = 0.10
    commission_rate: float = 0.0003
    stamp_tax_rate: float = 0.0005
    min_dv_ttm_watch: float = 2.5
    min_long_score_watch: float = 75.0
    min_turnover_rate_ma20: float = 0.3
    max_turnover_rate_ma20: float = 8.0
    min_total_mv: float = 800000.0
    min_circ_mv: float = 500000.0
    max_volatility_quantile: float = 0.80
    listing_years: int = 2
    prefilter_min_dv_ttm: float = 2.5
    prefilter_min_total_mv: float = 800000.0
    prefilter_min_circ_mv: float = 500000.0


def parse_date(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    return pd.to_datetime(value, format="%Y%m%d")


def percentile_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    rank = values.rank(pct=True)
    if not higher_is_better:
        rank = 1.0 - rank
    return (rank * 100).fillna(0.0)


def normalize_ts_code(path: Path) -> str:
    return path.stem


def load_stock_basic() -> pd.DataFrame:
    if not STOCK_BASIC_PATH.exists():
        return pd.DataFrame(columns=["ts_code", "name", "industry", "list_date"])
    frame = pd.read_parquet(STOCK_BASIC_PATH)
    frame["list_date"] = pd.to_datetime(frame["list_date"].astype(str), format="%Y%m%d", errors="coerce")
    return frame[["ts_code", "name", "industry", "list_date"]].copy()


def load_daily_monthly_features(
    start: pd.Timestamp,
    end: pd.Timestamp | None,
    stock_basic: pd.DataFrame,
    candidate_symbols: set[str] | None = None,
    *,
    use_cache: bool = True,
    include_daily_returns: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_key_source = "|".join(
        [
            start.strftime("%Y%m%d"),
            end.strftime("%Y%m%d") if end is not None else "none",
            "qfq_ohlc_price_v1",
            ",".join(sorted(candidate_symbols or [])),
        ]
    )
    cache_key = hashlib.sha1(cache_key_source.encode("utf-8")).hexdigest()[:16]
    feature_cache = RESEARCH_CACHE_DIR / f"daily_monthly_features_{cache_key}.parquet"
    returns_cache = RESEARCH_CACHE_DIR / f"daily_returns_{cache_key}.parquet"
    cache_ready = feature_cache.exists() and (
        returns_cache.exists() or not include_daily_returns
    )
    if use_cache and cache_ready:
        print(f"loading cached daily features: {feature_cache.name}", flush=True)
        cached_returns = (
            pd.read_parquet(returns_cache)
            if include_daily_returns
            else pd.DataFrame()
        )
        return pd.read_parquet(feature_cache), cached_returns

    history_start = start - pd.Timedelta(days=450)
    frames: list[pd.DataFrame] = []
    returns: list[pd.DataFrame] = []
    stock_meta = stock_basic.set_index("ts_code") if not stock_basic.empty else pd.DataFrame()

    store = MarketDataStore(MarketDataStoreConfig.from_env(root=DAILY_DIR.parent))
    market = store.read_market_range(
        DAILY_DIR.name,
        start_date=history_start.strftime("%Y%m%d"),
        end_date=end.strftime("%Y%m%d") if end is not None else None,
        symbols=candidate_symbols,
        columns=["ts_code", "trade_date", "open", "high", "low", "close", "pct_chg"],
    )
    if market.empty:
        legacy_frames = []
        for path in sorted(DAILY_DIR.glob("*.parquet")):
            frame = pd.read_parquet(path)
            if "ts_code" not in frame.columns or frame["ts_code"].isna().all():
                frame["ts_code"] = path.stem
            legacy_frames.append(frame)
        market = pd.concat(legacy_frames, ignore_index=True, sort=False) if legacy_frames else pd.DataFrame()
    processed = 0
    grouped = market.groupby("ts_code", sort=True) if not market.empty else []
    for _, (ts_code, source_frame) in enumerate(grouped, start=1):
        if candidate_symbols is not None and ts_code not in candidate_symbols:
            continue
        processed += 1
        try:
            df = source_frame.copy()
        except Exception:
            continue
        if df.empty:
            continue
        if "ts_code" not in df.columns or df["ts_code"].isna().all():
            df["ts_code"] = ts_code
        df["ts_code"] = df["ts_code"].fillna(ts_code)
        df["date"] = pd.to_datetime(df["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
        df = df.dropna(subset=["date", "close"]).sort_values("date")
        df = df[df["date"] >= history_start]
        if end is not None:
            df = df[df["date"] <= end]
        if len(df) < 130:
            continue

        close = pd.to_numeric(df["close"], errors="coerce")
        ret = close.pct_change()
        ret = ret.fillna(pd.to_numeric(df["pct_chg"], errors="coerce") / 100.0)
        df["close"] = close
        df["ret_1d"] = ret
        df["ma_20"] = close.rolling(20).mean()
        df["ma_60"] = close.rolling(60).mean()
        df["ma_120"] = close.rolling(120).mean()
        df["median_close_60"] = close.rolling(60).median()
        df["return_120d"] = close.pct_change(120)
        df["volatility_60d"] = ret.rolling(60).std() * np.sqrt(252)
        downside = ret.where(ret < 0, 0.0)
        df["downside_volatility_60d"] = downside.rolling(60).std() * np.sqrt(252)
        df["ma_120_slope_20d"] = df["ma_120"] / df["ma_120"].shift(20) - 1

        if include_daily_returns:
            daily_return = df.loc[
                df["date"] >= start,
                ["date", "trade_date", "ts_code", "ret_1d", "close", "ma_20", "ma_60", "ma_120"],
            ].copy()
            returns.append(daily_return)

        monthly_idx = df.groupby(df["date"].dt.to_period("M"))["date"].idxmax()
        monthly = df.loc[
            monthly_idx,
            [
                "date",
                "trade_date",
                "ts_code",
                "close",
                "ma_20",
                "ma_60",
                "ma_120",
                "median_close_60",
                "return_120d",
                "volatility_60d",
                "downside_volatility_60d",
                "ma_120_slope_20d",
            ],
        ].copy()
        monthly = monthly[monthly["date"] >= start]
        if stock_meta is not None and not stock_meta.empty and ts_code in stock_meta.index:
            meta = stock_meta.loc[ts_code]
            monthly["name"] = meta.get("name")
            monthly["industry"] = meta.get("industry")
            monthly["list_date"] = meta.get("list_date")
        frames.append(monthly)

        if processed % 100 == 0:
            print(f"processed candidate daily files: {processed} usable_symbols={len(frames)}", flush=True)

    if not frames:
        raise RuntimeError(f"No usable daily data found under {DAILY_DIR}")
    features = pd.concat(frames, ignore_index=True)
    daily_returns = pd.concat(returns, ignore_index=True) if returns else pd.DataFrame()
    if use_cache:
        RESEARCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        features.to_parquet(feature_cache, index=False)
        if include_daily_returns:
            daily_returns.to_parquet(returns_cache, index=False)
    return features, daily_returns


def select_candidate_symbols_from_daily_basic(
    daily_basic: pd.DataFrame,
    stock_basic: pd.DataFrame,
    config: BacktestConfig,
) -> set[str]:
    if daily_basic.empty:
        return set()
    frame = daily_basic.copy()
    min_dv_ttm = 0.0 if config.variant in GROWTH_VARIANTS else (
        0.5 if config.variant in MARKET_REGIME_VARIANTS else config.prefilter_min_dv_ttm
    )
    frame = frame[
        (pd.to_numeric(frame["dv_ttm"], errors="coerce").fillna(0) >= min_dv_ttm)
        & (pd.to_numeric(frame["total_mv"], errors="coerce") >= config.prefilter_min_total_mv)
        & (pd.to_numeric(frame["circ_mv"], errors="coerce") >= config.prefilter_min_circ_mv)
    ]
    symbols = set(frame["ts_code"].dropna().astype(str).unique())
    if not stock_basic.empty:
        names = stock_basic.set_index("ts_code")["name"].astype(str)
        st_symbols = set(names[names.str.contains("ST|退", regex=True, na=False)].index)
        symbols -= st_symbols
    return symbols


def filter_daily_basic_point_in_time(daily_basic: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    """Keep only stocks that satisfy universe constraints on each signal date."""
    if daily_basic.empty:
        return daily_basic
    min_dv_ttm = 0.0 if config.variant in GROWTH_VARIANTS else (
        0.5 if config.variant in MARKET_REGIME_VARIANTS else config.prefilter_min_dv_ttm
    )
    frame = daily_basic.copy()
    mask = (
        (pd.to_numeric(frame["dv_ttm"], errors="coerce").fillna(0.0) >= min_dv_ttm)
        & (pd.to_numeric(frame["total_mv"], errors="coerce") >= config.prefilter_min_total_mv)
        & (pd.to_numeric(frame["circ_mv"], errors="coerce") >= config.prefilter_min_circ_mv)
    )
    return frame[mask].copy()


def load_daily_basic_monthly(start: pd.Timestamp, end: pd.Timestamp | None) -> tuple[pd.DataFrame, dict]:
    source_dir = DAILY_BASIC_DIR if any(DAILY_BASIC_DIR.glob("*.parquet")) else DAILY_BASIC_CACHE_DIR
    files = sorted(source_dir.glob("*.parquet"))
    if source_dir == DAILY_BASIC_CACHE_DIR:
        files = sorted(source_dir.glob("tushare_daily_basic_*.parquet"))
    if not files:
        return pd.DataFrame(), {"source_dir": str(source_dir), "files": 0}

    frames: list[pd.DataFrame] = []
    dates: list[str] = []
    for path in files:
        date_text = path.stem.replace("tushare_daily_basic_", "")
        if len(date_text) != 8:
            date_text = path.stem
        date = parse_date(date_text)
        if date is None or date < start or (end is not None and date > end):
            continue
        try:
            df = pd.read_parquet(path)
        except Exception:
            continue
        if df.empty or "ts_code" not in df.columns:
            continue
        if "trade_date" not in df.columns:
            df["trade_date"] = date_text
        frames.append(
            df[
                [
                    "ts_code",
                    "trade_date",
                    "turnover_rate",
                    "turnover_rate_f",
                    "pe_ttm",
                    "pb",
                    "ps_ttm",
                    "dv_ratio",
                    "dv_ttm",
                    "total_mv",
                    "circ_mv",
                ]
            ].copy()
        )
        dates.append(date_text)

    if not frames:
        return pd.DataFrame(), {"source_dir": str(source_dir), "files": len(files)}

    basic = pd.concat(frames, ignore_index=True)
    basic["date"] = pd.to_datetime(basic["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    monthly_dates = basic.groupby(basic["date"].dt.to_period("M"))["date"].max().rename("date").reset_index(drop=True)
    monthly = basic[basic["date"].isin(set(monthly_dates))].copy()
    monthly = monthly.sort_values(["ts_code", "date"]).reset_index(drop=True)
    dv = pd.to_numeric(monthly["dv_ttm"], errors="coerce")
    monthly["dv_ttm_mean_36m"] = dv.groupby(monthly["ts_code"]).transform(lambda s: s.rolling(36, min_periods=12).mean())
    monthly["dv_ttm_std_36m"] = dv.groupby(monthly["ts_code"]).transform(lambda s: s.rolling(36, min_periods=12).std())
    monthly["dv_ttm_stability_36m"] = (
        monthly["dv_ttm_mean_36m"] / monthly["dv_ttm_std_36m"].replace(0, np.nan)
    ).clip(0, 10)
    coverage = {
        "source_dir": str(source_dir),
        "files": len(files),
        "loaded_trade_dates": len(set(dates)),
        "first_trade_date": min(dates) if dates else None,
        "last_trade_date": max(dates) if dates else None,
        "monthly_rebalance_dates": int(monthly["date"].nunique()),
    }
    return monthly, coverage


def load_market_regime(start: pd.Timestamp, end: pd.Timestamp | None) -> pd.DataFrame:
    if not INDEX_300_PATH.exists():
        return pd.DataFrame()
    index = pd.read_parquet(INDEX_300_PATH)
    index["date"] = pd.to_datetime(index["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    index = index.dropna(subset=["date", "close"]).sort_values("date")
    close = pd.to_numeric(index["close"], errors="coerce")
    index["index_ma_60"] = close.rolling(60).mean()
    index["index_ma_120"] = close.rolling(120).mean()
    index["index_ma_120_slope_20d"] = index["index_ma_120"] / index["index_ma_120"].shift(20) - 1
    index["index_return_20d"] = close.pct_change(20)
    index["index_return_60d"] = close.pct_change(60)
    index["index_return_120d"] = close.pct_change(120)
    index["index_drawdown_60d"] = close / close.rolling(60).max() - 1
    index["index_overheat"] = (index["index_return_60d"] > 0.22) & (index["index_return_120d"] > 0.35)
    index["market_regime"] = "neutral"
    risk_on = (close > index["index_ma_120"]) & (index["index_ma_120_slope_20d"] > 0) & (index["index_return_60d"] > 0)
    risk_off = (close < index["index_ma_120"]) & (index["index_ma_120_slope_20d"] < 0)
    index.loc[risk_on, "market_regime"] = "risk_on"
    index.loc[risk_off, "market_regime"] = "risk_off"
    if end is not None:
        index = index[index["date"] <= end]
    index = index[index["date"] >= start]
    monthly_idx = index.groupby(index["date"].dt.to_period("M"))["date"].idxmax()
    return index.loc[
        monthly_idx,
        [
            "date",
            "market_regime",
            "index_ma_120_slope_20d",
            "index_return_20d",
            "index_return_60d",
            "index_return_120d",
            "index_drawdown_60d",
            "index_overheat",
        ],
    ].copy()


def load_financial_asof(features: pd.DataFrame) -> pd.DataFrame:
    if not FINA_INDICATOR_PATH.exists():
        features["roe"] = np.nan
        features["debt_to_assets"] = np.nan
        features["netprofit_margin"] = np.nan
        features["or_yoy"] = np.nan
        features["basic_eps_yoy"] = np.nan
        return features

    fina = pd.read_parquet(FINA_INDICATOR_PATH)
    if fina.empty:
        return features
    cols = ["ts_code", "ann_date", "roe", "debt_to_assets", "netprofit_margin", "or_yoy", "basic_eps_yoy"]
    fina = fina[[col for col in cols if col in fina.columns]].copy()
    fina["ann_date"] = pd.to_datetime(fina["ann_date"].astype(str), format="%Y%m%d", errors="coerce")
    fina = fina.dropna(subset=["ann_date"]).sort_values(["ann_date", "ts_code"])

    out = features.sort_values(["date", "ts_code"]).copy()
    merged_parts: list[pd.DataFrame] = []
    for ts_code, group in out.groupby("ts_code", sort=False):
        finance = fina[fina["ts_code"] == ts_code].sort_values("ann_date")
        if finance.empty:
            group = group.copy()
            group["roe"] = np.nan
            group["debt_to_assets"] = np.nan
            group["netprofit_margin"] = np.nan
            group["or_yoy"] = np.nan
            group["basic_eps_yoy"] = np.nan
            merged_parts.append(group)
            continue
        merged = pd.merge_asof(
            group.sort_values("date"),
            finance.drop(columns=["ts_code"]).sort_values("ann_date"),
            left_on="date",
            right_on="ann_date",
            direction="backward",
        )
        merged["ts_code"] = ts_code
        merged_parts.append(merged)
    return pd.concat(merged_parts, ignore_index=True)


def add_empty_analyst_forecast_columns(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    defaults = {
        "analyst_report_count_180d": 0.0,
        "analyst_org_count_180d": 0.0,
        "analyst_institution_count_180d": 0.0,
        "analyst_research_report_count_180d": 0.0,
        "analyst_consensus_report_count_180d": 0.0,
        "analyst_eps_mean_180d": np.nan,
        "analyst_pe_mean_180d": np.nan,
        "analyst_target_price_mean_180d": np.nan,
        "analyst_net_profit_mean_180d": np.nan,
        "analyst_revenue_mean_180d": np.nan,
        "analyst_eps_revision_180d": np.nan,
        "analyst_target_upside_180d": np.nan,
        "analyst_forward_years_180d": 0.0,
        "analyst_forward_eps_growth_180d": np.nan,
        "analyst_forward_revenue_growth_180d": np.nan,
        "analyst_forward_net_profit_growth_180d": np.nan,
        "analyst_forward_pe_180d": np.nan,
    }
    for column, value in defaults.items():
        if column not in out.columns:
            out[column] = value
    return out


def load_raw_analyst_reports() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if ANALYST_FORECAST_PATH.exists():
        forecast = pd.read_parquet(ANALYST_FORECAST_PATH)
        if not forecast.empty and "ts_code" in forecast.columns and "report_date" in forecast.columns:
            frames.append(forecast)
    if not frames:
        return pd.DataFrame()

    reports = pd.concat(frames, ignore_index=True, sort=False)
    keep_columns = [
        "source",
        "ts_code",
        "report_date",
        "report_title",
        "org_name",
        "author_name",
        "quarter",
        "forecast_year",
        "eps",
        "pe",
        "target_price",
        "net_profit",
        "revenue",
        "report_count",
        "snapshot_only",
    ]
    for column in keep_columns:
        if column not in reports.columns:
            reports[column] = np.nan
    reports = reports[keep_columns].copy()
    reports["report_date"] = pd.to_datetime(reports["report_date"], errors="coerce")
    reports = reports.dropna(subset=["report_date", "ts_code"])
    reports = reports.drop_duplicates(
        ["source", "ts_code", "report_date", "org_name", "author_name", "forecast_year"],
        keep="last",
    )
    return reports


def load_analyst_forecast_asof(features: pd.DataFrame) -> pd.DataFrame:
    """Merge analyst forecasts without using reports published after signal date."""
    reports = load_raw_analyst_reports()
    if reports.empty:
        return add_empty_analyst_forecast_columns(features)

    reports = reports.copy()
    reports["report_date"] = pd.to_datetime(reports["report_date"], errors="coerce")
    reports = reports.dropna(subset=["report_date"])
    min_signal_date = features["date"].min()
    max_signal_date = features["date"].max()
    reports = reports[
        (reports["report_date"] <= max_signal_date)
        & (reports["report_date"] >= min_signal_date - pd.Timedelta(days=180))
    ].copy()
    if reports.empty:
        return add_empty_analyst_forecast_columns(features)
    for column in ["eps", "pe", "target_price", "net_profit", "revenue"]:
        if column in reports.columns:
            reports[column] = pd.to_numeric(reports[column], errors="coerce")
        else:
            reports[column] = np.nan
    reports["forecast_year"] = pd.to_numeric(reports["forecast_year"], errors="coerce")
    if "org_name" not in reports.columns:
        reports["org_name"] = np.nan
    reports = reports.sort_values(["ts_code", "report_date"])
    reports_by_symbol = {ts_code: group for ts_code, group in reports.groupby("ts_code", sort=False)}

    def window_mean(valid_cumsum: np.ndarray, value_cumsum: np.ndarray, start: int, end: int) -> float:
        count = valid_cumsum[end] - valid_cumsum[start]
        if count <= 0:
            return np.nan
        return float((value_cumsum[end] - value_cumsum[start]) / count)

    def build_numeric_cumsums(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        valid = np.isfinite(values)
        valid_cumsum = np.concatenate([[0], np.cumsum(valid.astype(float))])
        value_cumsum = np.concatenate([[0.0], np.cumsum(np.where(valid, values, 0.0))])
        return valid_cumsum, value_cumsum

    rows: list[dict] = []
    for ts_code, group in features.groupby("ts_code", sort=False):
        symbol_reports = reports_by_symbol.get(ts_code)
        if symbol_reports is None or symbol_reports.empty:
            for date in group["date"]:
                rows.append({"ts_code": ts_code, "date": date})
            continue

        symbol_reports = symbol_reports.sort_values("report_date")
        report_dates = symbol_reports["report_date"].to_numpy(dtype="datetime64[ns]")
        ord_dates = report_dates.astype("datetime64[ns]").astype("int64")
        org_values = symbol_reports["org_name"].to_numpy()
        metric_cumsums: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for metric in ["eps", "pe", "target_price", "net_profit", "revenue"]:
            values = pd.to_numeric(symbol_reports[metric], errors="coerce").to_numpy(dtype=float)
            valid_cumsum, value_cumsum = build_numeric_cumsums(values)
            metric_cumsums[metric] = (values, valid_cumsum, value_cumsum)

        for _, row in group[["date", "close"]].iterrows():
            date = row["date"]
            close = float(row["close"]) if pd.notna(row["close"]) else np.nan
            window_start = date - pd.Timedelta(days=180)
            recent_start = date - pd.Timedelta(days=90)
            prior_start = date - pd.Timedelta(days=180)
            date_ord = np.datetime64(date, "ns").astype("int64")
            window_start_ord = np.datetime64(window_start, "ns").astype("int64")
            recent_start_ord = np.datetime64(recent_start, "ns").astype("int64")
            prior_start_ord = np.datetime64(prior_start, "ns").astype("int64")
            hi = int(np.searchsorted(ord_dates, date_ord, side="right"))
            lo_180 = int(np.searchsorted(ord_dates, window_start_ord, side="left"))
            lo_90 = int(np.searchsorted(ord_dates, recent_start_ord, side="left"))
            lo_prior = int(np.searchsorted(ord_dates, prior_start_ord, side="left"))
            _, eps_valid, eps_sum = metric_cumsums["eps"]
            eps_recent = window_mean(eps_valid, eps_sum, lo_90, hi)
            eps_prior = window_mean(eps_valid, eps_sum, lo_prior, lo_90)
            eps_revision = np.nan
            if pd.notna(eps_recent) and pd.notna(eps_prior) and abs(float(eps_prior)) > 1e-9:
                eps_revision = float(eps_recent / eps_prior - 1.0)
            _, target_valid, target_sum = metric_cumsums["target_price"]
            target_mean = window_mean(target_valid, target_sum, lo_180, hi)
            target_upside = np.nan
            if pd.notna(target_mean) and np.isfinite(close) and close > 0:
                target_upside = float(target_mean / close - 1.0)
            visible_orgs = pd.Series(org_values[lo_180:hi]).dropna()
            visible = symbol_reports.iloc[lo_180:hi]
            detailed_reports = visible[
                visible["source"].isin(["akshare_em_research", "akshare_cninfo_rating"])
                & visible["org_name"].notna()
            ].copy()
            institution_count = detailed_reports["org_name"].nunique(dropna=True)
            research_report_count = len(
                detailed_reports.drop_duplicates(
                    ["source", "report_date", "org_name", "author_name", "report_title"],
                    keep="last",
                )
            )
            consensus_report_counts = pd.to_numeric(visible["report_count"], errors="coerce").dropna()
            consensus_report_count = float(consensus_report_counts.max()) if not consensus_report_counts.empty else 0.0
            forward = visible[visible["forecast_year"] >= date.year].copy()
            forward_years = forward["forecast_year"].dropna().nunique()
            forward_eps_growth = np.nan
            forward_revenue_growth = np.nan
            forward_net_profit_growth = np.nan
            forward_pe = np.nan
            if not forward.empty:
                by_year = forward.groupby("forecast_year", dropna=True)[["eps", "revenue", "net_profit", "pe"]].mean()
                current_year = float(date.year)
                next_year = float(date.year + 1)
                if current_year in by_year.index and next_year in by_year.index:
                    current = by_year.loc[current_year]
                    following = by_year.loc[next_year]
                    if pd.notna(current.get("eps")) and pd.notna(following.get("eps")) and abs(float(current["eps"])) > 1e-9:
                        forward_eps_growth = float(following["eps"] / current["eps"] - 1.0)
                    if pd.notna(current.get("revenue")) and pd.notna(following.get("revenue")) and abs(float(current["revenue"])) > 1e-9:
                        forward_revenue_growth = float(following["revenue"] / current["revenue"] - 1.0)
                    if pd.notna(current.get("net_profit")) and pd.notna(following.get("net_profit")) and abs(float(current["net_profit"])) > 1e-9:
                        forward_net_profit_growth = float(following["net_profit"] / current["net_profit"] - 1.0)
                pe_window = by_year.loc[by_year.index.isin([current_year, next_year]), "pe"].replace([np.inf, -np.inf], np.nan).dropna()
                if not pe_window.empty:
                    forward_pe = float(pe_window.mean())
            rows.append(
                {
                    "ts_code": ts_code,
                    "date": date,
                    "analyst_report_count_180d": float(max(0, hi - lo_180)),
                    "analyst_org_count_180d": float(visible_orgs.nunique(dropna=True)),
                    "analyst_institution_count_180d": float(institution_count),
                    "analyst_research_report_count_180d": float(research_report_count),
                    "analyst_consensus_report_count_180d": consensus_report_count,
                    "analyst_eps_mean_180d": window_mean(*metric_cumsums["eps"][1:], lo_180, hi),
                    "analyst_pe_mean_180d": window_mean(*metric_cumsums["pe"][1:], lo_180, hi),
                    "analyst_target_price_mean_180d": target_mean,
                    "analyst_net_profit_mean_180d": window_mean(*metric_cumsums["net_profit"][1:], lo_180, hi),
                    "analyst_revenue_mean_180d": window_mean(*metric_cumsums["revenue"][1:], lo_180, hi),
                    "analyst_eps_revision_180d": eps_revision,
                    "analyst_target_upside_180d": target_upside,
                    "analyst_forward_years_180d": float(forward_years),
                    "analyst_forward_eps_growth_180d": forward_eps_growth,
                    "analyst_forward_revenue_growth_180d": forward_revenue_growth,
                    "analyst_forward_net_profit_growth_180d": forward_net_profit_growth,
                    "analyst_forward_pe_180d": forward_pe,
                }
            )

    forecast = pd.DataFrame(rows)
    out = features.merge(forecast, on=["date", "ts_code"], how="left")
    return add_empty_analyst_forecast_columns(out)


def get_analyst_forecast_coverage() -> dict:
    reports = load_raw_analyst_reports()
    if reports.empty:
        return {
            "paths": [str(ANALYST_FORECAST_PATH)],
            "exists": ANALYST_FORECAST_PATH.exists(),
            "rows": 0,
            "symbols": 0,
        }
    reports["report_date"] = pd.to_datetime(reports["report_date"], errors="coerce")
    by_source = reports.groupby("source", dropna=False)["ts_code"].agg(["count", "nunique"]).reset_index()
    return {
        "paths": [str(ANALYST_FORECAST_PATH)],
        "exists": True,
        "rows": int(len(reports)),
        "symbols": int(reports["ts_code"].nunique()),
        "first_report_date": reports["report_date"].min().strftime("%Y%m%d"),
        "last_report_date": reports["report_date"].max().strftime("%Y%m%d"),
        "by_source": by_source.to_dict(orient="records"),
    }


def load_cashflow_quality_asof(features: pd.DataFrame) -> pd.DataFrame:
    cashflow_path = PROJECT_ROOT / "data/raw/cashflow.parquet"
    income_path = PROJECT_ROOT / "data/raw/income.parquet"
    if not cashflow_path.exists() or not income_path.exists():
        features["cashflow_quality"] = np.nan
        return features
    cashflow = pd.read_parquet(
        cashflow_path,
        columns=["ts_code", "ann_date", "end_date", "report_type", "n_cashflow_act", "net_profit"],
    )
    income = pd.read_parquet(
        income_path,
        columns=["ts_code", "ann_date", "end_date", "report_type", "n_income_attr_p"],
    )
    finance = cashflow.merge(
        income,
        on=["ts_code", "ann_date", "end_date", "report_type"],
        how="left",
    )
    finance["ann_date"] = pd.to_datetime(finance["ann_date"].astype(str), format="%Y%m%d", errors="coerce")
    numerator = pd.to_numeric(finance["n_cashflow_act"], errors="coerce")
    denominator = pd.to_numeric(finance["n_income_attr_p"], errors="coerce").replace(0, np.nan)
    finance["cashflow_quality"] = numerator / denominator
    finance = finance.dropna(subset=["ann_date"]).sort_values(["ann_date", "ts_code"])

    out = features.sort_values(["date", "ts_code"]).copy()
    merged_parts: list[pd.DataFrame] = []
    for ts_code, group in out.groupby("ts_code", sort=False):
        symbol_finance = finance[finance["ts_code"] == ts_code][["ann_date", "cashflow_quality"]].sort_values("ann_date")
        if symbol_finance.empty:
            group = group.copy()
            group["cashflow_quality"] = np.nan
            merged_parts.append(group)
            continue
        merged = pd.merge_asof(
            group.sort_values("date"),
            symbol_finance,
            left_on="date",
            right_on="ann_date",
            direction="backward",
        )
        merged["ts_code"] = ts_code
        merged_parts.append(merged)
    return pd.concat(merged_parts, ignore_index=True)


def build_scores(features: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    out = features.copy()
    out["listing_years"] = (out["date"] - out["list_date"]).dt.days / 365.25
    out["target_price"] = pd.concat(
        [
            out["ma_60"] * 1.02,
            out["ma_120"] * 1.05,
            out["median_close_60"],
        ],
        axis=1,
    ).min(axis=1)
    out["pullback_entry"] = (out["close"] <= out["target_price"]) & (out["close"] >= out["ma_120"])

    scored: list[pd.DataFrame] = []
    for _, group in out.groupby("date", sort=True):
        group = group.copy()
        if config.variant in MARKET_REGIME_VARIANTS:
            yield_score = percentile_score(group["dv_ttm"], True)
            stability_score = percentile_score(group["dv_ttm_stability_36m"], True)
            group["dividend_score"] = (yield_score * 0.55 + stability_score * 0.45).fillna(yield_score)
        else:
            group["dividend_score"] = percentile_score(group["dv_ttm"], True)
        normal_dividend = group["dv_ttm"].between(2.0, 6.0, inclusive="both")
        caution_dividend = group["dv_ttm"].between(6.0, 9.0, inclusive="right")
        abnormal_dividend = group["dv_ttm"] > 9.0
        group.loc[normal_dividend, "dividend_score"] += 10
        group.loc[caution_dividend, "dividend_score"] += 2
        group.loc[abnormal_dividend, "dividend_score"] -= 25
        group["dividend_score"] = group["dividend_score"].clip(0, 100)

        value = (
            percentile_score(1 / group["pe_ttm"].where(group["pe_ttm"] > 0), True) * 0.4
            + percentile_score(1 / group["pb"].where(group["pb"] > 0), True) * 0.35
            + percentile_score(1 / group["ps_ttm"].where(group["ps_ttm"] > 0), True) * 0.25
        )
        group["value_score"] = value

        if config.variant == "v2_quality_cashflow":
            quality = (
                percentile_score(group["roe"], True) * 0.35
                + percentile_score(group["netprofit_margin"], True) * 0.20
                + percentile_score(group["or_yoy"], True) * 0.10
                + percentile_score(group["debt_to_assets"], False) * 0.15
                + percentile_score(group["cashflow_quality"].clip(-3, 3), True) * 0.20
            )
        else:
            quality = (
                percentile_score(group["roe"], True) * 0.45
                + percentile_score(group["netprofit_margin"], True) * 0.25
                + percentile_score(group["or_yoy"], True) * 0.15
                + percentile_score(group["debt_to_assets"], False) * 0.15
            )
        group["quality_score"] = quality.fillna(50)

        trend = (
            percentile_score(group["return_120d"], True) * 0.45
            + percentile_score(group["ma_120_slope_20d"], True) * 0.35
            + (group["close"] > group["ma_120"]).astype(float) * 20
        )
        group["trend_score"] = trend.clip(0, 100)

        risk = (
            percentile_score(group["volatility_60d"], False) * 0.55
            + percentile_score(group["downside_volatility_60d"], False) * 0.45
        )
        group["risk_score"] = risk
        group["compounder_score"] = np.nan
        group["market_trend_strength"] = np.nan
        if "index_return_60d" in group.columns and "index_ma_120_slope_20d" in group.columns:
            group["market_trend_strength"] = (
                50
                + pd.to_numeric(group["index_return_60d"], errors="coerce").fillna(0.0) * 220
                + pd.to_numeric(group["index_ma_120_slope_20d"], errors="coerce").fillna(0.0) * 650
            ).clip(0, 100)
        group["market_overheat"] = group["index_overheat"].fillna(False).astype(bool) if "index_overheat" in group.columns else False
        group["market_pullback_warning"] = (
            (
                pd.to_numeric(group["index_return_20d"], errors="coerce").fillna(0.0) < -0.04
            )
            | (
                pd.to_numeric(group["index_drawdown_60d"], errors="coerce").fillna(0.0) < -0.08
            )
            if "index_return_20d" in group.columns and "index_drawdown_60d" in group.columns
            else False
        )
        analyst_forecast = (
            percentile_score(group["analyst_report_count_180d"], True) * 0.25
            + percentile_score(group["analyst_org_count_180d"], True) * 0.20
            + percentile_score(group["analyst_eps_revision_180d"].clip(-0.5, 0.5), True) * 0.25
            + percentile_score(1 / group["analyst_pe_mean_180d"].where(group["analyst_pe_mean_180d"] > 0), True) * 0.15
            + percentile_score(group["analyst_target_upside_180d"].clip(-0.5, 1.0), True) * 0.15
        )
        has_forecast = group["analyst_report_count_180d"].fillna(0) > 0
        group["analyst_forecast_score"] = np.where(has_forecast, analyst_forecast.fillna(0), np.nan)
        has_forward_forecast = group["analyst_forward_years_180d"].fillna(0) > 0
        group["analyst_forward_growth_score"] = np.where(
            has_forward_forecast,
            (
                percentile_score(group["analyst_forward_eps_growth_180d"].clip(-0.5, 1.5), True) * 0.35
                + percentile_score(group["analyst_forward_revenue_growth_180d"].clip(-0.5, 1.5), True) * 0.30
                + percentile_score(group["analyst_forward_net_profit_growth_180d"].clip(-0.8, 2.0), True) * 0.25
                + percentile_score(group["analyst_report_count_180d"], True) * 0.10
            ).fillna(0),
            np.nan,
        )
        group["analyst_forward_value_score"] = np.where(
            has_forward_forecast,
            (
                percentile_score(1 / group["analyst_forward_pe_180d"].where(group["analyst_forward_pe_180d"] > 0), True) * 0.40
                + percentile_score(group["analyst_target_upside_180d"].clip(-0.5, 1.0), True) * 0.20
                + percentile_score(group["analyst_forward_net_profit_growth_180d"].clip(-0.5, 1.0), True) * 0.20
                + percentile_score(group["analyst_report_count_180d"], True) * 0.20
            ).fillna(0),
            np.nan,
        )
        group["analyst_quality_score"] = np.where(
            has_forecast,
            (
                percentile_score(group["analyst_report_count_180d"], True) * 0.45
                + percentile_score(group["analyst_org_count_180d"], True) * 0.35
                + percentile_score(group["analyst_eps_revision_180d"].clip(-0.5, 0.5), True) * 0.20
            ).fillna(0),
            np.nan,
        )
        group["analyst_negative_warning"] = (
            (
                (group["analyst_report_count_180d"].fillna(0) >= 2)
                & (group["analyst_eps_revision_180d"].fillna(0.0) <= -0.18)
            )
            | (
                (group["analyst_forward_years_180d"].fillna(0) >= 1)
                & (
                    (group["analyst_forward_revenue_growth_180d"].fillna(0.0) <= -0.15)
                    | (group["analyst_forward_net_profit_growth_180d"].fillna(0.0) <= -0.20)
                )
            )
        )
        if config.variant in MARKET_REGIME_VARIANTS:
            if config.variant in FORECAST_DUAL_SCORE_VARIANTS:
                group["long_score"] = (
                    group["dividend_score"] * 0.10
                    + group["quality_score"] * 0.25
                    + group["value_score"] * 0.15
                    + group["trend_score"] * 0.18
                    + group["risk_score"] * 0.12
                    + group["analyst_forward_value_score"].fillna(50) * 0.12
                    + group["analyst_quality_score"].fillna(50) * 0.08
                )
            else:
                group["long_score"] = (
                    group["dividend_score"] * 0.15
                    + group["quality_score"] * 0.30
                    + group["value_score"] * 0.20
                    + group["trend_score"] * 0.20
                    + group["risk_score"] * 0.15
                )
        else:
            group["long_score"] = (
                group["dividend_score"] * 0.30
                + group["quality_score"] * 0.25
                + group["value_score"] * 0.20
                + group["trend_score"] * 0.15
                + group["risk_score"] * 0.10
            )
        if config.variant in FORECAST_RANK_VARIANTS:
            group["forecast_core_rank_score"] = (
                group["long_score"]
                + (group["analyst_forward_value_score"].fillna(50) - 50) * 0.18
                + (group["analyst_quality_score"].fillna(50) - 50) * 0.10
                + np.where(group["analyst_forward_net_profit_growth_180d"].fillna(-1.0) >= 0.08, 2.0, 0.0)
                + np.where(group["analyst_forward_pe_180d"].fillna(999) <= group["pe_ttm"].fillna(999) * 0.95, 1.5, 0.0)
            ).clip(0, 115)
        elif config.variant in FORECAST_LIGHT_RANK_VARIANTS:
            group["forecast_core_rank_score"] = (
                group["long_score"]
                + (group["analyst_forward_value_score"].fillna(50) - 50) * 0.06
                + (group["analyst_quality_score"].fillna(50) - 50) * 0.04
                + np.where(group["analyst_forward_net_profit_growth_180d"].fillna(-1.0) >= 0.08, 0.8, 0.0)
            ).clip(0, 110)
        elif config.variant in FORECAST_GUARDRAIL_VARIANTS:
            group["forecast_core_rank_score"] = (
                group["long_score"]
                - np.where(group["analyst_negative_warning"], 6.0, 0.0)
                + np.where(group["analyst_quality_score"].fillna(0) >= 72, 1.0, 0.0)
            ).clip(0, 110)
        else:
            group["forecast_core_rank_score"] = group["long_score"]
        vol_cutoff = group["volatility_60d"].quantile(config.max_volatility_quantile)
        base_liquidity = (
            (group["total_mv"] >= config.min_total_mv)
            & (group["circ_mv"] >= config.min_circ_mv)
            & (group["turnover_rate"] >= config.min_turnover_rate_ma20)
            & (group["turnover_rate"] <= config.max_turnover_rate_ma20)
            & (group["volatility_60d"] <= vol_cutoff)
            & (group["listing_years"] >= config.listing_years)
        )
        if config.variant in MARKET_REGIME_VARIANTS:
            riskon_recovery = config.variant == "v35_pit_universe_riskon_recovery_sleeve"
            risk_on_entry = (
                (group["market_regime"] == "risk_on")
                & (group["long_score"] >= (70 if riskon_recovery else 72))
                & (group["close"] > group["ma_120"])
                & (group["ma_120_slope_20d"] >= (-0.015 if riskon_recovery else -0.01))
                & (group["close"] <= group["target_price"] * (1.14 if riskon_recovery else 1.10))
            )
            neutral_entry = (
                (group["market_regime"] == "neutral")
                & (group["long_score"] >= (75 if riskon_recovery else 76))
                & (group["close"] > group["ma_120"])
                & (group["ma_120_slope_20d"] >= 0)
                & (group["close"] <= group["target_price"] * (1.06 if riskon_recovery else 1.04))
            )
            risk_off_entry = (
                (group["market_regime"] == "risk_off")
                & (group["long_score"] >= 84)
                & (group["dividend_score"] >= 70)
                & (group["quality_score"] >= 65)
                & (group["pullback_entry"])
            )
            group["eligible"] = base_liquidity & (risk_on_entry | neutral_entry | risk_off_entry)
            if config.variant in FORECAST_GUARDRAIL_VARIANTS:
                cyclical_or_yield = (
                    group["industry"].astype(str).str.contains("煤|钢|有色|化工|石油|银行|证券|保险", regex=True, na=False)
                    | (group["dividend_score"] >= 80)
                    | (group["value_score"] >= 82)
                )
                group["eligible"] = group["eligible"] & ~(group["analyst_negative_warning"] & cyclical_or_yield)
            if config.variant in TEA_MASTER_VARIANTS and config.variant != "v39_tea_master_optimized":
                tea_quality_guard = (
                    (group["quality_score"] >= 62)
                    & (group["trend_score"] >= 55)
                    & (group["risk_score"] >= 35)
                    & (group["roe"].fillna(0) >= 6)
                    & (group["debt_to_assets"].fillna(100) <= 82)
                    & (group["pe_ttm"].fillna(999) <= 80)
                    & (group["pb"].fillna(999) <= 10)
                    & (group["turnover_rate"].fillna(0) >= 0.5)
                    & (group["turnover_rate"].fillna(99) <= 6.0)
                    & (group["close"] >= group["ma_120"] * 0.98)
                )
                if config.variant == "v36_tea_master_concentrated":
                    tea_quality_guard = tea_quality_guard & (group["long_score"] >= 76) & (group["risk_score"] >= 45)
                elif config.variant == "v37_tea_master_core_satellite":
                    tea_quality_guard = tea_quality_guard & (group["long_score"] >= 74)
                else:
                    tea_quality_guard = tea_quality_guard & (group["long_score"] >= 72)
                group["eligible"] = group["eligible"] & tea_quality_guard
        else:
            group["eligible"] = (
                (group["long_score"] >= config.min_long_score_watch)
                & (group["dv_ttm"] >= config.min_dv_ttm_watch)
                & (group["close"] > group["ma_120"])
                & (group["ma_120_slope_20d"] >= 0)
                & (group["pullback_entry"])
                & base_liquidity
            )
        if config.variant in GROWTH_VARIANTS:
            historical_growth_score = (
                percentile_score(group["or_yoy"], True) * 0.25
                + percentile_score(group["basic_eps_yoy"], True) * 0.20
                + percentile_score(group["roe"], True) * 0.20
                + percentile_score(group["return_120d"], True) * 0.20
                + percentile_score(group["ma_120_slope_20d"], True) * 0.15
            ).fillna(0).clip(0, 100)
            if config.variant in {
                "v11_analyst_growth_sleeve",
                "v14_selective_mega_growth_sleeve",
            } | ANALYST_QUALITY_RANK_VARIANTS:
                if config.variant in FORECAST_DUAL_SCORE_VARIANTS:
                    group["growth_score"] = (
                        historical_growth_score * 0.45
                        + group["analyst_forward_growth_score"].fillna(50) * 0.35
                        + group["analyst_forecast_score"].fillna(50) * 0.10
                        + group["trend_score"] * 0.10
                    ).clip(0, 100)
                elif config.variant in FORECAST_RANK_VARIANTS:
                    group["growth_score"] = (
                        historical_growth_score * 0.60
                        + group["analyst_forward_growth_score"].fillna(50) * 0.25
                        + group["analyst_forecast_score"].fillna(50) * 0.10
                        + group["trend_score"] * 0.05
                    ).clip(0, 100)
                else:
                    group["growth_score"] = (
                        historical_growth_score * 0.75
                        + group["analyst_forecast_score"].fillna(50) * 0.25
                    ).clip(0, 100)
                if config.variant in FORECAST_GUARDRAIL_VARIANTS:
                    group["growth_score"] = (
                        historical_growth_score * 0.82
                        + group["analyst_forecast_score"].fillna(50) * 0.10
                        + group["trend_score"] * 0.08
                        - np.where(group["analyst_negative_warning"], 6.0, 0.0)
                    ).clip(0, 100)
            elif config.variant in {"v12_quality_growth_sleeve", "v13_mega_quality_growth_sleeve"}:
                size_score = percentile_score(group["total_mv"], True)
                group["growth_score"] = (
                    historical_growth_score * 0.45
                    + group["quality_score"] * 0.20
                    + group["trend_score"] * 0.20
                    + group["analyst_forecast_score"].fillna(50) * 0.10
                    + size_score * 0.05
                ).clip(0, 100)
            else:
                group["growth_score"] = historical_growth_score
            group["growth_entry"] = (
                (group["market_regime"] == "risk_on")
                & base_liquidity
                & (group["growth_score"] >= 78)
                & (group["quality_score"] >= 55)
                & (group["trend_score"] >= 70)
                & (group["roe"] >= 5)
                & ((group["or_yoy"] >= 15) | (group["basic_eps_yoy"] >= 20))
                & (group["close"] > group["ma_120"])
                & (group["ma_120_slope_20d"] >= 0)
                & (group["pe_ttm"].fillna(999) <= 80)
                & (group["pb"].fillna(999) <= 12)
            )
            if config.variant == "v10_bluechip_growth_sleeve":
                group["growth_entry"] = (
                    (group["market_regime"] == "risk_on")
                    & base_liquidity
                    & (group["growth_score"] >= 76)
                    & (group["quality_score"] >= 55)
                    & (group["trend_score"] >= 65)
                    & (group["risk_score"] >= 30)
                    & (group["roe"] >= 5)
                    & ((group["or_yoy"] >= 10) | (group["basic_eps_yoy"] >= 15))
                    & (group["close"] > group["ma_120"])
                    & (group["ma_120_slope_20d"] >= 0)
                    & (group["pe_ttm"].fillna(999) <= 120)
                    & (group["pb"].fillna(999) <= 15)
                    & (group["total_mv"] >= 5000000)
                )
            if config.variant in {
                "v11_analyst_growth_sleeve",
                "v14_selective_mega_growth_sleeve",
            } | ANALYST_QUALITY_RANK_VARIANTS:
                analyst_supported = (
                    (group["analyst_report_count_180d"].fillna(0) >= 2)
                    & (group["analyst_org_count_180d"].fillna(0) >= 2)
                    & (group["analyst_forecast_score"].fillna(0) >= 55)
                    & (group["analyst_eps_revision_180d"].fillna(-1.0) >= -0.10)
                )
                historical_supported = (group["or_yoy"] >= 12) | (group["basic_eps_yoy"] >= 15)
                forecast_supported = (
                    (group["analyst_forward_growth_score"].fillna(0) >= 62)
                    & (group["analyst_forward_years_180d"].fillna(0) >= 1)
                    & (
                        (group["analyst_forward_eps_growth_180d"].fillna(-1.0) >= 0.08)
                        | (group["analyst_forward_revenue_growth_180d"].fillna(-1.0) >= 0.10)
                        | (group["analyst_forward_net_profit_growth_180d"].fillna(-1.0) >= 0.10)
                    )
                )
                support_condition = historical_supported | analyst_supported
                if config.variant in FORECAST_DUAL_SCORE_VARIANTS | FORECAST_RANK_VARIANTS | FORECAST_LIGHT_RANK_VARIANTS:
                    support_condition = support_condition | forecast_supported
                group["growth_entry"] = (
                    (group["market_regime"] == "risk_on")
                    & base_liquidity
                    & (
                        group["growth_score"]
                        >= np.where(
                            config.variant == "v26_market_trend_compounder_sleeve",
                            76,
                            np.where(
                                config.variant == "v27_cautious_compounder_sleeve",
                                82,
                                np.where(
                                    config.variant == "v28_overheat_guarded_compounder_sleeve",
                                    80,
                                    np.where(config.variant == "v35_pit_universe_riskon_recovery_sleeve", 76, 78),
                                ),
                            ),
                        )
                    )
                    & (group["quality_score"] >= 55)
                    & (group["trend_score"] >= np.where(config.variant == "v35_pit_universe_riskon_recovery_sleeve", 68, 70))
                    & (group["roe"] >= 5)
                    & support_condition
                    & ~(
                        group["analyst_negative_warning"]
                        & (group["analyst_report_count_180d"].fillna(0) >= 2)
                    )
                    & (group["close"] > group["ma_120"])
                    & (group["ma_120_slope_20d"] >= 0)
                    & (group["pe_ttm"].fillna(999) <= np.where(config.variant in COMPOUNDER_VARIANTS, 120, 100))
                    & (group["pb"].fillna(999) <= np.where(config.variant in COMPOUNDER_VARIANTS, 18, 15))
                )
                if config.variant in TEA_MASTER_VARIANTS:
                    tea_growth_threshold = {
                        "v36_tea_master_concentrated": 999,
                        "v37_tea_master_core_satellite": 82,
                        "v38_tea_master_balanced": 78,
                        "v39_tea_master_optimized": 76,
                    }[config.variant]
                    group["growth_entry"] = (
                        group["growth_entry"]
                        & (group["growth_score"] >= tea_growth_threshold)
                        & (group["quality_score"] >= 62)
                        & (group["trend_score"] >= 72)
                        & (group["risk_score"] >= 30)
                        & (group["market_regime"] == "risk_on")
                        & ~group["market_overheat"].fillna(False)
                        & ~group["market_pullback_warning"].fillna(False)
                    )
                if config.variant == "v28_overheat_guarded_compounder_sleeve":
                    group["growth_entry"] = (
                        group["growth_entry"]
                        & ~group["market_pullback_warning"].fillna(False)
                        & (
                            ~group["market_overheat"].fillna(False)
                            | (
                                (group["growth_score"] >= 88)
                                & (group["trend_score"] >= 85)
                                & (group["quality_score"] >= 75)
                            )
                        )
                    )
            if config.variant in {
                "v14_selective_mega_growth_sleeve",
            } | ANALYST_QUALITY_RANK_VARIANTS:
                mega_supported = (
                    ((group["or_yoy"] >= 8) | (group["basic_eps_yoy"] >= 10) | (group["roe"] >= 12))
                    & (group["market_regime"] == "risk_on")
                    & base_liquidity
                    & (group["total_mv"] >= 50000000)
                    & (group["quality_score"] >= 62)
                    & (group["trend_score"] >= 65)
                    & (group["risk_score"] >= 35)
                    & (group["close"] > group["ma_120"])
                    & (group["ma_120_slope_20d"] >= 0)
                    & (group["pe_ttm"].fillna(999) <= 80)
                    & (group["pb"].fillna(999) <= 10)
                )
                group["mega_quality_growth_entry"] = mega_supported
            else:
                group["mega_quality_growth_entry"] = False
            if config.variant in ANALYST_QUALITY_RANK_VARIANTS:
                group["growth_rank_score"] = (
                    group["growth_score"]
                    + np.where(group["mega_quality_growth_entry"], 8.0, 0.0)
                    + np.where(group["analyst_forecast_score"].fillna(0) >= 70, 3.0, 0.0)
                    + np.where(
                        config.variant in FORECAST_DUAL_SCORE_VARIANTS | FORECAST_RANK_VARIANTS,
                        (group["analyst_forward_growth_score"].fillna(50) - 50) * 0.20,
                        np.where(
                            config.variant in FORECAST_LIGHT_RANK_VARIANTS,
                            (group["analyst_forward_growth_score"].fillna(50) - 50) * 0.08,
                            0.0,
                        ),
                    )
                ).clip(0, 110)
                if config.variant in COMPOUNDER_VARIANTS:
                    group["compounder_score"] = (
                        group["growth_score"] * 0.30
                        + group["quality_score"] * 0.22
                        + group["trend_score"] * 0.28
                        + percentile_score(group["total_mv"], True) * 0.08
                        + group["risk_score"] * 0.07
                        + group["market_trend_strength"].fillna(50) * 0.05
                        + np.where(group["analyst_quality_score"].fillna(0) >= 65, 2.0, 0.0)
                        + np.where(group["analyst_forward_growth_score"].fillna(0) >= 65, 2.0, 0.0)
                        - np.where(group["analyst_negative_warning"], 8.0, 0.0)
                    ).clip(0, 120)
                    group["growth_rank_score"] = (
                        group["compounder_score"]
                        + np.where(group["mega_quality_growth_entry"], 3.0, 0.0)
                    ).clip(0, 125)
            else:
                group["growth_rank_score"] = group["growth_score"]
                group["compounder_score"] = np.nan
            if config.variant in {"v12_quality_growth_sleeve", "v13_mega_quality_growth_sleeve"}:
                quality_growth_supported = (
                    (group["or_yoy"] >= 8)
                    | (group["basic_eps_yoy"] >= 10)
                    | ((group["roe"] >= 12) & (group["return_120d"] >= 5))
                    | (group["analyst_forecast_score"].fillna(0) >= 70)
                )
                group["growth_entry"] = (
                    (group["market_regime"] == "risk_on")
                    & base_liquidity
                    & (group["growth_score"] >= 72)
                    & (group["quality_score"] >= 58)
                    & (group["trend_score"] >= 58)
                    & (group["risk_score"] >= 20)
                    & (group["roe"] >= 5)
                    & quality_growth_supported
                    & (group["close"] > group["ma_120"] * 0.98)
                    & (group["ma_120_slope_20d"] >= -0.005)
                    & (group["pe_ttm"].fillna(999) <= 150)
                    & (group["pb"].fillna(999) <= 20)
                    & (group["total_mv"] >= 1500000)
                )
        else:
            group["growth_score"] = np.nan
            group["growth_entry"] = False
        if config.variant == "v2_quality_cashflow":
            group["eligible"] = (
                group["eligible"]
                & (group["roe"] > 5)
                & (group["debt_to_assets"].fillna(100) < 85)
                & (group["cashflow_quality"].fillna(0) > 0)
            )
        scored.append(group)
    return pd.concat(scored, ignore_index=True)


def make_monthly_targets(scored: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    if config.variant in {"v3_stateful_hold"} | MARKET_REGIME_VARIANTS:
        return make_stateful_monthly_targets(scored, config)

    targets: list[pd.DataFrame] = []
    for date, group in scored.groupby("date", sort=True):
        picks = group[group["eligible"]].sort_values(
            ["long_score", "dividend_score", "risk_score"], ascending=False
        )
        picks = picks.head(config.max_positions).copy()
        if picks.empty:
            continue
        weight = min(config.max_symbol_weight, config.target_total_weight / max(len(picks), 1))
        picks["target_weight"] = weight
        picks["rebalance_date"] = date
        targets.append(
            picks[
                [
                    "rebalance_date",
                    "trade_date",
                    "ts_code",
                    "name",
                    "industry",
                    "close",
                    "target_price",
                    "target_weight",
                    "long_score",
                    "dividend_score",
                    "quality_score",
                    "value_score",
                    "trend_score",
                    "risk_score",
                    "dv_ttm",
                    "pe_ttm",
                    "pb",
                    "roe",
                ]
            ]
        )
    if not targets:
        return pd.DataFrame()
    return pd.concat(targets, ignore_index=True)


def format_target_rows(frame: pd.DataFrame, date: pd.Timestamp, config: BacktestConfig) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    target_total_weight = config.target_total_weight
    if config.variant in MARKET_REGIME_VARIANTS and "market_regime" in out.columns:
        regimes = out["market_regime"].dropna().astype(str)
        regime = regimes.iloc[0] if not regimes.empty else "neutral"
        target_total_weight = {"risk_on": 0.85, "neutral": 0.60, "risk_off": 0.30}.get(regime, 0.60)
    weight = min(config.max_symbol_weight, target_total_weight / max(len(out), 1))
    out["target_weight"] = weight
    out["rebalance_date"] = date
    columns = [
        "rebalance_date",
        "trade_date",
        "ts_code",
        "name",
        "industry",
        "market_regime",
        "close",
        "target_price",
        "target_weight",
        "long_score",
        "dividend_score",
        "quality_score",
        "value_score",
        "trend_score",
        "risk_score",
        "analyst_forecast_score",
        "analyst_report_count_180d",
        "analyst_org_count_180d",
        "analyst_eps_revision_180d",
        "analyst_target_upside_180d",
        "dv_ttm",
        "dv_ttm_stability_36m",
        "pe_ttm",
        "pb",
        "roe",
    ]
    return out[[col for col in columns if col in out.columns]]


def make_stateful_monthly_targets(scored: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    if config.variant in GROWTH_VARIANTS:
        return make_growth_sleeve_monthly_targets(scored, config)

    targets: list[pd.DataFrame] = []
    current: set[str] = set()
    for date, group in scored.groupby("date", sort=True):
        group = group.copy()
        group_by_symbol = group.set_index("ts_code", drop=False)

        survivors: list[str] = []
        for symbol in current:
            if symbol not in group_by_symbol.index:
                continue
            row = group_by_symbol.loc[symbol]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            if config.variant in MARKET_REGIME_VARIANTS:
                regime = str(row.get("market_regime", "neutral"))
                exit_signal = (
                    row["long_score"] < 58
                    or row["close"] < row["ma_120"] * (0.94 if regime == "risk_on" else 0.98)
                    or (regime == "risk_off" and row["quality_score"] < 55)
                    or (regime == "risk_off" and row["risk_score"] < 35)
                )
            else:
                exit_signal = (
                    row["long_score"] < 62
                    or row["dv_ttm"] < 1.8
                    or row["close"] < row["ma_120"] * 0.97
                    or row["ma_120_slope_20d"] < -0.03
                )
            if not exit_signal:
                survivors.append(symbol)

        entry_pool = group[group["eligible"] & ~group["ts_code"].isin(survivors)].sort_values(
            ["long_score", "dividend_score", "quality_score", "risk_score"], ascending=False
        )
        slots = max(config.max_positions - len(survivors), 0)
        entrants = entry_pool.head(slots)["ts_code"].tolist()
        selected_symbols = survivors + entrants
        if not selected_symbols:
            current = set()
            continue

        selected = group[group["ts_code"].isin(selected_symbols)].copy()
        selected["_order"] = selected["ts_code"].map({symbol: idx for idx, symbol in enumerate(selected_symbols)})
        selected = selected.sort_values("_order").drop(columns=["_order"])
        if config.variant == "v5_industry_cap":
            selected = (
                selected.sort_values(["industry", "long_score", "risk_score"], ascending=[True, False, False])
                .groupby("industry", dropna=False)
                .head(4)
                .sort_values(["long_score", "risk_score"], ascending=False)
                .head(config.max_positions)
            )
        current = set(selected["ts_code"])
        targets.append(format_target_rows(selected, date, config))

    if not targets:
        return pd.DataFrame()
    return pd.concat(targets, ignore_index=True)


def make_growth_sleeve_monthly_targets(scored: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    targets: list[pd.DataFrame] = []
    current: set[str] = set()
    for date, group in scored.groupby("date", sort=True):
        group = group.copy()
        group_by_symbol = group.set_index("ts_code", drop=False)
        regimes = group["market_regime"].dropna().astype(str) if "market_regime" in group.columns else pd.Series(dtype=str)
        regime = regimes.iloc[0] if not regimes.empty else "neutral"
        market_trend_strength = float(group["market_trend_strength"].dropna().iloc[0]) if "market_trend_strength" in group.columns and not group["market_trend_strength"].dropna().empty else 50.0
        market_overheat = bool(group["market_overheat"].fillna(False).iloc[0]) if "market_overheat" in group.columns and not group.empty else False
        market_pullback_warning = bool(group["market_pullback_warning"].fillna(False).iloc[0]) if "market_pullback_warning" in group.columns and not group.empty else False
        strong_risk_on = regime == "risk_on" and market_trend_strength >= 70.0
        healthy_strong_risk_on = strong_risk_on and not market_overheat and not market_pullback_warning

        survivors: list[str] = []
        for symbol in current:
            if symbol not in group_by_symbol.index:
                continue
            row = group_by_symbol.loc[symbol]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            is_growth = bool(row.get("growth_entry", False)) or (
                config.variant == "v16_mega_rank_growth_sleeve"
                and bool(row.get("mega_quality_growth_entry", False))
            )
            if is_growth:
                exit_signal = (
                    row["growth_score"] < 58
                    or row["close"] < row["ma_120"] * 0.94
                    or (regime != "risk_on" and row["trend_score"] < 55)
                    or (config.variant == "v32_empty_bear_sleeve" and (regime == "risk_off" or market_pullback_warning))
                )
            else:
                exit_signal = (
                    row["long_score"] < 58
                    or row["close"] < row["ma_120"] * (0.94 if regime == "risk_on" else 0.98)
                    or (regime == "risk_off" and row["quality_score"] < 55)
                    or (regime == "risk_off" and row["risk_score"] < 35)
                    or (config.variant == "v32_empty_bear_sleeve" and (regime == "risk_off" or market_pullback_warning))
                )
            if not exit_signal:
                survivors.append(symbol)

        if config.variant in CAPPED_GROWTH_VARIANTS:
            if config.variant == "v14_selective_mega_growth_sleeve":
                value_slots = 13 if regime == "risk_on" else config.max_positions
                growth_slots = 2 if regime == "risk_on" else 0
            elif config.variant in {"v12_quality_growth_sleeve", "v13_mega_quality_growth_sleeve"}:
                value_slots = 10 if regime == "risk_on" else config.max_positions
                growth_slots = 6 if regime == "risk_on" else 0
            elif config.variant == "v25_forecast_guardrail_grid_sleeve":
                value_slots = 12 if regime == "risk_on" else config.max_positions
                growth_slots = 3 if strong_risk_on else (1 if regime == "risk_on" else 0)
            elif config.variant == "v26_market_trend_compounder_sleeve":
                value_slots = 10 if regime == "risk_on" else config.max_positions
                growth_slots = 5 if strong_risk_on else (2 if regime == "risk_on" else 0)
            elif config.variant == "v27_cautious_compounder_sleeve":
                value_slots = 12 if regime == "risk_on" else config.max_positions
                growth_slots = 3 if strong_risk_on else (1 if regime == "risk_on" else 0)
            elif config.variant == "v28_overheat_guarded_compounder_sleeve":
                value_slots = 12 if regime == "risk_on" else config.max_positions
                growth_slots = 4 if healthy_strong_risk_on else (1 if regime == "risk_on" and not market_pullback_warning else 0)
            elif config.variant == "v30_concentrated_trend_sleeve":
                value_slots = 7 if regime == "risk_on" else (5 if regime == "neutral" else 2)
                growth_slots = 2 if healthy_strong_risk_on else (1 if regime == "risk_on" and not market_pullback_warning else 0)
            elif config.variant == "v31_bull_bear_exposure_sleeve":
                value_slots = 14 if regime == "risk_on" else (8 if regime == "neutral" else 3)
                growth_slots = 6 if healthy_strong_risk_on else (3 if regime == "risk_on" and not market_pullback_warning else 0)
            elif config.variant == "v32_empty_bear_sleeve":
                if regime == "risk_off" or market_pullback_warning:
                    value_slots = 0
                    growth_slots = 0
                else:
                    value_slots = 12 if regime == "risk_on" else 5
                    growth_slots = 4 if healthy_strong_risk_on else 0
            elif config.variant in {
                "v33_bull_boost_defensive_bear_sleeve",
                "v34_pit_universe_guarded_sleeve",
                "v35_pit_universe_riskon_recovery_sleeve",
                "v40_tea_master_regime_grid",
                "v41_tea_master_pit_guarded",
            }:
                value_slots = 13 if regime == "risk_on" else (12 if regime == "neutral" else 8)
                growth_slots = 5 if healthy_strong_risk_on else (2 if regime == "risk_on" and not market_pullback_warning else 0)
            elif config.variant == "v42_tea_master_growth_throttle":
                value_slots = 13 if regime == "risk_on" else (12 if regime == "neutral" else 8)
                growth_slots = 2 if healthy_strong_risk_on else 0
            elif config.variant == "v43_tea_master_core_only":
                value_slots = 13 if regime == "risk_on" else (12 if regime == "neutral" else 8)
                growth_slots = 0
            elif config.variant == "v44_tea_master_defensive_neutral":
                value_slots = 13 if regime == "risk_on" else (9 if regime == "neutral" else 5)
                growth_slots = 0
            elif config.variant == "v45_tea_master_riskon_quality":
                value_slots = 11 if regime == "risk_on" else (9 if regime == "neutral" else 5)
                growth_slots = 0
            elif config.variant == "v36_tea_master_concentrated":
                value_slots = 4 if regime == "risk_on" else (3 if regime == "neutral" else 1)
                growth_slots = 0
            elif config.variant == "v37_tea_master_core_satellite":
                value_slots = 4 if regime == "risk_on" else (3 if regime == "neutral" else 1)
                growth_slots = 1 if healthy_strong_risk_on else 0
            elif config.variant == "v38_tea_master_balanced":
                value_slots = 5 if regime == "risk_on" else (4 if regime == "neutral" else 2)
                growth_slots = 2 if healthy_strong_risk_on else (1 if regime == "risk_on" and not market_pullback_warning else 0)
            elif config.variant == "v39_tea_master_optimized":
                value_slots = 11 if regime == "risk_on" else (10 if regime == "neutral" else 6)
                growth_slots = 4 if healthy_strong_risk_on else (1 if regime == "risk_on" and not market_pullback_warning else 0)
            else:
                value_slots = 12 if regime == "risk_on" else config.max_positions
                growth_slots = 4 if regime == "risk_on" else 0
        else:
            value_slots = 10 if regime == "risk_on" else config.max_positions
            growth_slots = 5 if regime == "risk_on" else 0

        value_sort_column = (
            "forecast_core_rank_score"
            if config.variant in FORECAST_RANK_VARIANTS | FORECAST_LIGHT_RANK_VARIANTS | FORECAST_GUARDRAIL_VARIANTS
            else "long_score"
        )
        value_pool = group[group["eligible"] & ~group["ts_code"].isin(survivors)].sort_values(
            [value_sort_column, "long_score", "dividend_score", "quality_score", "risk_score"], ascending=False
        )
        value_slots = max(value_slots - len(survivors), 0)
        value_entrants = value_pool.head(value_slots)["ts_code"].tolist()

        growth_entrants: list[str] = []
        if growth_slots:
            if config.variant == "v16_mega_rank_growth_sleeve":
                growth_candidate = group["growth_entry"] | group["mega_quality_growth_entry"]
            else:
                growth_candidate = group["growth_entry"]
            growth_filter = (
                growth_candidate
                & ~group["ts_code"].isin(survivors)
                & ~group["ts_code"].isin(value_entrants)
            )
            mega_entrants: list[str] = []
            if config.variant == "v13_mega_quality_growth_sleeve":
                mega_pool = group[
                    growth_filter
                    & (group["total_mv"] >= 50000000)
                    & (group["quality_score"] >= 55)
                    & (group["trend_score"] >= 55)
                    & (group["roe"] >= 5)
                ].sort_values(
                    ["quality_score", "total_mv", "trend_score", "growth_score"],
                    ascending=False,
                )
                mega_entrants = mega_pool.head(min(2, growth_slots))["ts_code"].tolist()
            elif config.variant == "v14_selective_mega_growth_sleeve":
                mega_pool = group[
                    group["mega_quality_growth_entry"]
                    & ~group["ts_code"].isin(survivors)
                    & ~group["ts_code"].isin(value_entrants)
                ].sort_values(
                    ["quality_score", "trend_score", "total_mv", "growth_score"],
                    ascending=False,
                )
                mega_entrants = mega_pool.head(growth_slots)["ts_code"].tolist()
            growth_sort_column = (
                "growth_rank_score"
                if config.variant in ANALYST_QUALITY_RANK_VARIANTS
                else "growth_score"
            )
            growth_pool = group[
                growth_filter
                & ~group["ts_code"].isin(mega_entrants)
            ].sort_values([growth_sort_column, "growth_score", "trend_score", "quality_score"], ascending=False)
            if config.variant == "v14_selective_mega_growth_sleeve":
                growth_entrants = mega_entrants
            else:
                growth_entrants = growth_pool.head(growth_slots)["ts_code"].tolist()
                growth_entrants = (mega_entrants + growth_entrants)[:growth_slots]

        selected_symbols = survivors + value_entrants + growth_entrants
        if not selected_symbols:
            current = set()
            continue
        selected = group[group["ts_code"].isin(selected_symbols)].copy()
        selected["sleeve"] = np.where(selected["ts_code"].isin(growth_entrants), "growth", "core")
        selected["_order"] = selected["ts_code"].map({symbol: idx for idx, symbol in enumerate(selected_symbols)})
        selected = selected.sort_values("_order").drop(columns=["_order"])
        if config.variant in CAPPED_GROWTH_VARIANTS:
            core_mask = selected["sleeve"] == "core"
            growth_mask = selected["sleeve"] == "growth"
            if regime == "risk_on":
                if config.variant == "v14_selective_mega_growth_sleeve":
                    core_total = 0.70
                    growth_total = 0.10
                elif config.variant in {"v12_quality_growth_sleeve", "v13_mega_quality_growth_sleeve"}:
                    core_total = 0.60
                    growth_total = 0.20
                elif config.variant == "v25_forecast_guardrail_grid_sleeve":
                    core_total = 0.73 if strong_risk_on else 0.78
                    growth_total = 0.12 if strong_risk_on else 0.05
                elif config.variant == "v26_market_trend_compounder_sleeve":
                    core_total = 0.64 if strong_risk_on else 0.75
                    growth_total = 0.21 if strong_risk_on else 0.08
                elif config.variant == "v27_cautious_compounder_sleeve":
                    core_total = 0.72 if strong_risk_on else 0.78
                    growth_total = 0.13 if strong_risk_on else 0.04
                elif config.variant == "v28_overheat_guarded_compounder_sleeve":
                    if healthy_strong_risk_on:
                        core_total = 0.68
                        growth_total = 0.16
                    elif market_overheat or market_pullback_warning:
                        core_total = 0.60
                        growth_total = 0.02
                    else:
                        core_total = 0.76
                        growth_total = 0.04
                elif config.variant == "v29_overheat_throttle_grid_sleeve":
                    if market_pullback_warning:
                        core_total = 0.55
                        growth_total = 0.00
                    elif market_overheat:
                        core_total = 0.62
                        growth_total = 0.04
                    else:
                        core_total = 0.70
                        growth_total = 0.15
                elif config.variant == "v30_concentrated_trend_sleeve":
                    if market_pullback_warning:
                        core_total = 0.45
                        growth_total = 0.00
                    elif market_overheat:
                        core_total = 0.55
                        growth_total = 0.03
                    elif healthy_strong_risk_on:
                        core_total = 0.70
                        growth_total = 0.14
                    else:
                        core_total = 0.62
                        growth_total = 0.06
                elif config.variant == "v31_bull_bear_exposure_sleeve":
                    if market_pullback_warning:
                        core_total = 0.45
                        growth_total = 0.00
                    elif market_overheat:
                        core_total = 0.65
                        growth_total = 0.05
                    elif healthy_strong_risk_on:
                        core_total = 0.82
                        growth_total = 0.15
                    else:
                        core_total = 0.75
                        growth_total = 0.10
                elif config.variant == "v32_empty_bear_sleeve":
                    if market_pullback_warning:
                        core_total = 0.20
                        growth_total = 0.00
                    elif market_overheat:
                        core_total = 0.45
                        growth_total = 0.00
                    elif healthy_strong_risk_on:
                        core_total = 0.75
                        growth_total = 0.12
                    else:
                        core_total = 0.62
                        growth_total = 0.00
                elif config.variant in {
                    "v33_bull_boost_defensive_bear_sleeve",
                    "v34_pit_universe_guarded_sleeve",
                    "v35_pit_universe_riskon_recovery_sleeve",
                    "v40_tea_master_regime_grid",
                    "v41_tea_master_pit_guarded",
                }:
                    if market_pullback_warning:
                        core_total = 0.55
                        growth_total = 0.00
                    elif market_overheat:
                        core_total = 0.62
                        growth_total = 0.04
                    elif healthy_strong_risk_on:
                        core_total = 0.78
                        growth_total = 0.15
                    else:
                        core_total = 0.70
                        growth_total = 0.10
                elif config.variant == "v42_tea_master_growth_throttle":
                    if market_pullback_warning:
                        core_total = 0.55
                        growth_total = 0.00
                    elif market_overheat:
                        core_total = 0.62
                        growth_total = 0.00
                    elif healthy_strong_risk_on:
                        core_total = 0.76
                        growth_total = 0.06
                    else:
                        core_total = 0.70
                        growth_total = 0.00
                elif config.variant == "v43_tea_master_core_only":
                    if market_pullback_warning:
                        core_total = 0.55
                    elif market_overheat:
                        core_total = 0.62
                    elif healthy_strong_risk_on:
                        core_total = 0.78
                    else:
                        core_total = 0.70
                    growth_total = 0.00
                elif config.variant == "v44_tea_master_defensive_neutral":
                    if market_pullback_warning:
                        core_total = 0.45
                    elif market_overheat:
                        core_total = 0.56
                    elif healthy_strong_risk_on:
                        core_total = 0.78
                    else:
                        core_total = 0.66
                    growth_total = 0.00
                elif config.variant == "v45_tea_master_riskon_quality":
                    if market_pullback_warning:
                        core_total = 0.45
                    elif market_overheat:
                        core_total = 0.54
                    elif healthy_strong_risk_on:
                        core_total = 0.72
                    else:
                        core_total = 0.58
                    growth_total = 0.00
                elif config.variant == "v36_tea_master_concentrated":
                    if market_pullback_warning:
                        core_total = 0.45
                    elif market_overheat:
                        core_total = 0.55
                    elif healthy_strong_risk_on:
                        core_total = 0.75
                    else:
                        core_total = 0.65
                    growth_total = 0.0
                elif config.variant == "v37_tea_master_core_satellite":
                    if market_pullback_warning:
                        core_total = 0.48
                        growth_total = 0.00
                    elif market_overheat:
                        core_total = 0.58
                        growth_total = 0.00
                    elif healthy_strong_risk_on:
                        core_total = 0.72
                        growth_total = 0.08
                    else:
                        core_total = 0.64
                        growth_total = 0.00
                elif config.variant == "v38_tea_master_balanced":
                    if market_pullback_warning:
                        core_total = 0.50
                        growth_total = 0.00
                    elif market_overheat:
                        core_total = 0.60
                        growth_total = 0.02
                    elif healthy_strong_risk_on:
                        core_total = 0.72
                        growth_total = 0.10
                    else:
                        core_total = 0.66
                        growth_total = 0.04
                elif config.variant == "v39_tea_master_optimized":
                    if market_pullback_warning:
                        core_total = 0.55
                        growth_total = 0.00
                    elif market_overheat:
                        core_total = 0.62
                        growth_total = 0.03
                    elif healthy_strong_risk_on:
                        core_total = 0.76
                        growth_total = 0.12
                    else:
                        core_total = 0.70
                        growth_total = 0.08
                else:
                    core_total = 0.70
                    growth_total = 0.15
            elif regime == "neutral":
                if config.variant == "v30_concentrated_trend_sleeve":
                    core_total = 0.35
                elif config.variant == "v31_bull_bear_exposure_sleeve":
                    core_total = 0.45
                elif config.variant == "v32_empty_bear_sleeve":
                    core_total = 0.20
                elif config.variant in {
                    "v33_bull_boost_defensive_bear_sleeve",
                    "v34_pit_universe_guarded_sleeve",
                    "v35_pit_universe_riskon_recovery_sleeve",
                    "v40_tea_master_regime_grid",
                    "v41_tea_master_pit_guarded",
                    "v42_tea_master_growth_throttle",
                    "v43_tea_master_core_only",
                }:
                    core_total = 0.60
                elif config.variant == "v44_tea_master_defensive_neutral":
                    core_total = 0.42
                elif config.variant == "v45_tea_master_riskon_quality":
                    core_total = 0.38
                elif config.variant == "v36_tea_master_concentrated":
                    core_total = 0.50
                elif config.variant == "v37_tea_master_core_satellite":
                    core_total = 0.52
                elif config.variant == "v38_tea_master_balanced":
                    core_total = 0.55
                elif config.variant == "v39_tea_master_optimized":
                    core_total = 0.58
                else:
                    core_total = 0.60
                growth_total = 0.0
            else:
                if config.variant == "v30_concentrated_trend_sleeve":
                    core_total = 0.12
                elif config.variant == "v31_bull_bear_exposure_sleeve":
                    core_total = 0.05
                elif config.variant == "v32_empty_bear_sleeve":
                    core_total = 0.0
                elif config.variant in {
                    "v33_bull_boost_defensive_bear_sleeve",
                    "v34_pit_universe_guarded_sleeve",
                    "v35_pit_universe_riskon_recovery_sleeve",
                    "v40_tea_master_regime_grid",
                    "v41_tea_master_pit_guarded",
                    "v42_tea_master_growth_throttle",
                    "v43_tea_master_core_only",
                }:
                    core_total = 0.25
                elif config.variant in {"v44_tea_master_defensive_neutral", "v45_tea_master_riskon_quality"}:
                    core_total = 0.12
                elif config.variant == "v36_tea_master_concentrated":
                    core_total = 0.12
                elif config.variant == "v37_tea_master_core_satellite":
                    core_total = 0.15
                elif config.variant == "v38_tea_master_balanced":
                    core_total = 0.20
                elif config.variant == "v39_tea_master_optimized":
                    core_total = 0.25
                else:
                    core_total = 0.30
                growth_total = 0.0
            selected["target_weight"] = 0.0
            if core_mask.any():
                if config.variant == "v36_tea_master_concentrated":
                    core_cap = 0.25
                elif config.variant in {"v37_tea_master_core_satellite", "v38_tea_master_balanced"}:
                    core_cap = 0.20
                elif config.variant == "v39_tea_master_optimized":
                    core_cap = 0.10
                else:
                    core_cap = 0.12 if config.variant == "v30_concentrated_trend_sleeve" else 0.10
                selected.loc[core_mask, "target_weight"] = min(core_cap, core_total / int(core_mask.sum()))
            if growth_mask.any() and growth_total > 0:
                if config.variant in TEA_MASTER_VARIANTS:
                    growth_cap = 0.08 if config.variant == "v37_tea_master_core_satellite" else 0.06
                else:
                    growth_cap = (
                        0.07
                        if config.variant == "v31_bull_bear_exposure_sleeve" and healthy_strong_risk_on
                        else (0.06 if config.variant in {"v26_market_trend_compounder_sleeve", "v30_concentrated_trend_sleeve"} and strong_risk_on else 0.05)
                    )
                selected.loc[growth_mask, "target_weight"] = min(growth_cap, growth_total / int(growth_mask.sum()))
            selected["rebalance_date"] = date
            columns = [
                "rebalance_date",
                "trade_date",
                "ts_code",
                "name",
                "industry",
                "market_regime",
                "sleeve",
                "close",
                "target_price",
                "target_weight",
                "long_score",
                "forecast_core_rank_score",
                "growth_score",
                "growth_rank_score",
                "compounder_score",
                "growth_entry",
                "mega_quality_growth_entry",
                "market_trend_strength",
                "market_overheat",
                "market_pullback_warning",
                "analyst_negative_warning",
                "analyst_forecast_score",
                "analyst_forward_growth_score",
                "analyst_forward_value_score",
                "analyst_quality_score",
                "analyst_report_count_180d",
                "analyst_org_count_180d",
                "analyst_eps_revision_180d",
                "analyst_target_upside_180d",
                "analyst_forward_eps_growth_180d",
                "analyst_forward_revenue_growth_180d",
                "analyst_forward_net_profit_growth_180d",
                "analyst_forward_pe_180d",
                "dividend_score",
                "quality_score",
                "value_score",
                "trend_score",
                "risk_score",
                "dv_ttm",
                "dv_ttm_stability_36m",
                "pe_ttm",
                "pb",
                "roe",
            ]
            targets.append(selected[[col for col in columns if col in selected.columns]])
            current = set(selected["ts_code"])
            continue
        current = set(selected["ts_code"])
        targets.append(format_target_rows(selected, date, config))

    if not targets:
        return pd.DataFrame()
    return pd.concat(targets, ignore_index=True)


def build_staged_pending_weights(
    target_frame: pd.DataFrame,
    current_weights: dict[str, float],
    config: BacktestConfig,
) -> dict[str, float]:
    if config.variant not in STAGED_ENTRY_VARIANTS:
        return dict(zip(target_frame["ts_code"], target_frame["target_weight"], strict=False))

    pending: dict[str, float] = {}
    for _, row in target_frame.iterrows():
        symbol = str(row["ts_code"])
        current_weight = float(current_weights.get(symbol, 0.0))
        target_weight = float(row.get("target_weight", 0.0) or 0.0)
        if target_weight <= current_weight:
            pending[symbol] = target_weight
            continue

        close = float(row.get("close", np.nan))
        target_price = float(row.get("target_price", np.nan))
        sleeve = str(row.get("sleeve", "core"))
        regime = str(row.get("market_regime", "neutral"))
        long_score = float(row.get("long_score", np.nan))
        growth_rank_score = float(row.get("growth_rank_score", row.get("growth_score", np.nan)))

        fast_staged = config.variant == "v20_fast_staged_grid_sleeve"
        if sleeve == "growth":
            tranche = (0.65 if fast_staged else 0.45) if regime == "risk_on" else 0.0
        elif regime == "risk_on":
            tranche = 0.78 if fast_staged else 0.60
        elif regime == "risk_off":
            tranche = 0.45 if fast_staged else 0.35
        else:
            tranche = 0.65 if fast_staged else 0.50

        if np.isfinite(close) and np.isfinite(target_price) and target_price > 0:
            price_ratio = close / target_price
            if price_ratio <= 0.97:
                tranche += 0.20 if fast_staged else 0.30
            elif price_ratio <= 1.00:
                tranche += 0.15 if fast_staged else 0.20
            elif price_ratio >= (1.06 if sleeve == "growth" else 1.08):
                if fast_staged:
                    tranche = min(tranche, 0.45 if sleeve == "growth" else 0.55)
                else:
                    tranche = min(tranche, 0.25 if sleeve == "growth" else 0.30)

        if sleeve == "growth" and np.isfinite(growth_rank_score) and growth_rank_score >= 85:
            tranche += 0.10
        elif sleeve != "growth" and np.isfinite(long_score) and long_score >= 82:
            tranche += 0.10

        tranche = float(np.clip(tranche, 0.20, 1.0))
        pending[symbol] = current_weight + (target_weight - current_weight) * tranche
    return pending


def style_grid_parameters(info: dict, row: pd.Series, current_regime: str) -> dict:
    sleeve = str(info.get("sleeve", "core"))
    dividend_score = float(info.get("dividend_score", np.nan))
    value_score = float(info.get("value_score", np.nan))
    quality_score = float(info.get("quality_score", np.nan))
    risk_score = float(info.get("risk_score", np.nan))
    trend_score = float(info.get("trend_score", np.nan))
    market_trend_strength = float(info.get("market_trend_strength", np.nan))
    volatility = float(row.get("volatility_60d", np.nan))

    if sleeve == "growth":
        profile = "growth"
        core_ratio = 0.70
        sell_ma20, sell_ma60 = 1.10, 1.16
        buy_ma20, buy_ma60 = 1.02, 1.04
    elif (
        (np.isfinite(dividend_score) and dividend_score >= 80)
        or (np.isfinite(value_score) and value_score >= 80)
    ) and (not np.isfinite(risk_score) or risk_score >= 55):
        profile = "defensive_value"
        core_ratio = 0.85
        sell_ma20, sell_ma60 = 1.14, 1.20
        buy_ma20, buy_ma60 = 1.01, 1.03
    elif (np.isfinite(risk_score) and risk_score < 45) or (np.isfinite(volatility) and volatility > 0.04):
        profile = "high_volatility"
        core_ratio = 0.70
        sell_ma20, sell_ma60 = 1.08, 1.14
        buy_ma20, buy_ma60 = 1.025, 1.05
    else:
        profile = "quality_core"
        core_ratio = 0.80
        sell_ma20, sell_ma60 = 1.12, 1.18
        buy_ma20, buy_ma60 = 1.015, 1.035

    if current_regime == "risk_off":
        core_ratio = min(0.90, core_ratio + 0.05)
    elif sleeve == "growth" and np.isfinite(market_trend_strength) and market_trend_strength >= 75 and np.isfinite(trend_score) and trend_score >= 75:
        core_ratio = min(0.78, core_ratio + 0.05)
        sell_ma20 += 0.03
        sell_ma60 += 0.04
    if np.isfinite(quality_score) and quality_score < 55:
        core_ratio = min(core_ratio, 0.75)
    if sleeve == "growth" and np.isfinite(trend_score) and trend_score < 60:
        core_ratio = min(core_ratio, 0.65)

    return {
        "profile": profile,
        "core_ratio": core_ratio,
        "sell_ma20": sell_ma20,
        "sell_ma60": sell_ma60,
        "buy_ma20": buy_ma20,
        "buy_ma60": buy_ma60,
    }


def run_portfolio_backtest(
    daily_returns: pd.DataFrame,
    targets: pd.DataFrame,
    scored: pd.DataFrame,
    config: BacktestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = daily_returns.sort_values(["date", "ts_code"]).copy()
    daily["ret_1d"] = pd.to_numeric(daily["ret_1d"], errors="coerce").fillna(0.0)
    dates = sorted(daily["date"].dropna().unique())
    target_map = {date: frame.copy() for date, frame in targets.groupby("rebalance_date")}
    score_lookup = scored.set_index(["date", "ts_code"])["dv_ttm"].to_dict()

    current_weights: dict[str, float] = {}
    base_weights: dict[str, float] = {}
    position_info: dict[str, dict] = {}
    pending_weights: dict[str, float] | None = None
    pending_info: dict[str, dict] | None = None
    rows: list[dict] = []
    trades: list[dict] = []
    equity = 1.0
    previous_weights: dict[str, float] = {}
    current_regime = "neutral"

    returns_by_date = {date: frame for date, frame in daily.groupby("date", sort=True)}
    for date in dates:
        if pending_weights is not None:
            turnover = sum(
                abs(pending_weights.get(symbol, 0.0) - previous_weights.get(symbol, 0.0))
                for symbol in set(pending_weights) | set(previous_weights)
            )
            sell_turnover = sum(
                max(previous_weights.get(symbol, 0.0) - pending_weights.get(symbol, 0.0), 0.0)
                for symbol in set(pending_weights) | set(previous_weights)
            )
            cost = turnover * config.commission_rate + sell_turnover * config.stamp_tax_rate
            equity *= 1.0 - cost
            current_weights = pending_weights
            base_weights = pending_weights.copy()
            if pending_info is not None:
                position_info = {
                    symbol: pending_info.get(symbol, position_info.get(symbol, {}))
                    for symbol in current_weights
                }
            trades.append(
                {
                    "date": date,
                    "kind": "monthly_rebalance",
                    "turnover": turnover,
                    "sell_turnover": sell_turnover,
                    "cost": cost,
                    "positions": len(current_weights),
                    "total_weight": sum(current_weights.values()),
                }
            )
            previous_weights = current_weights.copy()
            pending_weights = None
            pending_info = None

        day = returns_by_date.get(date)
        price_return = 0.0
        dividend_return = 0.0
        if day is not None and current_weights:
            ret_map = dict(zip(day["ts_code"], day["ret_1d"]))
            for symbol, weight in current_weights.items():
                price_return += weight * float(ret_map.get(symbol, 0.0))
                dv_ttm = score_lookup.get((date, symbol))
                if pd.notna(dv_ttm):
                    dividend_return += weight * float(dv_ttm) / 100.0 / 252.0
        daily_return = price_return + dividend_return
        equity *= 1.0 + daily_return
        rows.append(
            {
                "date": date,
                "equity": equity,
                "daily_return": daily_return,
                "price_return": price_return,
                "dividend_return": dividend_return,
                "positions": len(current_weights),
                "total_weight": sum(current_weights.values()) if current_weights else 0.0,
                "market_regime": current_regime,
            }
        )

        if config.variant in {"v6_t_overlay", "v7_t_overlay_light"} | STYLE_GRID_VARIANTS and day is not None and current_weights:
            day_by_symbol = day.set_index("ts_code", drop=False)
            adjusted = current_weights.copy()
            overlay_turnover = 0.0
            overlay_sell_turnover = 0.0
            overlay_actions = 0
            overlay_profiles: dict[str, int] = {}
            for symbol, base_weight in base_weights.items():
                if symbol not in adjusted or symbol not in day_by_symbol.index:
                    continue
                row = day_by_symbol.loc[symbol]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[-1]
                close = float(row.get("close", np.nan))
                ma20 = float(row.get("ma_20", np.nan))
                ma60 = float(row.get("ma_60", np.nan))
                ma120 = float(row.get("ma_120", np.nan))
                if not np.isfinite(close) or not np.isfinite(ma20) or not np.isfinite(ma60) or not np.isfinite(ma120):
                    continue
                if config.variant in STYLE_GRID_VARIANTS:
                    params = style_grid_parameters(position_info.get(symbol, {}), row, current_regime)
                    core_weight = base_weight * float(params["core_ratio"])
                    sell_signal = close > ma20 * float(params["sell_ma20"]) or close > ma60 * float(params["sell_ma60"])
                    buy_signal = (
                        current_regime != "risk_off"
                        and close >= ma120 * 0.98
                        and (close <= ma20 * float(params["buy_ma20"]) or close <= ma60 * float(params["buy_ma60"]))
                    )
                    if position_info.get(symbol, {}).get("sleeve") == "growth":
                        buy_signal = buy_signal and float(position_info.get(symbol, {}).get("trend_score", 0.0) or 0.0) >= 60
                    profile = str(params["profile"])
                else:
                    core_weight = base_weight * (0.80 if config.variant == "v7_t_overlay_light" else 0.70)
                    sell_signal = (close > ma20 * (1.12 if config.variant == "v7_t_overlay_light" else 1.08) or close > ma60 * (1.18 if config.variant == "v7_t_overlay_light" else 1.14))
                    buy_signal = (
                        current_regime != "risk_off"
                        and close >= ma120 * 0.98
                        and (close <= ma20 * (1.015 if config.variant == "v7_t_overlay_light" else 1.01) or close <= ma60 * 1.03)
                    )
                    profile = config.variant
                full_weight = base_weight
                current_weight = adjusted.get(symbol, 0.0)
                new_weight = current_weight
                if sell_signal and current_weight > core_weight:
                    new_weight = core_weight
                elif buy_signal and current_weight < full_weight:
                    new_weight = full_weight
                if abs(new_weight - current_weight) > 1e-9:
                    delta = abs(new_weight - current_weight)
                    overlay_turnover += delta
                    overlay_sell_turnover += max(current_weight - new_weight, 0.0)
                    adjusted[symbol] = new_weight
                    overlay_actions += 1
                    overlay_profiles[profile] = overlay_profiles.get(profile, 0) + 1
            if overlay_actions:
                cost = overlay_turnover * config.commission_rate + overlay_sell_turnover * config.stamp_tax_rate
                equity *= 1.0 - cost
                current_weights = adjusted
                previous_weights = current_weights.copy()
                trades.append(
                    {
                        "date": date,
                        "kind": "t_overlay",
                        "turnover": overlay_turnover,
                        "sell_turnover": overlay_sell_turnover,
                        "cost": cost,
                        "positions": len(current_weights),
                        "total_weight": sum(current_weights.values()),
                        "actions": overlay_actions,
                        "profiles": json.dumps(overlay_profiles, ensure_ascii=False),
                    }
                )

        if date in target_map:
            target_frame = target_map[date]
            pending_weights = build_staged_pending_weights(target_frame, current_weights, config)
            info_columns = [
                "ts_code",
                "sleeve",
                "dividend_score",
                "quality_score",
                "value_score",
                "trend_score",
                "risk_score",
                "growth_score",
                "growth_rank_score",
                "compounder_score",
                "market_trend_strength",
                "market_overheat",
                "market_pullback_warning",
                "analyst_negative_warning",
                "long_score",
            ]
            pending_info = (
                target_frame[[col for col in info_columns if col in target_frame.columns]]
                .set_index("ts_code")
                .to_dict(orient="index")
            )
            if "market_regime" in target_frame.columns and not target_frame["market_regime"].dropna().empty:
                current_regime = str(target_frame["market_regime"].dropna().iloc[0])

    return pd.DataFrame(rows), pd.DataFrame(trades)


def load_benchmark(start: pd.Timestamp, end: pd.Timestamp | None) -> pd.DataFrame:
    if not INDEX_300_PATH.exists():
        return pd.DataFrame()
    index = pd.read_parquet(INDEX_300_PATH)
    index["date"] = pd.to_datetime(index["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    index = index.dropna(subset=["date"]).sort_values("date")
    index = index[index["date"] >= start]
    if end is not None:
        index = index[index["date"] <= end]
    index["benchmark_return"] = pd.to_numeric(index["close"], errors="coerce").pct_change().fillna(0.0)
    index["benchmark_equity"] = (1.0 + index["benchmark_return"]).cumprod()
    return index[["date", "benchmark_return", "benchmark_equity"]].copy()


def max_drawdown(equity: pd.Series) -> tuple[float, pd.Timestamp | None, pd.Timestamp | None]:
    if equity.empty:
        return 0.0, None, None
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    trough_idx = drawdown.idxmin()
    peak_idx = equity.loc[:trough_idx].idxmax()
    return float(drawdown.loc[trough_idx]), None, None


def summarize(equity: pd.DataFrame, trades: pd.DataFrame, benchmark: pd.DataFrame) -> dict:
    if equity.empty:
        return {}
    start_date = equity["date"].min()
    end_date = equity["date"].max()
    years = max((end_date - start_date).days / 365.25, 1 / 365.25)
    final_equity = float(equity["equity"].iloc[-1])
    annual_return = final_equity ** (1 / years) - 1
    mdd, _, _ = max_drawdown(equity["equity"])
    daily_ret = equity["daily_return"]
    sharpe = np.nan
    if daily_ret.std() > 0:
        sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252)
    bench_final = np.nan
    bench_annual = np.nan
    bench_mdd = np.nan
    if not benchmark.empty:
        bench_final = float(benchmark["benchmark_equity"].iloc[-1])
        bench_annual = bench_final ** (1 / years) - 1
        bench_mdd, _, _ = max_drawdown(benchmark["benchmark_equity"])
    return {
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "years": years,
        "final_equity": final_equity,
        "annual_return": annual_return,
        "max_drawdown": mdd,
        "sharpe": sharpe,
        "price_return_sum": float(equity["price_return"].sum()),
        "dividend_return_sum": float(equity["dividend_return"].sum()),
        "avg_positions": float(equity["positions"].mean()),
        "avg_total_weight": float(equity["total_weight"].mean()),
        "rebalance_count": int(len(trades)),
        "avg_turnover": float(trades["turnover"].mean()) if not trades.empty else 0.0,
        "benchmark_final_equity": bench_final,
        "benchmark_annual_return": bench_annual,
        "benchmark_max_drawdown": bench_mdd,
    }


def write_report(
    config: BacktestConfig,
    coverage: dict,
    summary: dict,
    targets: pd.DataFrame,
    equity: pd.DataFrame,
    trades: pd.DataFrame,
) -> None:
    report_dir = REPORT_DIR / config.variant
    report_dir.mkdir(parents=True, exist_ok=True)
    targets.to_csv(report_dir / "l1_dividend_quality_targets.csv", index=False)
    equity.to_csv(report_dir / "l1_dividend_quality_equity.csv", index=False)
    trades.to_csv(report_dir / "l1_dividend_quality_rebalances.csv", index=False)
    (report_dir / "l1_dividend_quality_summary.json").write_text(
        json.dumps({"config": config.__dict__, "coverage": coverage, "summary": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    missing_note = ""
    if coverage.get("first_trade_date") and coverage["first_trade_date"] > config.start:
        missing_note = (
            f"\n注意：本地 daily_basic 最早日期为 {coverage['first_trade_date']}，"
            f"早于该日期的区间无法执行股息率/估值过滤。"
        )

    top_latest = pd.DataFrame()
    if not targets.empty:
        latest_date = targets["rebalance_date"].max()
        top_latest = targets[targets["rebalance_date"] == latest_date].head(15)

    lines = [
        "# L1 股息质量稳健版回测报告",
        "",
        f"策略变体：`{config.variant}`",
        f"请求起始日期：{config.start}",
        f"实际回测区间：{summary.get('start_date')} 至 {summary.get('end_date')}",
        missing_note,
        "",
        "## 数据覆盖",
        "",
        f"- daily_basic 来源：`{coverage.get('source_dir')}`",
        f"- daily_basic 交易日数量：{coverage.get('loaded_trade_dates')}",
        f"- daily_basic 覆盖：{coverage.get('first_trade_date')} 至 {coverage.get('last_trade_date')}",
        f"- analyst forecast 覆盖：{coverage.get('report_rc', {}).get('symbols', 0)} 只股票，"
        f"{coverage.get('report_rc', {}).get('rows', 0)} 行，"
        f"{coverage.get('report_rc', {}).get('first_report_date', '-')} 至 "
        f"{coverage.get('report_rc', {}).get('last_report_date', '-')}",
        "",
        "## 绩效摘要",
        "",
        f"- 最终净值：{summary.get('final_equity', np.nan):.4f}",
        f"- 年化收益：{summary.get('annual_return', np.nan):.2%}",
        f"- 最大回撤：{summary.get('max_drawdown', np.nan):.2%}",
        f"- 日度 Sharpe：{summary.get('sharpe', np.nan):.2f}",
        f"- 平均持股数：{summary.get('avg_positions', np.nan):.2f}",
        f"- 平均仓位：{summary.get('avg_total_weight', np.nan):.2%}",
        f"- 调仓次数：{summary.get('rebalance_count', 0)}",
        f"- 平均换手：{summary.get('avg_turnover', 0):.2%}",
        f"- 价格收益贡献粗略和：{summary.get('price_return_sum', np.nan):.2%}",
        f"- 股息收益贡献近似和：{summary.get('dividend_return_sum', np.nan):.2%}",
        "",
        "## 沪深300对照",
        "",
        f"- 沪深300最终净值：{summary.get('benchmark_final_equity', np.nan):.4f}",
        f"- 沪深300年化收益：{summary.get('benchmark_annual_return', np.nan):.2%}",
        f"- 沪深300最大回撤：{summary.get('benchmark_max_drawdown', np.nan):.2%}",
        "",
        "## 最新一期持仓候选",
        "",
    ]
    if top_latest.empty:
        lines.append("最新一期没有满足谨慎买入条件的候选。")
    else:
        lines.append(top_latest.to_markdown(index=False))
    lines.extend(
        [
            "",
            "## 解释限制",
            "",
            "- v6/v7 已实现日级做T状态机；其他版本为长期核心仓或月度成长袖珍仓回测。",
            "- 股息收益使用 `dv_ttm / 252` 做近似日收益补偿，用于弥补非全收益价格序列的分红缺口。",
            "- 财务指标使用 `ann_date <= signal_date` 的 as-of 合并，避免直接使用未来财报。",
            "- 券商研报预测使用 `report_date <= signal_date` 且近 180 天窗口聚合，避免使用未来发布的预测。",
            "- v34 使用每个信号日可见的 daily_basic 股票池，不再用全周期预筛股票宇宙。",
            "- 当前回测以月度再平衡为主，执行价近似为下一交易日收益开始前的权重切换；尚未模拟分钟级做T和真实 T+1 执行约束。",
        ]
    )
    (report_dir / "l1_dividend_quality_backtest.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest L1 long-term dividend quality strategy.")
    parser.add_argument("--start", default="20130101")
    parser.add_argument("--end", default=None)
    parser.add_argument(
        "--variant",
        choices=[
            "baseline",
            "v2_quality_cashflow",
            "v3_stateful_hold",
            "v4_market_regime",
            "v5_industry_cap",
            "v6_t_overlay",
            "v7_t_overlay_light",
            "v8_growth_sleeve",
            "v9_growth_sleeve_capped",
            "v10_bluechip_growth_sleeve",
            "v11_analyst_growth_sleeve",
            "v12_quality_growth_sleeve",
            "v13_mega_quality_growth_sleeve",
            "v14_selective_mega_growth_sleeve",
            "v15_analyst_quality_rank_sleeve",
            "v16_mega_rank_growth_sleeve",
            "v17_staged_entry_sleeve",
            "v18_style_grid_overlay_sleeve",
            "v19_style_grid_full_entry_sleeve",
            "v20_fast_staged_grid_sleeve",
            "v21_forecast_dual_score_sleeve",
            "v22_forecast_dual_score_grid_sleeve",
            "v23_forecast_rank_grid_sleeve",
            "v24_forecast_tiebreak_grid_sleeve",
            "v25_forecast_guardrail_grid_sleeve",
            "v26_market_trend_compounder_sleeve",
            "v27_cautious_compounder_sleeve",
            "v28_overheat_guarded_compounder_sleeve",
            "v29_overheat_throttle_grid_sleeve",
            "v30_concentrated_trend_sleeve",
            "v31_bull_bear_exposure_sleeve",
            "v32_empty_bear_sleeve",
            "v33_bull_boost_defensive_bear_sleeve",
            "v34_pit_universe_guarded_sleeve",
            "v35_pit_universe_riskon_recovery_sleeve",
            "v36_tea_master_concentrated",
            "v37_tea_master_core_satellite",
            "v38_tea_master_balanced",
            "v39_tea_master_optimized",
            "v40_tea_master_regime_grid",
            "v41_tea_master_pit_guarded",
            "v42_tea_master_growth_throttle",
            "v43_tea_master_core_only",
            "v44_tea_master_defensive_neutral",
            "v45_tea_master_riskon_quality",
        ],
        default="baseline",
    )
    args = parser.parse_args()

    config = BacktestConfig(variant=args.variant, start=args.start, end=args.end)
    requested_start = parse_date(config.start)
    requested_end = parse_date(config.end)
    if requested_start is None:
        raise ValueError("--start must use YYYYMMDD")

    print("loading stock metadata", flush=True)
    stock_basic = load_stock_basic()
    print("loading daily_basic monthly features", flush=True)
    daily_basic, coverage = load_daily_basic_monthly(requested_start, requested_end)
    if daily_basic.empty:
        raise RuntimeError("No daily_basic data found; dividend strategy cannot be backtested.")
    if config.variant in PIT_UNIVERSE_VARIANTS:
        candidate_symbols = None
        coverage["point_in_time_universe"] = True
        print("prefilter candidate symbols: disabled for point-in-time universe", flush=True)
    else:
        candidate_symbols = select_candidate_symbols_from_daily_basic(daily_basic, stock_basic, config)
        coverage["point_in_time_universe"] = False
        print(f"prefilter candidate symbols: {len(candidate_symbols)}", flush=True)
    print("loading daily monthly features", flush=True)
    daily_features, daily_returns = load_daily_monthly_features(
        requested_start,
        requested_end,
        stock_basic,
        candidate_symbols=candidate_symbols,
    )
    executable_start_text = max(config.start, coverage.get("first_trade_date") or config.start)
    executable_start = parse_date(executable_start_text)
    daily_features = daily_features[daily_features["date"] >= executable_start].copy()
    daily_returns = daily_returns[daily_returns["date"] >= executable_start].copy()
    daily_basic = daily_basic[daily_basic["date"] >= executable_start].copy()
    daily_features = daily_features.sort_values(["date", "ts_code", "trade_date"]).drop_duplicates(
        ["date", "ts_code"],
        keep="last",
    )
    daily_returns = daily_returns.sort_values(["date", "ts_code", "trade_date"]).drop_duplicates(
        ["date", "ts_code"],
        keep="last",
    )
    daily_basic = daily_basic.sort_values(["date", "ts_code", "trade_date"]).drop_duplicates(
        ["date", "ts_code"],
        keep="last",
    )
    if config.variant in PIT_UNIVERSE_VARIANTS:
        before_rows = len(daily_basic)
        daily_basic = filter_daily_basic_point_in_time(daily_basic, config)
        coverage["point_in_time_universe_rows_before"] = int(before_rows)
        coverage["point_in_time_universe_rows_after"] = int(len(daily_basic))

    print("merging daily_basic", flush=True)
    merged = daily_features.merge(
        daily_basic.drop(columns=["trade_date"]),
        on=["date", "ts_code"],
        how="inner",
    )
    if merged.empty:
        raise RuntimeError("daily and daily_basic have no overlapping monthly rows.")

    print("merging financial indicators as-of", flush=True)
    merged = load_financial_asof(merged)
    if config.variant in GROWTH_VARIANTS:
        print("merging analyst forecasts as-of", flush=True)
        merged = load_analyst_forecast_asof(merged)
        coverage["report_rc"] = get_analyst_forecast_coverage()
    else:
        merged = add_empty_analyst_forecast_columns(merged)
    if config.variant in MARKET_REGIME_VARIANTS:
        print("merging market regime", flush=True)
        market_regime = load_market_regime(merged["date"].min(), merged["date"].max())
        if market_regime.empty:
            merged["market_regime"] = "neutral"
            merged["index_ma_120_slope_20d"] = np.nan
            merged["index_return_20d"] = np.nan
            merged["index_return_60d"] = np.nan
            merged["index_return_120d"] = np.nan
            merged["index_drawdown_60d"] = np.nan
            merged["index_overheat"] = False
        else:
            merged = merged.merge(market_regime, on="date", how="left")
            merged["market_regime"] = merged["market_regime"].fillna("neutral")
    if config.variant == "v2_quality_cashflow":
        print("merging cashflow quality as-of", flush=True)
        merged = load_cashflow_quality_asof(merged)
    else:
        merged["cashflow_quality"] = np.nan
    print("building scores", flush=True)
    scored = build_scores(merged, config)
    targets = make_monthly_targets(scored, config)
    if targets.empty:
        raise RuntimeError("No candidates passed L1 dividend quality filters.")

    print("running portfolio backtest", flush=True)
    equity, trades = run_portfolio_backtest(daily_returns, targets, scored, config)
    benchmark = load_benchmark(equity["date"].min(), equity["date"].max())
    summary = summarize(equity, trades, benchmark)
    write_report(config, coverage, summary, targets, equity, trades)
    print(json.dumps({"coverage": coverage, "summary": summary, "report_dir": str(REPORT_DIR / config.variant)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
