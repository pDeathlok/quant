#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
B1 entry/exit grid analysis.

Signal date is T+0. The simulation buys at T+1 open, evaluates exits from
T+2 onward, and sells at the configured expiry close if no earlier exit fires.
When an intraday stop and take-profit can both be touched on the same day, the
conservative assumption is stop first.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd

import build_training_data_parallel as btd
from quant.data import MarketDataStore, MarketDataStoreConfig, read_partitioned_symbol_file
from quant.features.variable_library import (
    EXTRA_FEATURE_COLUMNS,
    calc_bbi as project_calc_bbi,
    calculate_project_extra_features,
    merge_daily_basic_features,
)
from quant.research.b1_backtest import (
    ExitRule,
    add_future_prices,
    max_drawdown_from_daily_returns,
    simulate_exit,
    summarize_returns,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT_ROOT / "data/features/b1/candidates_strict_no_volume_20240101.parquet"
DEFAULT_DAILY_DIR = PROJECT_ROOT / "data/raw/daily"
DEFAULT_DAILY_BASIC_DIR = PROJECT_ROOT / "data/raw/daily_basic"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports/b1/research"
DEFAULT_CANDIDATE_CACHE = PROJECT_ROOT / "data/features/b1/candidates_strict_no_volume_20240101.parquet"
DEFAULT_FEATURE_CACHE = PROJECT_ROOT / "data/features/b1/training_xgb_project_vars.parquet"

FORMAL_MODEL_DIR = Path(
    os.getenv("B1_FORMAL_MODEL_DIR", str(PROJECT_ROOT / "models/production/b1"))
)


def _formal_model_path(primary: str, legacy: str | None = None) -> Path:
    primary_path = FORMAL_MODEL_DIR / primary
    if primary_path.exists() or legacy is None:
        return primary_path
    return FORMAL_MODEL_DIR / legacy


MODEL_PATHS = {
    "up5_es": _formal_model_path("up5_es.joblib"),
    "up8_es": _formal_model_path("up8_es.joblib"),
    "up10": _formal_model_path("up10_es.joblib", "up10.joblib"),
    "down2_es": _formal_model_path("down2_es.joblib"),
    "down3_es": _formal_model_path("down3_es.joblib"),
}

MODEL_SELECTED_FEATURES: set[str] | None = None


@dataclass(frozen=True)
class EntryRule:
    name: str
    score_col: str | None = None
    top_quantile: float | None = None
    up8_quantile: float | None = None
    up10_quantile: float | None = None
    down3_max_quantile: float | None = None
    max_per_day: int | None = None
    min_score: float | None = None
    min_up8: float | None = None
    min_up10: float | None = None
    max_down3: float | None = None


def load_candidates(path: Path, start_date: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= pd.to_datetime(start_date)].copy()
    if "label_b1_signal" in df.columns:
        df = df[df["label_b1_signal"] == 1].copy()
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


def load_stock_basic() -> pd.DataFrame:
    for path in [PROJECT_ROOT / "data/raw/stock_basic.parquet", PROJECT_ROOT / "data/cache/tushare_stock_basic_all.parquet"]:
        if path.exists():
            return pd.read_parquet(path)
    return pd.DataFrame(columns=["ts_code", "name", "industry", "market"])


def process_strict_b1_no_volume_file(args: tuple[str, str, dict[str, dict]]) -> pd.DataFrame | None:
    path_str, start_date, meta_by_ts_code = args
    path = Path(path_str)
    try:
        df = read_partitioned_symbol_file(path)
        return process_strict_b1_no_volume_frame((path.stem, df, start_date, meta_by_ts_code))
    except Exception as exc:
        print(f"  skip {path.name}: {exc}")
        return None


def process_strict_b1_no_volume_frame(
    args: tuple[str, pd.DataFrame, str, dict[str, dict]],
) -> pd.DataFrame | None:
    symbol, source_frame, start_date, meta_by_ts_code = args
    try:
        df = source_frame.copy()
        if "trade_date" in df.columns:
            df["date"] = pd.to_datetime(
                df["trade_date"].astype(str), format="%Y%m%d", errors="coerce"
            )
        else:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if "vol" in df.columns and "volume" not in df.columns:
            df = df.rename(columns={"vol": "volume"})
        if "ts_code" not in df.columns:
            df["ts_code"] = symbol
        df["symbol"] = str(symbol)
        df = df.sort_values("date").reset_index(drop=True)
        start_ts = pd.to_datetime(start_date)
        history_start = start_ts - pd.Timedelta(days=400)
        df = df[df["date"] >= history_start].reset_index(drop=True)
        if len(df) < 130:
            return None

        meta = meta_by_ts_code.get(str(symbol), {})
        for col in ["name", "industry", "market"]:
            if col in meta:
                df[col] = meta[col]

        name = str(df["name"].iloc[0]) if "name" in df.columns and len(df) else ""
        if "ST" in name.upper() or "退" in name:
            return None

        factors = calculate_minimal_model_features(df)
        result = pd.concat([df, factors], axis=1)
        pct_change = result["pct_chg"] if "pct_chg" in result.columns else result["close"].pct_change() * 100
        amplitude = (result["high"] - result["low"]) / result["low"] * 100
        bbi = project_calc_bbi(result["close"])
        signal_kdj_j = result["kdj_d_j"] if "kdj_d_j" in result.columns else btd.KDJ().compute(result)["J"]
        strict_no_volume = (
            (pct_change >= -2)
            & (pct_change <= 2)
            & (amplitude < 7)
            & (bbi > result["ma_60"])
            & (signal_kdj_j < 0)
        )
        result["label_b1_signal"] = strict_no_volume.astype(int)
        result = result[(result["date"] >= start_ts) & strict_no_volume].copy()
        return result if len(result) > 0 else None
    except Exception as exc:
        print(f"  skip {symbol}: {exc}")
        return None


def build_strict_b1_no_volume_candidates(
    daily_dir: Path,
    start_date: str,
    cache_path: Path | None = None,
    max_workers: int = 8,
    executor_type: str = "threads",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Derive formal candidates from the single unified B1 feature cache."""
    start_ts = pd.to_datetime(start_date)

    store = MarketDataStore(MarketDataStoreConfig.from_env(root=daily_dir.parent))
    market_latest = store.latest_dataset_trade_date(daily_dir.name)
    if DEFAULT_FEATURE_CACHE.exists():
        features = pd.read_parquet(DEFAULT_FEATURE_CACHE)
        features["date"] = pd.to_datetime(features["date"])
        feature_latest = features["date"].max() if not features.empty else pd.NaT
        if market_latest is None or (pd.notna(feature_latest) and feature_latest >= market_latest):
            derived = features[features["date"] >= start_ts].copy()
            # Legacy production models still name these two equivalent inputs
            # differently from the unified feature library.
            if "kdj_j" not in derived and "kdj_d_j" in derived:
                derived["kdj_j"] = derived["kdj_d_j"]
            if "turnover_ratio" not in derived and "volume_relative_60d" in derived:
                derived["turnover_ratio"] = derived["volume_relative_60d"]
            derived = derived.sort_values(["symbol", "date"]).reset_index(drop=True)
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                derived.to_parquet(cache_path, index=False)
                print(f"Candidate cache derived from unified B1 feature cache: {cache_path}")
            return derived
    raise RuntimeError(
        "Unified B1 feature cache is missing or stale; run refresh_b1_feature_cache.py "
        "before building formal candidates. Candidate-local factor rebuilding is disabled "
        "to prevent feature-definition drift."
    )


def predict_models(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for pred_col, model_path in MODEL_PATHS.items():
        model = joblib.load(model_path)
        feature_cols = list(model.feature_names_in_)
        if getattr(model, "selected_features_", None) is not None:
            selected = list(model.selected_features_)
        else:
            selected = [
                col
                for col, keep in zip(feature_cols, model.named_steps["feature_selection"].get_support())
                if keep
            ]
        missing_selected = [col for col in selected if col not in out.columns]
        if missing_selected:
            raise ValueError(f"{model_path} missing selected feature columns: {missing_selected[:10]}")
        for col in feature_cols:
            if col not in out.columns:
                out[col] = 0.0
        X = out[feature_cols].replace([np.inf, -np.inf], np.nan)
        pred = pd.Series(np.nan, index=out.index, dtype=float)
        if hasattr(model, "imputer") or "imputer" in getattr(model, "named_steps", {}):
            pred.loc[X.index] = model.predict_proba(X)[:, 1]
        else:
            valid = ~X.isna().any(axis=1)
            pred.loc[valid] = model.predict_proba(X.loc[valid])[:, 1]
        out[f"pred_{pred_col}"] = pred

    # Keep both the legacy formal-dashboard name and the unified research name.
    # Downstream strategy calibration uses the explicit ``_es`` suffix.
    out["pred_up10_es"] = out["pred_up10"]
    out["entry_score"] = (
        0.60 * out["pred_up8_es"]
        + 0.30 * out["pred_up10"]
        - 0.35 * out["pred_down3_es"]
    )
    required_predictions = [f"pred_{name}" for name in MODEL_PATHS]
    return out.dropna(subset=[*required_predictions, "entry_score"]).copy()


def get_model_selected_features() -> set[str]:
    global MODEL_SELECTED_FEATURES
    if MODEL_SELECTED_FEATURES is not None:
        return MODEL_SELECTED_FEATURES
    selected: set[str] = set()
    for model_path in MODEL_PATHS.values():
        model = joblib.load(model_path)
        if getattr(model, "selected_features_", None) is not None:
            selected.update(model.selected_features_)
        else:
            feature_cols = list(model.feature_names_in_)
            support = model.named_steps["feature_selection"].get_support()
            selected.update(col for col, keep in zip(feature_cols, support) if keep)
    MODEL_SELECTED_FEATURES = selected
    return selected


def calculate_minimal_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate only features selected by the fitted model pipelines."""
    selected = get_model_selected_features()
    factors = pd.DataFrame(index=df.index)
    close = df["close"]
    extra_feature_names = set(EXTRA_FEATURE_COLUMNS)
    if selected & extra_feature_names:
        extra = calculate_project_extra_features(df)
        for col in selected & extra_feature_names:
            if col in extra.columns:
                factors[col] = extra[col]

    for window in [5, 10, 20, 60, 120]:
        col = f"ma_{window}"
        if col in selected:
            factors[col] = btd.MA(window).compute(df)
    for window in [5, 10, 20]:
        col = f"ema_{window}"
        if col in selected:
            factors[col] = btd.EMA(window).compute(df)

    if "rsi_12" in selected:
        factors["rsi_12"] = btd.RSI(12).compute(df)
    if "kdj_j" in selected:
        factors["kdj_j"] = btd.KDJ().compute(df)["J"]

    boll_needed = {"bb_upper", "bb_lower"} & selected
    if boll_needed:
        bb = btd.BollingerBands().compute(df)
        factors["bb_upper"] = bb.iloc[:, 0]
        factors["bb_lower"] = bb.iloc[:, 2]

    if "atr_14" in selected:
        factors["atr_14"] = btd.ATR(14).compute(df)
    if "cci" in selected:
        factors["cci"] = btd.CCI().compute(df)
    for window in [6, 12, 24]:
        col = f"bias_{window}"
        if col in selected:
            factors[col] = btd.BIAS(window).compute(df)
    if "obv" in selected:
        factors["obv"] = btd.OBV().compute(df)
    if "psy_24" in selected:
        factors["psy_24"] = btd.PSY(24).compute(df)
    if "mass_index" in selected:
        factors["mass_index"] = btd.MassIndex().compute(df)
    if "parabolic_sar" in selected:
        factors["parabolic_sar"] = btd.ParabolicSAR().compute(df)

    vortex_needed = {"vortex_plus", "vortex_minus"} & selected
    if vortex_needed:
        vortex = btd.VortexIndicator().compute(df)
        factors["vortex_plus"] = vortex.iloc[:, 0]
        factors["vortex_minus"] = vortex.iloc[:, 1]

    kc_needed = {"keltner_upper", "keltner_lower", "keltner_width"} & selected
    if kc_needed:
        kc = btd.KeltnerChannel().compute(df)
        factors["keltner_upper"] = kc.iloc[:, 0]
        factors["keltner_lower"] = kc.iloc[:, 1]
        factors["keltner_width"] = (kc.iloc[:, 0] - kc.iloc[:, 1]) / kc.iloc[:, 2]

    for window in [1, 20]:
        col = f"amplitude_{window}"
        if col in selected:
            factors[col] = btd.Amplitude(window).compute(df)

    alpha_map = {
        "alpha003": btd.Alpha003Factor,
        "alpha004": btd.Alpha004Factor,
        "alpha005": btd.Alpha005Factor,
        "alpha006": btd.Alpha006Factor,
        "alpha009": btd.Alpha009Factor,
    }
    for col, cls in alpha_map.items():
        if col in selected:
            factors[col] = cls().compute(df)

    alpha191_map = {
        "alpha191_01": btd.Alpha191_01Factor,
        "alpha191_02": btd.Alpha191_02Factor,
        "alpha191_03": btd.Alpha191_03Factor,
        "alpha191_06": btd.Alpha191_06Factor,
        "alpha191_07": btd.Alpha191_07Factor,
        "alpha191_09": btd.Alpha191_09Factor,
        "alpha191_11": btd.Alpha191_11Factor,
        "alpha191_12": btd.Alpha191_12Factor,
        "alpha191_13": btd.Alpha191_13Factor,
        "alpha191_15": btd.Alpha191_15Factor,
    }
    for col, cls in alpha191_map.items():
        if col in selected:
            factors[col] = cls().compute(df)

    for window in [1, 5, 10, 60, 120]:
        col = f"return_{window}d"
        if col in selected:
            factors[col] = btd.ReturnFactor(window).compute(df)
    for window in [5, 20, 60]:
        col = f"momentum_{window}d"
        if col in selected:
            factors[col] = btd.MomentumSkip5Factor(window).compute(df)
    if "reversal_5d" in selected:
        factors["reversal_5d"] = btd.ReversalFactor(5).compute(df)
    for window in [20, 60]:
        col = f"volatility_{window}d"
        if col in selected:
            factors[col] = btd.Volatility(window).compute(df)
        col = f"downside_volatility_{window}d"
        if col in selected:
            factors[col] = btd.DownsideVolatility(window).compute(df)

    if "close" in selected:
        factors["close"] = close
    if "price_log" in selected:
        factors["price_log"] = np.log(close + 1)
    if "turnover_ratio" in selected:
        factors["turnover_ratio"] = df["volume"] / (df["volume"].rolling(60).mean() + 1)
    if "volume_relative_60d" in selected and "volume_relative_60d" not in factors.columns:
        factors["volume_relative_60d"] = df["volume"] / df["volume"].rolling(60).mean().replace(0, np.nan)

    return factors


def _daily_rank_pct(df: pd.DataFrame, col: str, ascending: bool) -> pd.Series:
    return df.groupby("date", group_keys=False)[col].rank(pct=True, ascending=ascending)


def add_entry_ranks(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rank_score_desc"] = _daily_rank_pct(out, "entry_score", ascending=True)
    out["rank_up8_desc"] = _daily_rank_pct(out, "pred_up8_es", ascending=True)
    out["rank_up10_desc"] = _daily_rank_pct(out, "pred_up10", ascending=True)
    out["rank_down3_asc"] = _daily_rank_pct(out, "pred_down3_es", ascending=True)
    return out


def build_rank_entry_rules() -> list[EntryRule]:
    return [
        EntryRule("B1_all"),
        EntryRule("up8_top30pct", up8_quantile=0.70),
        EntryRule("up8_top20pct_risk_low70pct", up8_quantile=0.80, down3_max_quantile=0.70),
        EntryRule("score_top30pct", score_col="entry_score", top_quantile=0.70),
        EntryRule("score_top20pct", score_col="entry_score", top_quantile=0.80),
        EntryRule("score_top10pct", score_col="entry_score", top_quantile=0.90),
        EntryRule("score_top20pct_risk_low70pct", score_col="entry_score", top_quantile=0.80, down3_max_quantile=0.70),
        EntryRule(
            "score_top20pct_up10_top50pct_risk_low70pct",
            score_col="entry_score",
            top_quantile=0.80,
            up10_quantile=0.50,
            down3_max_quantile=0.70,
        ),
        EntryRule("score_top10_per_day", score_col="entry_score", max_per_day=10),
        EntryRule("score_top5_per_day", score_col="entry_score", max_per_day=5),
    ]


def build_threshold_entry_rules() -> list[EntryRule]:
    return [
        EntryRule("B1_all"),
        EntryRule("score_ge_0.00", min_score=0.00),
        EntryRule("score_ge_0.05", min_score=0.05),
        EntryRule("score_ge_0.10", min_score=0.10),
        EntryRule("score_ge_0.15", min_score=0.15),
        EntryRule("score_ge_0.20", min_score=0.20),
        EntryRule("up8_ge_0.45", min_up8=0.45),
        EntryRule("up8_ge_0.50", min_up8=0.50),
        EntryRule("up8_ge_0.55", min_up8=0.55),
        EntryRule("up8_ge_0.50_down3_le_0.55", min_up8=0.50, max_down3=0.55),
        EntryRule("score_ge_0.10_down3_le_0.55", min_score=0.10, max_down3=0.55),
        EntryRule("score_ge_0.15_down3_le_0.55", min_score=0.15, max_down3=0.55),
        EntryRule("score_ge_0.10_up10_ge_0.20_down3_le_0.60", min_score=0.10, min_up10=0.20, max_down3=0.60),
        EntryRule("score_ge_0.15_up10_ge_0.20_down3_le_0.60", min_score=0.15, min_up10=0.20, max_down3=0.60),
    ]


def apply_entry_rule(df: pd.DataFrame, rule: EntryRule) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    if rule.top_quantile is not None:
        mask &= df["rank_score_desc"] >= rule.top_quantile
    if rule.up8_quantile is not None:
        mask &= df["rank_up8_desc"] >= rule.up8_quantile
    if rule.up10_quantile is not None:
        mask &= df["rank_up10_desc"] >= rule.up10_quantile
    if rule.down3_max_quantile is not None:
        mask &= df["rank_down3_asc"] <= rule.down3_max_quantile
    if rule.max_per_day is not None:
        daily_rank = df.groupby("date")[rule.score_col or "entry_score"].rank(method="first", ascending=False)
        mask &= daily_rank <= rule.max_per_day
    if rule.min_score is not None:
        mask &= df["entry_score"] >= rule.min_score
    if rule.min_up8 is not None:
        mask &= df["pred_up8_es"] >= rule.min_up8
    if rule.min_up10 is not None:
        mask &= df["pred_up10"] >= rule.min_up10
    if rule.max_down3 is not None:
        mask &= df["pred_down3_es"] <= rule.max_down3
    return mask


def build_exit_rules() -> list[ExitRule]:
    rules: list[ExitRule] = []
    for hold_days in [3, 5, 8]:
        rules.append(ExitRule(f"expiry_T{hold_days + 1}_close", "expiry", hold_days))

    for hold_days in [5, 8]:
        for tp in [0.05, 0.08, 0.10]:
            for sl in [0.02, 0.03, 0.04]:
                rules.append(ExitRule(f"fixed_tp{tp:.0%}_sl{sl:.0%}_T{hold_days + 1}", "fixed", hold_days, tp, sl))

    for hold_days in [5, 8]:
        for target in [0.05, 0.08, 0.10]:
            for trail in [0.02, 0.03, 0.05]:
                for sl in [0.02, 0.03]:
                    rules.append(
                        ExitRule(
                            f"trail_target{target:.0%}_dd{trail:.0%}_sl{sl:.0%}_T{hold_days + 1}",
                            "trailing",
                            hold_days,
                            target,
                            sl,
                            trail,
                        )
                    )
    return rules


def read_daily_file(daily_dir: Path, symbol: str) -> pd.DataFrame | None:
    path = daily_dir / f"{symbol}.parquet"
    df = read_partitioned_symbol_file(path)
    if df.empty:
        return None
    if "trade_date" in df.columns:
        df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    else:
        df["date"] = pd.to_datetime(df["date"])
    if "vol" in df.columns and "volume" not in df.columns:
        df = df.rename(columns={"vol": "volume"})
    df["symbol"] = symbol
    return df.sort_values("date").reset_index(drop=True)


def evaluate_grid(df: pd.DataFrame, entry_rules: Iterable[EntryRule], exit_rules: Iterable[ExitRule]) -> pd.DataFrame:
    rows = []
    periods = {
        "2024_test": (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31")),
        "oot_2025plus": (pd.Timestamp("2025-01-01"), pd.Timestamp.max),
        "all": (df["date"].min(), pd.Timestamp.max),
    }

    for entry_rule in entry_rules:
        entry_mask = apply_entry_rule(df, entry_rule)
        entry_df = df[entry_mask].copy()
        if entry_df.empty:
            continue
        print(f"entry {entry_rule.name}: {len(entry_df):,} candidates")
        for exit_rule in exit_rules:
            trades = simulate_exit(entry_df, exit_rule)
            if trades.empty:
                continue
            for period_name, (start, end) in periods.items():
                p_trades = trades[(trades["date"] >= start) & (trades["date"] <= end)]
                metrics = summarize_returns(p_trades)
                if not metrics:
                    continue
                rows.append({
                    "period": period_name,
                    "entry_rule": entry_rule.name,
                    "exit_rule": exit_rule.name,
                    "exit_kind": exit_rule.kind,
                    "hold_days": exit_rule.hold_days,
                    "take_profit": exit_rule.take_profit,
                    "stop_loss": exit_rule.stop_loss,
                    "trail_drawdown": exit_rule.trail_drawdown,
                    **metrics,
                })
    return pd.DataFrame(rows)


def write_report(summary: pd.DataFrame, output_dir: Path, timestamp: str) -> Path:
    report_path = output_dir / f"b1_entry_exit_grid_report_{timestamp}.md"
    key_cols = [
        "period", "entry_rule", "exit_rule", "trades", "avg_return_pct",
        "win_rate", "daily_sharpe", "max_drawdown_pct", "profit_factor",
        "stop_rate", "take_profit_rate", "trailing_stop_rate", "expiry_rate",
    ]
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# B1买入/卖出时机交叉对比报告\n\n")
        f.write("口径：T+0 产生B1候选，T+1开盘买入，T+2起检查卖出条件；同日同时触发止损/止盈时按先止损处理。\n\n")
        for period in ["2024_test", "oot_2025plus", "all"]:
            part = summary[summary["period"] == period].copy()
            if part.empty:
                continue
            part = part[part["trades"] >= 100]
            f.write(f"## {period} Top 20 by daily_sharpe\n\n")
            top = part.sort_values(["daily_sharpe", "avg_return_pct"], ascending=False).head(20)
            f.write(top[key_cols].to_markdown(index=False, floatfmt=".4f"))
            f.write("\n\n")
            f.write(f"## {period} Top 20 by avg_return_pct\n\n")
            top = part.sort_values(["avg_return_pct", "daily_sharpe"], ascending=False).head(20)
            f.write(top[key_cols].to_markdown(index=False, floatfmt=".4f"))
            f.write("\n\n")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B1 entry/exit grid analysis")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="B1 candidate feature parquet")
    parser.add_argument("--daily-dir", type=Path, default=DEFAULT_DAILY_DIR, help="Raw daily parquet directory")
    parser.add_argument("--daily-basic-dir", type=Path, default=DEFAULT_DAILY_BASIC_DIR, help="Tushare daily_basic parquet directory")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--start-date", default="2024-01-01", help="First signal date to analyze")
    parser.add_argument(
        "--candidate-mode",
        choices=["prebuilt", "strict_no_volume"],
        default="prebuilt",
        help="prebuilt uses --data; strict_no_volume rebuilds strict B1 candidates without the volume condition",
    )
    parser.add_argument(
        "--entry-mode",
        choices=["rank", "threshold"],
        default="rank",
        help="rank uses daily top-N/top-percent rules; threshold uses fixed score/probability thresholds",
    )
    parser.add_argument("--candidate-cache", type=Path, default=None, help="Optional parquet cache for rebuilt candidates")
    parser.add_argument(
        "--force-candidate-refresh",
        action="store_true",
        help="Rewrite the derived candidate cache from the unified feature cache.",
    )
    parser.add_argument(
        "--build-candidates-only",
        action="store_true",
        help="Stop after writing/validating the strict candidate cache.",
    )
    parser.add_argument("--max-workers", type=int, default=8, help="Workers for strict candidate rebuild")
    parser.add_argument(
        "--executor",
        choices=["threads", "processes"],
        default="threads",
        help="Executor for strict candidate rebuild; threads avoid heavy pickle overhead",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.candidate_mode == "strict_no_volume":
        cache_path = args.candidate_cache
        if cache_path is None:
            cache_path = DEFAULT_CANDIDATE_CACHE
        print(f"Building strict B1 candidates without volume condition from {args.daily_dir}")
        candidates = build_strict_b1_no_volume_candidates(
            args.daily_dir,
            args.start_date,
            cache_path,
            args.max_workers,
            args.executor,
            force_refresh=args.force_candidate_refresh,
        )
    else:
        print(f"Loading candidates: {args.data}")
        candidates = load_candidates(args.data, args.start_date)
    print(f"Candidates after {args.start_date}: {len(candidates):,}")
    if args.build_candidates_only:
        print(
            f"Candidate range: {pd.to_datetime(candidates['date']).min():%Y-%m-%d}"
            f" to {pd.to_datetime(candidates['date']).max():%Y-%m-%d}"
        )
        return
    candidates = merge_daily_basic_features(candidates, args.daily_basic_dir)

    print("Predicting entry models...")
    candidates = predict_models(candidates)
    candidates = add_entry_ranks(candidates)
    print(f"Candidates with valid predictions: {len(candidates):,}")
    print("Prediction quantiles:")
    print(candidates[["entry_score", "pred_up8_es", "pred_up10", "pred_down3_es"]].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_string())

    max_hold_days = max(rule.hold_days for rule in build_exit_rules())
    print(f"Adding future prices from {args.daily_dir}...")
    candidates = add_future_prices(candidates, args.daily_dir, max_hold_days)
    print(f"Candidates with future entry prices: {len(candidates):,}")

    print("Evaluating grid...")
    entry_rules = build_threshold_entry_rules() if args.entry_mode == "threshold" else build_rank_entry_rules()
    summary = evaluate_grid(candidates, entry_rules, build_exit_rules())
    csv_path = args.output_dir / f"b1_entry_exit_grid_summary_{timestamp}.csv"
    summary.to_csv(csv_path, index=False)
    report_path = write_report(summary, args.output_dir, timestamp)

    print(f"Summary CSV: {csv_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
