#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Model broad z-skill tactics and backtest model-filtered executions.

This is the z-skill analogue of the B1 model workflow:
- Build a Tushare-only candidate dataset with the project variable library.
- For each broad/high-frequency tactic, train XGBoost models for up5/up8/down3.
- Split train/test by stock code before 2025, reserve 2025+ as OOT.
- Use model probability thresholds as entry filters.
- Cross entry filters with open filters and exit rules, with non-overlap trades.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier
from xgboost.callback import TrainingCallback

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "research"))

import build_training_data_parallel as btd
from analyze_b1_entry_exit_grid import ExitRule, add_future_prices, simulate_exit, summarize_returns
from analyze_b1_xgb_entry_exit_grid import DEFAULT_DAILY_DIR, DEFAULT_OUTPUT_DIR, drop_overlapping_trades
from analyze_z_skill_entry_exit_backtest import OpenFilter, apply_open_filter, build_open_filters
from quant.data.source_merge import normalize_tushare_daily
from quant.features.variable_library import PROJECT_FACTOR_COLUMNS, build_continuous_ohlc, calculate_project_extra_features
from quant.ml.label_maker import create_b1_labels
from quant.ml.xgb_research import XGBResearchModel

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


SIGNAL_CACHE = PROJECT_ROOT / "data/features/z_skill_daily_candidates.parquet"
FAMILY_SIGNAL_CACHE = PROJECT_ROOT / "data/features/b1/b1_family_rule_candidates.parquet"
DATASET_PATH = PROJECT_ROOT / "data/features/z_skill_model_dataset.parquet"
MODEL_DIR = PROJECT_ROOT / "models/research/z_skill"

PRIORITY_SIGNALS = [
    "B2",
    "BREATHING",
    "NANA",
    "YIDONG_DILIAN",
    "KEY_K",
    "GOLDEN_BOWL",
    "DUICHEN_VA",
    "ZAIHOU",
    "YUEYUE",
    "VIOLENCE_K",
]

RESEARCH_SIGNALS = [
    "KEY_K",
    "BREATHING",
    "DUICHEN_VA",
    "YIDONG_DILIAN",
    "ZAIHOU",
    "GOLDEN_BOWL",
    "NANA",
    "YUEYUE",
    "VIOLENCE_K",
    "DOUBLE_GUN",
]

B2_SOURCE_COLUMNS = [
    "b2_any_pchg4_vol15",
    "b2_oversold_pchg3_vol12",
    "b2_bbi_reclaim_vol12",
]

LABELS = {
    "up5": "label_t1_open_max_high_5pct",
    "up8": "label_t1_open_max_high_8pct",
    "down3": "label_t1_open_min_low_3pct_below_t0_low",
}


@dataclass(frozen=True)
class ThresholdEntryRule:
    name: str
    min_up5: float | None = None
    min_up8: float | None = None
    max_down3: float | None = None


class AucGapEarlyStopping(TrainingCallback):
    """Early stop on test AUC with a penalty for train/test AUC divergence."""

    def __init__(self, rounds: int = 35, gap_tolerance: float = 0.035, gap_penalty: float = 0.45):
        self.rounds = rounds
        self.gap_tolerance = gap_tolerance
        self.gap_penalty = gap_penalty
        self.best_score = -np.inf
        self.best_iteration = 0
        self.history: list[dict[str, float]] = []

    def after_iteration(self, model, epoch: int, evals_log: dict) -> bool:
        train_auc = float(evals_log["validation_0"]["auc"][-1])
        test_auc = float(evals_log["validation_1"]["auc"][-1])
        gap = max(0.0, train_auc - test_auc - self.gap_tolerance)
        score = test_auc - self.gap_penalty * gap
        self.history.append(
            {
                "iteration": float(epoch),
                "train_auc": train_auc,
                "test_auc": test_auc,
                "auc_gap": train_auc - test_auc,
                "early_stop_score": score,
            }
        )
        if score > self.best_score:
            self.best_score = score
            self.best_iteration = epoch
        return epoch - self.best_iteration >= self.rounds


def build_entry_rules() -> list[ThresholdEntryRule]:
    return [
        ThresholdEntryRule("model_all"),
        ThresholdEntryRule("up5_ge_0.55", min_up5=0.55),
        ThresholdEntryRule("up5_ge_0.60", min_up5=0.60),
        ThresholdEntryRule("up5_ge_0.65", min_up5=0.65),
        ThresholdEntryRule("up8_ge_0.35", min_up8=0.35),
        ThresholdEntryRule("up8_ge_0.40", min_up8=0.40),
        ThresholdEntryRule("down3_le_0.45", max_down3=0.45),
        ThresholdEntryRule("up5_ge_0.55_down3_le_0.55", min_up5=0.55, max_down3=0.55),
        ThresholdEntryRule("up5_ge_0.60_down3_le_0.50", min_up5=0.60, max_down3=0.50),
        ThresholdEntryRule("up8_ge_0.35_down3_le_0.55", min_up8=0.35, max_down3=0.55),
        ThresholdEntryRule("up8_ge_0.40_down3_le_0.50", min_up8=0.40, max_down3=0.50),
        ThresholdEntryRule("up5_ge_0.55_up8_ge_0.35_down3_le_0.55", min_up5=0.55, min_up8=0.35, max_down3=0.55),
        ThresholdEntryRule("up5_ge_0.60_up8_ge_0.40_down3_le_0.50", min_up5=0.60, min_up8=0.40, max_down3=0.50),
    ]


def build_focused_entry_rules() -> list[ThresholdEntryRule]:
    return [
        ThresholdEntryRule("model_all"),
        ThresholdEntryRule("up5_ge_0.55", min_up5=0.55),
        ThresholdEntryRule("up5_ge_0.60", min_up5=0.60),
        ThresholdEntryRule("up5_ge_0.65", min_up5=0.65),
        ThresholdEntryRule("up8_ge_0.35", min_up8=0.35),
        ThresholdEntryRule("up8_ge_0.40", min_up8=0.40),
        ThresholdEntryRule("down3_le_0.45", max_down3=0.45),
        ThresholdEntryRule("up5_ge_0.60_down3_le_0.50", min_up5=0.60, max_down3=0.50),
        ThresholdEntryRule("up8_ge_0.40_down3_le_0.50", min_up8=0.40, max_down3=0.50),
        ThresholdEntryRule("up5_ge_0.55_up8_ge_0.35_down3_le_0.55", min_up5=0.55, min_up8=0.35, max_down3=0.55),
    ]


def build_exit_rules() -> list[ExitRule]:
    rules: list[ExitRule] = []
    for label, hold_days in [("T3", 2), ("T5", 4), ("T7", 6)]:
        rules.append(ExitRule(f"expiry_{label}_close", "expiry", hold_days))
    for label, hold_days in [("T5", 4), ("T7", 6)]:
        for tp in [0.03, 0.04, 0.06, 0.08]:
            for sl in [0.01, 0.015, 0.02]:
                for trigger in ["intraday", "close"]:
                    rules.append(ExitRule(f"fixed_tp{tp:.1%}_sl{sl:.1%}_{trigger}_{label}", "fixed", hold_days, tp, sl, stop_trigger=trigger))
        for target in [0.03, 0.04, 0.06]:
            for trail in [0.015, 0.02]:
                for sl in [0.01, 0.015]:
                    for trigger in ["intraday", "close"]:
                        rules.append(ExitRule(f"trail_target{target:.1%}_dd{trail:.1%}_sl{sl:.1%}_{trigger}_{label}", "trailing", hold_days, target, sl, trail, trigger))
    return rules


def build_focused_exit_rules() -> list[ExitRule]:
    return [
        ExitRule("expiry_T5_close", "expiry", 4),
        ExitRule("expiry_T7_close", "expiry", 6),
        ExitRule("fixed_tp3.0%_sl1.0%_intraday_T5", "fixed", 4, 0.03, 0.01, stop_trigger="intraday"),
        ExitRule("fixed_tp3.0%_sl1.0%_close_T5", "fixed", 4, 0.03, 0.01, stop_trigger="close"),
        ExitRule("fixed_tp4.0%_sl1.5%_intraday_T5", "fixed", 4, 0.04, 0.015, stop_trigger="intraday"),
        ExitRule("fixed_tp4.0%_sl1.5%_close_T5", "fixed", 4, 0.04, 0.015, stop_trigger="close"),
        ExitRule("fixed_tp4.0%_sl1.5%_intraday_T7", "fixed", 6, 0.04, 0.015, stop_trigger="intraday"),
        ExitRule("fixed_tp4.0%_sl1.5%_close_T7", "fixed", 6, 0.04, 0.015, stop_trigger="close"),
        ExitRule("fixed_tp6.0%_sl1.5%_intraday_T7", "fixed", 6, 0.06, 0.015, stop_trigger="intraday"),
        ExitRule("fixed_tp6.0%_sl1.5%_close_T7", "fixed", 6, 0.06, 0.015, stop_trigger="close"),
        ExitRule("fixed_tp8.0%_sl1.5%_intraday_T7", "fixed", 6, 0.08, 0.015, stop_trigger="intraday"),
        ExitRule("fixed_tp8.0%_sl1.5%_close_T7", "fixed", 6, 0.08, 0.015, stop_trigger="close"),
        ExitRule("trail_target3.0%_dd1.5%_sl1.0%_intraday_T5", "trailing", 4, 0.03, 0.01, 0.015, "intraday"),
        ExitRule("trail_target3.0%_dd1.5%_sl1.0%_close_T5", "trailing", 4, 0.03, 0.01, 0.015, "close"),
        ExitRule("trail_target4.0%_dd1.5%_sl1.5%_intraday_T5", "trailing", 4, 0.04, 0.015, 0.015, "intraday"),
        ExitRule("trail_target4.0%_dd1.5%_sl1.5%_close_T5", "trailing", 4, 0.04, 0.015, 0.015, "close"),
        ExitRule("trail_target4.0%_dd1.5%_sl1.5%_intraday_T7", "trailing", 6, 0.04, 0.015, 0.015, "intraday"),
        ExitRule("trail_target4.0%_dd1.5%_sl1.5%_close_T7", "trailing", 6, 0.04, 0.015, 0.015, "close"),
        ExitRule("trail_target6.0%_dd2.0%_sl1.5%_intraday_T7", "trailing", 6, 0.06, 0.015, 0.02, "intraday"),
        ExitRule("trail_target6.0%_dd2.0%_sl1.5%_close_T7", "trailing", 6, 0.06, 0.015, 0.02, "close"),
    ]


def apply_entry_rule(df: pd.DataFrame, rule: ThresholdEntryRule) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    if rule.min_up5 is not None:
        mask &= df["pred_up5"] >= rule.min_up5
    if rule.min_up8 is not None:
        mask &= df["pred_up8"] >= rule.min_up8
    if rule.max_down3 is not None:
        mask &= df["pred_down3"] <= rule.max_down3
    return mask


def _load_signal_cache(signals: list[str], start_date: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    z_signals = [signal for signal in signals if signal != "B2"]
    if z_signals:
        if not SIGNAL_CACHE.exists():
            raise FileNotFoundError(f"Missing signal cache: {SIGNAL_CACHE}. Run analyze_z_skill_entry_exit_backtest.py first.")
        signals_df = pd.read_parquet(SIGNAL_CACHE)
        signals_df["date"] = pd.to_datetime(signals_df["date"])
        missing = [signal for signal in z_signals if signal not in signals_df.columns]
        if missing:
            raise ValueError(f"Signal cache missing columns: {missing}")
        keep_cols = ["symbol", "date", "name", *z_signals]
        present = [col for col in keep_cols if col in signals_df.columns]
        z_df = signals_df.loc[signals_df["date"] >= pd.Timestamp(start_date), present].copy()
        frames.append(z_df[z_df[z_signals].any(axis=1)].copy())

    if "B2" in signals:
        if not FAMILY_SIGNAL_CACHE.exists():
            raise FileNotFoundError(f"Missing B1 family signal cache: {FAMILY_SIGNAL_CACHE}")
        family = pd.read_parquet(FAMILY_SIGNAL_CACHE)
        family["date"] = pd.to_datetime(family["date"])
        missing = [col for col in B2_SOURCE_COLUMNS if col not in family.columns]
        if missing:
            raise ValueError(f"B2 signal cache missing columns: {missing}")
        family["B2"] = family[B2_SOURCE_COLUMNS].fillna(False).any(axis=1)
        keep_cols = ["symbol", "date", "name", "B2"]
        present = [col for col in keep_cols if col in family.columns]
        b2_df = family.loc[(family["date"] >= pd.Timestamp(start_date)) & family["B2"], present].copy()
        frames.append(b2_df)

    if not frames:
        raise RuntimeError("No signal cache rows loaded")
    out = pd.concat(frames, ignore_index=True, sort=False)
    for signal in signals:
        if signal not in out.columns:
            out[signal] = False
        out[signal] = out[signal].fillna(False).astype(bool)
    out = out.sort_values(["date", "symbol"]).drop_duplicates(["symbol", "date"], keep="last")
    return out[out[signals].any(axis=1)].copy()


def _process_daily_for_dataset(args: tuple[str, pd.DataFrame, list[str], str]) -> pd.DataFrame | None:
    path_str, signal_rows, signals, start_date = args
    path = Path(path_str)
    try:
        if signal_rows.empty:
            return None
        daily = pd.read_parquet(path)
        daily = normalize_tushare_daily(daily, path.stem)
        daily = daily.sort_values("date").reset_index(drop=True)
        history_start = pd.Timestamp(start_date) - pd.Timedelta(days=450)
        daily = daily[daily["date"] >= history_start].reset_index(drop=True)
        if len(daily) < 130:
            return None
        name = str(daily["name"].dropna().iloc[0]) if "name" in daily.columns and daily["name"].notna().any() else ""
        if "ST" in name.upper() or "退" in name:
            return None

        factors = pd.concat([btd.calculate_factors_single_stock(daily), calculate_project_extra_features(daily)], axis=1)
        factors = factors.loc[:, ~factors.columns.duplicated(keep="last")]
        labels = create_b1_labels(daily, forward_days=5, exit_aware=True, use_new_labels=True)
        price = build_continuous_ohlc(daily)
        close_pos = ((price["close"] - price["low"]) / (price["high"] - price["low"]).replace(0, np.nan)).rename("close_pos")
        result = pd.concat([daily, factors, labels, close_pos], axis=1)
        result = result.loc[:, ~result.columns.duplicated(keep="last")]
        result["symbol"] = result["symbol"].fillna(result.get("ts_code", path.stem)).astype(str)

        signal_rows = signal_rows.copy()
        signal_rows["date"] = pd.to_datetime(signal_rows["date"])
        merged = result.merge(signal_rows[["symbol", "date", *signals]], on=["symbol", "date"], how="inner")
        if merged.empty:
            return None
        keep = [
            "symbol",
            "ts_code",
            "date",
            "name",
            "open",
            "high",
            "low",
            "close",
            "pct_chg",
            "close_pos",
            *signals,
            *PROJECT_FACTOR_COLUMNS,
            *LABELS.values(),
        ]
        present = list(dict.fromkeys(col for col in keep if col in merged.columns))
        merged = merged.loc[merged["date"] >= pd.Timestamp(start_date), present].copy()
        return merged if not merged.empty else None
    except Exception as exc:
        print(f"skip {path.name}: {exc}", flush=True)
        return None


def build_model_dataset(
    daily_dir: Path,
    signals: list[str],
    start_date: str,
    workers: int,
    force_refresh: bool,
    executor_type: str = "threads",
) -> pd.DataFrame:
    if DATASET_PATH.exists() and not force_refresh:
        data = pd.read_parquet(DATASET_PATH)
        data["date"] = pd.to_datetime(data["date"])
        expected = set(signals) | set(LABELS.values())
        if expected <= set(data.columns):
            return data[(data["date"] >= pd.Timestamp(start_date)) & data[signals].any(axis=1)].copy()
        print("z-skill model dataset missing expected columns; rebuilding", flush=True)

    signal_df = _load_signal_cache(signals, start_date)
    by_symbol = {symbol: group[["symbol", "date", *signals]].copy() for symbol, group in signal_df.groupby("symbol")}
    suffixes = (".SZ.parquet", ".SH.parquet", ".BJ.parquet")
    files = [path for path in sorted(daily_dir.glob("*.parquet")) if path.name.endswith(suffixes) and path.stem in by_symbol]
    frames: list[pd.DataFrame] = []
    started = perf_counter()
    executor_cls = ProcessPoolExecutor if executor_type == "processes" else ThreadPoolExecutor
    with executor_cls(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(_process_daily_for_dataset, (str(path), by_symbol[path.stem], signals, start_date)) for path in files]
        for n, future in enumerate(as_completed(futures), start=1):
            frame = future.result()
            if frame is not None and len(frame):
                frames.append(frame)
            if n % 500 == 0 or n == len(futures):
                elapsed = perf_counter() - started
                print(f"  z-skill model dataset: {n}/{len(futures)} files frames={len(frames)} elapsed={elapsed:.1f}s", flush=True)
    if not frames:
        raise RuntimeError("No z-skill model rows built")
    data = pd.concat(frames, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    data = data.replace([np.inf, -np.inf], np.nan)
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(DATASET_PATH, index=False)
    return data


def assign_splits(data: pd.DataFrame, oot_start: str, random_state: int) -> pd.DataFrame:
    out = data.copy()
    if "close_pos" not in out.columns and {"high", "low", "close"} <= set(out.columns):
        out["close_pos"] = (out["close"] - out["low"]) / (out["high"] - out["low"]).replace(0, np.nan)
    oot_ts = pd.Timestamp(oot_start)
    research = out[out["date"] < oot_ts]
    symbols = np.array(sorted(research["symbol"].dropna().astype(str).unique()))
    rng = np.random.default_rng(random_state)
    rng.shuffle(symbols)
    test_count = max(1, int(round(len(symbols) * 0.2)))
    test_symbols = set(symbols[:test_count])
    out["split"] = np.where(out["date"] >= oot_ts, "oot", np.where(out["symbol"].astype(str).isin(test_symbols), "test", "train"))
    return out


def _feature_columns(data: pd.DataFrame) -> list[str]:
    return [col for col in PROJECT_FACTOR_COLUMNS if col in data.columns]


def _safe_auc(y_true: pd.Series, proba: np.ndarray) -> float:
    if y_true.nunique(dropna=True) < 2:
        return np.nan
    return float(roc_auc_score(y_true, proba))


def _safe_ap(y_true: pd.Series, proba: np.ndarray) -> float:
    if y_true.nunique(dropna=True) < 2:
        return np.nan
    return float(average_precision_score(y_true, proba))


def train_one_model(data: pd.DataFrame, signal: str, label_name: str, feature_cols: list[str], model_dir: Path) -> tuple[XGBResearchModel | None, dict]:
    label_col = LABELS[label_name]
    subset = data[data[signal]].dropna(subset=[label_col]).copy()
    subset[label_col] = subset[label_col].astype(int)
    rows = {"signal": signal, "model": label_name, "label_col": label_col, "rows": int(len(subset))}
    if len(subset) < 500 or subset[label_col].nunique() < 2:
        rows["status"] = "skipped_insufficient_rows_or_classes"
        return None, rows

    train = subset[subset["split"] == "train"]
    test = subset[subset["split"] == "test"]
    oot = subset[subset["split"] == "oot"]
    if train[label_col].nunique() < 2 or test[label_col].nunique() < 2:
        rows["status"] = "skipped_bad_split_classes"
        return None, rows

    imputer = SimpleImputer(strategy="median")
    X_train_raw = train[feature_cols].replace([np.inf, -np.inf], np.nan)
    X_test_raw = test[feature_cols].replace([np.inf, -np.inf], np.nan)
    X_oot_raw = oot[feature_cols].replace([np.inf, -np.inf], np.nan)
    X_train_imp = imputer.fit_transform(X_train_raw)
    X_test_imp = imputer.transform(X_test_raw)
    X_oot_imp = imputer.transform(X_oot_raw) if len(oot) else np.empty((0, len(feature_cols)))

    k = min(90, X_train_imp.shape[1])
    selector = SelectKBest(f_classif, k=k)
    X_train = selector.fit_transform(X_train_imp, train[label_col])
    X_test = selector.transform(X_test_imp)
    X_oot = selector.transform(X_oot_imp) if len(oot) else X_oot_imp

    pos = int(train[label_col].sum())
    neg = int(len(train) - pos)
    scale_pos_weight = neg / max(pos, 1)
    early_stop = AucGapEarlyStopping()
    classifier = XGBClassifier(
        n_estimators=450,
        max_depth=4,
        learning_rate=0.035,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        callbacks=[early_stop],
        n_jobs=-1,
        random_state=42,
    )
    classifier.fit(X_train, train[label_col], eval_set=[(X_train, train[label_col]), (X_test, test[label_col])], verbose=False)
    selected_features = [feature_cols[i] for i, keep in enumerate(selector.get_support()) if keep]
    wrapper = XGBResearchModel(feature_cols, selected_features, imputer, selector, classifier, early_stop.best_iteration)

    for split, frame, X in [("train", train, X_train), ("test", test, X_test), ("oot", oot, X_oot)]:
        if len(frame) == 0:
            continue
        if split == "oot" and frame[label_col].nunique() < 2:
            continue
        proba = classifier.predict_proba(X, iteration_range=(0, early_stop.best_iteration + 1))[:, 1]
        rows[f"{split}_rows"] = int(len(frame))
        rows[f"{split}_positive_rate"] = float(frame[label_col].mean())
        rows[f"{split}_auc"] = _safe_auc(frame[label_col], proba)
        rows[f"{split}_ap"] = _safe_ap(frame[label_col], proba)
    rows["status"] = "ok"
    rows["best_iteration"] = int(early_stop.best_iteration)
    rows["early_stop_best_score"] = float(early_stop.best_score)
    rows["selected_feature_count"] = int(len(selected_features))

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(wrapper, model_dir / f"{signal}_{label_name}.joblib")
    return wrapper, rows


def train_models(data: pd.DataFrame, signals: list[str], model_dir: Path) -> tuple[dict[tuple[str, str], XGBResearchModel], pd.DataFrame]:
    feature_cols = _feature_columns(data)
    models: dict[tuple[str, str], XGBResearchModel] = {}
    reports: list[dict] = []
    for signal in signals:
        print(f"training signal={signal}", flush=True)
        for label_name in LABELS:
            model, report = train_one_model(data, signal, label_name, feature_cols, model_dir)
            reports.append(report)
            print(
                f"  {signal}/{label_name}: {report.get('status')} "
                f"test_auc={report.get('test_auc')} oot_auc={report.get('oot_auc')}",
                flush=True,
            )
            if model is not None:
                models[(signal, label_name)] = model
    return models, pd.DataFrame(reports)


def load_models(signals: list[str], model_dir: Path, output_dir: Path) -> tuple[dict[tuple[str, str], XGBResearchModel], pd.DataFrame]:
    models: dict[tuple[str, str], XGBResearchModel] = {}
    rows: list[dict] = []
    report_path = output_dir / "latest_z_skill_model_training_report.csv"
    if report_path.exists():
        rows = pd.read_csv(report_path).to_dict("records")
    for signal in signals:
        for label_name in LABELS:
            path = model_dir / f"{signal}_{label_name}.joblib"
            if not path.exists():
                raise FileNotFoundError(f"Missing trained model: {path}")
            models[(signal, label_name)] = joblib.load(path)
    return models, pd.DataFrame(rows)


def add_predictions(data: pd.DataFrame, models: dict[tuple[str, str], XGBResearchModel], signals: list[str]) -> pd.DataFrame:
    out = data.copy()
    for signal in signals:
        mask = out[signal].fillna(False).astype(bool)
        for label_name in LABELS:
            col = f"pred_{label_name}"
            out.loc[mask, col] = np.nan
            model = models.get((signal, label_name))
            if model is None or not mask.any():
                continue
            out.loc[mask, col] = model.predict_proba(out.loc[mask, model.feature_names_in_])[:, 1]
    return out


def evaluate_model_strategies(candidates: pd.DataFrame, signals: list[str], min_entry_rows: int, focused_grid: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    open_filters = build_open_filters()
    exit_rules = build_focused_exit_rules() if focused_grid else build_exit_rules()
    entry_rules = build_focused_entry_rules() if focused_grid else build_entry_rules()
    rows: list[dict] = []
    trade_samples: list[pd.DataFrame] = []

    for signal in signals:
        base = candidates[candidates[signal]].dropna(subset=["pred_up5", "pred_up8", "pred_down3"]).copy()
        if base.empty:
            continue
        print(f"evaluating model-filtered signal={signal} rows={len(base):,}", flush=True)
        for entry_rule in entry_rules:
            entry_base = base[apply_entry_rule(base, entry_rule)].copy()
            if len(entry_base) < min_entry_rows:
                continue
            for open_filter in open_filters:
                entry_df = entry_base[apply_open_filter(entry_base, open_filter)].copy()
                if len(entry_df) < min_entry_rows:
                    continue
                for exit_rule in exit_rules:
                    trades = simulate_exit(entry_df, exit_rule)
                    if trades.empty:
                        continue
                    meta_cols = ["date", "symbol", "split", "close", "entry_open", "close_pos", "pred_up5", "pred_up8", "pred_down3"]
                    trades = trades.merge(entry_df[meta_cols], on=["date", "symbol"], how="left")
                    raw_trades = len(trades)
                    trades = drop_overlapping_trades(trades)
                    skipped = raw_trades - len(trades)
                    if entry_rule.name in {"up5_ge_0.60_down3_le_0.50", "up8_ge_0.40_down3_le_0.50"} and exit_rule.name in {"expiry_T3_close", "fixed_tp4.0%_sl1.5%_intraday_T5"}:
                        sample = trades.copy()
                        sample["signal"] = signal
                        sample["entry_rule"] = entry_rule.name
                        sample["open_filter"] = open_filter.name
                        sample["exit_rule"] = exit_rule.name
                        trade_samples.append(sample)
                    for split in ["train", "test", "oot"]:
                        part = trades[trades["split"] == split]
                        metrics = summarize_returns(part)
                        if not metrics:
                            continue
                        rows.append(
                            {
                                "signal": signal,
                                "entry_rule": entry_rule.name,
                                "open_filter": open_filter.name,
                                "open_filter_description": open_filter.description,
                                "exit_rule": exit_rule.name,
                                "exit_kind": exit_rule.kind,
                                "hold_days": exit_rule.hold_days,
                                "take_profit": exit_rule.take_profit,
                                "stop_loss": exit_rule.stop_loss,
                                "stop_trigger": exit_rule.stop_trigger,
                                "trail_drawdown": exit_rule.trail_drawdown,
                                "split": split,
                                "raw_trades": raw_trades,
                                "skipped_overlaps": skipped,
                                "overlap_skip_rate": skipped / raw_trades if raw_trades else np.nan,
                                "min_return_pct": float(part["return_pct"].min()) if not part.empty else np.nan,
                                "max_return_pct": float(part["return_pct"].max()) if not part.empty else np.nan,
                                **metrics,
                            }
                        )
    details = pd.concat(trade_samples, ignore_index=True) if trade_samples else pd.DataFrame()
    return pd.DataFrame(rows), details


def _score(row: pd.Series) -> float:
    return (
        float(row.get("avg_return_pct") or 0) * 0.45
        + min(float(row.get("profit_factor") or 0), 4) * 1.3
        + float(row.get("win_rate") or 0) * 2.5
        + float(row.get("max_drawdown_pct") or -100) / 20
        + float(row.get("min_return_pct") or -100) / 30
        + min(float(row.get("trades") or 0), 500) / 500
    )


def choose_model_playbooks(summary: pd.DataFrame) -> pd.DataFrame:
    oot = summary[summary["split"] == "oot"].copy()
    if oot.empty:
        return oot
    oot["selection_score"] = oot.apply(_score, axis=1)
    rows = []
    for signal, part in oot.groupby("signal"):
        tradable = part[
            (part["trades"] >= 80)
            & (part["avg_return_pct"] > 0)
            & (part["profit_factor"].fillna(0) >= 1.35)
            & (part["max_drawdown_pct"] >= -35)
            & (part["min_return_pct"] >= -25)
        ].copy()
        if tradable.empty:
            eligible = part.sort_values(["selection_score", "profit_factor", "avg_return_pct"], ascending=[False, False, False]).head(1).copy()
            eligible["action_level"] = "模型观察"
        else:
            eligible = tradable.sort_values(["selection_score", "profit_factor", "avg_return_pct"], ascending=[False, False, False]).head(1).copy()
            best = eligible.iloc[0]
            eligible["action_level"] = "可小仓实操" if best["profit_factor"] >= 1.5 and best["avg_return_pct"] >= 0.6 else "谨慎实操"
        rows.append(eligible)
    return pd.concat(rows, ignore_index=True).sort_values(["action_level", "selection_score"], ascending=[True, False])


def _entry_rule_thresholds(rule_name: str) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for part in str(rule_name).split("_"):
        pass
    tokens = str(rule_name).split("_")
    for i in range(len(tokens) - 2):
        key = tokens[i]
        op = tokens[i + 1]
        value = tokens[i + 2]
        if key in {"up5", "up8", "down3"} and op in {"ge", "le"}:
            try:
                thresholds[f"{key}_{op}"] = float(value)
            except ValueError:
                continue
    return thresholds


def _entry_rule_pass(row: pd.Series, rule_name: str) -> bool:
    if str(rule_name) == "model_all":
        return True
    thresholds = _entry_rule_thresholds(rule_name)
    if "up5_ge" in thresholds and float(row.get("pred_up5") or 0) < thresholds["up5_ge"]:
        return False
    if "up8_ge" in thresholds and float(row.get("pred_up8") or 0) < thresholds["up8_ge"]:
        return False
    if "down3_le" in thresholds and float(row.get("pred_down3") or 1) > thresholds["down3_le"]:
        return False
    return True


def write_latest_scored_candidates(predicted: pd.DataFrame, signals: list[str], playbooks: pd.DataFrame, output_dir: Path) -> Path:
    if predicted.empty:
        scored = pd.DataFrame()
    else:
        latest_date = predicted["date"].max()
        latest = predicted[predicted["date"] == latest_date].copy()
        playbook_by_signal = {str(row["signal"]): row for _, row in playbooks.iterrows()}
        rows: list[dict] = []
        for signal in signals:
            if signal not in latest.columns:
                continue
            part = latest[latest[signal].fillna(False).astype(bool)].copy()
            playbook = playbook_by_signal.get(signal)
            if playbook is None:
                continue
            for _, row in part.iterrows():
                passed = _entry_rule_pass(row, str(playbook.get("entry_rule")))
                rows.append(
                    {
                        "symbol": row.get("symbol"),
                        "date": row.get("date"),
                        "name": row.get("name"),
                        "signal": signal,
                        "pred_up5": row.get("pred_up5"),
                        "pred_up8": row.get("pred_up8"),
                        "pred_down3": row.get("pred_down3"),
                        "entry_rule": playbook.get("entry_rule"),
                        "model_pass": passed,
                        "action_level": playbook.get("action_level"),
                        "selection_score": playbook.get("selection_score"),
                    }
                )
        scored = pd.DataFrame(rows)
    path = output_dir / "latest_z_skill_model_scored_candidates.parquet"
    scored.to_parquet(path, index=False)
    return path


def fmt_pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.2f}%"


def fmt_rate(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value) * 100:.2f}%"


def fmt_num(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{int(round(float(value))):,}"


def markdown_table(rows: list[dict], headers: list[str]) -> str:
    lines = ["|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("|" + "|".join(str(row.get(header, "")) for header in headers) + "|")
    return "\n".join(lines)


def write_report(model_report: pd.DataFrame, summary: pd.DataFrame, playbooks: pd.DataFrame, output_dir: Path, timestamp: str) -> Path:
    path = output_dir / f"z_skill_model_entry_exit_backtest_{timestamp}.md"
    oot = summary[(summary["split"] == "oot") & (summary["trades"] >= 80)].copy()
    oot["selection_score"] = oot.apply(_score, axis=1)
    top = oot.sort_values(["selection_score", "profit_factor", "avg_return_pct"], ascending=[False, False, False]).head(40)

    with path.open("w", encoding="utf-8") as f:
        f.write("# z-skill 高频战法建模与买卖策略评估\n\n")
        f.write("## 建模口径\n\n")
        f.write("- 优先策略：命中多但规则边际较弱的 z-skill 战法。\n")
        f.write("- 特征：项目变量库 147 个 Tushare/技术派生变量，按单只股票历史滚动计算。\n")
        f.write("- 标签：T+1 开盘后 5 日内 up5/up8/down3，复用 B1 exit-aware 标签口径。\n")
        f.write("- 切分：2020-2024 中按股票代码随机 8:2 切 train/test，2025+ 为 OOT。\n")
        f.write("- 训练：XGBoost，使用 test AUC 与 train/test AUC gap 的 early stop，避免过拟合。\n")
        f.write("- 策略评估：模型概率阈值 + T+1 开盘过滤 + 卖出规则网格；同一股票未卖出前不重复买入。\n\n")

        f.write("## 推荐模型实操清单\n\n")
        rows = []
        for _, row in playbooks.iterrows():
            rows.append(
                {
                    "分层": row["action_level"],
                    "战法": row["signal"],
                    "买入模型阈值": row["entry_rule"],
                    "开盘过滤": row["open_filter_description"],
                    "卖出": row["exit_rule"],
                    "交易数": fmt_num(row["trades"]),
                    "均值": fmt_pct(row["avg_return_pct"]),
                    "胜率": fmt_rate(row["win_rate"]),
                    "最大回撤": fmt_pct(row["max_drawdown_pct"]),
                    "PF": f"{row['profit_factor']:.2f}" if pd.notna(row["profit_factor"]) else "",
                    "最差单笔": fmt_pct(row["min_return_pct"]),
                }
            )
        f.write(markdown_table(rows, ["分层", "战法", "买入模型阈值", "开盘过滤", "卖出", "交易数", "均值", "胜率", "最大回撤", "PF", "最差单笔"]))
        f.write("\n\n")

        f.write("## 模型质量摘要\n\n")
        quality_rows = []
        for _, row in model_report[model_report["status"] == "ok"].iterrows():
            quality_rows.append(
                {
                    "战法": row["signal"],
                    "模型": row["model"],
                    "样本": fmt_num(row["rows"]),
                    "test_auc": f"{row.get('test_auc', np.nan):.4f}" if pd.notna(row.get("test_auc")) else "",
                    "oot_auc": f"{row.get('oot_auc', np.nan):.4f}" if pd.notna(row.get("oot_auc")) else "",
                    "best_iter": fmt_num(row.get("best_iteration")),
                }
            )
        f.write(markdown_table(quality_rows, ["战法", "模型", "样本", "test_auc", "oot_auc", "best_iter"]))
        f.write("\n\n")

        f.write("## OOT 综合 Top 40\n\n")
        top_rows = []
        for _, row in top.iterrows():
            top_rows.append(
                {
                    "战法": row["signal"],
                    "买入": row["entry_rule"],
                    "开盘": row["open_filter"],
                    "卖出": row["exit_rule"],
                    "交易数": fmt_num(row["trades"]),
                    "均值": fmt_pct(row["avg_return_pct"]),
                    "胜率": fmt_rate(row["win_rate"]),
                    "最大回撤": fmt_pct(row["max_drawdown_pct"]),
                    "PF": f"{row['profit_factor']:.2f}" if pd.notna(row["profit_factor"]) else "",
                    "最差单笔": fmt_pct(row["min_return_pct"]),
                }
            )
        f.write(markdown_table(top_rows, ["战法", "买入", "开盘", "卖出", "交易数", "均值", "胜率", "最大回撤", "PF", "最差单笔"]))
        f.write("\n")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train z-skill models and backtest model-filtered strategies")
    parser.add_argument("--daily-dir", type=Path, default=DEFAULT_DAILY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--oot-start", default="2025-01-01")
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--executor", choices=["threads", "processes"], default="threads")
    parser.add_argument("--force-dataset", action="store_true")
    parser.add_argument("--reuse-models", action="store_true")
    parser.add_argument("--focused-grid", action="store_true")
    parser.add_argument("--signals", nargs="*", default=PRIORITY_SIGNALS)
    parser.add_argument("--min-entry-rows", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    signals = list(dict.fromkeys(args.signals))
    print(f"signals: {signals}", flush=True)
    print("building/loading model dataset", flush=True)
    data = build_model_dataset(args.daily_dir, signals, args.start_date, args.workers, args.force_dataset, args.executor)
    data = assign_splits(data, args.oot_start, random_state=42)
    print(f"model dataset rows={len(data):,} features={len(_feature_columns(data))}", flush=True)

    if args.reuse_models:
        print("loading trained models", flush=True)
        models, model_report = load_models(signals, args.model_dir, args.output_dir)
        model_report_path = args.output_dir / "latest_z_skill_model_training_report.csv"
    else:
        models, model_report = train_models(data, signals, args.model_dir)
        model_report_path = args.output_dir / f"z_skill_model_training_report_{timestamp}.csv"
        latest_model_report = args.output_dir / "latest_z_skill_model_training_report.csv"
        model_report.to_csv(model_report_path, index=False)
        model_report.to_csv(latest_model_report, index=False)

    print("adding model predictions", flush=True)
    predicted = add_predictions(data, models, signals)
    max_hold_days = max(rule.hold_days for rule in build_exit_rules())
    print("adding future prices", flush=True)
    predicted = add_future_prices(predicted, args.daily_dir, max_hold_days=max_hold_days)

    print("evaluating model strategy grid", flush=True)
    summary, details = evaluate_model_strategies(predicted, signals, args.min_entry_rows, focused_grid=args.focused_grid)
    playbooks = choose_model_playbooks(summary)

    summary_path = args.output_dir / f"z_skill_model_entry_exit_backtest_{timestamp}.csv"
    latest_summary = args.output_dir / "latest_z_skill_model_entry_exit_backtest.csv"
    playbook_path = args.output_dir / f"z_skill_model_operational_playbook_{timestamp}.csv"
    latest_playbook = args.output_dir / "latest_z_skill_model_operational_playbook.csv"
    detail_path = args.output_dir / f"z_skill_model_trade_samples_{timestamp}.csv"
    latest_detail = args.output_dir / "latest_z_skill_model_trade_samples.csv"
    summary.to_csv(summary_path, index=False)
    summary.to_csv(latest_summary, index=False)
    playbooks.to_csv(playbook_path, index=False)
    playbooks.to_csv(latest_playbook, index=False)
    latest_scored = write_latest_scored_candidates(predicted, signals, playbooks, args.output_dir)
    if not details.empty:
        details.to_csv(detail_path, index=False)
        details.to_csv(latest_detail, index=False)

    report_path = write_report(model_report, summary, playbooks, args.output_dir, timestamp)
    latest_report = args.output_dir / "latest_z_skill_model_entry_exit_backtest.md"
    shutil.copyfile(report_path, latest_report)
    metadata = {
        "timestamp": timestamp,
        "signals": signals,
        "dataset": str(DATASET_PATH),
        "model_dir": str(args.model_dir),
        "model_report": str(model_report_path),
        "summary": str(summary_path),
        "playbook": str(playbook_path),
        "latest_scored_candidates": str(latest_scored),
        "report": str(report_path),
    }
    (args.output_dir / "latest_z_skill_model_run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
