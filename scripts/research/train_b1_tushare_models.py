#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train B1 production models from Tushare-only daily data.

The B1 production path intentionally uses only:
- Tushare daily fields from data/raw/daily/*.parquet
- Tushare daily_basic fields from data/raw/daily_basic/*.parquet
- Tushare stock_basic metadata already merged into daily files
- local technical/price-volume factors derived from those fields

No AkShare fields are read or required.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from xgboost.callback import TrainingCallback
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "research"))

import build_training_data_parallel as btd
from quant.data import MarketDataStore, MarketDataStoreConfig
from quant.data.source_merge import normalize_tushare_daily
from quant.features.b1_gate import calculate_b1_gate
from quant.features.daily_factor_layer import BASE_FACTOR_COLUMNS, attach_daily_base_factors
from quant.features.variable_library import (
    PROJECT_FACTOR_COLUMNS,
    merge_daily_basic_features,
)
from quant.ml.label_maker import create_b1_labels
from quant.ml.xgb_research import XGBResearchModel


B1_FEATURE_COLUMNS = PROJECT_FACTOR_COLUMNS

LABELS = {
    "up5_es": "label_t1_open_max_high_5pct",
    "up8_es": "label_t1_open_max_high_8pct",
    "up10_es": "label_t1_open_max_high_10pct",
    "down2_es": "label_t1_open_min_low_2pct_below_t0_low",
    "down3_es": "label_t1_open_min_low_3pct_below_t0_low",
}

MODEL_PARAMS = {
    "up5_es": {"n_estimators": 500, "max_depth": 4, "learning_rate": 0.035},
    "up8_es": {"n_estimators": 550, "max_depth": 4, "learning_rate": 0.035},
    "up10_es": {"n_estimators": 600, "max_depth": 4, "learning_rate": 0.03},
    "down2_es": {"n_estimators": 500, "max_depth": 4, "learning_rate": 0.035},
    "down3_es": {"n_estimators": 500, "max_depth": 4, "learning_rate": 0.035},
}


def process_daily_file(args: tuple[str, str]) -> pd.DataFrame | None:
    path_str, start_date = args
    path = Path(path_str)
    start_ts = pd.to_datetime(start_date)
    history_start = start_ts - pd.Timedelta(days=450)
    store = MarketDataStore(MarketDataStoreConfig(backend="parquet", root=path.parent.parent))
    df = store.read_market_range(path.parent.name, start_date=history_start.strftime("%Y%m%d"), symbols=[path.stem])
    return process_daily_frame((path.stem, df, start_date))


def process_daily_frame(
    args: tuple[str, pd.DataFrame, str] | tuple[str, pd.DataFrame, str, bool],
) -> pd.DataFrame | None:
    symbol, df, start_date, *options = args
    raise_errors = bool(options[0]) if options else False
    try:
        start_ts = pd.to_datetime(start_date)
        history_start = start_ts - pd.Timedelta(days=450)
        df = normalize_tushare_daily(df, symbol)
        if "vol" in df.columns and "volume" in df.columns:
            df = df.drop(columns=["vol"])
        df = df.sort_values("date").reset_index(drop=True)
        df = df[df["date"] >= history_start].reset_index(drop=True)
        if len(df) < 130:
            return None

        name = str(df["name"].iloc[0]) if "name" in df.columns and len(df) else ""
        if "ST" in name.upper() or "退" in name:
            return None

        shared = attach_daily_base_factors(
            df,
            symbol=symbol,
            compute_if_missing=True,
            persist_missing=False,
        )
        # The B1 gate is much cheaper than the full project factor set and the
        # forward-label build.  During an incremental daily refresh only a tiny
        # fraction of symbols pass the gate, so reject the rest before doing
        # the expensive work.  The mask intentionally uses the same continuous
        # OHLC and shared KDJ definitions as the full path below.
        b1_signal = calculate_b1_gate(
            df,
            shared_factors=shared,
        )["b1_gate"]
        if not bool((b1_signal & (df["date"] >= start_ts)).any()):
            return None

        shared_cols = [col for col in BASE_FACTOR_COLUMNS if col in shared.columns]
        factors = pd.concat([btd.calculate_factors_single_stock(df), shared[shared_cols]], axis=1)
        factors = factors.loc[:, ~factors.columns.duplicated(keep="last")]
        labels = create_b1_labels(df, forward_days=5, exit_aware=True, use_new_labels=True)
        result = pd.concat([df, factors, labels], axis=1)

        keep_cols = [
            "ts_code",
            "trade_date",
            "date",
            "symbol",
            "name",
            "industry",
            "market",
            *B1_FEATURE_COLUMNS,
            *LABELS.values(),
        ]
        present = [col for col in keep_cols if col in result.columns]
        out = result.loc[(result["date"] >= start_ts) & b1_signal, present].copy()
        return out if len(out) else None
    except Exception as exc:
        if raise_errors:
            raise RuntimeError(f"{symbol}: {exc}") from exc
        print(f"skip {symbol}: {exc}", flush=True)
        return None


def build_dataset(
    daily_dir: Path,
    start_date: str,
    workers: int,
    limit: int | None = None,
    executor_type: str = "threads",
    adaptive_workers: bool = False,
    min_workers: int = 16,
    max_workers: int | None = None,
    worker_step: int = 16,
    load_target: float = 0.80,
    load_hard_limit: float = 1.20,
    max_symbol_error_rate: float | None = None,
    allow_empty: bool = False,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    start_ts = pd.to_datetime(start_date)
    history_start = (start_ts - pd.Timedelta(days=450)).strftime("%Y%m%d")
    store = MarketDataStore(MarketDataStoreConfig(backend="parquet", root=daily_dir.parent))
    market = store.read_market_range(
        daily_dir.name,
        start_date=history_start,
        symbols=symbols,
    )
    if market.empty:
        raise RuntimeError(f"No canonical Tushare daily rows found for {history_start}+")
    source_dates = pd.to_datetime(
        market.get("date", market.get("trade_date")),
        errors="coerce",
    )
    source_latest_trade_date = source_dates.max()
    symbols = sorted(market["ts_code"].dropna().astype(str).unique().tolist())
    if limit:
        symbols = symbols[:limit]
        market = market[market["ts_code"].astype(str).isin(symbols)]

    frames: list[pd.DataFrame] = []
    symbol_errors: list[str] = []
    tasks = [
        (str(symbol), group.reset_index(drop=True), start_date, True)
        for symbol, group in market.groupby("ts_code", sort=True)
    ]
    executor_cls = ThreadPoolExecutor if executor_type == "threads" else ProcessPoolExecutor

    def consume_batch(
        batch: list[tuple[str, pd.DataFrame, str, bool]],
        batch_workers: int,
        processed_before: int,
    ) -> tuple[int, float]:
        started = perf_counter()
        with executor_cls(max_workers=batch_workers) as executor:
            futures = {
                executor.submit(process_daily_frame, task): task[0]
                for task in batch
            }
            for offset, future in enumerate(as_completed(futures), start=1):
                symbol = futures[future]
                try:
                    frame = future.result()
                except Exception as exc:
                    symbol_errors.append(f"{symbol}: {exc}")
                    frame = None
                if frame is not None and len(frame):
                    frames.append(frame)
                n = processed_before + offset
                if n % 500 == 0 or n == len(tasks):
                    print(f"processed {n}/{len(tasks)} daily files, frames={len(frames)}", flush=True)
        return processed_before + len(batch), perf_counter() - started

    if not adaptive_workers:
        consume_batch(tasks, workers, 0)
    else:
        cpu_count = os.cpu_count() or 4
        current_workers = max(min_workers, workers)
        max_workers = max_workers or current_workers
        processed = 0
        batch_no = 0
        while processed < len(tasks):
            batch_no += 1
            current_workers = min(max_workers, max(min_workers, current_workers))
            batch_size = min(max(120, current_workers * 3), 600, len(tasks) - processed)
            batch = tasks[processed: processed + batch_size]
            processed, elapsed = consume_batch(batch, current_workers, processed)
            throughput = len(batch) / max(elapsed, 1e-6)
            try:
                load_ratio = os.getloadavg()[0] / cpu_count
            except (AttributeError, OSError):
                load_ratio = 0.0

            next_workers = current_workers
            if load_ratio and load_ratio > load_hard_limit:
                next_workers = max(min_workers, current_workers - worker_step)
            elif load_ratio < load_target and current_workers < max_workers:
                next_workers = min(max_workers, current_workers + worker_step)

            print(
                "adaptive workers: "
                f"batch={batch_no} size={len(batch)} elapsed={elapsed:.1f}s "
                f"throughput={throughput:.2f}/s load_ratio={load_ratio:.2f} "
                f"workers={current_workers}->{next_workers}",
                flush=True,
            )
            current_workers = next_workers

    allowed_error_rate = (
        float(os.getenv("B1_FEATURE_MAX_SYMBOL_ERROR_RATE", "0.001"))
        if max_symbol_error_rate is None
        else float(max_symbol_error_rate)
    )
    error_rate = len(symbol_errors) / max(len(tasks), 1)
    if error_rate > allowed_error_rate:
        samples = "; ".join(symbol_errors[:10])
        raise RuntimeError(
            "B1 feature coverage gate failed: "
            f"errors={len(symbol_errors)}/{len(tasks)} ({error_rate:.4%}) "
            f"> allowed={allowed_error_rate:.4%}; samples={samples}"
        )
    if not frames and not allow_empty:
        raise RuntimeError("No B1 training rows were produced")
    if frames:
        data = (
            pd.concat(frames, ignore_index=True)
            .sort_values(["date", "symbol"])
            .reset_index(drop=True)
            .replace([np.inf, -np.inf], np.nan)
        )
    else:
        data = pd.DataFrame()
    data.attrs["source_symbol_count"] = len(tasks)
    data.attrs["source_latest_trade_date"] = (
        source_latest_trade_date.strftime("%Y-%m-%d")
        if pd.notna(source_latest_trade_date)
        else None
    )
    data.attrs["symbol_error_count"] = len(symbol_errors)
    data.attrs["symbol_error_rate"] = error_rate
    data.attrs["symbol_error_samples"] = symbol_errors[:10]
    return data


class AucGapEarlyStopping(TrainingCallback):
    """Early stop on test AUC with a penalty for train/test AUC divergence."""

    def __init__(self, rounds: int = 40, gap_tolerance: float = 0.03, gap_penalty: float = 0.5):
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
        self.history.append({
            "iteration": float(epoch),
            "train_auc": train_auc,
            "test_auc": test_auc,
            "auc_gap": train_auc - test_auc,
            "early_stop_score": score,
        })
        if score > self.best_score:
            self.best_score = score
            self.best_iteration = epoch
        return epoch - self.best_iteration >= self.rounds


def make_classifier(model_name: str, scale_pos_weight: float, early_stop: AucGapEarlyStopping) -> XGBClassifier:
    params = MODEL_PARAMS[model_name]
    return XGBClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
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


def assign_symbol_splits(data: pd.DataFrame, oot_start: str, test_size: float, random_state: int) -> pd.DataFrame:
    out = data.copy()
    if "symbol" not in out.columns:
        out["symbol"] = out["ts_code"]
    out["symbol"] = out["symbol"].fillna(out["ts_code"]).astype(str)
    split_key = "ts_code" if "ts_code" in out.columns else "symbol"
    out[split_key] = out[split_key].astype(str)
    oot_start_ts = pd.Timestamp(oot_start)
    research = out[out["date"] < oot_start_ts]
    symbols = np.array(sorted(research[split_key].dropna().astype(str).unique()))
    rng = np.random.default_rng(random_state)
    shuffled = symbols.copy()
    rng.shuffle(shuffled)
    test_count = max(1, int(round(len(shuffled) * test_size)))
    test_symbols = set(shuffled[:test_count])
    train_symbols = set(shuffled[test_count:])

    out["split"] = "oot"
    research_mask = out["date"] < oot_start_ts
    out.loc[research_mask & out[split_key].isin(train_symbols), "split"] = "train"
    out.loc[research_mask & out[split_key].isin(test_symbols), "split"] = "test"
    return out


def validate_label_splits(data: pd.DataFrame) -> None:
    problems = []
    for model_name, label_col in LABELS.items():
        subset = data.dropna(subset=[label_col])
        for split in ["train", "test", "oot"]:
            part = subset[subset["split"] == split]
            if part.empty:
                problems.append(f"{model_name}/{split}: empty")
            elif part[label_col].nunique() < 2:
                problems.append(f"{model_name}/{split}: only class {part[label_col].iloc[0]}")
    if problems:
        raise RuntimeError("Invalid label split: " + "; ".join(problems))


def resolve_worker_count(
    requested_workers: int,
    buffer_cores: int,
    worker_multiplier: int,
    max_auto_workers: int,
) -> int:
    """Resolve feature-build concurrency while leaving CPU buffer for the machine."""
    if requested_workers > 0:
        return requested_workers
    cpu_count = os.cpu_count() or 4
    usable_cores = max(1, cpu_count - max(0, buffer_cores))
    return min(max_auto_workers, max(16, usable_cores * max(1, worker_multiplier)))


def _binary_metrics(y_true: pd.Series, pred: np.ndarray, proba: np.ndarray) -> dict:
    if len(y_true) == 0:
        return {}
    return {
        "rows": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
        "auc": float(roc_auc_score(y_true, proba)) if y_true.nunique() > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y_true, proba)) if y_true.nunique() > 1 else float("nan"),
        "classification_report": classification_report(y_true, pred, output_dict=True, zero_division=0),
    }


def train_models(data: pd.DataFrame, output_dir: Path, report_dir: Path, select_k: int | None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, dict] = {}

    for model_name, label_col in LABELS.items():
        cols = [col for col in B1_FEATURE_COLUMNS if col in data.columns]
        subset = data.loc[:, [*cols, label_col, "date", "symbol", "split"]].dropna(subset=[label_col]).copy()
        train = subset[subset["split"] == "train"]
        test = subset[subset["split"] == "test"]
        oot = subset[subset["split"] == "oot"]
        if train.empty or test.empty or oot.empty:
            raise RuntimeError(f"{model_name} has empty train/test/oot split")

        X_train_raw = train[cols]
        y_train = train[label_col].astype(int)
        X_test_raw = test[cols]
        y_test = test[label_col].astype(int)
        if y_train.nunique() < 2:
            raise RuntimeError(f"{model_name} train split has only one class for {label_col}")

        model_select_k = None if select_k is None else min(select_k, len(cols))
        imputer = SimpleImputer(strategy="median")
        X_train_imputed = imputer.fit_transform(X_train_raw.replace([np.inf, -np.inf], np.nan))
        X_test_imputed = imputer.transform(X_test_raw.replace([np.inf, -np.inf], np.nan))
        selector = None
        selected = cols
        if model_select_k is not None:
            selector = SelectKBest(score_func=f_classif, k=model_select_k)
            X_train_model = selector.fit_transform(X_train_imputed, y_train)
            X_test_model = selector.transform(X_test_imputed)
            support = selector.get_support()
            selected = [col for col, keep in zip(cols, support) if keep]
        else:
            X_train_model = X_train_imputed
            X_test_model = X_test_imputed

        positives = max(1, int(y_train.sum()))
        negatives = max(1, int(len(y_train) - y_train.sum()))
        scale_pos_weight = negatives / positives
        early_stop = AucGapEarlyStopping(rounds=40, gap_tolerance=0.03, gap_penalty=0.5)
        classifier = make_classifier(model_name, scale_pos_weight, early_stop)
        classifier.fit(
            X_train_model,
            y_train,
            eval_set=[(X_train_model, y_train), (X_test_model, y_test)],
            verbose=False,
        )
        classifier.set_params(callbacks=None)
        classifier.callbacks = None
        model = XGBResearchModel(
            feature_names_in_=cols,
            selected_features_=selected,
            imputer=imputer,
            selector=selector,
            classifier=classifier,
            best_iteration=early_stop.best_iteration,
        )

        split_reports = {}
        for split_name, split_df in [("train", train), ("test", test), ("oot", oot)]:
            X_split = split_df[cols]
            y_split = split_df[label_col].astype(int)
            pred = model.predict(X_split)
            proba = model.predict_proba(X_split)[:, 1]
            split_reports[split_name] = _binary_metrics(y_split, pred, proba)

        importances = classifier.feature_importances_
        top_features = sorted(
            [
                {"feature": feature, "importance": float(importance)}
                for feature, importance in zip(selected, importances)
            ],
            key=lambda item: item["importance"],
            reverse=True,
        )[:40]

        model_path = output_dir / f"{model_name}.joblib"
        joblib.dump(model, model_path)
        reports[model_name] = {
            "label": label_col,
            "model_path": str(model_path),
            "features": len(cols),
            "feature_selection_k": "all" if model_select_k is None else model_select_k,
            "selected_feature_count": len(selected),
            "selected_features": selected,
            "scale_pos_weight": float(scale_pos_weight),
            "xgboost_params": MODEL_PARAMS[model_name],
            "best_iteration": int(classifier.best_iteration) if getattr(classifier, "best_iteration", None) is not None else None,
            "early_stop_best_iteration": int(early_stop.best_iteration),
            "early_stop_best_score": float(early_stop.best_score),
            "early_stop_history_tail": early_stop.history[-10:],
            "train_test_auc_gap": float(split_reports["train"]["auc"] - split_reports["test"]["auc"]),
            "test_oot_auc_gap": float(split_reports["test"]["auc"] - split_reports["oot"]["auc"]),
            "top_features": top_features,
            "splits": split_reports,
        }
        print(
            f"trained {model_name}: "
            f"train={len(train)} test={len(test)} oot={len(oot)} "
            f"test_auc={split_reports['test']['auc']:.4f} oot_auc={split_reports['oot']['auc']:.4f}",
            flush=True,
        )

    feature_cols = [col for col in B1_FEATURE_COLUMNS if col in data.columns]
    reports["dataset"] = {
        "rows": int(len(data)),
        "train_rows": int((data["split"] == "train").sum()),
        "test_rows": int((data["split"] == "test").sum()),
        "oot_rows": int((data["split"] == "oot").sum()),
        "train_symbols": int(data.loc[data["split"] == "train", "symbol"].nunique()),
        "test_symbols": int(data.loc[data["split"] == "test", "symbol"].nunique()),
        "oot_symbols": int(data.loc[data["split"] == "oot", "symbol"].nunique()),
        "date_min": str(data["date"].min().date()),
        "date_max": str(data["date"].max().date()),
        "configured_feature_count": len(B1_FEATURE_COLUMNS),
        "available_feature_count": len(feature_cols),
        "feature_missing_rate_top50": {
            col: float(rate)
            for col, rate in data[feature_cols].isna().mean().sort_values(ascending=False).head(50).items()
        },
        "source_policy": "tushare_only_daily_daily_basic",
        "model_type": "xgboost",
    }
    report_path = report_dir / "training_report.json"
    report_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"training report written: {report_path}", flush=True)
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Train B1 XGBoost research models from Tushare-only daily data.")
    parser.add_argument("--daily-dir", type=Path, default=PROJECT_ROOT / "data/raw/daily")
    parser.add_argument("--daily-basic-dir", type=Path, default=PROJECT_ROOT / "data/raw/daily_basic")
    parser.add_argument("--start", default="20200101")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="0 means auto initial workers: min(max-auto-workers, max(16, (cpu_count-buffer_cores)*worker_multiplier))",
    )
    parser.add_argument("--auto-worker-buffer-cores", type=int, default=2)
    parser.add_argument("--auto-worker-multiplier", type=int, default=6)
    parser.add_argument("--max-auto-workers", type=int, default=160)
    parser.add_argument("--adaptive-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-workers", type=int, default=32)
    parser.add_argument("--worker-step", type=int, default=16)
    parser.add_argument("--load-target", type=float, default=0.80)
    parser.add_argument("--load-hard-limit", type=float, default=1.20)
    parser.add_argument("--executor", choices=["threads", "processes"], default="threads")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--oot-start", default="2025-01-01")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--dataset-out", type=Path, default=PROJECT_ROOT / "data/features/b1/training_xgb_project_vars.parquet")
    parser.add_argument(
        "--reuse-dataset",
        action="store_true",
        help="Train from dataset-out without rebuilding market features.",
    )
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models/research/b1_xgb_project_vars")
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "reports/b1/research/xgb_project_vars")
    parser.add_argument("--select-k", type=int, default=0, help="0 means use all available factors; positive values enable SelectKBest")
    args = parser.parse_args()

    workers = resolve_worker_count(
        args.workers,
        args.auto_worker_buffer_cores,
        args.auto_worker_multiplier,
        args.max_auto_workers,
    )
    print(
        f"feature build workers={workers} adaptive={args.adaptive_workers} "
        f"(cpu={os.cpu_count() or 4}, buffer={args.auto_worker_buffer_cores}, "
        f"multiplier={args.auto_worker_multiplier}, max_auto={args.max_auto_workers})",
        flush=True,
    )
    if args.reuse_dataset:
        if not args.dataset_out.exists():
            raise FileNotFoundError(f"Training dataset does not exist: {args.dataset_out}")
        data = pd.read_parquet(args.dataset_out)
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        print(f"reusing training dataset: {args.dataset_out} rows={len(data)}", flush=True)
    else:
        data = build_dataset(
            args.daily_dir,
            args.start,
            workers=workers,
            limit=args.limit,
            executor_type=args.executor,
            adaptive_workers=args.adaptive_workers,
            min_workers=args.min_workers,
            max_workers=args.max_auto_workers,
            worker_step=args.worker_step,
            load_target=args.load_target,
            load_hard_limit=args.load_hard_limit,
        )
        data = merge_daily_basic_features(data, args.daily_basic_dir)
    data = assign_symbol_splits(data, args.oot_start, args.test_size, args.random_state)
    validate_label_splits(data)
    args.dataset_out.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(args.dataset_out, index=False)
    print(f"training dataset written: {args.dataset_out} rows={len(data)}", flush=True)
    select_k = None if args.select_k == 0 else args.select_k
    train_models(data, args.model_dir, args.report_dir, select_k=select_k)


if __name__ == "__main__":
    main()
