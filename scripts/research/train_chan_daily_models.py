#!/usr/bin/env python
"""Train model filters for Chan daily buy signals and evaluate candidate pools."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quant.features.market_sentiment import (
    build_limit_proxy_features,
    normalize_ts_code,
    read_top_list_features,
)
from quant.data import read_partitioned_symbol_file


DEFAULT_DAILY_DIR = PROJECT_ROOT / "data/stocks_daily"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports/chan_daily"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models/research/chan_daily"
DEFAULT_TOP_LIST_DIR = PROJECT_ROOT / "data/raw/moneyflow"
DEFAULT_DAILY_BASIC_DIR = PROJECT_ROOT / "data/raw/daily_basic"


BASE_FEATURES = [
    "chan_score",
    "chan_center_width",
    "chan_stroke_amplitude",
    "entry_gap_pct",
    "ret_1d",
    "ret_3d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "close_pos_20",
    "ma5_dist",
    "ma10_dist",
    "ma20_dist",
    "ma60_dist",
    "ma20_slope_5d",
    "volume_rel5",
    "volume_rel20",
    "volume_z20",
    "turnover_rate",
    "turnover_rate_ma20",
    "turnover_rate_rel20",
    "volatility_20d",
    "amount_rel20",
    "market_up_ratio",
    "market_median_ret_1d",
    "limit_up_count_proxy",
    "limit_up_ratio_proxy",
    "limit_down_ratio_proxy",
    "strong_up_ratio_proxy",
    "market_sentiment_5d",
    "market_panic_5d",
    "top_list_count",
    "top_net_amount_ratio",
    "top_net_rate",
    "db_turnover_rate",
    "db_turnover_rate_f",
    "db_volume_ratio",
    "db_total_mv_log",
    "db_circ_mv_log",
    "db_float_mv_ratio",
    "db_free_float_share_ratio",
    "db_pe_ttm_inv",
    "db_pb_inv",
    "db_ps_ttm_inv",
    "db_total_mv_pct_rank",
    "db_turnover_rate_pct_rank",
    "db_volume_ratio_pct_rank",
    "db_pb_pct_rank",
]


def read_daily_file(path: Path) -> pd.DataFrame:
    df = read_partitioned_symbol_file(path)
    if "trade_date" in df.columns:
        trade_dates = pd.to_datetime(df["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
        if "date" not in df.columns:
            df["date"] = trade_dates
        else:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").fillna(trade_dates)
    else:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").dropna(subset=["date"]).reset_index(drop=True)
    if "volume" not in df.columns and "vol" in df.columns:
        df = df.rename(columns={"vol": "volume"})
    elif "vol" in df.columns and ("volume" not in df.columns or df["volume"].isna().all()):
        df["volume"] = df["vol"]
    if "symbol" not in df.columns or df["symbol"].isna().all():
        df["symbol"] = df["ts_code"].astype(str) if "ts_code" in df.columns else path.stem
    df["ts_code"] = df["symbol"].map(normalize_ts_code)
    if "amount" not in df.columns or df["amount"].isna().all():
        df["amount"] = df.get("turnover", np.nan)
    return df


def add_stock_features(daily: pd.DataFrame) -> pd.DataFrame:
    out = daily.copy()
    close = out["close"].astype(float)
    volume = out["volume"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    out["ret_1d"] = close.pct_change(1) * 100
    out["ret_3d"] = close.pct_change(3) * 100
    out["ret_5d"] = close.pct_change(5) * 100
    out["ret_10d"] = close.pct_change(10) * 100
    out["ret_20d"] = close.pct_change(20) * 100
    for window in [5, 10, 20, 60]:
        ma = close.rolling(window, min_periods=max(3, window // 2)).mean()
        out[f"ma{window}"] = ma
        out[f"ma{window}_dist"] = close / ma.replace(0, np.nan) - 1
    out["ma20_slope_5d"] = out["ma20"].pct_change(5)
    high20 = high.rolling(20, min_periods=10).max()
    low20 = low.rolling(20, min_periods=10).min()
    out["close_pos_20"] = (close - low20) / (high20 - low20).replace(0, np.nan)
    out["volume_rel5"] = volume / volume.rolling(5, min_periods=3).mean().replace(0, np.nan)
    out["volume_rel20"] = volume / volume.rolling(20, min_periods=10).mean().replace(0, np.nan)
    out["volume_z20"] = (volume - volume.rolling(20, min_periods=10).mean()) / volume.rolling(20, min_periods=10).std().replace(0, np.nan)
    out["turnover_rate_ma20"] = out.get("turnover_rate", pd.Series(np.nan, index=out.index)).rolling(20, min_periods=10).mean()
    out["turnover_rate_rel20"] = out.get("turnover_rate", np.nan) / out["turnover_rate_ma20"].replace(0, np.nan)
    out["amount_rel20"] = out.get("amount", np.nan) / out.get("amount", pd.Series(np.nan, index=out.index)).rolling(20, min_periods=10).mean().replace(0, np.nan)
    out["volatility_20d"] = out["ret_1d"].rolling(20, min_periods=10).std()
    return out


def build_feature_dataset(
    candidates: pd.DataFrame,
    trades: pd.DataFrame,
    daily_dir: Path,
    daily_basic_dir: Path,
    top_list_dir: Path,
    start_date: str,
) -> pd.DataFrame:
    signal = candidates[candidates["signal_chan_daily_long"].eq(1)].copy()
    signal["date"] = pd.to_datetime(signal["date"])
    signal["symbol"] = signal["symbol"].astype(str)

    label_source = trades[trades["rule"].isin(["hold_5d_close", "hold_10d_close", "hold_20d_close"])].copy()
    label_source["symbol"] = label_source["symbol"].astype(str).str.zfill(6)
    labels = label_source.pivot_table(
        index=["symbol", "date"],
        columns="rule",
        values="return_pct",
        aggfunc="first",
    ).reset_index()
    labels.columns.name = None
    entry_context = (
        label_source.sort_values(["symbol", "date"])
        .drop_duplicates(["symbol", "date"])[["symbol", "date", "entry_gap_pct", "entry_open"]]
    )
    labels = labels.merge(entry_context, on=["symbol", "date"], how="left")
    labels["date"] = pd.to_datetime(labels["date"])
    signal = signal.merge(labels, on=["symbol", "date"], how="inner")
    signal["target_win10"] = (signal["hold_10d_close"] > 0).astype(int)
    signal["target_big10"] = (signal["hold_10d_close"] >= 3.0).astype(int)
    signal["target_good"] = ((signal["hold_10d_close"] >= 2.0) & (signal["hold_5d_close"] > -2.0)).astype(int)

    frames: list[pd.DataFrame] = []
    for symbol, group in signal.groupby("symbol"):
        path = daily_dir / f"{symbol}.parquet"
        daily = add_stock_features(read_daily_file(path))
        keep = [
            "symbol",
            "ts_code",
            "date",
            "ret_1d",
            "ret_3d",
            "ret_5d",
            "ret_10d",
            "ret_20d",
            "close_pos_20",
            "ma5_dist",
            "ma10_dist",
            "ma20_dist",
            "ma60_dist",
            "ma20_slope_5d",
            "volume_rel5",
            "volume_rel20",
            "volume_z20",
            "turnover_rate",
            "turnover_rate_ma20",
            "turnover_rate_rel20",
            "volatility_20d",
            "amount_rel20",
        ]
        merged = group.merge(daily[[col for col in keep if col in daily.columns]], on=["symbol", "date"], how="left")
        frames.append(merged)
    if not frames:
        raise RuntimeError("No feature rows built")

    data = pd.concat(frames, ignore_index=True)
    market = build_limit_proxy_features(daily_dir, start=start_date)
    if not market.empty:
        data = data.merge(market, on="date", how="left")

    top = read_top_list_features(top_list_dir, start=start_date)
    if not top.empty:
        data["ts_code"] = data["symbol"].map(normalize_ts_code)
        data = data.merge(top, on=["ts_code", "date"], how="left")

    daily_basic = read_daily_basic_features(daily_basic_dir, data["date"])
    if not daily_basic.empty:
        data["ts_code"] = data["symbol"].map(normalize_ts_code)
        data = data.merge(daily_basic, on=["ts_code", "date"], how="left")

    for col in ["top_list_count", "top_net_amount_ratio", "top_net_rate"]:
        if col not in data.columns:
            data[col] = np.nan
    data["top_list_count"] = data["top_list_count"].fillna(0)
    data = add_candidate_cross_section_ranks(data)
    data = assign_stock_time_splits(data)
    return data.replace([np.inf, -np.inf], np.nan)


def assign_stock_time_splits(
    data: pd.DataFrame,
    oot_start: str = "2025-01-01",
    test_symbol_mod: int = 5,
) -> pd.DataFrame:
    """Split by symbol before OOT, then reserve future dates as OOT.

    Train/test are isolated by stock to avoid the same symbol appearing in both
    model fitting and validation. OOT remains a pure time split.
    """
    out = data.copy()
    symbol_num = pd.to_numeric(out["symbol"].astype(str).str.extract(r"(\d+)")[0], errors="coerce").fillna(0).astype(int)
    pre_oot_test = symbol_num.mod(test_symbol_mod).eq(0)
    out["split"] = np.where(out["date"] >= pd.Timestamp(oot_start), "oot", np.where(pre_oot_test, "test", "train"))
    return out


def read_daily_basic_features(daily_basic_dir: Path, dates: pd.Series) -> pd.DataFrame:
    needed_dates = {
        pd.Timestamp(date).strftime("%Y%m%d")
        for date in pd.to_datetime(dates, errors="coerce").dropna().unique()
    }
    frames: list[pd.DataFrame] = []
    keep = [
        "ts_code",
        "trade_date",
        "turnover_rate",
        "turnover_rate_f",
        "volume_ratio",
        "pe_ttm",
        "pb",
        "ps_ttm",
        "total_share",
        "float_share",
        "free_share",
        "total_mv",
        "circ_mv",
    ]
    for trade_date in sorted(needed_dates):
        path = daily_basic_dir / f"{trade_date}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        present = [col for col in keep if col in df.columns]
        if {"ts_code", "trade_date"} <= set(present):
            frames.append(df[present].copy())
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    for col in out.columns:
        if col not in {"ts_code", "trade_date", "date"}:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["db_total_mv_log"] = np.log(out["total_mv"].replace(0, np.nan)) if "total_mv" in out.columns else np.nan
    out["db_circ_mv_log"] = np.log(out["circ_mv"].replace(0, np.nan)) if "circ_mv" in out.columns else np.nan
    if {"circ_mv", "total_mv"} <= set(out.columns):
        out["db_float_mv_ratio"] = out["circ_mv"] / out["total_mv"].replace(0, np.nan)
    if {"free_share", "float_share"} <= set(out.columns):
        out["db_free_float_share_ratio"] = out["free_share"] / out["float_share"].replace(0, np.nan)
    for source, target in [
        ("pe_ttm", "db_pe_ttm_inv"),
        ("pb", "db_pb_inv"),
        ("ps_ttm", "db_ps_ttm_inv"),
    ]:
        out[target] = 1 / out[source].replace(0, np.nan) if source in out.columns else np.nan
    rename = {
        "turnover_rate": "db_turnover_rate",
        "turnover_rate_f": "db_turnover_rate_f",
        "volume_ratio": "db_volume_ratio",
    }
    out = out.rename(columns=rename)
    keep_out = [
        "ts_code",
        "date",
        "db_turnover_rate",
        "db_turnover_rate_f",
        "db_volume_ratio",
        "db_total_mv_log",
        "db_circ_mv_log",
        "db_float_mv_ratio",
        "db_free_float_share_ratio",
        "db_pe_ttm_inv",
        "db_pb_inv",
        "db_ps_ttm_inv",
    ]
    return out[[col for col in keep_out if col in out.columns]].drop_duplicates(["ts_code", "date"])


def add_candidate_cross_section_ranks(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    rank_specs = [
        ("db_total_mv_log", "db_total_mv_pct_rank", True),
        ("db_turnover_rate", "db_turnover_rate_pct_rank", True),
        ("db_volume_ratio", "db_volume_ratio_pct_rank", True),
        ("db_pb_inv", "db_pb_pct_rank", True),
    ]
    for source, target, ascending in rank_specs:
        if source in out.columns:
            out[target] = out.groupby("date")[source].rank(pct=True, ascending=ascending)
        else:
            out[target] = np.nan
    return out


def train_model(data: pd.DataFrame, target: str, features: list[str], model_dir: Path) -> tuple[XGBClassifier, SimpleImputer, dict]:
    train = data[data["split"].eq("train")].copy()
    test = data[data["split"].eq("test")].copy()
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train[features])
    y_train = train[target].astype(int)
    x_test = imputer.transform(test[features])
    y_test = test[target].astype(int)
    model = XGBClassifier(
        n_estimators=240,
        max_depth=3,
        learning_rate=0.045,
        subsample=0.78,
        colsample_bytree=0.78,
        min_child_weight=20,
        reg_lambda=3.0,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=42,
        n_jobs=4,
    )
    model.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=False)
    metrics = {}
    for split in ["train", "test", "oot"]:
        part = data[data["split"].eq(split)].copy()
        if part.empty or part[target].nunique() < 2:
            continue
        pred = model.predict_proba(imputer.transform(part[features]))[:, 1]
        metrics[f"{split}_auc"] = float(roc_auc_score(part[target], pred))
        metrics[f"{split}_ap"] = float(average_precision_score(part[target], pred))
        metrics[f"{split}_rows"] = int(len(part))
        metrics[f"{split}_positive_rate"] = float(part[target].mean())
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "imputer": imputer, "features": features, "target": target}, model_dir / f"{target}.joblib")
    return model, imputer, metrics


def add_predictions(data: pd.DataFrame, models: dict[str, tuple[XGBClassifier, SimpleImputer]], features: list[str]) -> pd.DataFrame:
    out = data.copy()
    for target, (model, imputer) in models.items():
        out[f"pred_{target}"] = model.predict_proba(imputer.transform(out[features]))[:, 1]
    return out


def summarize_pool(df: pd.DataFrame, name: str) -> dict:
    returns = df["hold_10d_close"].astype(float)
    gross_profit = returns[returns > 0].sum()
    gross_loss = -returns[returns < 0].sum()
    return {
        "pool": name,
        "rows": int(len(df)),
        "avg_return_10d": float(returns.mean()),
        "median_return_10d": float(returns.median()),
        "win_rate_10d": float((returns > 0).mean()),
        "big_win_rate_10d": float((returns >= 3).mean()),
        "profit_factor_10d": float(gross_profit / gross_loss) if gross_loss > 0 else np.inf,
        "avg_return_5d": float(df["hold_5d_close"].mean()),
        "avg_return_20d": float(df["hold_20d_close"].mean()),
    }


def evaluate_rules(data: pd.DataFrame) -> pd.DataFrame:
    ranked = data.copy()
    for col in ["pred_target_good", "pred_target_big10", "pred_target_win10"]:
        ranked[f"{col}_rank"] = ranked.groupby("split")[col].rank(pct=True)
    rules = {
        "baseline_all": pd.Series(True, index=ranked.index),
        "chan_score_ge95": ranked["chan_score"] >= 95,
        "buy3_score_ge95": (ranked["chan_signal_name"] == "三买确认") & (ranked["chan_score"] >= 95),
        "buy3_score_ge95_gap_0_3": (ranked["chan_signal_name"] == "三买确认") & (ranked["chan_score"] >= 95) & ranked["entry_gap_pct"].between(0, 3),
        "model_good_top30": ranked["pred_target_good_rank"] >= 0.70,
        "model_good_top20": ranked["pred_target_good_rank"] >= 0.80,
        "model_good_top10": ranked["pred_target_good_rank"] >= 0.90,
        "model_big_good_top20": (ranked["pred_target_big10_rank"] >= 0.80) & (ranked["pred_target_good_rank"] >= 0.80),
        "buy3_model_good_top30": (ranked["chan_signal_name"] == "三买确认") & (ranked["pred_target_good_rank"] >= 0.70),
        "buy3_model_good_top20": (ranked["chan_signal_name"] == "三买确认") & (ranked["pred_target_good_rank"] >= 0.80),
        "buy3_score_ge95_model_top50": (ranked["chan_signal_name"] == "三买确认") & (ranked["chan_score"] >= 95) & (ranked["pred_target_good_rank"] >= 0.50),
        "buy3_score_ge95_model_top30_gap_le3": (
            (ranked["chan_signal_name"] == "三买确认")
            & (ranked["chan_score"] >= 95)
            & (ranked["pred_target_good_rank"] >= 0.70)
            & (ranked["entry_gap_pct"] <= 3)
        ),
    }
    rows = []
    for split in ["train", "test", "oot"]:
        part = ranked[ranked["split"].eq(split)]
        for name, mask in rules.items():
            selected = part[mask.reindex(part.index).fillna(False)]
            if len(selected) < 50:
                continue
            rows.append({"split": split, **summarize_pool(selected, name)})
    return pd.DataFrame(rows).sort_values(["split", "avg_return_10d", "win_rate_10d"], ascending=[True, False, False])


def analyze_cases(data: pd.DataFrame, output_dir: Path) -> None:
    out = data.copy()
    out["pred_target_good_rank"] = out.groupby("split")["pred_target_good"].rank(pct=True)
    selected = out[
        (out["split"].eq("oot"))
        & (out["chan_signal_name"].eq("三买确认"))
        & (out["pred_target_good_rank"] >= 0.80)
    ].copy()
    winners = selected.sort_values("hold_10d_close", ascending=False).head(80)
    losers = selected.sort_values("hold_10d_close", ascending=True).head(80)
    winners.to_csv(output_dir / "chan_model_oot_winner_cases.csv", index=False)
    losers.to_csv(output_dir / "chan_model_oot_loser_cases.csv", index=False)
    compare_cols = [
        "chan_score",
        "entry_gap_pct",
        "ret_5d",
        "ret_20d",
        "close_pos_20",
        "volume_rel20",
        "market_up_ratio",
        "limit_up_ratio_proxy",
        "market_sentiment_5d",
        "top_net_amount_ratio",
    ]
    rows = []
    for col in compare_cols:
        if col in selected.columns:
            rows.append(
                {
                    "feature": col,
                    "winners_median": winners[col].median(),
                    "losers_median": losers[col].median(),
                    "winner_minus_loser": winners[col].median() - losers[col].median(),
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "chan_model_case_feature_contrast.csv", index=False)


def write_feature_importance(
    models: dict[str, tuple[XGBClassifier, SimpleImputer]],
    features: list[str],
    output_dir: Path,
) -> None:
    rows = []
    for target, (model, _) in models.items():
        for feature, gain in zip(features, model.feature_importances_):
            rows.append({"target": target, "feature": feature, "importance": float(gain)})
    pd.DataFrame(rows).sort_values(["target", "importance"], ascending=[True, False]).to_csv(
        output_dir / "chan_model_feature_importance.csv",
        index=False,
    )


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_parquet(args.report_dir / "chan_daily_candidates.parquet")
    trades = pd.read_csv(args.report_dir / "chan_daily_light_trades.csv", dtype={"symbol": str}, parse_dates=["date", "exit_date"])
    trades["symbol"] = trades["symbol"].astype(str).str.zfill(6)
    dataset_path = args.output_dir / "chan_model_dataset.parquet"
    if dataset_path.exists() and not args.force_rebuild:
        data = pd.read_parquet(dataset_path)
    else:
        data = build_feature_dataset(candidates, trades, args.daily_dir, args.daily_basic_dir, args.top_list_dir, args.start_date)
        data.to_parquet(dataset_path, index=False)
        data.to_csv(args.output_dir / "chan_model_dataset_sample.csv", index=False)

    train_part = data[data["split"].eq("train")]
    features = [
        col for col in BASE_FEATURES
        if col in data.columns and train_part[col].notna().any()
    ]
    models = {}
    model_metrics = {}
    for target in ["target_win10", "target_big10", "target_good"]:
        model, imputer, metrics = train_model(data, target, features, args.model_dir)
        models[target] = (model, imputer)
        model_metrics[target] = metrics
    scored = add_predictions(data, models, features)
    scored.to_parquet(args.output_dir / "chan_model_scored_candidates.parquet", index=False)
    scored.to_csv(args.output_dir / "chan_model_scored_candidates.csv", index=False)
    evaluation = evaluate_rules(scored)
    evaluation.to_csv(args.output_dir / "chan_model_filter_evaluation.csv", index=False)
    analyze_cases(scored, args.output_dir)
    write_feature_importance(models, features, args.output_dir)
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "features": features,
        "rows": int(len(scored)),
        "model_metrics": model_metrics,
        "top_evaluation": evaluation.head(20).to_dict("records"),
    }
    (args.output_dir / "chan_model_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(model_metrics, ensure_ascii=False, indent=2))
    print(evaluation.head(30).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-dir", type=Path, default=DEFAULT_DAILY_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR / "model_filter")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--top-list-dir", type=Path, default=DEFAULT_TOP_LIST_DIR)
    parser.add_argument("--daily-basic-dir", type=Path, default=DEFAULT_DAILY_BASIC_DIR)
    parser.add_argument("--start-date", default="2015-01-01")
    parser.add_argument("--force-rebuild", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
